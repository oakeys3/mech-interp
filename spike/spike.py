"""
Feasibility spike for "Agentic Failures as Interpretability Problems".

PURPOSE
  Decide, on evidence, whether Pythia-1.4B (base) or Qwen2.5-1.5B-Instruct
  can produce a usable distribution of code-generation failures with a
  working self-correction loop, BEFORE building the real harness or
  collecting the 500-instance dataset.

  This is throwaway code. Its only outputs are:
    (1) first-attempt and post-correction pass rates per model, and
    (2) a JSONL transcript of every attempt, for manual inspection.

  The numbers tell you about the capability floor.
  Reading the transcripts by hand tells you whether the F1-F4 taxonomy is
  real and whether the model engages with an error trace at all.

SAFETY
  This executes model-generated code. Run it on a throwaway machine or in a
  container. Execution is isolated to a subprocess with a timeout, which is
  the minimum, not real sandboxing.

USAGE
  python spike.py --model EleutherAI/pythia-1.4b            --dataset humaneval --n 25
  python spike.py --model Qwen/Qwen2.5-1.5B-Instruct --instruct --dataset humaneval --n 25
  python spike.py --model EleutherAI/pythia-1.4b            --dataset mbpp      --n 25
  python spike.py --model Qwen/Qwen2.5-1.5B-Instruct --instruct --dataset mbpp --n 25
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

TIMEOUT_S = 10
MAX_NEW_TOKENS = 512
STOP_MARKERS = ["\ndef ", "\nclass ", "\nif __name__", "\nassert ", "\nprint("]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_problems(dataset, n):
    if dataset == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test").select(range(n))
        return [
            {
                "task_id": r["task_id"],
                "kind": "humaneval",
                "prompt": r["prompt"],          # signature + docstring
                "test": r["test"],              # defines check(...)
                "entry_point": r["entry_point"],
            }
            for r in ds
        ]
    if dataset == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", split="test").select(range(n))
        return [
            {
                "task_id": f"mbpp/{r['task_id']}",
                "kind": "mbpp",
                "text": r["text"],
                "setup": (r.get("test_setup_code") or ""),
                "tests": r["test_list"],
            }
            for r in ds
        ]
    raise ValueError(dataset)


# --------------------------------------------------------------------------- #
# Model wrapper (greedy decoding for reproducible pass@1)
# --------------------------------------------------------------------------- #
class LM:
    def __init__(self, name, instruct, device):
        self.tok = AutoTokenizer.from_pretrained(name)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=dtype
        ).to(device)
        self.model.eval()
        self.instruct = instruct
        self.device = device
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    @torch.no_grad()
    def _gen(self, inputs):
        # inputs is a dict-like with input_ids (+ attention_mask)
        out = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=self.tok.pad_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        return self.tok.decode(out[0][prompt_len:], skip_special_tokens=True)

    def complete(self, prompt):  # base path: raw continuation
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        return self._gen(inputs)

    def chat(self, messages):    # instruct path: chat template
        inputs = self.tok.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(self.device)
        return self._gen(inputs)


# --------------------------------------------------------------------------- #
# Code extraction
# --------------------------------------------------------------------------- #
def truncate_at_markers(text):
    cut = len(text)
    for m in STOP_MARKERS:
        i = text.find(m)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def extract_block(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


# Instruct models often append their own (self-graded) tests after the function.
# Strip them so we grade the function against the benchmark's tests, not the
# model's. Cuts at the first top-level test-scaffolding line; keeps helper defs.
_SELFTEST = ["\nassert ", "\nif __name__", "\n# Test", "\n# test", "\nprint("]


def strip_self_tests(code):
    cut = len(code)
    for m in _SELFTEST:
        i = code.find(m)
        if i != -1:
            cut = min(cut, i)
    return code[:cut].rstrip()


# --------------------------------------------------------------------------- #
# Prompt + program construction. NOTE the asymmetry: base completes a body,
# instruct writes a whole function. This is intentional, not a bug.
# --------------------------------------------------------------------------- #
def first_prompt(lm, p):
    if p["kind"] == "humaneval":
        if lm.instruct:
            return [{"role": "user", "content":
                     "Complete this Python function. Return only the full "
                     "function in a ```python code block, including any imports.\n\n"
                     + p["prompt"]}]
        return p["prompt"]
    # mbpp
    tests = "\n".join(p["tests"])
    body = (f"Write a Python function for this task. It must pass these tests.\n\n"
            f"Task: {p['text']}\n\nTests:\n{tests}\n")
    if lm.instruct:
        return [{"role": "user", "content": body + "\nReturn only the function in a "
                 "```python code block."}]
    return body + "\n```python\n"  # nudge the base model toward code


def build_program(lm, p, raw_generation):
    if p["kind"] == "humaneval":
        if lm.instruct:
            code = strip_self_tests(extract_block(raw_generation))
            return f"{code}\n\n{p['test']}\n\ncheck({p['entry_point']})\n"
        body = truncate_at_markers(raw_generation)
        return f"{p['prompt']}{body}\n\n{p['test']}\n\ncheck({p['entry_point']})\n"
    # mbpp
    code = strip_self_tests(extract_block(raw_generation)) if lm.instruct else truncate_at_markers(raw_generation)
    setup = (p["setup"] + "\n") if p["setup"] else ""
    return f"{setup}{code}\n\n" + "\n".join(p["tests"]) + "\n"


def correction_prompt(lm, p, prev_generation, error):
    err = error.strip()[-800:]  # tail of the traceback is the informative part
    if lm.instruct:
        problem = p["prompt"] if p["kind"] == "humaneval" else p["text"]
        return [
            {"role": "user", "content": "Solve this task:\n\n" + problem},
            {"role": "assistant", "content": prev_generation},
            {"role": "user", "content":
             f"That failed when executed:\n\n{err}\n\nReturn a corrected full "
             "function in a ```python code block."},
        ]
    # base: append the error as a comment and ask for a corrected version
    head = p["prompt"] if p["kind"] == "humaneval" else p["text"]
    commented = "\n".join("# " + line for line in err.splitlines())
    return (f"{head}{prev_generation}\n\n# The code above failed:\n{commented}\n"
            "# Corrected version:\n")


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def run_program(src):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path], capture_output=True, text=True, timeout=TIMEOUT_S
        )
        if proc.returncode == 0:
            return True, ""
        return False, proc.stderr
    except subprocess.TimeoutExpired:
        return False, f"TimeoutExpired after {TIMEOUT_S}s"
    finally:
        os.unlink(path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--instruct", action="store_true")
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    global MAX_NEW_TOKENS
    MAX_NEW_TOKENS = args.max_new_tokens

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = args.out or f"spike_{args.dataset}_{args.model.split('/')[-1]}.jsonl"

    problems = load_problems(args.dataset, args.n)
    lm = LM(args.model, args.instruct, device)

    n_pass1 = n_pass2 = 0
    with open(out_path, "w") as fh:
        for i, p in enumerate(problems):
            gen1 = lm.chat(first_prompt(lm, p)) if lm.instruct else lm.complete(first_prompt(lm, p))
            ok1, err1 = run_program(build_program(lm, p, gen1))

            rec = {"task_id": p["task_id"], "gen1": gen1, "pass1": ok1, "err1": err1}
            n_pass1 += ok1

            if not ok1:
                cp = correction_prompt(lm, p, gen1, err1)
                gen2 = lm.chat(cp) if lm.instruct else lm.complete(cp)
                ok2, err2 = run_program(build_program(lm, p, gen2))
                rec.update({"gen2": gen2, "pass2": ok2, "err2": err2})
                n_pass2 += ok2

            fh.write(json.dumps(rec) + "\n")
            status = "PASS" if ok1 else ("FIX " if rec.get("pass2") else "FAIL")
            print(f"[{i+1:>3}/{len(problems)}] {status}  {p['task_id']}")

    total = len(problems)
    print("\n----- summary -----")
    print(f"model            : {args.model}{'  (instruct)' if args.instruct else ''}")
    print(f"dataset / n      : {args.dataset} / {total}")
    print(f"attempt-1 pass   : {n_pass1}/{total}  ({n_pass1/total:.0%})")
    print(f"after correction : {(n_pass1+n_pass2)}/{total}  ({(n_pass1+n_pass2)/total:.0%})")
    print(f"transcripts      : {out_path}")
    print("\nNow read the FAIL transcripts by hand. The numbers are the floor check;")
    print("the transcripts tell you whether F1-F4 is real and whether correction happens.")


if __name__ == "__main__":
    main()
