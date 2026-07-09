# Research Log

Running log of decisions and findings, newest entry last. Every major
methodological choice gets an entry. Friction with upstream libraries is
tagged `[OSS CANDIDATE]` for later review.

---

## 2026-07-07 — Repo initialized; spike findings imported

**Repo scaffolded** to the full project layout (data/probing/patching/analysis,
config, tests, CI). Placeholder modules document their intended role; nothing
past Phase 0 is implemented, by design — Phase 1 infrastructure waits until the
Phase 0 decision gates close.

**Spike findings carried forward** (from `spike/spike.py`, now frozen):

- **Pythia-1.4B eliminated.** In self-correction mode it echoes tracebacks back
  as comments instead of generating corrected code. The failure mode is
  illegible for probing — there is no correction attempt to label F3/F4.
- **Qwen2.5-1.5B-Instruct is the candidate primary model.** Went 2/3 on first
  attempts with one genuine, traceable F2 → F3/F4 progression.
- **Self-test stripping is mandatory.** Qwen appends its own `assert` blocks to
  solutions; left in place they pollute test results. The fix lives in the
  spike and must carry into `data/collection/harness.py`.
  `[OSS CANDIDATE]` — if this is a general instruct-model problem on EvalPlus,
  the stripping logic may be worth upstreaming.
- **EvalPlus over raw HumanEval.** HumanEval's weak test coverage lets
  mathematically incorrect solutions pass, which would inject label noise into
  the probing signal. EvalPlus's augmented suites close this.
- `[OSS CANDIDATE]` — `transformers` 5.x `apply_chat_template` returns a dict;
  the `**inputs` unpacking pattern for generation is underdocumented and cost
  debugging time in the spike.

**Methodological principle confirmed:** disposable spike before infrastructure.
Repeat at every major phase transition.

**Open blocker (Phase 0 gate):** the `--n 25` run of Qwen on HumanEval is
pending. Qwen is not committed until it shows a usable failure-rate
distribution (target 30–60%). Below 20% → need a harder benchmark or higher
temperature; above 80% → model too weak to show meaningful self-correction
dynamics.

Next: run `py -3.12 spike/spike.py --model Qwen/Qwen2.5-1.5B-Instruct
--instruct --dataset humaneval --n 25`, inspect the failure-rate distribution
and per-problem transcripts, then close the model/benchmark decision gates.

---

## 2026-07-07 (late) — n=25 confirmation run results

Run completed: Qwen2.5-1.5B-Instruct, HumanEval, 25 problems, one correction
round, CPU (~2.3 min/problem). Transcript: `spike/spike_humaneval_qwen25_n25.jsonl`
(gitignored; regenerate with the command above).

**Headline numbers:**

- Attempt-1 pass rate 18/25 (**28% failure rate**) — nominally just under the
  30–60% target band, but with n=25 the 95% CI is roughly 12–49%, and EvalPlus's
  stricter test suites will only push the effective failure rate up. Treating
  this as inside the band.
- Post-correction pass rate: still 18/25 — **0/7 corrections succeeded**.

**Per-failure breakdown** (error type on attempt 1 → did the correction change
the logic? → error type on attempt 2):

| Problem | err1 | logic changed? | err2 |
|---------|------|----------------|------|
| HumanEval/1 | AssertionError | no — re-emitted same code + self-tests | AssertionError |
| HumanEval/6 | RecursionError | yes | RecursionError |
| HumanEval/9 | AssertionError | yes | AssertionError |
| HumanEval/10 | IndexError | yes | IndexError |
| HumanEval/17 | AssertionError | yes | **SyntaxError (regression)** |
| HumanEval/18 | AssertionError | yes | AssertionError |
| HumanEval/19 | ValueError | yes | AssertionError |

**What this means:**

- Unlike Pythia, Qwen genuinely engages with tracebacks: 6/7 corrections made
  real logic changes. The failure modes are legible — F3 examples (failed
  corrections) will be plentiful, including at least one clean regression
  (HumanEval/17: wrong-output → syntax error).
- **Open risk:** 0/7 correction successes means the "correction succeeds"
  outcome is so far unobserved. If that holds at scale, probing F3 against
  successful corrections has no contrast class. Mitigations available: the real
  collection loop uses 3 correction rounds (spike used 1), MBPP problems are
  easier, and n=25 is small. Watch this in Phase 1; revisit if success stays ~0%.
- **Design note for Phase 1:** F4 (spurious correction of passing code) cannot
  occur in the spike's loop, which only triggers correction on failure. To
  collect F4 examples, `collect.py` must also ask the model to review/refine a
  subset of *passing* attempts.
- Self-test assertion stripping confirmed necessary again — gen2 outputs came
  back with appended `assert` blocks.

**Gate recommendation (decision pending):** confirm Qwen2.5-1.5B-Instruct as
primary model and EvalPlus as benchmark. Failure rate is usable, failure modes
are probe-legible, and no candidate model at this size is obviously better.

---

## 2026-07-08 — PHASE 0 CLOSED: model and benchmark locked

Decision: **Qwen2.5-1.5B-Instruct** as primary model, **EvalPlus**
(HumanEval+ / MBPP+) as benchmark suite. Evidence in the 2026-07-07 entries;
config markers removed in `configs/experiment_config.yaml`. No fallback model
carried — if Qwen fails at Phase 1 scale, that is a new decision, not a
pre-made one.

Phase 1 (dataset collection) begins. First target: `data/collection/harness.py`
with the self-test stripping fix ported from the spike, under unit test —
stripping was re-confirmed necessary in the n=25 run (gen2 outputs appended
`assert` blocks again).
