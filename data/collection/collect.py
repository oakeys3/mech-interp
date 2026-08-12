"""Main dataset collection script.

Runs the primary model over EvalPlus problems in a multi-round self-correction
loop and appends one JSONL record per attempt. See ``data/SCHEMA.md`` for the
record format.

Two design constraints drive the structure:

1. **Colab disconnects.** Production collection runs on a free Colab GPU, whose
   sessions die after a few hours. Records are appended as each attempt
   completes, and ``load_completed_keys`` lets a restart skip finished work.

2. **Testability without a GPU.** The trajectory loop takes the generator and
   the grading function as arguments rather than constructing them, so tests
   drive it with stubs. Nothing in ``run_trajectory`` imports torch.

Grading is POSIX-only (see ``harness.py``), so end-to-end runs happen under
WSL or on Linux.

Usage:
    py -3.12 -m data.collection.collect --subset humaneval --limit 5 --dry-run
    python -m data.collection.collect --subset humaneval --out data/raw/humaneval.jsonl
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from data.collection.harness import GradeResult, grade, load_problem_suite, prepare_solution

Messages = list[dict[str, str]]

# Only the tail of a failure message is useful, and long feedback crowds out
# the problem statement in a 1.5B model's context.
MAX_FEEDBACK_CHARS = 800


class Generator(Protocol):
    """Anything that can turn chat messages into a completion."""

    def chat(self, messages: Messages) -> str:
        """Generate a response to a chat message list.

        Args:
            messages: OpenAI-style ``{"role", "content"}`` dicts.

        Returns:
            The model's raw text response.
        """
        ...


GradeFn = Callable[[str, dict, dict, str], GradeResult]


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

def initial_messages(problem: dict[str, Any]) -> Messages:
    """Build the first-attempt prompt for a problem.

    Args:
        problem: An EvalPlus problem dict.

    Returns:
        Chat messages asking for a complete solution in a code fence.
    """
    return [
        {
            "role": "user",
            "content": (
                "Complete this Python function. Return only the full function "
                "in a ```python code block, including any imports.\n\n"
                + problem["prompt"]
            ),
        }
    ]


def correction_messages(
    problem: dict[str, Any], previous_generation: str, feedback: str
) -> Messages:
    """Build a correction prompt showing the model its failed attempt.

    The previous attempt is replayed as an assistant turn so the model sees
    its own output in context rather than as quoted text.

    Args:
        problem: An EvalPlus problem dict.
        previous_generation: The raw generation that failed.
        feedback: Failure description from ``build_feedback``.

    Returns:
        Chat messages requesting a corrected solution.
    """
    return [
        *initial_messages(problem),
        {"role": "assistant", "content": previous_generation},
        {
            "role": "user",
            "content": (
                f"That solution failed the tests:\n\n{feedback}\n\n"
                "Return a corrected full function in a ```python code block."
            ),
        },
    ]


def review_messages(problem: dict[str, Any], previous_generation: str) -> Messages:
    """Build the review prompt used to elicit F4 (spurious correction).

    The wording deliberately gives the model an explicit out — it is told it
    may return the solution unchanged. A prompt that implied a bug existed
    would manufacture F4 rather than observe it, and the resulting label would
    say more about the prompt than about the model.

    Args:
        problem: An EvalPlus problem dict.
        previous_generation: The raw generation that passed.

    Returns:
        Chat messages requesting a review of passing code.
    """
    return [
        *initial_messages(problem),
        {"role": "assistant", "content": previous_generation},
        {
            "role": "user",
            "content": (
                "Review your solution for correctness and edge cases. If it is "
                "already correct, return it unchanged. Return the full function "
                "in a ```python code block."
            ),
        },
    ]


# --------------------------------------------------------------------------- #
# Failure feedback
# --------------------------------------------------------------------------- #

def build_feedback(
    problem: dict[str, Any], expected_output: dict[str, Any], result: GradeResult
) -> str:
    """Describe a failure in terms the model can act on.

    EvalPlus reports per-input pass/fail flags rather than a traceback (see
    RESEARCH_LOG.md 2026-07-09), so feedback is reconstructed as the first
    failing input and its expected output. This is weaker than a traceback for
    crashes — the model is not told which exception was raised.

    Args:
        problem: An EvalPlus problem dict.
        expected_output: That problem's groundtruth entry.
        result: The grading outcome.

    Returns:
        A short failure description, truncated to ``MAX_FEEDBACK_CHARS``.
    """
    for suite in ("base", "plus"):
        status = result.base_status if suite == "base" else result.plus_status
        if status == "pass":
            continue
        if status == "timeout":
            return "Your solution timed out. It is too slow or does not terminate."

        details = result.base_details if suite == "base" else result.plus_details
        inputs = problem[f"{suite}_input"]
        expected = expected_output[suite]
        for index, ok in enumerate(details or []):
            if ok:
                continue
            if index >= len(inputs):
                break
            message = f"Wrong result for input {inputs[index]!r}."
            if index < len(expected):
                message += f" Expected {expected[index]!r}."
            return message[:MAX_FEEDBACK_CHARS]
        return "Your solution did not pass the tests."
    return ""


# --------------------------------------------------------------------------- #
# Trajectory loop
# --------------------------------------------------------------------------- #

def run_trajectory(
    problem: dict[str, Any],
    expected_output: dict[str, Any],
    generator: Generator,
    grade_fn: GradeFn,
    *,
    subset: str,
    sample: int = 0,
    correction_rounds: int = 3,
    review_passing: bool = True,
    model: str = "unknown",
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    """Run one problem through the self-correction loop.

    The loop generates, grades, and — on failure — feeds the failure back for
    up to ``correction_rounds`` retries. If the trajectory ends in a pass and
    ``review_passing`` is set, one extra review round runs; that round is the
    only place F4 can occur.

    Args:
        problem: An EvalPlus problem dict.
        expected_output: That problem's groundtruth entry.
        generator: Anything satisfying the ``Generator`` protocol.
        grade_fn: Callable with ``harness.grade``'s signature.
        subset: ``humaneval`` or ``mbpp``.
        sample: Index of this sample when a problem is run more than once.
        correction_rounds: Maximum correction attempts after the first.
        review_passing: Whether to add a review round after a pass.
        model: Model id, recorded for provenance.
        temperature: Sampling temperature, recorded for provenance.

    Returns:
        One record per attempt, in order, with ``final`` set on the last.
    """
    trajectory_id = uuid.uuid4().hex
    records: list[dict[str, Any]] = []
    messages = initial_messages(problem)
    stage = "initial"

    for round_index in range(correction_rounds + 1):
        raw_generation = generator.chat(messages)
        solution = prepare_solution(raw_generation)
        result = grade_fn(subset, problem, expected_output, solution)
        feedback = "" if result.passed else build_feedback(problem, expected_output, result)

        records.append(
            {
                "trajectory_id": trajectory_id,
                "task_id": problem["task_id"],
                "subset": subset,
                "sample": sample,
                "round": round_index,
                "stage": stage,
                "prompt_messages": messages,
                "raw_generation": raw_generation,
                "solution": solution,
                "base_status": result.base_status,
                "plus_status": result.plus_status,
                "passed": result.passed,
                "feedback": feedback,
                "final": False,
                "model": model,
                "temperature": temperature,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        if result.passed:
            break
        if round_index == correction_rounds:
            break
        messages = correction_messages(problem, raw_generation, feedback)
        stage = "correction"

    if review_passing and records[-1]["passed"]:
        messages = review_messages(problem, records[-1]["raw_generation"])
        raw_generation = generator.chat(messages)
        solution = prepare_solution(raw_generation)
        result = grade_fn(subset, problem, expected_output, solution)
        records.append(
            {
                **records[-1],
                "round": records[-1]["round"] + 1,
                "stage": "review",
                "prompt_messages": messages,
                "raw_generation": raw_generation,
                "solution": solution,
                "base_status": result.base_status,
                "plus_status": result.plus_status,
                "passed": result.passed,
                "feedback": (
                    "" if result.passed else build_feedback(problem, expected_output, result)
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    records[-1]["final"] = True
    return records


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #

def load_completed_keys(path: Path) -> set[tuple[str, int]]:
    """Find which ``(task_id, sample)`` pairs already finished.

    Only trajectories with a ``final`` record count as complete. A trajectory
    cut short by a dead session has no such record and is re-run. A truncated
    final line is expected after an abrupt disconnect and is skipped rather
    than treated as corruption.

    Args:
        path: JSONL file written by a previous run. May not exist.

    Returns:
        The set of completed ``(task_id, sample)`` keys.
    """
    if not path.exists():
        return set()

    completed: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("final"):
                completed.add((record["task_id"], record.get("sample", 0)))
    return completed


def append_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Append attempt records to the JSONL log and flush to disk.

    Flushing matters: an unflushed buffer is lost when a Colab session is
    killed, which would silently undo the checkpointing.

    Args:
        path: Destination JSONL file; parent directories are created.
        records: Attempt records to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
        handle.flush()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def sort_task_ids(task_ids: list[str]) -> list[str]:
    """Order task ids numerically rather than lexicographically.

    EvalPlus ids look like ``HumanEval/10``, so a plain string sort puts
    problem 10 before problem 2. That would silently make ``--limit 25`` a
    different problem set than the spike's first 25.

    Args:
        task_ids: Unordered EvalPlus task ids.

    Returns:
        The ids sorted by prefix, then by trailing integer.
    """

    def key(task_id: str) -> tuple[str, int]:
        prefix, _, number = task_id.rpartition("/")
        return prefix, int(number) if number.isdigit() else 0

    return sorted(task_ids, key=key)


def build_generator(model: str, temperature: float, max_new_tokens: int) -> Generator:
    """Construct the real HuggingFace-backed generator.

    Imported lazily so that the loop, the tests, and ``--dry-run`` never pay
    for torch.

    Args:
        model: HuggingFace model id.
        temperature: 0 means greedy decoding; above 0 enables sampling.
        max_new_tokens: Generation length cap.

    Returns:
        A ready-to-use Generator.
    """
    from data.collection.model import HuggingFaceGenerator

    return HuggingFaceGenerator(model, temperature, max_new_tokens)


def main() -> None:
    """Run collection over an EvalPlus subset, resuming any previous run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=["humaneval", "mbpp"], required=True)
    parser.add_argument("--out", default=None, help="JSONL output path")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--correction-rounds", type=int, default=3)
    parser.add_argument(
        "--samples-per-problem",
        type=int,
        default=1,
        help="Trajectories per problem; >1 requires temperature > 0",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only the first N problems")
    parser.add_argument("--no-review", action="store_true", help="Skip F4 review rounds")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the work that would run, without loading the model",
    )
    args = parser.parse_args()

    if args.samples_per_problem > 1 and args.temperature <= 0:
        parser.error("--samples-per-problem > 1 needs --temperature > 0 (greedy repeats)")

    out_path = Path(args.out or f"data/raw/{args.subset}_trajectories.jsonl")
    problems, expected = load_problem_suite(args.subset)
    task_ids = sort_task_ids(list(problems))[: args.limit]

    completed = load_completed_keys(out_path)
    todo = [
        (task_id, sample)
        for task_id in task_ids
        for sample in range(args.samples_per_problem)
        if (task_id, sample) not in completed
    ]

    print(f"problems: {len(task_ids)}  already done: {len(completed)}  to run: {len(todo)}")
    if args.dry_run:
        for task_id, sample in todo[:10]:
            print(f"  would run {task_id} sample {sample}")
        return

    generator = build_generator(args.model, args.temperature, args.max_new_tokens)
    for index, (task_id, sample) in enumerate(todo, start=1):
        records = run_trajectory(
            problems[task_id],
            expected[task_id],
            generator,
            grade,
            subset=args.subset,
            sample=sample,
            correction_rounds=args.correction_rounds,
            review_passing=not args.no_review,
            model=args.model,
            temperature=args.temperature,
        )
        append_records(out_path, records)
        outcome = "PASS" if records[-1]["passed"] else "FAIL"
        stages = "->".join(record["stage"] for record in records)
        print(f"[{index}/{len(todo)}] {outcome:4} {task_id} sample {sample}  {stages}")

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
