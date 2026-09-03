"""Unit tests for the collection loop.

The loop is exercised with a stub generator and a stub grader, so these tests
need neither a GPU, a model download, nor a POSIX-only EvalPlus sandbox.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from data.collection.collect import (
    append_records,
    baseline_fraction,
    build_feedback,
    correction_messages,
    initial_messages,
    load_completed_keys,
    review_messages,
    run_trajectory,
    sort_task_ids,
)
from data.collection.harness import GradeResult

PROBLEM: dict[str, Any] = {
    "task_id": "HumanEval/0",
    "prompt": "def has_close_elements(numbers, threshold):\n    ",
    "base_input": [[[1.0, 2.0], 0.5], [[1.0, 2.8], 0.3]],
    "plus_input": [[[3.0, 3.1], 0.2]],
}
EXPECTED: dict[str, Any] = {"base": [True, False], "plus": [True]}

PASS = GradeResult("pass", "pass", (1, 1), (1,))
FAIL = GradeResult("fail", "fail", (1, 0), (0,))
TIMEOUT = GradeResult("timeout", "timeout", None, None)


class FakeGenerator:
    """Returns scripted responses and remembers the prompts it was given."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.seen: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.seen.append(messages)
        if not self.responses:
            raise AssertionError("generator called more times than scripted")
        return self.responses.pop(0)


class FakeGrader:
    """Returns scripted grading outcomes in order."""

    def __init__(self, results: list[GradeResult]) -> None:
        self.results = list(results)

    def __call__(self, subset, problem, expected, solution) -> GradeResult:
        if not self.results:
            raise AssertionError("grader called more times than scripted")
        return self.results.pop(0)


def code(body: str) -> str:
    """Wrap a snippet in a python fence the way the model would."""
    return f"```python\n{body}\n```"


def run(responses: list[str], results: list[GradeResult], **kwargs):
    """Run a trajectory with stubs, returning (records, generator)."""
    generator = FakeGenerator(responses)
    records = run_trajectory(
        PROBLEM,
        EXPECTED,
        generator,
        FakeGrader(results),
        subset="humaneval",
        **kwargs,
    )
    return records, generator


# --------------------------------------------------------------------------- #
# Trajectory shape
# --------------------------------------------------------------------------- #

def test_immediate_pass_then_review() -> None:
    """A first-attempt pass still gets one review round — the F4 opportunity."""
    records, _ = run([code("a = 1"), code("a = 1")], [PASS, PASS])
    assert [r["stage"] for r in records] == ["initial", "review"]
    assert [r["round"] for r in records] == [0, 1]
    assert all(r["passed"] for r in records)


def test_review_can_be_disabled() -> None:
    """With review off, a passing trajectory stops at the initial attempt."""
    records, _ = run([code("a = 1")], [PASS], review_passing=False)
    assert [r["stage"] for r in records] == ["initial"]


def test_persistent_failure_stops_at_round_limit() -> None:
    """Failures retry exactly correction_rounds times, then stop."""
    records, _ = run([code("bad")] * 3, [FAIL] * 3, correction_rounds=2)
    assert [r["stage"] for r in records] == ["initial", "correction", "correction"]
    assert [r["round"] for r in records] == [0, 1, 2]
    assert not any(r["passed"] for r in records)


def test_recovery_then_review() -> None:
    """Fail, correct successfully, then review the now-passing solution."""
    records, _ = run([code("bad"), code("good"), code("good")], [FAIL, PASS, PASS])
    assert [r["stage"] for r in records] == ["initial", "correction", "review"]


def test_spurious_correction_is_recorded() -> None:
    """F4: a review round breaks previously passing code."""
    records, _ = run([code("good"), code("broken")], [PASS, FAIL])
    last = records[-1]
    assert last["stage"] == "review"
    assert last["passed"] is False
    assert records[0]["passed"] is True


def test_only_last_record_is_final() -> None:
    """`final` is the resume marker, so exactly one record may carry it."""
    records, _ = run([code("bad")] * 4, [FAIL] * 4, correction_rounds=3)
    assert [r["final"] for r in records] == [False, False, False, True]


def test_records_share_trajectory_id_and_carry_provenance() -> None:
    """Attempts group by trajectory_id and record how they were produced."""
    records, _ = run(
        [code("bad"), code("good"), code("good")],
        [FAIL, PASS, PASS],
        model="Qwen/Qwen2.5-1.5B-Instruct",
        temperature=0.2,
        sample=3,
    )
    assert len({r["trajectory_id"] for r in records}) == 1
    assert all(r["model"] == "Qwen/Qwen2.5-1.5B-Instruct" for r in records)
    assert all(r["temperature"] == 0.2 for r in records)
    assert all(r["sample"] == 3 for r in records)


def test_solution_is_cleaned_before_grading() -> None:
    """The graded solution has fences and self-tests stripped."""
    raw = "```python\ndef f():\n    return 1\n\nassert f() == 1\n```"
    records, _ = run([raw], [PASS], review_passing=False)
    assert records[0]["solution"] == "def f():\n    return 1"
    assert records[0]["raw_generation"] == raw


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

def test_correction_prompt_replays_failure() -> None:
    """The failed attempt comes back as an assistant turn, plus the feedback."""
    _, generator = run([code("bad"), code("good"), code("good")], [FAIL, PASS, PASS])
    correction = generator.seen[1]
    assert [m["role"] for m in correction] == ["user", "assistant", "user"]
    assert correction[1]["content"] == code("bad")
    assert "failed the tests" in correction[2]["content"]


def test_review_prompt_offers_no_change() -> None:
    """The review wording must not imply a bug exists, or F4 is manufactured."""
    messages = review_messages(PROBLEM, code("good"))
    assert "already correct, return it unchanged" in messages[-1]["content"]


def test_prompt_messages_are_stored_verbatim() -> None:
    """Phase 2 replays these exact prompts to capture activations."""
    records, generator = run([code("bad"), code("good"), code("good")], [FAIL, PASS, PASS])
    assert [r["prompt_messages"] for r in records] == generator.seen


def test_initial_prompt_contains_problem() -> None:
    """The problem statement is passed through unmodified."""
    assert PROBLEM["prompt"] in initial_messages(PROBLEM)[0]["content"]
    assert correction_messages(PROBLEM, "gen", "why")[0] == initial_messages(PROBLEM)[0]


# --------------------------------------------------------------------------- #
# Feedback
# --------------------------------------------------------------------------- #

def test_feedback_names_first_failing_input() -> None:
    """Feedback points at the first failing case, with its expected output."""
    message = build_feedback(PROBLEM, EXPECTED, FAIL)
    assert repr(PROBLEM["base_input"][1]) in message
    assert repr(EXPECTED["base"][1]) in message


def test_feedback_reports_timeout_distinctly() -> None:
    """A timeout has no failing input to point at, so it gets its own text."""
    assert "timed out" in build_feedback(PROBLEM, EXPECTED, TIMEOUT)


def test_feedback_falls_back_when_details_missing() -> None:
    """No per-input flags still yields a usable message, never a crash."""
    result = GradeResult("fail", "fail", None, None)
    assert build_feedback(PROBLEM, EXPECTED, result) == "Your solution did not pass the tests."


def test_passing_result_has_no_feedback() -> None:
    """Nothing to report when both suites pass."""
    assert build_feedback(PROBLEM, EXPECTED, PASS) == ""


def test_plus_only_failure_is_reported() -> None:
    """A base-suite pass that fails the plus suite is still a failure."""
    result = GradeResult("pass", "fail", (1, 1), (0,))
    message = build_feedback(PROBLEM, EXPECTED, result)
    assert repr(PROBLEM["plus_input"][0]) in message


# --------------------------------------------------------------------------- #
# Huge values
# --------------------------------------------------------------------------- #

# Python 3.11+ refuses to stringify integers past 4300 digits, and EvalPlus's
# augmented inputs really do reach that size (HumanEval/83, /139). This crashed
# a live collection run at problem 84 of 164.
HUGE = 10 ** 5000


def test_baseline_survives_huge_integers() -> None:
    """Grouping expected outputs must not stringify them."""
    expected = {"base": [HUGE, HUGE, 1], "plus": []}
    assert baseline_fraction(expected) == 2 / 3


def test_baseline_survives_huge_integers_inside_lists() -> None:
    """The same value nested in an unhashable container is just as fatal."""
    expected = {"base": [[HUGE], [HUGE], [1]], "plus": []}
    assert baseline_fraction(expected) == 2 / 3


def test_baseline_distinguishes_bool_from_int() -> None:
    """hash(True) == hash(1), so the key carries the type."""
    assert baseline_fraction({"base": [True, 1], "plus": []}) == 0.5


def test_baseline_handles_unhashable_values() -> None:
    """Expected outputs are frequently lists."""
    assert baseline_fraction({"base": [[1, 2], [1, 2], [3]], "plus": []}) == 2 / 3


def test_feedback_survives_huge_integers() -> None:
    """Building the correction message must not crash on giant expected values."""
    problem = {**PROBLEM, "base_input": [[HUGE], [2]]}
    expected = {"base": [HUGE, 2], "plus": [1]}
    message = build_feedback(problem, expected, GradeResult("fail", "fail", (0, 1), (1,)))
    assert "too large to display" in message


# --------------------------------------------------------------------------- #
# Checkpoint / resume
# --------------------------------------------------------------------------- #

def test_missing_file_has_nothing_completed(tmp_path: Path) -> None:
    """A first run starts from an empty set, not an error."""
    assert load_completed_keys(tmp_path / "absent.jsonl") == set()


def test_completed_trajectory_is_skipped(tmp_path: Path) -> None:
    """Only trajectories with a final record count as done."""
    path = tmp_path / "out.jsonl"
    records, _ = run([code("a")], [PASS], review_passing=False)
    append_records(path, records)
    assert load_completed_keys(path) == {("HumanEval/0", 0)}


def test_incomplete_trajectory_is_rerun(tmp_path: Path) -> None:
    """A session killed mid-trajectory leaves no final record, so it re-runs."""
    path = tmp_path / "out.jsonl"
    records, _ = run([code("a")], [PASS], review_passing=False)
    records[-1]["final"] = False
    append_records(path, records)
    assert load_completed_keys(path) == set()


def test_truncated_final_line_is_tolerated(tmp_path: Path) -> None:
    """An abrupt disconnect can cut the last line mid-write."""
    path = tmp_path / "out.jsonl"
    records, _ = run([code("a")], [PASS], review_passing=False)
    append_records(path, records)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"task_id": "HumanEval/1", "fin')
    assert load_completed_keys(path) == {("HumanEval/0", 0)}


def test_samples_are_tracked_independently(tmp_path: Path) -> None:
    """Two samples of one problem are separate units of work."""
    path = tmp_path / "out.jsonl"
    for sample in (0, 1):
        records, _ = run([code("a")], [PASS], review_passing=False, sample=sample)
        append_records(path, records)
    assert load_completed_keys(path) == {("HumanEval/0", 0), ("HumanEval/0", 1)}


def test_appended_records_are_valid_jsonl(tmp_path: Path) -> None:
    """Every record must survive a JSON round-trip unchanged."""
    path = tmp_path / "out.jsonl"
    records, _ = run([code("bad"), code("good"), code("good")], [FAIL, PASS, PASS])
    append_records(path, records)
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert [json.loads(line) for line in lines] == records


def test_task_ids_sort_numerically() -> None:
    """`--limit N` must mean the first N problems, not the first N by string."""
    ids = ["HumanEval/10", "HumanEval/2", "HumanEval/1", "HumanEval/20"]
    assert sort_task_ids(ids) == [
        "HumanEval/1",
        "HumanEval/2",
        "HumanEval/10",
        "HumanEval/20",
    ]


def test_append_creates_parent_directory(tmp_path: Path) -> None:
    """data/raw/ may not exist on a fresh clone or a fresh Colab session."""
    path = tmp_path / "nested" / "dir" / "out.jsonl"
    records, _ = run([code("a")], [PASS], review_passing=False)
    append_records(path, records)
    assert path.exists()
