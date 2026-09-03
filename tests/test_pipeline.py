"""End-to-end pipeline test: collect -> grade -> label, on real EvalPlus data.

The model is scripted, but everything else is real: real HumanEval+ problems,
real EvalPlus grading, real feedback construction, real labeling. This is the
check that the pieces fit together before a GPU run is started.

POSIX only — EvalPlus grading does not work on Windows (see harness.py).
"""

from __future__ import annotations

import os

import pytest

from data.collection.collect import run_trajectory
from data.collection.harness import grade, load_problem_suite
from data.collection.label import label_records

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="EvalPlus sandbox is POSIX-only"
)

TASK_ID = "HumanEval/0"


class ScriptedGenerator:
    """Returns prepared solutions in order, standing in for the model."""

    def __init__(self, solutions: list[str]) -> None:
        self.solutions = list(solutions)

    def chat(self, messages: list[dict[str, str]]) -> str:
        return f"```python\n{self.solutions.pop(0)}\n```"


@pytest.fixture(scope="module")
def suite():
    """Load HumanEval+ once for the module."""
    pytest.importorskip("evalplus")
    return load_problem_suite("humaneval")


# Right approach (pairwise distance) with a real bug: it only compares
# adjacent elements, so it misses close pairs that are not neighbours. This
# beats the constant baseline, which is what separates F2 from F1.
BUGGY_BODY = """    for i in range(len(numbers) - 1):
        if abs(numbers[i] - numbers[i + 1]) < threshold:
            return True
    return False
"""


def test_partial_credit_failure_then_recovery(suite) -> None:
    """A right-approach-but-buggy attempt labels F2, and its fix labels P."""
    problems, expected = suite
    problem = problems[TASK_ID]
    canonical = problem["prompt"] + problem["canonical_solution"]

    generator = ScriptedGenerator(
        [problem["prompt"] + BUGGY_BODY, canonical, canonical]
    )
    records = run_trajectory(
        problem,
        expected[TASK_ID],
        generator,
        grade,
        subset="humaneval",
        correction_rounds=3,
    )

    labeled = label_records(records)
    assert [record["label"] for record in labeled] == ["F2", "P", "P"]
    assert [record["stage"] for record in labeled] == ["initial", "correction", "review"]


def test_feedback_is_built_from_real_grading(suite) -> None:
    """The correction prompt carries a concrete failing input, not a placeholder."""
    problems, expected = suite
    problem = problems[TASK_ID]

    generator = ScriptedGenerator(
        [problem["prompt"] + BUGGY_BODY, problem["prompt"] + BUGGY_BODY]
    )
    records = run_trajectory(
        problem,
        expected[TASK_ID],
        generator,
        grade,
        subset="humaneval",
        correction_rounds=1,
    )

    feedback = records[0]["feedback"]
    assert "Wrong result for input" in feedback
    assert feedback in records[1]["prompt_messages"][-1]["content"]


def test_constant_output_labels_plan_failure(suite) -> None:
    """A constant answer is a plan failure even though it matches some tests.

    ``return False`` is right on every test whose answer is False — roughly
    half of them. Only the comparison against the constant baseline stops this
    being mislabeled as an implementation failure.
    """
    problems, expected = suite
    problem = problems[TASK_ID]

    generator = ScriptedGenerator([problem["prompt"] + "    return False\n"])
    records = run_trajectory(
        problem,
        expected[TASK_ID],
        generator,
        grade,
        subset="humaneval",
        correction_rounds=0,
    )

    labeled = label_records(records)
    assert [record["label"] for record in labeled] == ["F1"]
    assert labeled[0]["base_details"] is not None
    assert labeled[0]["baseline_fraction"] > 0


def test_records_survive_labeling_as_json(suite) -> None:
    """Grading details must reach the labeler intact through the record."""
    import json

    problems, expected = suite
    problem = problems[TASK_ID]
    canonical = problem["prompt"] + problem["canonical_solution"]

    generator = ScriptedGenerator([canonical, canonical])
    records = run_trajectory(
        problem, expected[TASK_ID], generator, grade, subset="humaneval"
    )

    round_tripped = [json.loads(json.dumps(record)) for record in records]
    assert round_tripped == records
    assert [r["label"] for r in label_records(round_tripped)] == ["P", "P"]


# EvalPlus's augmented inputs push these two problems' expected outputs past
# Python's 4300-digit integer-to-string limit. Keying or formatting them via
# repr killed a live collection run at problem 84 of 164.
@pytest.mark.parametrize("task_id", ["HumanEval/83", "HumanEval/139"])
def test_problems_with_huge_expected_outputs(suite, task_id) -> None:
    """Collection must survive problems whose answers are astronomically large."""
    problems, expected = suite
    problem = problems[task_id]

    generator = ScriptedGenerator([problem["prompt"] + "    return 0\n"])
    records = run_trajectory(
        problem,
        expected[task_id],
        generator,
        grade,
        subset="humaneval",
        correction_rounds=0,
    )

    assert records[0]["baseline_fraction"] is not None
    assert isinstance(records[0]["feedback"], str)
    assert label_records(records)[0]["label"] in {"F1", "F2"}
