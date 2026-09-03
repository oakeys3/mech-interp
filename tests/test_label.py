"""Unit tests for per-stage failure labeling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from data.collection.label import (
    classify_initial_failure,
    group_trajectories,
    label_attempt,
    label_records,
    read_jsonl,
    summarize,
    passed_fraction,
)


def attempt(
    *,
    stage: str = "initial",
    passed: bool = False,
    base_details: tuple[int, ...] | None = (0, 0),
    plus_details: tuple[int, ...] | None = (0,),
    round_index: int = 0,
    trajectory_id: str = "t1",
    final: bool = False,
    baseline_fraction: float | None = None,
) -> dict[str, Any]:
    """Build a minimal attempt record shaped like collect.py's output."""
    return {
        "trajectory_id": trajectory_id,
        "task_id": "HumanEval/0",
        "round": round_index,
        "stage": stage,
        "passed": passed,
        "base_details": base_details,
        "plus_details": plus_details,
        "baseline_fraction": baseline_fraction,
        "final": final,
    }


# --------------------------------------------------------------------------- #
# Stage-determined labels
# --------------------------------------------------------------------------- #

def test_passing_attempt_is_p() -> None:
    """Any passing attempt is P regardless of stage."""
    for stage in ("initial", "correction", "review"):
        assert label_attempt(attempt(stage=stage, passed=True)) == "P"


def test_failed_correction_is_f3() -> None:
    """A correction that still fails is a self-correction failure."""
    assert label_attempt(attempt(stage="correction")) == "F3"


def test_failed_review_is_f4() -> None:
    """Review only runs on passing code, so breaking it is spurious."""
    assert label_attempt(attempt(stage="review")) == "F4"


def test_unknown_stage_raises() -> None:
    """An unrecognized stage is a schema bug, not something to guess at."""
    with pytest.raises(ValueError, match="unknown stage"):
        label_attempt(attempt(stage="brainstorm"))


# --------------------------------------------------------------------------- #
# F1 vs F2
# --------------------------------------------------------------------------- #

def test_no_tests_passed_is_plan_failure() -> None:
    """Zero partial credit reads as a wrong approach."""
    assert classify_initial_failure(attempt(base_details=(0, 0), plus_details=(0,))) == "F1"


def test_partial_credit_is_implementation_failure() -> None:
    """Some inputs handled means the approach is broadly right."""
    assert classify_initial_failure(attempt(base_details=(1, 0), plus_details=(0,))) == "F2"


def test_plus_suite_alone_can_earn_partial_credit() -> None:
    """Both suites count toward the judgement."""
    assert classify_initial_failure(attempt(base_details=(0, 0), plus_details=(1,))) == "F2"


def test_missing_details_is_plan_failure() -> None:
    """Timeouts report no per-input flags; nothing completed, so F1."""
    assert classify_initial_failure(attempt(base_details=None, plus_details=None)) == "F1"


def test_pass_fraction_combines_suites() -> None:
    """The fraction spans base and plus inputs together."""
    assert passed_fraction(attempt(base_details=(1, 0), plus_details=(1, 0))) == 0.5
    assert passed_fraction(attempt(base_details=None, plus_details=None)) is None


def test_partial_credit_must_beat_the_baseline() -> None:
    """A constant answer that matches half the tests is still a plan failure."""
    record = attempt(base_details=(1, 0), plus_details=(1, 0), baseline_fraction=0.5)
    assert classify_initial_failure(record) == "F1"


def test_beating_the_baseline_is_implementation_failure() -> None:
    """Credit above what a constant scores is evidence the approach works."""
    record = attempt(base_details=(1, 1), plus_details=(1, 0), baseline_fraction=0.5)
    assert classify_initial_failure(record) == "F2"


def test_missing_baseline_falls_back_to_partial_credit() -> None:
    """Records predating the baseline field must not be mislabeled."""
    record = attempt(base_details=(1, 0), plus_details=(0,), baseline_fraction=None)
    assert classify_initial_failure(record) == "F2"


def test_initial_failure_routes_through_label_attempt() -> None:
    """label_attempt delegates the only inferred distinction."""
    assert label_attempt(attempt(base_details=(1, 0))) == "F2"
    assert label_attempt(attempt(base_details=(0, 0), plus_details=(0,))) == "F1"


# --------------------------------------------------------------------------- #
# Trajectory grouping
# --------------------------------------------------------------------------- #

def test_incomplete_trajectory_is_dropped() -> None:
    """No final record means the session died mid-run."""
    records = [attempt(trajectory_id="t1", final=False)]
    assert group_trajectories(records) == []


def test_attempts_are_sorted_by_round() -> None:
    """Records may be appended out of order; trajectories must not be."""
    records = [
        attempt(trajectory_id="t1", round_index=2, stage="correction", final=True),
        attempt(trajectory_id="t1", round_index=0),
        attempt(trajectory_id="t1", round_index=1, stage="correction"),
    ]
    grouped = group_trajectories(records)
    assert [record["round"] for record in grouped[0]] == [0, 1, 2]


def test_separate_trajectories_stay_separate() -> None:
    """Grouping keys on trajectory_id, not task_id."""
    records = [
        attempt(trajectory_id="t1", final=True),
        attempt(trajectory_id="t2", final=True),
    ]
    assert len(group_trajectories(records)) == 2


# --------------------------------------------------------------------------- #
# End-to-end labeling
# --------------------------------------------------------------------------- #

def test_realistic_trajectory_labels() -> None:
    """The canonical F2 -> F3 -> P arc from the spike transcripts."""
    records = [
        attempt(round_index=0, base_details=(1, 0)),
        attempt(round_index=1, stage="correction"),
        attempt(round_index=2, stage="correction", passed=True, final=True),
    ]
    labeled = label_records(records)
    assert [record["label"] for record in labeled] == ["F2", "F3", "P"]


def test_spurious_correction_trajectory() -> None:
    """A pass followed by a broken review is the F4 arc."""
    records = [
        attempt(round_index=0, passed=True),
        attempt(round_index=1, stage="review", final=True),
    ]
    assert [record["label"] for record in label_records(records)] == ["P", "F4"]


def test_every_attempt_carries_the_full_sequence() -> None:
    """Records stay self-describing after the dataset is shuffled."""
    records = [
        attempt(round_index=0, base_details=(1, 0)),
        attempt(round_index=1, stage="correction", passed=True, final=True),
    ]
    labeled = label_records(records)
    assert all(record["trajectory_labels"] == ["F2", "P"] for record in labeled)


def test_original_fields_are_preserved() -> None:
    """Labeling adds fields; it never drops collection provenance."""
    records = [attempt(round_index=0, passed=True, stage="initial", final=True)]
    labeled = label_records(records)[0]
    assert labeled["task_id"] == "HumanEval/0"
    assert labeled["trajectory_id"] == "t1"


# --------------------------------------------------------------------------- #
# Reporting and IO
# --------------------------------------------------------------------------- #

def test_summary_flags_short_classes() -> None:
    """The summary is how the >=50-per-class exit criterion gets checked."""
    records = [
        attempt(round_index=0, base_details=(1, 0)),
        attempt(round_index=1, stage="correction", passed=True, final=True),
    ]
    report = summarize(label_records(records))
    assert "F2: 1  short by 49" in report
    assert "F4: 0  short by 50" in report
    assert "trajectories: 1" in report


def test_read_jsonl_tolerates_truncated_line(tmp_path: Path) -> None:
    """Collection files can end mid-write when a Colab session dies."""
    path = tmp_path / "raw.jsonl"
    path.write_text(
        json.dumps(attempt(final=True)) + "\n" + '{"trajectory_id": "t2", "sta',
        encoding="utf-8",
    )
    assert len(read_jsonl(path)) == 1
