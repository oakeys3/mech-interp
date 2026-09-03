"""Dataset report for a labeled trajectory file.

Answers the questions that decide whether Phase 2 can start: how the labels are
distributed, what shapes trajectories take, whether self-correction ever works,
and — because the review round is the only source of F4 — what the model
actually does when asked to review its own passing code.

Usage:
    py -3.12 -m analysis.dataset_report data/processed/humaneval_labeled.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from data.collection.label import LABELS, MIN_PER_FAILURE_CLASS, group_trajectories, read_jsonl


def label_distribution(records: list[dict[str, Any]]) -> str:
    """Report label counts against the per-class exit criterion.

    Args:
        records: Labeled attempt records.

    Returns:
        A printable section.
    """
    counts = Counter(record["label"] for record in records)
    lines = ["LABEL DISTRIBUTION", ""]
    total = len(records)
    for label in LABELS:
        count = counts.get(label, 0)
        share = f"{100 * count / total:5.1f}%" if total else "    -"
        if label == "P":
            status = ""
        elif count >= MIN_PER_FAILURE_CLASS:
            status = "  OK"
        else:
            status = f"  SHORT by {MIN_PER_FAILURE_CLASS - count}"
        lines.append(f"  {label:>2}: {count:5d}  {share}{status}")
    return "\n".join(lines)


def trajectory_shapes(trajectories: list[list[dict[str, Any]]], top: int = 8) -> str:
    """Report the most common label sequences.

    Args:
        trajectories: Grouped, round-ordered trajectories.
        top: How many distinct shapes to list.

    Returns:
        A printable section.
    """
    shapes = Counter(
        "->".join(attempt["label"] for attempt in trajectory) for trajectory in trajectories
    )
    lines = ["TRAJECTORY SHAPES", ""]
    for shape, count in shapes.most_common(top):
        lines.append(f"  {count:4d}  {shape}")
    if len(shapes) > top:
        lines.append(f"  ... {len(shapes) - top} more distinct shapes")
    return "\n".join(lines)


def correction_effectiveness(trajectories: list[list[dict[str, Any]]]) -> str:
    """Report how often a failed first attempt is eventually corrected.

    The spike saw 0/7 corrections succeed with one round. This is the check on
    whether three rounds changed that — and whether F3 has a contrast class of
    successful corrections to be probed against.

    Args:
        trajectories: Grouped, round-ordered trajectories.

    Returns:
        A printable section.
    """
    started_failed = 0
    recovered = 0
    for trajectory in trajectories:
        if trajectory[0]["label"] == "P":
            continue
        started_failed += 1
        if any(attempt["label"] == "P" for attempt in trajectory):
            recovered += 1
    lines = ["SELF-CORRECTION", ""]
    lines.append(f"  trajectories starting in failure : {started_failed}")
    lines.append(f"  eventually corrected             : {recovered}")
    if started_failed:
        lines.append(f"  correction success rate          : {100 * recovered / started_failed:.1f}%")
    return "\n".join(lines)


def review_behaviour(trajectories: list[list[dict[str, Any]]]) -> str:
    """Diagnose what the model does when asked to review passing code.

    F4 can only arise here, so a zero F4 count is either a real finding about
    the model or a sign the review round is not doing anything. The
    distinguishing evidence is whether the model edited its solution at all:
    the review prompt explicitly permits returning it unchanged.

    Args:
        trajectories: Grouped, round-ordered trajectories.

    Returns:
        A printable section.
    """
    reviews = 0
    unchanged = 0
    changed_still_passing = 0
    broke_it = 0
    for trajectory in trajectories:
        for previous, attempt in zip(trajectory, trajectory[1:]):
            if attempt["stage"] != "review":
                continue
            reviews += 1
            if attempt["solution"].strip() == previous["solution"].strip():
                unchanged += 1
            elif attempt["passed"]:
                changed_still_passing += 1
            else:
                broke_it += 1
    lines = ["REVIEW ROUND (the only source of F4)", ""]
    lines.append(f"  review attempts            : {reviews}")
    lines.append(f"  returned code unchanged    : {unchanged}")
    lines.append(f"  edited, still passing      : {changed_still_passing}")
    lines.append(f"  edited, broke it (= F4)    : {broke_it}")
    if reviews and unchanged == reviews:
        lines.append("")
        lines.append("  The model never edited passing code, so F4 cannot occur.")
        lines.append("  The review prompt permits returning it unchanged; that is")
        lines.append("  a prompt-design finding, not a collection bug.")
    elif not reviews:
        lines.append("")
        lines.append("  No review attempts at all — was collection run with --no-review?")
    return "\n".join(lines)


def repeated_attempts(trajectories: list[list[dict[str, Any]]]) -> str:
    """Count attempts that re-emit the previous solution verbatim.

    This decides how much of the F3 class is real. A trajectory that fails
    three correction rounds contributes three F3 labels, but if the model
    simply resubmitted the same code each time — behaviour already seen in the
    spike on HumanEval/1 — those are duplicate rows, not independent examples,
    and a probe trained on them is learning one example counted three times.

    Args:
        trajectories: Grouped, round-ordered trajectories.

    Returns:
        A printable section.
    """
    pairs = 0
    identical = 0
    for trajectory in trajectories:
        for previous, attempt in zip(trajectory, trajectory[1:]):
            if attempt["stage"] != "correction":
                continue
            pairs += 1
            if attempt["solution"].strip() == previous["solution"].strip():
                identical += 1
    distinct_f3 = sum(
        1
        for trajectory in trajectories
        for previous, attempt in zip(trajectory, trajectory[1:])
        if attempt["stage"] == "correction"
        and attempt["label"] == "F3"
        and attempt["solution"].strip() != previous["solution"].strip()
    )
    lines = ["CORRECTION NOVELTY", ""]
    lines.append(f"  correction attempts             : {pairs}")
    lines.append(f"  identical to previous attempt   : {identical}")
    if pairs:
        lines.append(f"  re-emission rate                : {100 * identical / pairs:.1f}%")
    lines.append(f"  F3 rows with genuinely new code : {distinct_f3}")
    lines.append("")
    lines.append("  Independent trajectories matter more than raw row counts:")
    lines.append("  attempts within one trajectory are correlated, so train/test")
    lines.append("  splits must be made per trajectory, never per attempt.")
    return "\n".join(lines)


def main() -> None:
    """Print a full report for a labeled trajectory file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Labeled JSONL from label.py")
    args = parser.parse_args()

    records = read_jsonl(args.input)
    trajectories = group_trajectories(records)

    print(f"file        : {args.input}")
    print(f"trajectories: {len(trajectories)}")
    print(f"attempts    : {len(records)}")
    print()
    for section in (
        label_distribution(records),
        trajectory_shapes(trajectories),
        correction_effectiveness(trajectories),
        repeated_attempts(trajectories),
        review_behaviour(trajectories),
    ):
        print(section)
        print()


if __name__ == "__main__":
    main()
