"""Per-stage automated labeling logic.

Assigns one label from {P, F1, F2, F3, F4} to every attempt record produced by
``collect.py``. A trajectory is a sequence of labels, e.g. ``[F2, F3, P]``.

    P  — pass: correct solution, all tests pass
    F1 — plan failure: structurally wrong approach
    F2 — implementation failure: correct plan, broken execution
    F3 — self-correction failure: correction attempted but wrong or regresses
    F4 — spurious correction: "corrects" passing code, introducing a new bug

Three of the five labels fall out of the collection stage and need no
judgement: a passing attempt is P, a failed ``correction`` attempt is F3, and
a failed ``review`` attempt is F4 (the review stage only ever runs on code that
already passed, so breaking it is spurious by construction).

Only F1 vs. F2 requires inference — see ``classify_initial_failure``.

Usage:
    py -3.12 -m data.collection.label data/raw/humaneval_trajectories.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

# An initial attempt that passes no test at all is read as a wrong approach;
# partial credit is read as a broadly right approach with broken execution.
# Exposed as a constant because it is a research assumption, not a detail: it
# must survive the manual spot-check before the dataset is trusted.
PARTIAL_CREDIT_THRESHOLD = 0.0

LABELS = ("P", "F1", "F2", "F3", "F4")
MIN_PER_FAILURE_CLASS = 50


def passed_fraction(record: dict[str, Any]) -> float | None:
    """Fraction of individual test inputs the attempt satisfied.

    Combines the base and plus suites, since both are per-input flag lists of
    the same kind and more evidence is better for this judgement.

    Args:
        record: One attempt record.

    Returns:
        A value in [0, 1], or None when EvalPlus reported no per-input flags
        (which happens on timeouts and on some crashes).
    """
    flags: list[int] = []
    for key in ("base_details", "plus_details"):
        details = record.get(key)
        if details:
            flags.extend(int(flag) for flag in details)
    if not flags:
        return None
    return sum(flags) / len(flags)


def classify_initial_failure(record: dict[str, Any]) -> str:
    """Decide whether a failed first attempt is a plan or implementation failure.

    The heuristic is partial credit *relative to a trivial baseline*: the
    approach is read as broadly right (F2) when the solution beats what a
    constant-output solution would score, and as wrong (F1) when it does not —
    including timeouts, where no test completes at all.

    The baseline matters. Measured against raw partial credit, ``return False``
    on a boolean problem scores about 50% purely by matching every test whose
    answer happens to be False, and would be misread as a working-but-buggy
    implementation. ``baseline_fraction`` in ``collect.py`` records what the
    best constant scores so that credit has to be earned.

    This is still a proxy, not ground truth. It cannot see a solution that is
    coincidentally right on easy inputs while being algorithmically wrong, and
    it will call a syntax error F1 even when the intended plan was sound. The
    spec requires 50 manually spot-checked trajectories at >=85% agreement
    before these labels are used for probing; treat the threshold as a
    parameter to revisit if that check fails.

    Args:
        record: One failed attempt record with ``stage == "initial"``.

    Returns:
        "F1" or "F2".
    """
    fraction = passed_fraction(record)
    if fraction is None:
        return "F1"
    # Records collected before baseline_fraction existed fall back to plain
    # partial credit rather than being silently mislabeled against a zero.
    baseline = record.get("baseline_fraction")
    floor = PARTIAL_CREDIT_THRESHOLD if baseline is None else baseline
    return "F2" if fraction > floor else "F1"


def label_attempt(record: dict[str, Any]) -> str:
    """Assign one taxonomy label to a single attempt.

    Args:
        record: One attempt record from the collection JSONL.

    Returns:
        One of P, F1, F2, F3, F4.

    Raises:
        ValueError: If the record carries an unrecognized stage.
    """
    if record["passed"]:
        return "P"

    stage = record["stage"]
    if stage == "review":
        return "F4"
    if stage == "correction":
        return "F3"
    if stage == "initial":
        return classify_initial_failure(record)
    raise ValueError(f"unknown stage: {stage!r}")


def group_trajectories(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group attempt records into complete trajectories, in round order.

    Trajectories without a ``final`` record are incomplete — the collecting
    session died mid-run — and are dropped, per ``data/SCHEMA.md``.

    Args:
        records: Attempt records, in any order.

    Returns:
        One list of attempts per complete trajectory, each sorted by round.
        Trajectories appear in the order they first occur in ``records``.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["trajectory_id"], []).append(record)

    complete = []
    for attempts in grouped.values():
        if not any(attempt.get("final") for attempt in attempts):
            continue
        complete.append(sorted(attempts, key=lambda attempt: attempt["round"]))
    return complete


def label_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Label every attempt of every complete trajectory.

    Args:
        records: Raw attempt records read from a collection JSONL.

    Returns:
        Labeled copies of the attempts belonging to complete trajectories.
        Each gains a ``label`` field and a ``trajectory_labels`` field holding
        the whole trajectory's label sequence, so a single record stays
        self-describing once the dataset is shuffled for probing.
    """
    labeled: list[dict[str, Any]] = []
    for attempts in group_trajectories(records):
        sequence = [label_attempt(attempt) for attempt in attempts]
        for attempt, label in zip(attempts, sequence):
            labeled.append({**attempt, "label": label, "trajectory_labels": sequence})
    return labeled


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, tolerating a truncated final line.

    Args:
        path: File to read.

    Returns:
        The parsed records.
    """
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def summarize(labeled: list[dict[str, Any]]) -> str:
    """Report the class distribution against the per-class exit criterion.

    Args:
        labeled: Labeled attempt records.

    Returns:
        A printable multi-line summary.
    """
    counts = Counter(record["label"] for record in labeled)
    trajectories = len({record["trajectory_id"] for record in labeled})
    lines = [
        f"trajectories: {trajectories}   attempts: {len(labeled)}",
        "",
        "label counts:",
    ]
    for label in LABELS:
        count = counts.get(label, 0)
        if label == "P":
            note = ""
        elif count >= MIN_PER_FAILURE_CLASS:
            note = "  OK"
        else:
            note = f"  short by {MIN_PER_FAILURE_CLASS - count}"
        lines.append(f"  {label}: {count}{note}")
    return "\n".join(lines)


def main() -> None:
    """Label a raw collection JSONL and write the processed dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Raw trajectory JSONL from collect.py")
    parser.add_argument("--out", type=Path, default=None, help="Labeled JSONL output path")
    args = parser.parse_args()

    records = read_jsonl(args.input)
    labeled = label_records(records)

    out_path = args.out or Path("data/processed") / f"{args.input.stem}_labeled.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in labeled:
            handle.write(json.dumps(record) + "\n")

    print(summarize(labeled))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
