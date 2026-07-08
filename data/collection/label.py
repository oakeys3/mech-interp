"""Per-stage automated labeling logic.

Assigns one label from {P, F1, F2, F3, F4} to every (problem, stage, attempt)
tuple. A trajectory is a sequence of labels, e.g. ``[F2, F3, P]``.

    P  — pass: correct solution, all tests pass
    F1 — plan failure: structurally wrong approach
    F2 — implementation failure: correct plan, broken execution
    F3 — self-correction failure: correction attempted but wrong or regresses
    F4 — spurious correction: "corrects" passing code, introducing a new bug

Status: NOT IMPLEMENTED — Phase 1 begins after the Phase 0 decision gates
(model + benchmark confirmation) are closed. See spec.md.
"""
