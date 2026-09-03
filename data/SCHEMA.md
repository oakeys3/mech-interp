# Trajectory JSONL schema (v1)

One JSON object **per attempt**, not per trajectory. A trajectory is recovered
by grouping on `trajectory_id` and sorting by `round`.

Why per-attempt: the labeling unit is `(problem, stage, attempt)`, and each
attempt is one training example for probing. Per-attempt lines also let the
collector append as it goes, which is what makes checkpoint/resume work.

The file is **append-only**. Nothing is ever rewritten in place, so a Colab
disconnect can at worst truncate the final line.

## Fields

| Field | Type | Meaning |
|-------|------|---------|
| `trajectory_id` | str | uuid4 hex; groups attempts of one run |
| `task_id` | str | EvalPlus id, e.g. `HumanEval/0` |
| `subset` | str | `humaneval` or `mbpp` |
| `sample` | int | Which sample of this problem (0-based) |
| `round` | int | 0 = first attempt; 1..n = subsequent attempts |
| `stage` | str | `initial`, `correction`, or `review` |
| `prompt_messages` | list | The exact chat messages sent to the model |
| `raw_generation` | str | Unmodified model output |
| `solution` | str | Cleaned code that was actually graded |
| `base_status` | str | EvalPlus status on the original tests |
| `plus_status` | str | EvalPlus status on the augmented tests |
| `base_details` | list\|null | Per-input 0/1 flags for the original tests |
| `plus_details` | list\|null | Per-input 0/1 flags for the augmented tests |
| `passed` | bool | True only if **both** suites pass |
| `feedback` | str | Failure message shown to the model next round (`""` if passed) |
| `final` | bool | True on the last attempt of a trajectory |
| `model` | str | Model id used |
| `temperature` | float | Sampling temperature |
| `timestamp` | str | UTC ISO-8601 |

## Two fields that carry design weight

**`prompt_messages`** stores the full message list, not just the problem id.
Phase 2 must replay the exact prompt to capture activations; storing it here
means `extract.py` never has to reconstruct prompts and risk drifting from
what was actually run.

**`final`** is the resume marker. On restart the collector skips any
`(task_id, sample)` that already has a `final: true` record. A trajectory
without one is incomplete (the session died mid-run); it is re-run under a new
`trajectory_id`, and consumers drop trajectories that lack a `final` record.

## Stages

- `initial` — first attempt at the problem.
- `correction` — response to a failed attempt, with failure feedback.
- `review` — the model is shown its own **passing** solution and asked to
  review it. This is the only way F4 (spurious correction) can occur; the
  ordinary loop never revisits passing code.

## Label mapping (consumed by `label.py`)

Labels are assigned downstream, not by the collector. The collector records
what happened; `label.py` decides what to call it.

| Situation | Label |
|-----------|-------|
| `passed` | P |
| Failed `initial` attempt | F1 or F2 (needs plan-vs-execution judgement) |
| Failed `correction` attempt | F3 |
| Failed `review` attempt after a pass | F4 |

F1 vs. F2 is decided from `base_details`/`plus_details`: an attempt that
satisfies at least one test input is read as F2 (approach right, execution
broken), and one that satisfies none — timeouts included — as F1. Those two
fields exist for exactly this reason and must not be dropped from the record.
