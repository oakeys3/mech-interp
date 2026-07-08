# Agentic Failures as Interpretability Problems

Mechanistic interpretability of agentic code-generation failures in LLMs — sparse probing, circuit decomposition, and causal activation patching.

## Research question

**Do internal model representations predict distinct categories of code-generation failures, and can we intervene on those representations to causally suppress failure?**

The methodology is probe-then-patch, in three stages:

1. **Sparse probing** — L1-regularized linear probes on residual stream activations find *which specific directions* are failure-predictive: not just whether a boundary exists, but where it lives and how sparse it is.
2. **Component decomposition** — decompose the residual stream by component (per-head outputs, MLP outputs, residual stream at each sublayer) with TransformerLens, attributing failure prediction to specific circuits.
3. **Causal validation** — patch the identified failure-predictive directions and measure whether failure rates shift, converting a correlation finding into a mechanism finding.

## Failure taxonomy

Every trajectory is labeled **per stage** — each (problem, stage, attempt) tuple gets its own label, so a single run is a sequence like `[F2, F3, P]`.

| Code | Name | Definition |
|------|------|------------|
| F1 | Plan failure | Structurally wrong approach — wrong algorithm, wrong data structure, misread spec |
| F2 | Implementation failure | Correct plan, broken execution — off-by-one, wrong API, syntax error, type mismatch |
| F3 | Self-correction failure | Model sees traceback, attempts correction, but the correction is wrong or regresses |
| F4 | Spurious correction | Model "corrects" passing code, introducing a new bug where none existed |
| P | Pass | Correct solution, all tests pass |

## Status

**Phase 0 — model & benchmark confirmation (in progress).**

Findings so far, from the feasibility spike ([spike/spike.py](spike/spike.py)):

- **Pythia-1.4B is eliminated.** Its self-correction mode echoes tracebacks as comments rather than generating corrected code — the failure mode is illegible for probing.
- **Qwen2.5-1.5B-Instruct is the leading candidate**: 2/3 on first attempts with one genuine, traceable F2 → F3/F4 progression. Pending confirmation at n=25 (target: 30–60% failure rate across HumanEval problems).
- **Harness fix**: instruct models append self-test assertions that pollute grading; these must be stripped before test execution.
- **Benchmark**: EvalPlus suites (HumanEval+ / MBPP+) rather than raw HumanEval, whose weak test coverage lets mathematically incorrect solutions pass and would inject label noise into probing.

See [RESEARCH_LOG.md](RESEARCH_LOG.md) for the running log and [spec.md](spec.md) for the full research spec, timeline, and success criteria.

## Repository layout

```
spike/       feasibility spike (frozen — do not modify)
data/        collection harness, labeling, raw + processed datasets
probing/     activation extraction, L1 sparse probes, component decomposition
patching/    activation patching and causal validation
analysis/    visualization, circuit reports, exploratory notebooks
configs/     experiment configuration (single source of truth)
tests/       unit tests (run in CI on every push)
```

## Setup

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest tests/ -v
```

**Safety note:** the collection harness and spike execute model-generated code in subprocesses with timeouts. That is isolation, not sandboxing — run on a throwaway machine or in a container.
