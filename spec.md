# Research Spec — Agentic Failures as Interpretability Problems

**Hard deadline:** August 14, 2026. Interview-ready results targeted for late July.

## 1. Mission

One sharp research question: **do internal model representations predict distinct categories of code-generation failures in LLMs, and can we intervene on those representations to causally suppress failure?**

Probe-then-patch, in three stages:

- **A (Sparse probing):** Fit L1-regularized linear probes on residual stream activations to find *which specific directions* are failure-predictive — not just whether a boundary exists, but where it lives and how sparse it is.
- **B (Component decomposition):** Decompose the residual stream by component (per-head outputs, MLP layer outputs, residual stream at each sublayer) using TransformerLens, attributing failure prediction to specific circuits rather than the stream wholesale.
- **C (Causal validation):** Patch the identified failure-predictive directions and measure whether failure rates shift. This converts a correlation finding into a mechanism finding. Scoped to F1+F2 first; F3+F4 are stretch goals.

## 2. Failure taxonomy

Labels are assigned **per stage**, not per trajectory: each (problem, stage, attempt) tuple gets its own label, and a trajectory is a label sequence, e.g. `[F2, F3, P]`.

| Code | Name | Definition |
|------|------|------------|
| F1 | Plan failure | Structurally wrong approach — wrong algorithm, wrong data structure, misread problem spec |
| F2 | Implementation failure | Correct plan, broken execution — off-by-one, wrong API, syntax error, type mismatch |
| F3 | Self-correction failure | Model sees traceback, attempts correction, but correction is wrong or regresses |
| F4 | Spurious correction | Model "corrects" passing code, introducing a new bug where none existed |
| P | Pass | Correct solution, all tests pass |

## 3. Phased timeline

Phases are time-boxed, not open-ended. Each phase has explicit exit criteria; the next phase does not begin until they are met.

### Phase 0 — Confirm model & benchmark (days 1–3)

- Run Qwen2.5-1.5B-Instruct at `--n 25` on HumanEval; inspect failure-rate distribution and per-problem transcripts.
- Decision gates: confirm Qwen as primary model (usable failure rate = 30–60%; <20% → harder benchmark or higher temperature; >80% → model too weak) and EvalPlus as benchmark suite.
- Lock choices into `configs/experiment_config.yaml`.

**Exit:** model and benchmark committed; no further deliberation.

### Phase 1 — Dataset collection (days 4–12)

- `data/collection/harness.py` — EvalPlus test runner with self-test stripping.
- `data/collection/collect.py` — batch generation with multi-round self-correction (up to 3 rounds per problem).
- `data/collection/label.py` — per-stage automated labeling (P/F1–F4).
- Spot-check 50 trajectories manually against automated labels; log everything to JSONL with a version-controlled schema.

**Exit:** ≥500–800 labeled (problem, stage, attempt) tuples, ≥50 examples per failure class (oversample harder problems if needed), spot-check accuracy ≥85%.

### Phase 2 — Activation extraction & sparse probing (days 13–22)

- `probing/extract.py` — TransformerLens hooks capturing residual stream, per-head outputs, MLP outputs at every layer.
- `probing/probe.py` — L1 logistic regression (one-vs-rest over F1–F4), `solver='liblinear'`, C ∈ {0.001, 0.01, 0.1, 1.0}; report accuracy, active-direction count, sparsity ratio.
- `probing/decompose.py` — probes on per-component activations to attribute signal to specific heads/MLPs.
- `probing/evaluate.py` — cross-validation, confusion matrices, sparsity-pattern heatmaps.
- Key question: do F1/F2/F3/F4 activate *different* sparse directions, or share a common "failure subspace"?

**Exit:** probe accuracy ≥65% held-out (chance = 20%), ≥1 failure type with an interpretable sparse pattern, component decomposition identifying signal-carrying heads/MLPs.

### Phase 3 — Causal validation (days 23–32)

Priority order (cut from the bottom if time runs short): F2, F1, then F3/F4 as stretch.

- `patching/patch.py` — zero or replace failure-predictive directions mid-forward-pass.
- `patching/causal_validate.py` — per intervention: rerun generation, measure pass-rate shift vs. baseline.
- Both directions: patch out the failure direction (pass rate up?) and patch a pass direction into failing trajectories (now passes?).

**Exit:** F2 shows a statistically significant causal effect (p < 0.05, Δ pass rate ≥ 10pp).

### Phase 4 — Write-up & public presentation (days 33–40, ends Aug 14)

Final README with methodology diagram and findings, final spec with results vs. criteria, 2–3 page research report (PDF), reproducible notebooks, setup instructions, closing research-log entry.

## 4. Success criteria

| Criterion | Bar |
|-----------|-----|
| Working dataset | ≥500 labeled trajectories, ≥50 per failure class |
| Sparse probe signal | ≥65% probe accuracy, ≥1 interpretable sparsity pattern |
| Component attribution | ≥1 failure type attributed to specific heads/MLPs |
| Causal result | F2 patching shows ≥10pp Δ pass rate (p < 0.05) |
| Public repo | Clean, documented, runnable by a stranger |
| OSS contribution | ≥1 PR opened or merged to TransformerLens or EvalPlus |

## 5. Scope management

If August 14 is at risk, cut in this order: F4 causal validation → F3 causal validation → component decomposition depth → dataset size. The F2 causal result and the public write-up are never cut — they are the core artifact.

## 6. Environment notes

- Windows; invoke Python as `py -3.12` explicitly. All scripts runnable by copy-paste from a terminal.
- JSONL for all intermediate artifacts — human-readable, inspectable without tooling.
- `transformers` 5.x: `apply_chat_template` returns a dict — unpack with `**inputs` for generation.
- TransformerLens: `model.run_with_hooks()` with `names_filter` for efficient activation capture.
- HuggingFace dataset IDs are namespaced: `openai/openai_humaneval`, `google-research-datasets/mbpp`.
- Model-generated code executes in subprocesses with timeouts — isolation, not sandboxing.
