# Research Log

Running log of decisions and findings, newest entry last. Every major
methodological choice gets an entry. Friction with upstream libraries is
tagged `[OSS CANDIDATE]` for later review.

---

## 2026-07-07 — Repo initialized; spike findings imported

**Repo scaffolded** to the full project layout (data/probing/patching/analysis,
config, tests, CI). Placeholder modules document their intended role; nothing
past Phase 0 is implemented, by design — Phase 1 infrastructure waits until the
Phase 0 decision gates close.

**Spike findings carried forward** (from `spike/spike.py`, now frozen):

- **Pythia-1.4B eliminated.** In self-correction mode it echoes tracebacks back
  as comments instead of generating corrected code. The failure mode is
  illegible for probing — there is no correction attempt to label F3/F4.
- **Qwen2.5-1.5B-Instruct is the candidate primary model.** Went 2/3 on first
  attempts with one genuine, traceable F2 → F3/F4 progression.
- **Self-test stripping is mandatory.** Qwen appends its own `assert` blocks to
  solutions; left in place they pollute test results. The fix lives in the
  spike and must carry into `data/collection/harness.py`.
  `[OSS CANDIDATE]` — if this is a general instruct-model problem on EvalPlus,
  the stripping logic may be worth upstreaming.
- **EvalPlus over raw HumanEval.** HumanEval's weak test coverage lets
  mathematically incorrect solutions pass, which would inject label noise into
  the probing signal. EvalPlus's augmented suites close this.
- `[OSS CANDIDATE]` — `transformers` 5.x `apply_chat_template` returns a dict;
  the `**inputs` unpacking pattern for generation is underdocumented and cost
  debugging time in the spike.

**Methodological principle confirmed:** disposable spike before infrastructure.
Repeat at every major phase transition.

**Open blocker (Phase 0 gate):** the `--n 25` run of Qwen on HumanEval is
pending. Qwen is not committed until it shows a usable failure-rate
distribution (target 30–60%). Below 20% → need a harder benchmark or higher
temperature; above 80% → model too weak to show meaningful self-correction
dynamics.

Next: run `py -3.12 spike/spike.py --model Qwen/Qwen2.5-1.5B-Instruct
--instruct --dataset humaneval --n 25`, inspect the failure-rate distribution
and per-problem transcripts, then close the model/benchmark decision gates.
