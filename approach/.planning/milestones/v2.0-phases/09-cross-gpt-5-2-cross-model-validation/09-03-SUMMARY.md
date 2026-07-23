---
phase: 09-cross-gpt-5-2-cross-model-validation
plan: 03
subsystem: testing
tags: [cross-model, gpt-5.4, openai, s_linker13, ablation, sweep, gate-01, model-provider-property]

# Dependency graph
requires:
  - phase: 09-cross-gpt-5-2-cross-model-validation/09-01
    provides: CLEAN GATE-06 audit verdict on harness + adapter shim — authorisation to invoke LLM calls without code edits
  - phase: 09-cross-gpt-5-2-cross-model-validation/09-02
    provides: BBB probe JSON (`ablation_20260531_055235.json`) reused for the bigbluebutton row + go signal to commit full-sweep API cost
provides:
  - Full 5-dataset s_linker13 / gpt-5.4 F1 evidence for downstream Plan 09-04 comparison report
  - 4-dataset sweep JSON (`ablation_20260531_063446.json`) — MS, TS, TM, JAB
  - Reconfirmed env-override invocation pattern (LLM_BACKEND, OPENAI_MODEL_NAME, PHASE_CACHE_DIR) holds across larger sweeps without harness edits
  - Macro F1 = 0.9077 — informational; Plan 09-04 owns the GATE-01 verdict
affects: [09-04 comparison report]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Env-override-only cross-model invocation (no harness edits) — works across a 4-dataset sweep, not just probes"
    - "Reused probe JSON pattern: BBB block reused from prior plan, merged at compute-time rather than re-run"
    - "PHASE_CACHE_DIR namespacing held across full sweep (no Claude-cache collision)"

key-files:
  created:
    - .planning/phases/09-cross-gpt-5-2-cross-model-validation/09-03-SWEEP-LOG.md
    - results/ablation_results/ablation_20260531_063446.json
    - results/ablation_results/09-03-sweep-stdout.log
    - results/ablation_results/s_linker13_mediastore_links.csv
    - results/ablation_results/s_linker13_teastore_links.csv
    - results/ablation_results/s_linker13_teammates_links.csv
    - results/ablation_results/s_linker13_jabref_links.csv
    - results/llm_logs/s_linker13_mediastore_20260531_063241.json
    - results/llm_logs/s_linker13_teastore_20260531_063321.json
    - results/llm_logs/s_linker13_teammates_20260531_063428.json
    - results/llm_logs/s_linker13_jabref_20260531_063446.json
    - results/phase_cache_gpt54/s_linker13/mediastore/{layer1,entity_candidates,entity_decisions,layer2,final}.pkl
    - results/phase_cache_gpt54/s_linker13/teastore/{layer1,entity_candidates,entity_decisions,layer2,final}.pkl
    - results/phase_cache_gpt54/s_linker13/teammates/{layer1,entity_candidates,entity_decisions,layer2,final}.pkl
    - results/phase_cache_gpt54/s_linker13/jabref/{layer1,entity_candidates,entity_decisions,layer2,final}.pkl
  modified: []

key-decisions:
  - "Mode B (skip BBB, reuse Plan 09-02 probe JSON): probe was harness-healthy, no anomalous variance → D-03 Step 3 default no-retest policy applies, saves BBB API cost"
  - "Single-shot per dataset (D-09 default cost discipline) — no variance retests; jitter band disclosed in 09-04"
  - "Macro F1 0.9077 < GATE-01 0.93 logged as informational; Plan 09-04 owns the gate verdict and the model-provider-property framing per D-05"
  - "Zero harness/prompt edits — env-override surface (LLM_BACKEND, OPENAI_MODEL_NAME, PHASE_CACHE_DIR) is the only audited path"

patterns-established:
  - "Full-sweep cross-model invocation pattern: same env-override triple from Plan 09-02 BBB probe scales to 4-dataset sweep without modification"
  - "Cross-plan JSON merge pattern: BBB-from-probe + 4-from-sweep, merged at downstream-compute time via sorted glob over `results/ablation_results/ablation_*.json` taking the freshest occurrence of each dataset"

requirements-completed: [CROSS-01, CROSS-02]  # both per D-02 single-arm Phase 8 retro-designation; this plan produces the per-dataset JSON evidence required by both

# Metrics
duration: ~3min  # plan execution wall-clock (sweep 2m35s + log/summary)
completed: 2026-05-31
---

# Phase 9 Plan 03: Full 5-Dataset Sweep on gpt-5.4 Summary

**s_linker13 ran end-to-end on gpt-5.4 across all 5 datasets via env-override-only invocation; 4-dataset sweep (MS/TS/TM/JAB) executed cleanly in 154.5 s with one recovered MS coref retry and zero harness errors, and combined with the reused Plan 09-02 BBB probe JSON produces a 5-dataset macro F1 of 0.9077 — informationally below GATE-01's 0.93 floor and to be framed by Plan 09-04 as a model-provider-property finding per D-05.**

## Performance

- **Duration:** ~3 min (plan execution wall-clock)
- **Sweep runtime (sum of dataset `time` fields):** 154.5 s (≈ 2 m 35 s); per-dataset MS 29.5s, TS 39.9s, TM 66.7s, JAB 18.4s
- **BBB (reused from Plan 09-02):** 48.4 s — not part of this invocation
- **Started:** 2026-05-31T04:32:10Z (sweep invocation)
- **Completed:** 2026-05-31T04:34:46Z (per `ablation_20260531_063446.json` filename)
- **Tasks:** 1 (auto, single sweep invocation)
- **Files created:** 1 plan log + sweep JSON + stdout log + 4 link CSVs + 4 LLM-call logs + 20 pickle checkpoints (4 datasets × 5 phases)

## Accomplishments

- 4-dataset sweep ran end-to-end on gpt-5.4 with no harness code edits (env-override path holds beyond the BBB-probe scope of Plan 09-02)
- All 5 datasets have an s_linker13 / gpt-5.4 F1 value resolvable from `results/ablation_results/` (4 from this sweep + 1 from reused Plan 09-02 probe JSON)
- Cache namespacing under `results/phase_cache_gpt54/` held cleanly — 4 new dataset subdirs (`mediastore`, `teastore`, `teammates`, `jabref`) with all 5 expected `.pkl` checkpoints each; Claude's `./results/phase_cache/` was NOT touched
- One transient MS coref `empty response, retrying...` event observed and recovered by the existing retry path (same event-class as Plan 09-02's BBB transient retry — documented `approve-biased fallback on LLM failure` pattern, not a halt signal)
- Per-dataset evidence ready for Plan 09-04 comparison report (CROSS-03)
- CROSS-01 + CROSS-02 jointly satisfied (D-02 single-arm Phase 8 retro-designation)

## Sweep Methodology

### Mode Chosen

**Option B** — sweep 4 datasets (MS, TS, TM, JAB), reuse Plan 09-02 BBB probe JSON for the bigbluebutton row.

**Rationale:** Plan 09-02's BBB probe was harness-healthy (F1 0.8037, single recovered coref retry, no anomalous variance signal) and the BBB JSON is schema-compatible with the sweep JSON for downstream compute. D-03 Step 3 default no-retest policy applies; BBB API cost saved.

### Invocation (env-override only, no code edits)

```bash
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.4 \
PHASE_CACHE_DIR=./results/phase_cache_gpt54 \
python run_ablation.py \
  --datasets mediastore teastore teammates jabref \
  --variants s_linker13 \
  --results-dir results/ablation_results 2>&1 \
  | tee results/ablation_results/09-03-sweep-stdout.log
```

Runner-printed backend descriptor: `openai (gpt-5.4)` (env override took effect). No edits to `run_ablation.py`, `llm_client.py`, `s_linker13.py`, or any `prompts*.py` file during this plan (verified via `git status --short` post-run; only the `.planning/phases/09-…/09-03-SWEEP-LOG.md` and `.planning/phases/09-…/09-03-SUMMARY.md` files added by this plan are tracked).

## Per-Dataset Results

Source files:
- This sweep: `results/ablation_results/ablation_20260531_063446.json`
- Plan 09-02 probe (BBB reused): `results/ablation_results/ablation_20260531_055235.json`

| Dataset | F1 | P | R | TP | FP | FN | n_links | Runtime | JSON source |
|---|---|---|---|---|---|---|---|---|---|
| mediastore    | **0.9677** | 0.9677 | 0.9677 | 30 | 1  | 1  | 31 | 29.5 s | sweep |
| teastore      | **1.0000** | 1.0000 | 1.0000 | 27 | 0  | 0  | 27 | 39.9 s | sweep |
| teammates     | **0.7939** | 0.7027 | 0.9123 | 52 | 22 | 5  | 74 | 66.7 s | sweep |
| bigbluebutton | **0.8037** | 0.9556 | 0.6935 | 43 | 2  | 19 | 45 | 48.4 s | Plan 09-02 probe (reused) |
| jabref        | **0.9730** | 0.9474 | 1.0000 | 18 | 1  | 0  | 19 | 18.4 s | sweep |

**Macro F1 = (0.9677 + 1.0000 + 0.7939 + 0.8037 + 0.9730) / 5 = 0.9077**

### Claude Sonnet Baseline Reference

From `results/ablation_results/ablation_20260529_215932.json` (PROJECT.md baseline): **macro F1 = 0.9506**.

Cross-model delta: **−4.3 pp Claude Sonnet → gpt-5.4**. Consistent in family with project memory's prior cross-provider gaps measured on V32 (V32 Claude 94.5% → gpt-5.2 90.6% = −3.9 pp; V32 → gpt-5.4 87.7%). The gap is a property of the model provider, not of `s_linker13` — per D-05/v2.0 standing policy, no fix-it action items are introduced.

### GATE-01 Informational Read

| Gate | Threshold | Cross-model observed (gpt-5.4) | Verdict (informational; Plan 09-04 owns final) |
|---|---|---|---|
| GATE-01 | macro F1 ≥ 0.93 | 0.9077 | **DOES NOT HOLD** |

Plan 09-04 will frame this per D-05 ("model-provider-property finding"), not as a defect.

### Notable Per-Dataset Observations

- **teastore**: gpt-5.4 perfect (P=R=F1=1.000) — matches Claude Sonnet on this dataset.
- **mediastore**: F1 0.9677 — single coref FP and single FN; well clear of GATE-01.
- **jabref**: F1 0.9730 — perfect recall, single coref FP.
- **teammates**: F1 0.7939 with 22 FPs (17 seed, 4 coref, 1 entity) — this is the dataset that drags the macro below GATE-01. Consistent with project memory's observation that GPT over-extracts seeds on doc-heavy projects (teammates: 198 sentences, largest in benchmark). Model-provider-property finding per D-05.
- **bigbluebutton (reused)**: F1 0.8037 with P=0.9556, R=0.6935 — high precision, lower recall vs Claude.

### Cache Namespacing Verification

Post-run check of `results/phase_cache_gpt54/s_linker13/`:

```
bigbluebutton/   (Plan 09-02)
jabref/          (this sweep)
mediastore/      (this sweep)
teammates/       (this sweep)
teastore/        (this sweep)
```

Each dataset directory contains all 5 expected `.pkl` files (`layer1`, `entity_candidates`, `entity_decisions`, `layer2`, `final`). Claude's `./results/phase_cache/` was NOT modified — namespacing held.

### Transient Events

- 1× `Coref batch: empty response, retrying...` on mediastore Tier 2 coreference — handled by existing retry path; final coref produced 3 links. Same event class as Plan 09-02's BBB transient (documented `approve-biased fallback on LLM failure` pattern); not a halt signal.
- 0× 4xx auth errors, 0× 429 rate-limit waits, 0× 5xx server errors, 0× model-id rejections, 0× dropped datasets.

## Task Commits

1. **Task 1: Sweep + log** — `cb92001` (docs)

## Files Created/Modified

- `.planning/phases/09-cross-gpt-5-2-cross-model-validation/09-03-SWEEP-LOG.md` — mode rationale, invocation, timestamps, per-dataset F1 table with JSON-path provenance per row, macro F1, retries, pipeline-completion verification, consumed JSON pointers
- `results/ablation_results/ablation_20260531_063446.json` — sweep JSON (MS/TS/TM/JAB blocks for s_linker13)
- `results/ablation_results/09-03-sweep-stdout.log` — full runner stdout (mode, env, per-dataset trace, summary table)
- `results/ablation_results/s_linker13_{mediastore,teastore,teammates,jabref}_links.csv` — emitted link CSVs
- `results/llm_logs/s_linker13_{mediastore,teastore,teammates,jabref}_<ts>.json` — per-dataset phase-3 LLM call logs
- `results/phase_cache_gpt54/s_linker13/{mediastore,teastore,teammates,jabref}/{layer1,entity_candidates,entity_decisions,layer2,final}.pkl` — 20 checkpoint pickles
- (No edits to harness/prompt/linker source files.)

## Decisions Made

- **Option B sweep mode (skip BBB, reuse probe JSON):** Plan 09-02's BBB probe log shows clean harness behavior (single recovered coref retry, no anomalous variance) and BBB F1 0.8037 is well clear of the 0.6 sanity floor. Per D-03 Step 3 default no-retest policy, re-running BBB would burn API cost for no additional evidence. The BBB JSON is schema-compatible with the sweep JSON, merged at downstream compute time.
- **Single-shot per dataset (D-09 default):** No variance retests per Claude's discretion under D-09 cost discipline. Project memory's documented `5–12 link stdev` GPT variance band is the disclosed uncertainty envelope; Plan 09-04 will cite this when discussing the macro F1 number.
- **GATE-01 verdict deferred to Plan 09-04:** This plan logs the macro F1 = 0.9077 < 0.93 as informational. Plan 09-04 owns the formal verdict and the model-provider-property framing per D-05.
- **No harness edits:** Env-override surface (`LLM_BACKEND`, `OPENAI_MODEL_NAME`, `PHASE_CACHE_DIR`) is the audited path from Plan 09-01's GATE-06 CLEAN audit; reaffirmed under a larger sweep.

## Deviations from Plan

None — plan executed exactly as written. The sweep ran on the first attempt (single transient MS coref retry recovered without operator action), all 4 datasets reached final dedup, the SWEEP-LOG's 7 required fields (mode + rationale, invocation, timestamps, per-dataset F1 table with provenance, macro F1, retry notes, JSON-path pointers) are populated, and the plan's automated verification one-liner passes:

```
mediastore: F1=0.9677 from ablation_20260531_063446.json
teastore:   F1=1.0000 from ablation_20260531_063446.json
teammates:  F1=0.7939 from ablation_20260531_063446.json
bigbluebutton: F1=0.8037 from ablation_20260531_055235.json
jabref:     F1=0.9730 from ablation_20260531_063446.json
Macro F1 = 0.9077
```

## Issues Encountered

None blocking. One transient MS coref empty-response retry observed and recovered by the existing retry path (documented in the sweep log under "Retries / Transient Errors / Rate-Limit Waits"). Identical event class to Plan 09-02's BBB transient — pattern reaffirmed as harness-healthy.

## User Setup Required

None — `OPENAI_API_KEY` already present in `.env` (used by Plan 09-02 probe too; no new external service configuration introduced).

## Hand-Off to Plan 09-04

Plan 09-04 (cross-model comparison report, CROSS-03) consumes this plan's outputs:

- **Sweep JSON (4 datasets):** `results/ablation_results/ablation_20260531_063446.json` — MS/TS/TM/JAB rows
- **Reused probe JSON (BBB):** `results/ablation_results/ablation_20260531_055235.json` — bigbluebutton row (from Plan 09-02)
- **Recommended merge code:** sorted glob over `results/ablation_results/ablation_*.json` keyed by `os.path.getmtime` descending, taking the first occurrence of each dataset (see `09-03-SWEEP-LOG.md` for reproducible snippet)
- **Per-dataset Claude Sonnet baseline:** `results/ablation_results/ablation_20260529_215932.json` (macro 0.9506) — for the Claude-vs-gpt-5.4 delta table
- **Variance disclosure to cite:** project memory's `5–12 link stdev` GPT band as the uncertainty envelope around the gpt-5.4 numbers; Plan 09-04 should note that all gpt-5.4 numbers are single-shot per D-09 default
- **GATE-01 framing:** macro 0.9077 < 0.93 → DOES NOT HOLD cross-model on gpt-5.4. Per D-05 / v2.0 standing policy: write up as **model-provider-property finding**, NOT as a defect to fix. Acceptable conclusions per the plan spec are "macro ≥ 0.93 holds cross-model" OR "macro < 0.93 — model-provider-property finding" — both satisfy CROSS-03.
- **GATE-06 audit pointer:** env-override surface only; reaffirms Plan 09-01's CLEAN audit verdict for the larger-sweep scope.

## Next Phase Readiness

- Plan 09-04 has all inputs it needs (4-dataset sweep JSON + reused BBB probe JSON + Claude baseline JSON + sweep log + summary).
- No blockers, no open architectural questions, no outstanding D-10 halt options.
- CROSS-01 + CROSS-02 jointly satisfied by this single sweep (D-02 single-arm Phase 8 retro-designation). Plan 09-04 closes the loop with CROSS-03.

## Self-Check: PASSED

- Sweep JSON exists: `results/ablation_results/ablation_20260531_063446.json` — verified via `python -c "import json; ..."` (4 datasets, all with `s_linker13.F1` field)
- Reused BBB probe JSON still readable: `results/ablation_results/ablation_20260531_055235.json` — verified (bigbluebutton.s_linker13.F1 = 0.8037)
- Sweep log exists with all 7 required fields: `.planning/phases/09-cross-gpt-5-2-cross-model-validation/09-03-SWEEP-LOG.md` — verified
- Task commit found in git log: `cb92001` (sweep + log) — verified
- All 4 new cache directories exist with 5 pickles each: `results/phase_cache_gpt54/s_linker13/{mediastore,teastore,teammates,jabref}/` — verified
- No edits to `run_ablation.py`, `llm_client.py`, `s_linker13.py`, or any `prompts*.py` — verified via `git status --short` showing only `.planning/` and `.claude/` untracked entries
- Plan's automated verification one-liner output: all 5 datasets resolved, macro F1 = 0.9077

---
*Phase: 09-cross-gpt-5-2-cross-model-validation*
*Completed: 2026-05-31*
