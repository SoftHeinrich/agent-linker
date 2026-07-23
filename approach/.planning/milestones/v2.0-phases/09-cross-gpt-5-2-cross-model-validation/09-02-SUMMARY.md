---
phase: 09-cross-gpt-5-2-cross-model-validation
plan: 02
subsystem: testing
tags: [cross-model, gpt-5.4, openai, s_linker13, bigbluebutton, ablation, probe, gate]

# Dependency graph
requires:
  - phase: 09-cross-gpt-5-2-cross-model-validation/09-01
    provides: CLEAN GATE-06 audit verdict on harness + adapter shim — authorisation to invoke LLM calls without code edits
provides:
  - BBB probe JSON for s_linker13 on gpt-5.4 (`results/ablation_results/ablation_20260531_055235.json`)
  - Verified env-override invocation pattern (LLM_BACKEND, OPENAI_MODEL_NAME, PHASE_CACHE_DIR) for gpt-5.4 sweeps
  - Verified cache-namespacing (`results/phase_cache_gpt54/`) — no collision with Claude's `results/phase_cache/`
  - User adjudication on D-03 Step 2 / D-10 sanity gate (go → full sweep authorised)
affects: [09-03 full sweep, 09-04 comparison report]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Env-override invocation for cross-model runs (no harness code edits)"
    - "Per-model PHASE_CACHE_DIR namespacing to isolate checkpoints per backend"
    - "BBB-first hardest-dataset probe before committing to full sweep API cost"

key-files:
  created:
    - .planning/phases/09-cross-gpt-5-2-cross-model-validation/09-02-PROBE-LOG.md
    - results/ablation_results/ablation_20260531_055235.json
    - results/ablation_results/09-02-bbb-probe-stdout.log
    - results/ablation_results/s_linker13_bigbluebutton_links.csv
    - results/llm_logs/s_linker13_bigbluebutton_20260531_055235.json
    - results/phase_cache_gpt54/s_linker13/bigbluebutton/{layer1,entity_candidates,entity_decisions,layer2,final}.pkl
  modified: []

key-decisions:
  - "GO: BBB F1 0.8037 ≥ 0.6 sanity floor, no persistent harness errors → proceed to Plan 09-03 full 5-dataset sweep"
  - "Reuse this BBB probe JSON as the bigbluebutton row in the Plan 09-03 sweep (no BBB re-run; D-03 Step 3 path)"
  - "Env-override invocation surface is the audited path — no edits to run_ablation.py:436 setdefault"
  - "Single transient empty-response coref retry is harness-healthy, handled by existing retry path"

patterns-established:
  - "Cross-model BBB probe pattern: LLM_BACKEND=openai + OPENAI_MODEL_NAME=<model> + PHASE_CACHE_DIR=./results/phase_cache_<tag> + run_ablation.py --datasets bigbluebutton --variants <variant> | tee stdout.log"
  - "Per-backend cache namespacing prevents Claude/GPT pickle collision without code changes"

requirements-completed: []  # CROSS-01 is multi-plan; closes when 09-03 sweep completes per ROADMAP

# Metrics
duration: ~5min  # plan execution wall-clock (probe + log + adjudication); BBB run itself 48.4 s
completed: 2026-05-31
---

# Phase 9 Plan 02: BBB Probe on gpt-5.4 Summary

**BBB probe on gpt-5.4 cleared D-03 Step 2 sanity floor (F1 0.8037 ≥ 0.6) with zero harness errors — user issued `go` for the full 5-dataset sweep, with this probe JSON reused as the BBB row.**

## Performance

- **Duration:** ~5 min (plan execution wall-clock)
- **BBB probe runtime:** 48.4 s (gpt-5.4 end-to-end on s_linker13)
- **Started:** 2026-05-31T03:51:47Z (probe invocation)
- **Completed:** 2026-05-31T04:29:30Z (user adjudication recorded)
- **Tasks:** 2 (1 auto + 1 checkpoint:human-verify, both cleared)
- **Files modified/created:** 7 artifact files + 1 plan log + 1 plan summary

## Accomplishments

- BBB probe ran end-to-end on gpt-5.4 with no harness code edits (env-override path validated)
- BBB F1 = **0.8037** (P 0.9556, R 0.6935; TP 43, FP 2, FN 19) — **−1.7 pp** vs Claude Sonnet baseline (0.821), within BBB's documented jitter band
- Cache namespacing under `results/phase_cache_gpt54/` verified post-run; Claude's `results/phase_cache/` untouched
- D-03 Step 2 sanity floor (F1 ≥ 0.6) and D-10 cancellation rule (no persistent harness errors) both cleared
- User adjudicated `go`; Plan 09-03 is authorised to run a 4-dataset sweep (MS, TS, TM, JAB) and reuse this BBB JSON

## Probe Methodology

### Invocation (env-override only, no code edits)

```bash
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.4 \
PHASE_CACHE_DIR=./results/phase_cache_gpt54 \
python run_ablation.py --datasets bigbluebutton --variants s_linker13 \
  --results-dir results/ablation_results 2>&1 | tee results/ablation_results/09-02-bbb-probe-stdout.log
```

Three shell env overrides:

- `LLM_BACKEND=openai` → `run_ablation.py:get_backend()` (line 427) selects OpenAI dispatch.
- `OPENAI_MODEL_NAME=gpt-5.4` → wins over `setdefault` at `run_ablation.py:436`, `llm_client.py:112`, `s_linker13.py:171` (all three honour external env).
- `PHASE_CACHE_DIR=./results/phase_cache_gpt54` → namespaces s_linker13's per-phase pickle cache (`s_linker13.py:1160`) away from Claude's default `./results/phase_cache`.

Runner-printed backend descriptor: `openai (gpt-5.4)` (confirms override took effect).

### Why BBB-only first (D-03 Step 1)

- BBB is the **hardest dataset** with the widest historical variance band (~4 pp on Claude Sonnet, 5–12 link stdev on GPT per project memory).
- If the harness is broken (auth, model id rejection, rate limits, prompt-template breakage), BBB surfaces it cheapest and earliest.
- Clearing a minimum-sanity floor on BBB justifies the full-sweep API cost.

## Probe Results

Source JSON: `results/ablation_results/ablation_20260531_055235.json` (key path `bigbluebutton.s_linker13`).

| Metric | gpt-5.4 (this probe) | Claude Sonnet baseline | Delta |
|---|---|---|---|
| **F1** | **0.8037** | 0.821 | −1.7 pp |
| Precision | 0.9556 | — | — |
| Recall | 0.6935 | — | — |
| TP | 43 | — | — |
| FP | 2 | — | — |
| FN | 19 | — | — |
| n_links emitted | 45 | — | — |
| Runtime | 48.4 s | — | — |

- **FP breakdown:** 1 `coreference`, 1 `seed`.
- **Link sources:** `{seed: 37, entity: 5, coreference: 3}`.
- **Pipeline completion:** all 3 tiers (Knowledge Acquisition, Entity Pipeline + Coreference, Link Consolidation) ran end-to-end; checkpoints `layer1`, `entity_candidates`, `entity_decisions`, `layer2`, `final` all written; final dedup 45 links from 82 raw.

### Cache namespacing verification

Post-run `ls` confirmed pickles landed only under the fresh root:

```
results/phase_cache_gpt54/s_linker13/bigbluebutton/
├── entity_candidates.pkl  (~21.5 KB)
├── entity_decisions.pkl   (~1.3 KB)
├── final.pkl              (~2.5 KB)
├── layer1.pkl             (~3.6 KB)
└── layer2.pkl             (~8.3 KB)
```

`./results/phase_cache/` (Claude's default) was NOT touched — namespacing held.

### Transient events

- 1× empty-response retry on a coreference batch (`Coref batch: empty response, retrying...`) — handled by the existing retry path; **not** a halt-class event.
- 0× HTTP 4xx/5xx, 0× auth errors, 0× model-id rejections, 0× rate-limit waits.

## D-10 Floor Clearance

| D-10 condition | Threshold | Observed | Verdict |
|---|---|---|---|
| BBB F1 sanity floor | ≥ 0.6 | 0.8037 | **CLEAR** |
| No persistent harness errors | 0 unrecovered 401/403/429/5xx/404 | 0 (only 1 recovered empty-response coref retry) | **CLEAR** |

Both legs of D-10 cleared → no D-10 halt option triggered.

## User Adjudication

- **Option selected:** `go`
- **Timestamp (UTC):** 2026-05-31T04:29:30Z
- **Gating-rule citation:** D-03 Step 2 sanity floor cleared (BBB F1 0.8037 ≥ 0.6); D-10 cancellation rule cleared on both F1-floor and harness-health legs.
- **Effect:** Plan 09-03 authorised to run gpt-5.4 sweep on the remaining 4 datasets (MS, TS, TM, JAB), reusing this BBB JSON as the bigbluebutton row (D-03 Step 3 path — no BBB re-run).

## Task Commits

1. **Task 1: BBB probe invocation** — `9b17c31` (docs)
2. **Task 2: BBB probe go/halt decision** — `cc34f5f` (docs — user adjudication recorded)

**Plan metadata:** this SUMMARY is finalised in the next commit (`docs(09-02): finalize summary — go signal received`).

## Files Created/Modified

- `.planning/phases/09-cross-gpt-5-2-cross-model-validation/09-02-PROBE-LOG.md` — invocation command, BBB metrics, cache verification, retry log, user adjudication block
- `results/ablation_results/ablation_20260531_055235.json` — probe JSON with `bigbluebutton.s_linker13` block (reused by Plan 09-03)
- `results/ablation_results/09-02-bbb-probe-stdout.log` — full runner stdout
- `results/ablation_results/s_linker13_bigbluebutton_links.csv` — exported link CSV
- `results/llm_logs/s_linker13_bigbluebutton_20260531_055235.json` — phase-3 LLM call log
- `results/phase_cache_gpt54/s_linker13/bigbluebutton/{layer1,entity_candidates,entity_decisions,layer2,final}.pkl` — checkpoints

## Decisions Made

- **Reuse BBB probe JSON in Plan 09-03 (D-03 Step 3 path).** Re-running BBB would burn API cost for no additional evidence; the probe JSON is schema-compatible with the sweep JSON and can be merged at the comparison-report stage (Plan 09-04).
- **No harness edits.** The `setdefault` at `run_ablation.py:436` was left untouched even though it defaults to `gpt-5.2`; shell env wins, which is the audited surface from Plan 09-01.
- **One transient retry is healthy.** The empty-response coreference retry is part of the documented `approve-biased fallback on LLM failure` pattern (per CONTEXT §code_context); not a halt signal.

## Deviations from Plan

None — plan executed exactly as written. The probe ran on the first attempt, all logged fields populated, both tasks completed in order, user adjudicated `go` per D-03 Step 2.

## Issues Encountered

None blocking. One transient empty-response coreference retry observed and recovered by the existing retry path; documented in the probe log under "Retries / Transient Errors / Rate-Limit Waits".

## User Setup Required

None — `OPENAI_API_KEY` was already present in the environment; no new external service configuration introduced.

## Hand-Off to Plan 09-03

Plan 09-03 (full 5-dataset sweep on gpt-5.4) consumes this plan's outputs:

- **Reused JSON:** `results/ablation_results/ablation_20260531_055235.json` provides the `bigbluebutton.s_linker13` row directly. Plan 09-03 SHOULD NOT re-run BBB; instead it MUST merge this JSON with its new MS/TS/TM/JAB sweep JSON when computing macro F1 for the GATE-01 (macro ≥ 0.93) check.
- **Invocation pattern:** same env-override triple (`LLM_BACKEND=openai`, `OPENAI_MODEL_NAME=gpt-5.4`, `PHASE_CACHE_DIR=./results/phase_cache_gpt54`) with `--datasets mediastore teastore teammates jabref` (no bigbluebutton).
- **Cache root:** `results/phase_cache_gpt54/` already partially populated (BBB pickles); new dataset subdirectories will be created alongside without collision risk.
- **Variance disclosure:** Plan 09-04 comparison report must cite the BBB run as single-shot per D-09 default cost discipline; project memory's `5–12 link stdev` GPT variance band stands as the disclosed uncertainty envelope.

## Next Phase Readiness

- Plan 09-03 has all inputs it needs (probe JSON, invocation pattern, cache root, go signal).
- No blockers, no open architectural questions, no outstanding D-10 halt options.

## Self-Check: PASSED

- BBB probe JSON exists: `results/ablation_results/ablation_20260531_055235.json` (verified)
- Probe log exists with user adjudication block: `.planning/phases/09-cross-gpt-5-2-cross-model-validation/09-02-PROBE-LOG.md` (verified)
- Task commits found in git log: `9b17c31` (probe), `cc34f5f` (adjudication) (verified)
- Cache directory present under `results/phase_cache_gpt54/s_linker13/bigbluebutton/` (verified)
- No edits to `run_ablation.py`, `llm_client.py`, or `s_linker13.py` (env-override path only)

---
*Phase: 09-cross-gpt-5-2-cross-model-validation*
*Completed: 2026-05-31*
