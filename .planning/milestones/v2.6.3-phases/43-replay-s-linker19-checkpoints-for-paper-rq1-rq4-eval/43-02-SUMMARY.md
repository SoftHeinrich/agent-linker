---
phase: 43-replay-s-linker19-checkpoints-for-paper-rq1-rq4-eval
plan: 02
subsystem: approach/replay-stage
tags: [replay, csv-contract, rq1, rq3, rq4, zero-llm, gate-01]
requires:
  - results/phase_cache/s_linker19/{claude,openai}/<project>/{layer1..4,final}.pkl (read-only)
  - src/llm_sad_sam/core/data_types_v2.py (SadSamLink + CandidateLink dataclasses on sys.path for pickle.load)
  - transarc-emp/src/lib/transarc_error_analysis.py (load_code_model_files, load_gs_sam_code_maps for SAM->code expansion)
  - ardoco/.../benchmark/<project>/goldstandards/goldstandard_sad_<year>-sam_<year>.csv (read-only)
provides:
  - scripts/v2.6.3/replay_common.py — PROJECTS, BACKENDS, BACKEND_DISPLAY, PHASE_CACHE_ROOT, OUTPUT_ROOT, assert_no_llm_env, load_layer, load_all_layers, load_gold_links
  - scripts/v2.6.3/replay_s19_to_csv.py — main, replay_sad_sam_for_project, replay_sad_code_for_project (RQ1 emitter)
  - scripts/v2.6.3/replay_s19_rq3.py — main, compute_variant_metrics, compute_validator_audit_counts (RQ3 emitter)
  - scripts/v2.6.3/replay_s19_rq4.py — main, compute_overlap_decomposition (RQ4 emitter)
  - scripts/v2.6.3/README.md — schema contract for downstream Plans 03/04
  - results/v2.6.3/{claude,openai}/<project>/{sad-sam,sad-code,rq3,rq3_audit,rq4,rq4_upset}.csv — 60 CSVs total
  - .gitignore — re-include rule for results/v2.6.3/ (the rest of results/ stays ignored)
affects:
  - Plan 03 — will consume sad-sam.csv + sad-code.csv via transarc-emp metrics_api
  - Plan 04 — will consume rq3.csv + rq3_audit.csv + rq4.csv + rq4_upset.csv via new transarc-emp paper formatters
  - Plan 05 — GATE-01 byte-equality of s_linker19.py + s_linker13_min.py preserved (zero edits under src/)
tech-stack:
  added: []
  patterns:
    - Replay-from-pickle (zero new LLM calls); pickle deserialization via sys.path injection of src/
    - LLM-call hard-guard at script entry (assert_no_llm_env on OPENAI_API_KEY / ANTHROPIC_API_KEY / LLM_BACKEND)
    - .gitignore negation for a single result-tree branch while keeping the parent ignored
key-files:
  created:
    - scripts/v2.6.3/__init__.py
    - scripts/v2.6.3/replay_common.py
    - scripts/v2.6.3/replay_s19_to_csv.py
    - scripts/v2.6.3/replay_s19_rq3.py
    - scripts/v2.6.3/replay_s19_rq4.py
    - scripts/v2.6.3/README.md
    - results/v2.6.3/.gitkeep
    - results/v2.6.3/{claude,openai}/{mediastore,teastore,teammates,bigbluebutton,jabref}/{sad-sam,sad-code,rq3,rq3_audit,rq4,rq4_upset}.csv (60 files)
  modified:
    - .gitignore (added re-include rule for results/v2.6.3/)
decisions:
  - "Honored CONTEXT D-08 exactly: four RQ3 variants {Full, NoEntityValid, NoCitation, NoValidator} all derived from layer3/layer4 fields (no LLM calls)."
  - "Honored CONTEXT D-05 / D-06: RQ4 = 2 linkers (Entity, Coref) with a 3-cell UpSet decomposition (only_E, both, only_C against gold)."
  - "Resolved results/ gitignore by adding a tightened pattern (`/results/*` + `!/results/v2.6.3/`) so the v2.6.3 contract CSVs are tracked while phase_cache/ etc. stay ignored — preserves the user's original intent while satisfying the plan's commit requirement for the .gitkeep + CSV outputs."
  - "Used transarc-emp's load_gs_sam_code_maps for sad-code composition; documented this as the chosen SAM->code source (D-02 permits approach/ -> evaluation/ imports)."
metrics:
  duration: "~18 minutes"
  completed: 2026-06-05
  tasks_completed: 4
  files_committed: 67
---

# Phase 43 Plan 02: Replay-stage scripts + 60 contract CSVs Summary

Built the three Phase 43 replay emitters (`replay_s19_to_csv.py`,
`replay_s19_rq3.py`, `replay_s19_rq4.py`) plus the shared
`replay_common.py` helper module, generated all 60 contract CSVs across
5 projects × 2 backends, and pinned the four CSV schemas in
`scripts/v2.6.3/README.md` so Plans 03 and 04 have a stable contract to
read against. Zero new LLM calls (verified by hard-guards at script
entry); zero edits under `src/llm_sad_sam/` (GATE-01 preserved).

## Tasks Completed

| Task | Name                                                    | Commit  | Files                                                                                  |
| ---- | ------------------------------------------------------- | ------- | -------------------------------------------------------------------------------------- |
| 1    | replay-stage common helpers + output dir scaffold       | ee1ea80 | scripts/v2.6.3/__init__.py, scripts/v2.6.3/replay_common.py, results/v2.6.3/.gitkeep, .gitignore |
| 2    | RQ1 sad-sam + sad-code CSV emitter                      | 9780c08 | scripts/v2.6.3/replay_s19_to_csv.py                                                    |
| 3    | RQ3 + RQ4 replay emitters and CSV schema README         | 70a9b5c | scripts/v2.6.3/replay_s19_rq3.py, scripts/v2.6.3/replay_s19_rq4.py, scripts/v2.6.3/README.md |
| 4    | Generate 60 RQ1/RQ3/RQ4 CSVs across 5 projects × 2 backends | bf94508 | 60 files under results/v2.6.3/                                                          |

## CSV Schemas Pinned

| File             | Header                                                   | Rows per file              |
| ---------------- | -------------------------------------------------------- | -------------------------- |
| `sad-sam.csv`    | `modelElementID,sentence,source`                         | one per s_linker19 link    |
| `sad-code.csv`   | `sentence,codeID`                                        | sad-sam ⨝ gold sam-code    |
| `rq3.csv`        | `variant,tp,fp,fn,precision,recall,f1`                   | 4 (Full / NoEntityValid / NoCitation / NoValidator) |
| `rq3_audit.csv`  | `validator,killed_gold,killed_spurious,kept_gold,kept_spurious` | 2 (entity, coref)   |
| `rq4.csv`        | `linker,tps_caught,unique_tps,fps,delta_f1_if_removed`   | 2 (Entity, Coref)          |
| `rq4_upset.csv`  | `cell,count`                                             | 3 (only_E, both, only_C)   |

Total file count under `results/v2.6.3/`: **60** CSVs (5 projects × 2 backends × 6 files per pair).

## Spot-check numbers (Claude / mediastore)

- RQ3 `Full`: TP=27, FP=0, FN=4, F1=0.931 — matches final.pkl link count.
- RQ3 `NoEntityValid`: TP=27, FP=3, FN=4, F1=0.885 — entity validator killed 3 spurious candidates (consistent with rq3_audit `entity.killed_spurious=3`).
- RQ4 UpSet: only_E=22, both=1, only_C=4 — coref-only contributes 4 unique gold TPs.

## Zero-LLM Guard Verification

- `OPENAI_API_KEY=x python3 scripts/v2.6.3/replay_s19_rq3.py …` → exit 1 with RuntimeError citing CONTEXT D-01 / D-14.
- `LLM_BACKEND=openai python3 scripts/v2.6.3/replay_s19_rq3.py …` → exit 1 (same guard).
- `LLM_BACKEND=checkpoint python3 scripts/v2.6.3/replay_s19_rq3.py …` → exit 0 (offline replay allowed).
- No `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` set during the `--all` runs; verified by inspecting the live env.

## Deviations from Plan

### Rule 3 — Auto-fixed blocking issues

**1. `.gitignore` blocks the plan's required `results/v2.6.3/` commit**

- **Found during:** Task 1 (`git add results/v2.6.3/.gitkeep` rejected).
- **Issue:** The repo's `.gitignore` had a blanket `results/` rule, which made the plan's required output-tree commit impossible without `git add -f`. The plan's must_haves explicitly list `approach/results/v2.6.3/.gitkeep` as a tracked artefact and require the contract CSVs to be tracked outputs for downstream Plans 03/04.
- **Fix:** Replaced the single `results/` rule with `/results/*` + `!/results/v2.6.3/`. The change leaves every other subdir of `results/` (including `results/phase_cache/`, `results/ablation_results/`, and the dated JSON dumps) ignored exactly as before, but allows the v2.6.3 contract tree to be tracked. Verified by running `git check-ignore -v results/phase_cache/...final.pkl` → still ignored; `git check-ignore -v results/v2.6.3/.gitkeep` → no longer ignored. No use of `git add -f`.
- **Files modified:** `.gitignore`
- **Commit:** ee1ea80

### Informational deviations (not rule-flagged)

- The plan listed `approach/scripts/v2.6.3/eval_rq1_baselines.py` as pre-existing (from the LiSSA RQ1 work) and instructed me to leave it alone unless explicitly required. Confirmed it was not modified.

## Self-Check: PASSED

Verification commands and results (run from `/mnt/hostshare/ardoco-home/agent-linker`):

```
$ find results/v2.6.3 -name "sad-sam.csv"  | wc -l    → 10
$ find results/v2.6.3 -name "sad-code.csv" | wc -l    → 10
$ find results/v2.6.3 -name "rq3.csv"      | wc -l    → 10
$ find results/v2.6.3 -name "rq3_audit.csv"| wc -l    → 10
$ find results/v2.6.3 -name "rq4.csv"      | wc -l    → 10
$ find results/v2.6.3 -name "rq4_upset.csv"| wc -l    → 10
$ git diff --stat src/                                 → (empty)
$ git log --oneline -5 | grep -c "43-02"               → 4
```

All four `(43-02)` commits resolve to existing hashes (ee1ea80, 9780c08, 70a9b5c, bf94508). All 60 CSVs present. No `src/` modifications. README contains the four `## sad-*` / `## rq*` schema headings + `## Invariants` + the D-08 derivation block.
