---
phase: 17-confirmation-tier
plan: 1
subsystem: voyager-training
tags:
  - voyager
  - training-run
  - gpt-5.4
  - confirmation-tier
  - 3-split
dependency_graph:
  requires:
    - "results/voyager_v4_beta/mainline/final_bank.json (Phase 16 — warm-start reference)"
    - "scripts/voyager_train_tlr_v4_beta.py (v4 β harness)"
    - "src/llm_sad_sam/linkers/experimental/s_linker14_voyager.py"
  provides:
    - "results/voyager_v4_beta/split1_replication/range_summary.json"
    - "results/voyager_v4_beta/split1_replication/final_bank.json"
    - "results/voyager_v4_beta/split2_bbb_in_train/range_summary.json"
    - "results/voyager_v4_beta/split2_bbb_in_train/final_bank.json"
    - "results/voyager_v4_beta/split3_rotated_holdout/range_summary.json"
    - "results/voyager_v4_beta/split3_rotated_holdout/final_bank.json"
    - "logs/voyager_v4_beta/eval_split1.log"
    - "logs/voyager_v4_beta/eval_split2.log"
    - "logs/voyager_v4_beta/eval_split3.log"
  affects:
    - "17-P2 cross-split aggregation + final verdict"
tech_stack:
  patterns:
    - "Fresh-start per-split range runs (no warm-start from mainline banks)"
    - "Probation gate: commit patch only if delta >= 0"
    - "Bank format: slot_patterns wrapper (v4b)"
key_files:
  created: []
  modified:
    - "logs/voyager_v4_beta/confirmation_split1.log"
    - "logs/voyager_v4_beta/confirmation_split2.log"
    - "logs/voyager_v4_beta/confirmation_split3.log"
    - "logs/voyager_v4_beta/eval_split1.log"
    - "logs/voyager_v4_beta/eval_split2.log"
    - "logs/voyager_v4_beta/eval_split3.log"
decisions:
  - "All 3 splits ran 5 passes without convergence — D kept proposing patterns but probation gate blocked most"
  - "Split 2 bank empty: every pass rolled back by probation (BBB hard to train for)"
  - "Split 3 strongest bank (8 patterns) despite TM/JAB being smaller — TS+TM+JAB distillator generalizes better"
  - "results/ gitignored — range_summary.json and per-project banks on disk only; logs committed"
metrics:
  duration: "~4 hours (3 sequential range runs + 3 evals)"
  completed: "2026-06-01"
  tasks_completed: 9
  files_changed: 6
---

# Phase 17 Plan 1: 3-Split Confirmation Runs + Per-Split Banks + Per-Split Eval Summary

**One-liner:** 3-split Voyager v4 β confirmation runs (gpt-5.4, fresh-start) — 5 passes each, probation-gated banks, 5-dataset eval per split.

## Results

### Per-Split Range Summary

| Split | Train | Test (holdout) | Passes | Converged | Train Macro F1 | Bank Patterns |
|-------|-------|----------------|--------|-----------|----------------|---------------|
| split1_replication | MS+TS+TM | BBB+JAB | 5 | No | 0.8941 | 2 (2 slots) |
| split2_bbb_in_train | MS+TS+BBB | TM+JAB | 5 | No | 0.8855 | 0 (empty) |
| split3_rotated_holdout | TS+TM+JAB | MS+BBB | 5 | No | 0.9561 | 8 (5 slots) |

### Per-Split 5-Dataset Eval (gpt-5.4)

| Split | MS | TS | TM | BBB | JAB | Macro F1 |
|-------|----|----|-----|-----|-----|----------|
| split1 (2 patterns) | 96.7% | 94.5% | 83.3% | 77.2% | 100.0% | **90.3%** |
| split2 (0 patterns, axiom-only) | 95.1% | 93.1% | 82.6% | 76.5% | 97.3% | **88.9%** |
| split3 (8 patterns) | 91.5% | 94.3% | 85.0% | 78.6% | 100.0% | **89.9%** |

### Key Observations

- **No split converged** — D kept proposing patterns at each pass but probation gate (macro delta >= 0) rejected most. All 3 splits hit the 5-pass cap.
- **Split 1 trained F1 (0.8941) < Split 3 trained F1 (0.9561)** — split 3 train set (TS+TM+JAB) is easier to learn. Split 3 also built a richer bank (8 patterns vs 2).
- **Split 2 bank empty** — MS+TS+BBB train set: BBB is difficult; every proposed pattern batch failed probation (all deltas negative). Eval result (88.9%) is pure axiom-only performance.
- **Bank lift vs axiom-only**: split1 +1.4pp (90.3% vs 88.9%); split3 +1.0pp (89.9% vs 88.9%). Modest positive signal.
- **TM and BBB are consistent weak spots** across all splits (TM: 82-85%, BBB: 76-79%). These are the largest/hardest datasets.
- **JAB is stable at 97-100%** — small dataset, few ambiguous components, well-learned.
- **MS generalizes well** (91-97%) across all splits — simple architecture, few components.

## Artifacts Produced

All output files reside in `results/voyager_v4_beta/` (gitignored). Logs committed:

| Artifact | Path | Status |
|----------|------|--------|
| Split 1 range log | `logs/voyager_v4_beta/confirmation_split1.log` | committed 1a29509 |
| Split 1 eval log | `logs/voyager_v4_beta/eval_split1.log` | committed 23f4ed8 |
| Split 2 range log | `logs/voyager_v4_beta/confirmation_split2.log` | committed 4bf51ae |
| Split 2 eval log | `logs/voyager_v4_beta/eval_split2.log` | committed 0d32b52 |
| Split 3 range log | `logs/voyager_v4_beta/confirmation_split3.log` | committed b349374 |
| Split 3 eval log | `logs/voyager_v4_beta/eval_split3.log` | committed 70c87ab |

Untracked (on disk, gitignored):
- `results/voyager_v4_beta/split1_replication/{range_summary.json, mediastore_bank.json, teastore_bank.json, teammates_bank.json, final_bank.json}`
- `results/voyager_v4_beta/split2_bbb_in_train/{range_summary.json, mediastore_bank.json, teastore_bank.json, bigbluebutton_bank.json, final_bank.json}`
- `results/voyager_v4_beta/split3_rotated_holdout/{range_summary.json, teastore_bank.json, teammates_bank.json, jabref_bank.json, final_bank.json}`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Aggregation script format mismatch**
- **Found during:** Task 1b (pre-execution review)
- **Issue:** Plan's aggregation script used `bank.get(slot, [])` treating slots as top-level keys, but actual bank format wraps everything under `{"slot_patterns": {...}}`. Per confirmed format in `mainline/final_bank.json` and per-project banks.
- **Fix:** Used corrected format `bank.get("slot_patterns", {}).get(slot, [])` in all 3 aggregation runs.
- **Files modified:** Inline Python scripts (not persisted as separate files)

**2. [Rule 2 - Discovery] results/ gitignored**
- **Found during:** Task 1b commit
- **Issue:** `.gitignore` blocks `results/` — range_summary.json and bank files cannot be committed.
- **Fix:** Committed log files only (not gitignored). SUMMARY documents artifact locations on disk.
- **Files modified:** N/A

## Threat Flags

None. No new network endpoints, auth paths, or schema changes introduced.

## Known Stubs

None. All eval runs completed with real LLM output.

## Decisions Made

1. Fresh-start banks per split (no warm-start from mainline). Confirmed correct science.
2. Aggregation uses per-project bank union by pattern_id (first-seen wins for duplicates).
3. `slot_patterns` wrapper format used throughout (confirmed by reading actual bank files).
4. Empty bank for split2 is valid — captured as axiom-only baseline, useful for P2 cross-split analysis.

## Self-Check: PASSED

- `results/voyager_v4_beta/split1_replication/range_summary.json` — exists on disk (confirmed)
- `results/voyager_v4_beta/split1_replication/final_bank.json` — exists on disk (confirmed)
- `results/voyager_v4_beta/split2_bbb_in_train/range_summary.json` — exists on disk (confirmed)
- `results/voyager_v4_beta/split2_bbb_in_train/final_bank.json` — exists on disk (confirmed)
- `results/voyager_v4_beta/split3_rotated_holdout/range_summary.json` — exists on disk (confirmed)
- `results/voyager_v4_beta/split3_rotated_holdout/final_bank.json` — exists on disk (confirmed)
- `logs/voyager_v4_beta/eval_split1.log` — committed 23f4ed8
- `logs/voyager_v4_beta/eval_split2.log` — committed 0d32b52
- `logs/voyager_v4_beta/eval_split3.log` — committed 70c87ab
- All 3 `range_summary.json` contain `"split"` field matching expected split name
- All 3 eval logs contain per-dataset F1 for all 5 datasets
