---
quick_id: 260628-dnl
slug: promote-s20u-layered-to-s-linker21-canon
date: 2026-06-28
status: in-progress
---

# Quick Task 260628-dnl — SUMMARY

Promote `s_linker20_union_layered` → canonical **`s_linker21`** (paper Full) + run RQ1–4
results on **gpt-5.4** (no-reasoning). User decisions: canonical=True; launch live sweeps
now; gpt-5.4 only.

## Done (committed)

| Commit | Repo | What |
|--------|------|------|
| `34b3239` | agent-linker | `s_linker21.py` (verbatim copy of layered, class SLinker21, canonical=True) + registry (`s_linker21` canonical Full, `s_linker21_noknow` no_knowledge=True). |
| `d8f3508` | agent-linker | `run_s21_gpt_n3.sh`, `run_s21_noknow_gpt_n3.sh` (gpt-5.4, no-reasoning, N=3, per-run PHASE_CACHE_DIR). |
| `8a6ef4b` | transarc-emp | `mini-rq34/rq34.py`: `VARIANT` env-overridable via `RQ34_VARIANT`. |
| `bd163a3` | agent-linker | `extract_s20union_caches.py`: `--s21` / `--s21-noknow` paths + `variant_subdir` param (regression: s20_union still 30/30 PASS). |

**Validation:** GATE-01 holds (s13_min/s19/s20_union .py untouched). GATE-06 holds (rubric
copied verbatim from taboo-clean source). S21 behaviourally byte-identical to
s_linker20_union_layered (diff = docstring + class name + _VARIANT_NAME only).
**Smoke (live, gpt-5.4, no-reasoning):** jabref P=100/R=100/F1=100 (TP18/FP0), 10 calls/22s,
phase_cache layer1–4+final written in the exact `phase_cache/s_linker21/openai/<proj>/`
layout rq34 expects. Matches spike-004's gpt jabref→100.

## In flight

**Sweeps launched** as harness-tracked background job `bdjs32klu` (chained):
`run_s21_gpt_n3.sh && run_s21_noknow_gpt_n3.sh`. Outputs:
`results/v2.6.6_s21_gpt/run{1,2,3}/` (Full) + `results/v2.6.6_s21_noknow_gpt/run{1,2,3}/`
(No-Knowledge). Both gitignored. Progress: `logs/v2.6.6_s21_gpt/PROGRESS.log`,
`logs/v2.6.6_s21_noknow_gpt/PROGRESS.log`; completion markers `logs/*/.ALL_DONE`.
ETA ~3–5h (mostly built-in cooldowns). Projected ~$60–120.

## Post-sweep continuation (run when bdjs32klu completes / both .ALL_DONE exist)

All `$0`, no LLM. From `agent-linker/` unless noted.

```bash
# 0. confirm sweeps done
ls logs/v2.6.6_s21_gpt/.ALL_DONE logs/v2.6.6_s21_noknow_gpt/.ALL_DONE

# 1. extract S21 caches -> neutral JSON (expect 15/15 PASS each)
python scripts/extract_s20union_caches.py --s21
python scripts/extract_s20union_caches.py --s21-noknow

# 2. RQ3 + RQ4 (validator + module ablation, gpt-5.4)  [transarc-emp/mini-rq34]
cd ../transarc-emp/mini-rq34
RQ34_VARIANT=s_linker21 \
RQ34_OPENAI_SLOT=../../agent-linker/results/v2.6.6_s21_gpt \
  python3 rq34.py            # -> reports/rq3_*.csv, reports/rq4_*.csv (openai rows)
RQ34_VARIANT=s_linker21 \
RQ34_OPENAI_SLOT=../../agent-linker/results/v2.6.6_s21_gpt \
  python3 rq34_rq2.py        # RQ3/RQ4 composed to doc-to-code (size-aware)

# 3. RQ4 knowledge A/B: Full vs No-Knowledge macro delta (gpt-5.4)
#    Compare final-link macro-F1 of v2.6.6_s21_gpt vs v2.6.6_s21_noknow_gpt
#    (use each run's ablation_*.json macro, or score the extract JSONs).

# 4. RQ1 + RQ2 (SOTA + size-aware) — needs the sota build pointed at S21 extracts:
#    a) build aalinker dump from results/v2.6.6_extracts_s21 (gpt-5.4 Full) into a new
#       config slot (e.g. gpt-5.4_s21_full) OR overwrite gpt-5.4_full (S21 IS the new
#       canonical Full — paper-level decision; default to a NEW slot, non-destructive).
#       sota build_unified.py is hardwired to EXTRACTS=v2.6.6_extracts + config names;
#       adapt EXTRACTS/config for S21 (small edit) then run it.
#    b) add an "approach S21 (GPT-5.4)" row to mini-src/rq12.py ROSTER pointing at the
#       new config, then:  cd ../transarc-emp && python3 mini-src/rq12.py
#       -> reports/RQ12_BIGTABLE.csv + reports/RQ2_PANEL.csv

# 5. finalize: write final result CSVs into working/out/ (or quick dir), update this
#    SUMMARY (status: complete) + STATE.md Quick Tasks row + commit docs.
```

## Expected numbers (from spike-004, sanity targets)
- gpt-5.4 macro 89.4 (s20_union no-reasoning) → **93.2** (S21/layered), every dataset up,
  coref FP 13→2, implicit-FN flat. Per-dataset: ms 96.6, ts 98.7, tm 89.0, bbb 81.6, jab 100.
- Smoke jabref already hit 100 ✓.

## Notes / landmines
- `s_linker21_noknow` keeps `_VARIANT_NAME="s_linker21"`, so its phase_cache nests under
  `…/s_linker21/` too — kept in a SEPARATE base (`v2.6.6_s21_noknow_gpt`) so no clobber
  (mirrors the s20_union_noknow landmine). The extract `--s21-noknow` uses
  `variant_subdir="s_linker21"` accordingly.
- gpt-5.4 no-reasoning = `OPENAI_REASONING_EFFORT` unset (scripts `unset` it explicitly).
- If sweep has GIVEUP cells, re-run the script (resume via per-cell `.done` markers).
