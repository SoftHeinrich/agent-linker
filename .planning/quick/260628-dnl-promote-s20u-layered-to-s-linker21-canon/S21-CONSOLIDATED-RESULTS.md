# S21 Consolidated Results — all four sweeps (paper-fill reference)

`s_linker21` = canonical **Full** linker (promotion of spike-004 `s_linker20_union_layered`,
layered no-reasoning validator). This doc gathers **all four N=3 live sweeps** scored on the
same footing, so the paper numbers can be filled from one place.

| Sweep | backend | role (D-04 REVISED) | raw results dir | finished |
|-------|---------|---------------------|-----------------|----------|
| Full | GPT-5.4 (no reasoning) | **body** | `results/v2.6.6_s21_gpt` | 2026-06-28 10:53 |
| No-Knowledge | GPT-5.4 | body RQ4 A/B | `results/v2.6.6_s21_noknow_gpt` | 2026-06-28 11:50 |
| Full | Claude Sonnet (thinking off) | **appendix mirror** | `results/v2.6.6_s21_sonnet` | 2026-06-28 18:24 |
| No-Knowledge | Claude Sonnet | appendix RQ4 A/B | `results/v2.6.6_s21_noknow_sonnet` | 2026-06-28 19:44 |

All four: 3 runs × 5 projects, `.ALL_DONE`, 0 GIVEUP, extract faithfulness **15/15 PASS** each.
Every number below is the **mean of three independent runs**; macro = unweighted mean over the
five projects. GPT numbers reproduce `260628-dnl-RESULTS.md` exactly; Sonnet numbers are new
(scored 2026-06-28, this task).

---

## RQ1 — doc→model (SAD–SAM, link-level) macro

| system | macro P | macro R | macro **F1** |
|--------|--------:|--------:|------------:|
| **approach S21 (GPT-5.4)** — body | 0.9894 | 0.8913 | **0.9360** |
| **approach S21 (Claude)** — appendix | 0.9610 | 0.9010 | **0.9265** |
| approach s20_union (GPT-5.4) baseline | — | — | 0.8939 |
| approach s20_union (Claude) baseline | — | — | 0.9276 |
| Artemis (GPT-5.4) | — | — | 0.8355 |
| TransArC | — | — | 0.7990 |

### RQ1 doc→model per-project F1 (the SAD–SAM table, `tables/metrics_sad-sam.tex`)

| project | S21 GPT | S21 Claude | S21-noknow GPT | S21-noknow Claude |
|---------|--------:|-----------:|---------------:|------------------:|
| MediaStore | 0.9608 | 0.9437 | 0.7976 | 0.8202 |
| TeaStore | 0.9615 | 0.9701 | 0.8670 | 0.9013 |
| TeaMmates | 0.9008 | 0.9123 | 0.9187 | 0.9199 |
| BigBlueButton | 0.8571 | 0.8242 | 0.8072 | 0.7598 |
| JabRef | 1.0000 | 0.9820 | 1.0000 | 0.9820 |
| **Macro** | **0.9360** | **0.9265** | **0.8781** | **0.8766** |

## RQ1 — doc→code (file-level, composed with ArCoTL bridge) macro

| system | doc→code file **F1** |
|--------|--------------------:|
| **approach S21 (GPT-5.4)** — body | **0.9063** |
| **approach S21 (Claude)** — appendix | **0.9084** |
| approach s20_union (GPT-5.4) | 0.8746 |
| approach s20_union (Claude) | 0.9061 |
| Artemis (GPT-5.4) | 0.8488 |
| TransArC | 0.8026 |

## RQ2 — architecture-driven size-aware metrics (doc→code, mean of 3 runs)

| system | file F1 | sentence cov. | worst-comp F1 | harmonic-comp F1 |
|--------|--------:|--------------:|--------------:|-----------------:|
| **approach S21 (GPT-5.4)** | **0.9063** | 0.8417 | **0.7530** | **0.8838** |
| **approach S21 (Claude)** | 0.9084 | 0.8427 | 0.5760 | 0.7038 |
| approach s20_union (GPT-5.4) | 0.8746 | 0.8108 | 0.5908 | 0.7780 |
| approach s20_union (Claude) | 0.9061 | 0.8475 | 0.6949 | 0.8237 |
| Artemis (GPT-5.4) | 0.8488 | 0.7091 | 0.3455 | 0.4656 |
| TransArC | 0.8026 | 0.7511 | 0.5445 | 0.6734 |

> **Backend note worth flagging in the paper:** S21 GPT is best on every size-aware metric.
> S21 Claude has the highest *file* F1 (0.9084) but its size-aware *tail* is weaker
> (worst-comp 0.576, harmonic 0.704) — below both S21 GPT and the s20_union Claude baseline.
> The no-reasoning Sonnet config trades worst-component recall for file-level precision. This
> is the strongest argument for keeping GPT-5.4 in the body and Sonnet in the appendix.

## RQ3 — validator contribution (removal ΔF1, mean of 3 runs)

| config | GPT-5.4 F1 | GPT Δ | Claude F1 | Claude Δ |
|--------|-----------:|------:|----------:|---------:|
| Full | 0.9360 | — | 0.9265 | — |
| − entity validator | 0.8870 | +0.049 | 0.8837 | +0.043 |
| − citation/coref validator | 0.8766 | +0.059 | 0.9028 | +0.024 |
| − both validators | 0.8430 | **+0.093** | 0.8760 | **+0.051** |

Both validators contribute non-redundant gains on both backends; together +9.3pp (GPT) /
+5.1pp (Claude). On GPT the coref validator carries more weight; on Sonnet the entity
validator does.

## RQ4 — per-module + knowledge A/B (mean of 3 runs)

| module set | GPT-5.4 F1 | Claude F1 |
|------------|-----------:|----------:|
| entity-only | 0.8800 | 0.8607 |
| coref-only | 0.2422 | 0.6104 |
| Full (entity+coref) | 0.9360 | 0.9265 |

Knowledge module (alias table + ambiguity map): **GPT** Full 0.9360 vs No-Knowledge 0.8781
→ **+5.79pp**; **Claude** Full 0.9265 vs No-Knowledge 0.8766 → **+4.99pp**.
(coref-only is much stronger on Sonnet — 0.610 vs GPT 0.242 — i.e. gpt's coref candidates lean
harder on the entity gate to survive.)

---

## Output files (provenance)

**doc→model + doc→code + RQ2 (RQ1/RQ2):**
- GPT: `transarc-emp/reports/s21/RQ12_BIGTABLE_s21.csv`, `transarc-emp/reports/s21/RQ2_PANEL.csv`
  (S21 dumps `sota/recovered-links/{model-doc/aalinker,doc-code/aalinker-composed}/gpt-5.4_s21/`).
- Claude: same two CSVs (rq12 roster now has an `approach S21 (Claude)` row);
  dump slot `…/sonnet_s21/`, manifest `…/_manifest_s21_sonnet.csv`.
- **SOTA tracking:** both S21 backends are folded into `sota/recovered-links/UNIFIED_MANIFEST.csv`
  (now 125 rows: arcotl + `gpt-5.4_full`/`sonnet_full` baselines + `gpt-5.4_s21`/`sonnet_s21`).
  `build_s21_dump.py` rebuilds it by aggregating every per-task `_manifest*.csv`, so it stays
  complete without needing the (absent) upstream `transarc-emp/results` tree.

**RQ3/RQ4:**
- GPT: `transarc-emp/mini-rq34/reports_s21/` (Full) + `reports_s21_noknow/` (No-Knowledge).
- Claude: `transarc-emp/mini-rq34/reports_s21_sonnet/` (Full) + `reports_s21_noknow_sonnet/` (No-Knowledge).

**Neutral extracts (faithfulness-checked, gitignored raw runs):**
- GPT: `results/v2.6.6_extracts_s21[_noknow]/gpt/`
- Claude: `results/v2.6.6_extracts_s21_sonnet/sonnet/`, `results/v2.6.6_extracts_s21_noknow_sonnet/sonnet/`

## Reproduce (Sonnet path — the GPT path is in 260628-dnl-RESULTS.md)

```bash
# 1. extract sonnet caches -> neutral JSON (15/15 PASS each)
cd agent-linker
python scripts/extract_s20union_caches.py --s21-sonnet
python scripts/extract_s20union_caches.py --s21-noknow-sonnet

# 2. build the sonnet_s21 doc-code dump slot (env-overridable backend knobs)
cd ../transarc-emp
EXTRACTS_S21=../agent-linker/results/v2.6.6_extracts_s21_sonnet \
  S21_BE_DIR=sonnet S21_BE_TAG=claude S21_CONFIG=sonnet_s21 S21_MANIFEST_TAG=s21_sonnet \
  python3 mini-src/build_s21_dump.py
python3 mini-src/rq12.py --csv reports/s21/RQ12_BIGTABLE_s21.csv   # roster has S21 Claude row

# 3. RQ3/RQ4 (claude backend, reads phase-cache pickles directly)
cd mini-rq34
SF=../../agent-linker/results/v2.6.6_s21_sonnet
SN=../../agent-linker/results/v2.6.6_s21_noknow_sonnet
RQ34_VARIANT=s_linker21 RQ34_CLAUDE_SLOT=$SF python3 rq34.py     --backends claude --csv-root reports_s21_sonnet
RQ34_VARIANT=s_linker21 RQ34_CLAUDE_SLOT=$SF python3 rq34_rq2.py --backends claude --csv-root reports_s21_sonnet
RQ34_VARIANT=s_linker21 RQ34_CLAUDE_SLOT=$SN python3 rq34.py     --backends claude --csv-root reports_s21_noknow_sonnet --no-validate
```

## Paper-fill mapping (what each number replaces)

| paper target | current content | fill with |
|--------------|-----------------|-----------|
| `sections/results.tex` `0.xx` placeholders (RQ1 prose) | placeholders | S21 GPT 0.9360 (doc→model), 0.9063 (doc→code) vs Artemis 0.8355 / 0.8488 |
| `tables/metrics_sad-sam.tex` (per-project doc→model) | s20_union | S21 per-project table above (body=GPT, appendix=Claude) |
| `tables/metrics_sad-code.tex` (per-project doc→code) | s20_union | regenerate per-project from `gpt-5.4_s21` / `sonnet_s21` dump (rq12 reports macro only — per-project F1 not yet dumped to CSV) |
| `tables/cross_system.tex`, `tables/rq1_*.tex` | s20_union | S21 macro rows above |
| `table/rq2-summary.tex` | OUTDATED reasoning=medium | RQ2 panel above (S21 GPT body) |
| `table/rq3-validators.tex` | s20_union | RQ3 table above |
| `table/rq4-agents.tex` | s20_union | RQ4 module + knowledge table above |
| `appendix/rq3-rq4-mirror.tex` | Claude mirror | S21 Claude RQ3/RQ4 columns above |

**Still TODO for a complete float swap:** per-project doc→code file-F1 for the
`metrics_sad-code.tex` table is not yet emitted to a CSV (rq12 reports macro only). Everything
else needed for the body + appendix is in this doc.
