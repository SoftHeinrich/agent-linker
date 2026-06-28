# Quick Task 260628-dnl — S21 RQ1–4 RESULTS (gpt-5.4, no-reasoning, N=3 live)

`s_linker21` = canonical Full (promotion of spike-004 `s_linker20_union_layered`). All
numbers below are a **fresh live gpt-5.4 sweep** (not a replay): Full N=3 + No-Knowledge N=3,
both 5/5 datasets, 0 GIVEUP cells, extract faithfulness 15/15 PASS each. ~554 LLM calls.

## RQ1 — vs SOTA (macro over 5 projects, approach = mean of 3 runs)

| system | doc→model link F1 | doc→code file F1 |
|--------|------------------:|-----------------:|
| **approach S21 (GPT-5.4)** | **0.9360** | **0.9063** |
| approach s20_union (GPT-5.4, baseline) | 0.8939 | 0.8746 |
| approach s20_union (Claude) | 0.9276 | 0.9061 |
| Artemis (GPT-5.4) | 0.8355 | 0.8488 |
| TransArC | 0.7990 | 0.8026 |

S21 is the **top doc→model system** (+4.2pp over the gpt s20_union no-reasoning baseline,
above even the Claude approach) and ties the Claude approach on doc→code while beating gpt
s20_union (+3.2pp) and all baselines.

## RQ2 — architecture-driven size-aware metrics (doc→code)

| system | file F1 | worst-component F1 | harmonic-component F1 |
|--------|--------:|-------------------:|----------------------:|
| **approach S21 (GPT-5.4)** | **0.9063** | **0.7530** | **0.8838** |
| approach s20_union (GPT-5.4) | 0.8746 | 0.5908 | 0.7780 |
| approach s20_union (Claude) | 0.9061 | 0.6949 | 0.8237 |
| Artemis (GPT-5.4) | 0.8488 | 0.3455 | 0.4656 |
| TransArC | 0.8026 | 0.5445 | 0.6734 |

S21 is **best on every size-aware metric**, including worst-component F1 (the "recover every
component" lens): +16pp over gpt s20_union, +21pp over TransArC, +41pp over Artemis.

## RQ3 — validator contribution (avg of 3 runs, removal ΔF1)

| config | macro F1 | Δ vs Full |
|--------|--------:|----------:|
| Full | 0.9360 | — |
| − entity validator | 0.8870 | **+0.049** |
| − citation/coref validator | 0.8766 | **+0.059** |
| − both validators | 0.8430 | **+0.093** |

Both validators contribute non-redundant gains; together +9.3pp.

## RQ4 — per-module + knowledge A/B (avg of 3 runs)

| module set | macro F1 |
|-----------|--------:|
| entity-only | 0.8800 |
| coref-only | 0.2422 |
| Full (entity+coref) | 0.9360 |

Knowledge module (alias table + ambiguity map): Full **0.9360** vs No-Knowledge **0.8781**
→ **+5.79pp** macro contribution.

## Output files (provenance)

- RQ1/RQ2: `transarc-emp/reports/s21/RQ12_BIGTABLE_s21.csv`, `transarc-emp/reports/s21/RQ2_PANEL.csv`
  (S21 dump: `sota/recovered-links/model-doc/aalinker/gpt-5.4_s21/` +
  `doc-code/aalinker-composed/gpt-5.4_s21/`, built by `sota/recovered-links/build_s21.py`).
- RQ3/RQ4: `transarc-emp/mini-rq34/reports_s21/` (Full) + `reports_s21_noknow/` (No-Knowledge).
- Raw runs (gitignored): `agent-linker/results/v2.6.6_s21_gpt/`,
  `results/v2.6.6_s21_noknow_gpt/`; neutral extracts `results/v2.6.6_extracts_s21[_noknow]/`.

## Reproduce

```bash
# (sweeps already done; to re-score from frozen caches:)
cd agent-linker && python scripts/extract_s20union_caches.py --s21 && \
  python scripts/extract_s20union_caches.py --s21-noknow
cd ../transarc-emp/mini-rq34 && \
  RQ34_VARIANT=s_linker21 RQ34_OPENAI_SLOT=../../agent-linker/results/v2.6.6_s21_gpt \
    python3 rq34.py --backends openai --csv-root reports_s21 && \
  RQ34_VARIANT=s_linker21 RQ34_OPENAI_SLOT=../../agent-linker/results/v2.6.6_s21_gpt \
    python3 rq34_rq2.py --backends openai --csv-root reports_s21 && \
  RQ34_VARIANT=s_linker21 RQ34_OPENAI_SLOT=../../agent-linker/results/v2.6.6_s21_noknow_gpt \
    python3 rq34.py --backends openai --csv-root reports_s21_noknow --no-validate
cd ../../sota/recovered-links && python3 build_s21.py
cd ../../transarc-emp && mkdir -p reports/s21 && \
  python3 mini-src/rq12.py --csv reports/s21/RQ12_BIGTABLE_s21.csv
```
