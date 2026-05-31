---
phase: 06
plan: 06-08
status: partial-pivot-2
date: 2026-05-31
requirements: [EXT-01]
verdict: gate-05-fail-2 → design-pivot-2
---

# Plan 06-08 — Partial: Second GATE-05 Negative, Design Pivot #2

## Status

Plan 06-08 executed Task 1 (GATE-05 hard-tier dev loop on 4 alias-aware finalists) only. Tasks 2-5 superseded by a second user-directed design pivot. Both attempts at replacing `_has_standalone_mention` with an LLM primitive have failed GATE-05 on BBB:

| Approach | BBB F1 (best) | vs s_linker13 parent (0.8990) | vs pure-LLM floor (0.8108) | GATE-05 |
|---|---:|---:|---:|:---:|
| Pure-LLM (Plan 06-04) | 0.8108 | -8.8pp | — | FAIL |
| Alias-aware LLM (Plan 06-08) | 0.8319 (pre_alias) | -5.7pp | +2.1pp | FAIL |

Knowledge injection has measurable empirical benefit (+2.1pp over pure-LLM, 3 of 4 variants beat the rejected baseline) but the lift is insufficient to clear the parent regex baseline.

## GATE-05 Results — Alias-Aware Finalists

| Variant | TM F1 / Δ vs 0.9374 | BBB F1 / Δ vs 0.8890 | D-09 (>0.8108) | Dual-floor BBB (≥0.7473) |
|---|---|---|:---:|:---:|
| s_linker13g_pre_alias | 0.9310 (**-0.6pp**) | **0.8319** (-5.7pp) | PASS | PASS |
| s_linker13g_sem_alias | **0.9643** (+2.7pp) | 0.8000 (-8.9pp) | FAIL | PASS |
| s_linker13g_pre_full | 0.9231 (-1.4pp) | 0.8182 (-7.1pp) | PASS | PASS |
| s_linker13g_sem_full | 0.9204 (-1.7pp) | 0.8257 (-6.3pp) | PASS | PASS |

**Source data:** `results/ablation_results/ablation_ext01_gate05_alias.json` + 8 per-run JSONs/logs under `results/ablation_results/ext01_gate05_runs/` (gitignored, on disk).

## Empirical Findings (preserved for milestone documentation)

1. **Knowledge injection works but is insufficient.** 3 of 4 alias-aware variants beat the pure-LLM rejected baseline on BBB by 0.7–2.1pp. The alias map and linkmap context have real signal value, just not enough magnitude to recover the BBB recall the structural rule provided.
2. **Pareto trade-off is sharp.** sem_alias is the only variant to clear the TM GATE-05 floor (+2.7pp above 0.9374), but it has the worst BBB (-8.9pp). pre_alias has the best BBB but marginally misses TM (-0.6pp). No single configuration dominates.
3. **All 4 variants pass the dual-floor** (BBB ≥ 0.7473 vs s_linker12c). The failure is strictly vs the immediate parent (s_linker13), not vs the milestone-level floor.
4. **D-07's "coref antecedents" excluded by design** (cycle: coref consumes standalone_map upstream). The "full-knowledge" variants used `raw_seed_links` as the linkmap substitute. This may have undersold the full-knowledge hypothesis — but adding coref into the Tier-1 standalone map would require redesigning the Tier-1 DAG and is out of scope for this iteration.
5. **The failure mode is consistent across all 6 attempted variants** (2 pure-LLM + 4 alias-aware). 17 BBB FNs concentrated on HTML5 Client / HTML5 Server abbreviation references. Whatever the LLM standalone-mention primitive sees, it cannot bridge "the client" / "the server" / bare "HTML5" to the named BBB component as reliably as the regex's substring match.

## Disposition

- Plan 06-08: PARTIAL (Task 1 only — GATE-05 dev loop). Tasks 2-5 voided. This SUMMARY closes out 06-08.
- All 6 EXT-01 sub-variant files (`s_linker13g_pre.py`, `s_linker13g_sem.py`, `s_linker13g_pre_alias.py`, `s_linker13g_sem_alias.py`, `s_linker13g_pre_full.py`, `s_linker13g_sem_full.py`) retained as rejected-baseline ablation artifacts.
- Phase 6: design pivot #2 pending. Awaiting user direction on the new design angle.

## Files (on disk)

- 6 sub-variant `s_linker13g_*.py` files in `src/llm_sad_sam/linkers/experimental/` — rejected baselines
- `prompts_v2.py` — 6 standalone-mention prompts (2 pure-LLM from 06-01, 4 alias-aware from 06-05)
- `run_ablation.py` — 6 sub-variants registered (canonical=False)
- `06-DIFF-MATRIX.md` + `06-DIFF-MATRIX-ALIAS.md` — diff stage evidence
- `06-GATE-06-AUDIT.md` — generality audit (clean for all 6 prompt constants)
- `BENCHMARK_TABOO.md` — anti-pattern section added in 06-03
- `results/ablation_results/ablation_ext01_hardtier.json` + `ablation_ext01_gate05_alias.json` — GATE-05 evidence for both attempts (gitignored)
