---
phase: 06
plan: 07
subsystem: llm-linker / ext-01 / diff-stage
requirements: [EXT-01]
tags: [llm-linker, ext-01, d-02, diff-stage, alias-aware, gate-05]
one-liner: "Anchor-diff matrix for the 4 alias-aware variants vs regex + dual pure-LLM baselines, denominator-aware Jaccard skip retires the BBB-kurento artefact, user adjudication promotes all 4 variants to Plan 06-08."
dependency-graph:
  requires:
    - "Plan 06-03: scripts/ext01_diff_stage.py harness + drop rule + denominator-aware adjudication pattern"
    - "Plan 06-04: cached pure-LLM baseline pickles at results/phase_cache/s_linker13g_{pre,sem}/<ds>/standalone_map.pkl"
    - "Plan 06-06: four new alias-aware variant modules (s_linker13g_{pre,sem}_{alias,full}) registered in run_ablation.py"
    - "BENCHMARK_TABOO.md §'Tailored Code Anti-Patterns' (line 70) — denominator-aware-update precedent"
  provides:
    - "results/ablation_results/ablation_ext01_diff_alias.json — merged 3-baseline diff matrix"
    - ".planning/phases/06-.../06-DIFF-MATRIX-ALIAS.md — human-readable report + User adjudication + Final finalist set"
    - "results/phase_cache/s_linker13g_{pre_alias,sem_alias,pre_full,sem_full}/<ds>/standalone_map.pkl — populated for Plan 06-08 zero-recompute reuse"
  affects:
    - "Plan 06-08 GATE-05 dev loop + full sweep — finalist set = all 4 alias-aware variants"
tech-stack:
  added: []
  patterns:
    - "denominator-aware Jaccard skip (single module-level boolean, no per-cell tuning surface)"
    - "dual-baseline diff mode (regex drop-gating + pure-LLM informational)"
    - "structural-property skip (skip on |S_baseline|=0, not on benchmark cell names)"
key-files:
  created:
    - ".planning/phases/06-ext-01-project-agnostic-standalone-mention-llm-primitive/06-DIFF-MATRIX-ALIAS.md"
    - "results/ablation_results/ablation_ext01_diff_alias.json"
    - "results/ablation_results/ablation_ext01_diff_alias_regex.json"
    - "results/ablation_results/ablation_ext01_diff_alias_purellm_pre.json"
    - "results/ablation_results/ablation_ext01_diff_alias_purellm_sem.json"
  modified:
    - "scripts/ext01_diff_stage.py — denominator-aware skip + dual-baseline mode + 4 new variant entries"
decisions:
  - "Finalist set for Plan 06-08 = all 4 alias-aware variants (user adjudication 2026-05-30T16:56:44Z, option proceed-all-4)"
  - "Denominator-aware Jaccard skip retires the BBB-kurento artefact without per-cell tuning (DENOMINATOR_AWARE_J_SKIP = True; structural property |S_baseline|=0, NOT a cell-name allowlist)"
  - "Regex baseline = drop-decision-gating; pure-LLM baselines = informational (D-09 verification only)"
  - "D-09 BBB-identity finding (alias context does not change Tier-1b standalone map on BBB) is a downstream hypothesis, not a diff-stage drop signal"
metrics:
  duration: "Plan-07 execution wave (Task 1 harness extension + Task 2 dual-baseline run + Task 3 user adjudication)"
  tasks-completed: 3
  files-modified: 1
  files-created: 5
  completed-date: "2026-05-30"
---

# Phase 06 Plan 07: Alias-Aware Diff Matrix Summary

Anchor-diff stage (D-02) extended for the 4 alias-aware variants produced by Plan 06-06. The Plan 06-03 harness gains denominator-aware Jaccard skip + dual-baseline mode (regex AND pure-LLM-pre/sem) without per-cell tuning. All 4 variants pass the mechanical drop rule against the regex baseline; the user adjudicated `proceed-all-4`, promoting the full quartet to Plan 06-08.

## Per-variant rollup snapshots

### vs regex baseline (drop-decision-gating)

| Variant | MS min J | TS min J | TM min J | BBB min J | JAB min J | TM #C<0.5 | BBB max D |
|---------|---------:|---------:|---------:|----------:|----------:|----------:|----------:|
| s_linker13g_pre_alias | 1.000 | 0.333 | 0.429 | 0.600 | 1.000 | 1/8 | 2 |
| s_linker13g_sem_alias | 1.000 | 0.400 | 0.625 | 0.600 | 1.000 | 0/8 | 2 |
| s_linker13g_pre_full  | 1.000 | 0.333 | 0.429 | 0.600 | 0.800 | 2/8 | 2 |
| s_linker13g_sem_full  | 1.000 | 0.400 | 0.429 | 0.600 | 0.800 | 1/8 | 2 |

All four clear `min_J > 0.3`, `max_D ≤ 10`, `#C<0.5 ≤ 25%` on hard-tier datasets (TM, BBB).

### vs pure-llm-pre baseline (D-09 informational)

| Variant | TM min J | TM #C<0.5 | TM max D | BBB min J |
|---------|---------:|----------:|---------:|----------:|
| s_linker13g_pre_alias | 0.833 | 0/8 | 2 | 1.000 |
| s_linker13g_sem_alias | 0.429 | 1/8 | 5 | 1.000 |
| s_linker13g_pre_full  | 0.455 | 1/8 | 6 | 1.000 |
| s_linker13g_sem_full  | 0.727 | 0/8 | 4 | 1.000 |

### vs pure-llm-sem baseline (D-09 informational)

| Variant | TM min J | TM #C<0.5 | TM max D | BBB min J |
|---------|---------:|----------:|---------:|----------:|
| s_linker13g_pre_alias | 0.429 | 1/8 | 7 | 1.000 |
| s_linker13g_sem_alias | 0.750 | 0/8 | 4 | 1.000 |
| s_linker13g_pre_full  | 0.429 | 2/8 | 7 | 1.000 |
| s_linker13g_sem_full  | 0.429 | 1/8 | 6 | 1.000 |

## Drop decisions

Mechanical drop rule against regex baseline — **ZERO drops**:

| Variant | Drop? | Driver |
|---------|-------|--------|
| s_linker13g_pre_alias | NO | All thresholds clear |
| s_linker13g_sem_alias | NO | All thresholds clear |
| s_linker13g_pre_full  | NO | TM #C<0.5 = 2/8 = 25.0% (exactly at threshold, rule fires on strict `>`); cleared |
| s_linker13g_sem_full  | NO | All thresholds clear |

The denominator-aware skip (`DENOMINATOR_AWARE_J_SKIP = True`) retires the Plan 06-03 BBB-`kurento` artefact: on each variant, 1–7 components per dataset have `|S_baseline| = 0`; J on those cells is skipped, D is still enforced. No Pitfall-5 escalation triggered.

## D-09 evidence summary

**Question:** Do the alias-aware variants demonstrably differ from the rejected pure-LLM baselines (`s_linker13g_pre`, `s_linker13g_sem`)?

**TM (answer: yes, substantially):** All 4 variants accept MORE standalone mentions than the pure-LLM baselines on Storage / Client / Common / E2E / UI — "alias-recovery" signal. `*_full` more divergent than `*_alias` (linkmap injection nudges acceptance further). Max symmetric difference reaches 7 sentences per component.

**BBB (answer: no — identical Tier-1b output):** All 4 new variants produce the **identical** BBB standalone-mention map as both pure-LLM baselines (per-component J = 1.0 on every BBB component vs both pure-llm-pre and pure-llm-sem). The Recording-Service / kurento / WebRTC-SFU / HTML5-Server / HTML5-Client anchors are the same set across all 6 variants (4 new + 2 rejected baselines). **Implication:** if Plan 06-08 produces BBB F1 improvement, the source must be downstream tier behavior (alias/linkmap context affecting alias-matching / coref), NOT the Tier-1b standalone-map output. This is the hypothesis Plan 06-08 will test in the full pipeline.

**MS / JAB (answer: low-density, easy passes):** Near-identical anchors across regex, pure-LLM, and all 4 new variants. Expected to clear Plan 06-08 GATE-05 without contention.

**TM divergence pattern:** Consistently shaped — `|S_variant| > |S_baseline|` (alias-recovery direction). The `pre`-family shows more divergence vs regex than the `sem`-family (regex pre-filter prunes sentences before the LLM judge sees them). Against pure-LLM baselines, the `*_full` variants show more divergence than `*_alias` (linkmap context nudges the judge more).

## LLM cost actuals

Regex-baseline run only (pure-LLM-baseline runs are cache-only, zero LLM cost via cached pickles):

| Variant | MS | TS | TM | BBB | JAB | Total |
|---------|---:|---:|---:|---:|---:|---:|
| s_linker13g_pre_alias | 20 | 16 | 84 | 35 | 20 | 175 |
| s_linker13g_sem_alias | 20 | 16 | 87 | 35 | 20 | 178 |
| s_linker13g_pre_full  | 20 | 16 | 84 | 35 | 19 | 174 |
| s_linker13g_sem_full  | 20 | 15 | 70 | 35 | 20 | 160 |
| **Total**             | 80 | 63 | 325 | 140 | 79 | **687** |

~2x Plan 06-03's 333-pair budget (expected for 4 variants vs 2). Within the CONTEXT.md D-06 relaxed-budget envelope. All 20 variant pickles persisted under `results/phase_cache/s_linker13g_{pre_alias,sem_alias,pre_full,sem_full}/<ds>/standalone_map.pkl` for Plan 06-08 zero-recompute reuse on the hard-tier GATE-05 leg.

## User adjudication

**Timestamp (UTC):** 2026-05-30T16:56:44Z
**Option selected:** `proceed-all-4`

**User reasoning (verbatim):**

> Maximum empirical signal. Plan 06-08 resolves both axes (alias-only vs full-knowledge AND pre vs sem) in one sweep. The D-09 BBB-identity finding (alias context doesn't change Tier-1b standalone-map output on BBB) is a hypothesis to be tested downstream in the full pipeline, not grounds to drop *_full at the diff-stage gate.

## Final finalist set for Plan 06-08

| Variant | Context type | Tier-1b mode |
|---------|--------------|--------------|
| `s_linker13g_pre_alias` | alias-only | regex pre-filter |
| `s_linker13g_sem_alias` | alias-only | LLM-only |
| `s_linker13g_pre_full`  | alias + linkmap | regex pre-filter |
| `s_linker13g_sem_full`  | alias + linkmap | LLM-only |

Plan 06-08 GATE-05 dev loop + full sweep operate on all 4. Both the dotted-path axis (`pre` vs `sem`) and the knowledge-richness axis (`alias` vs `full`) are resolved in one sweep per CONTEXT.md D-04 / D-08.

## Pointers for Plan 06-08

- **Cached pickles** (zero-recompute on hard tier):
  - `results/phase_cache/s_linker13g_pre_alias/<ds>/standalone_map.pkl` (5 datasets)
  - `results/phase_cache/s_linker13g_sem_alias/<ds>/standalone_map.pkl` (5 datasets)
  - `results/phase_cache/s_linker13g_pre_full/<ds>/standalone_map.pkl` (5 datasets)
  - `results/phase_cache/s_linker13g_sem_full/<ds>/standalone_map.pkl` (5 datasets)
- **GATE-05 floors:**
  - BBB ≥ 0.8890 (legacy regex baseline floor)
  - BBB > 0.8108 (D-09 new floor: must beat the pure-LLM rejected baseline)
  - TM, MS, JAB: legacy GATE-05 floors per ROADMAP.md / 06-CONTEXT.md
- **D-09 BBB hypothesis to test in Plan 06-08:** the BBB F1 lift (if any) comes from downstream alias/linkmap context affecting alias-matching / coref tiers, NOT from Tier-1b standalone-map differences. If Plan 06-08 BBB F1 is still ~0.8108 across all 4 new variants, this confirms the negative finding and Plan 06-08 may close empty (Pitfall-5 path).
- **Denominator-aware-update precedent:** `BENCHMARK_TABOO.md` §"Tailored Code Anti-Patterns" (line 70) — the Plan 06-07 `DENOMINATOR_AWARE_J_SKIP` constant is the canonical structural-property fix (skip on `|S_baseline|=0`, NOT a per-cell allowlist). Future GATE updates that need denominator-aware adjustments should follow the same single-module-boolean pattern (no per-call override surface, audit via `grep -c`).

## Deviations from Plan

None — Tasks 1 and 2 executed exactly as written in 06-07-PLAN.md (prior agent passes). Task 3 executed exactly per `<action>` block: append `## User adjudication`, rename `Proposed finalist set` to `Final finalist set`, commit the markdown update. Per the resume objective, STATE.md and ROADMAP.md were intentionally NOT updated by this agent.

## Self-Check: PASSED

- File: `.planning/phases/06-ext-01-project-agnostic-standalone-mention-llm-primitive/06-DIFF-MATRIX-ALIAS.md` — FOUND (contains `## User adjudication` and `Final finalist set for Plan 06-08` sections)
- File: `results/ablation_results/ablation_ext01_diff_alias.json` — FOUND (178 KB merged matrix)
- File: `results/ablation_results/ablation_ext01_diff_alias_regex.json` — FOUND
- File: `results/ablation_results/ablation_ext01_diff_alias_purellm_pre.json` — FOUND
- File: `results/ablation_results/ablation_ext01_diff_alias_purellm_sem.json` — FOUND
- Pickles: `results/phase_cache/s_linker13g_{pre_alias,sem_alias,pre_full,sem_full}/<5 datasets>/` — FOUND (4 variants × 5 dataset dirs)
- Commit: `540b76f` (adjudication record) — verifiable via `git log --oneline`
