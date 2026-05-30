# Phase 6 — EXT-01 D-02 Diff Matrix (Plan 06-07 — Alias-Aware Variants)

**Generated:** 2026-05-30T16:46:31
**Source:** `results/ablation_results/ablation_ext01_diff_alias.json` (merged top-level: `results_by_baseline` with 3 keys: `regex`, `pure-llm-pre`, `pure-llm-sem`)
**Intermediate JSONs:** `ablation_ext01_diff_alias_regex.json`, `ablation_ext01_diff_alias_purellm_pre.json`, `ablation_ext01_diff_alias_purellm_sem.json`
**Harness:** `scripts/ext01_diff_stage.py` (Plan 06-07 extensions over the Plan 06-03 harness — denominator-aware Jaccard skip + dual-baseline mode + 4 new variant classes)
**Method:** RESEARCH.md §"Empirical Matrix Operationalization" + Plan 06-03 03-DIFF-MATRIX adjudication. Offline anchor-set Jaccard vs **two** baselines: the original regex baseline (drop-decision-gating) AND the rejected pure-LLM baselines (Plan 06-04: `s_linker13g_pre`, `s_linker13g_sem`) for D-09 verification. No full-pipeline invocation; only each variant's Tier-1b `_compute_standalone_mention_map`.

**Key naming choice for merged JSON:** intermediate files use `purellm_pre` / `purellm_sem` in their filename (underscored, no hyphen), but the merged JSON's `results_by_baseline` keys are `regex`, `pure-llm-pre`, `pure-llm-sem` (matching the CLI choice values). Downstream consumers should index by the merged-JSON spellings.

## Threshold rules (verbatim from RESEARCH.md lines 319-321)

Drop variant if (on TM or BBB):
- `min_jaccard_per_comp < 0.3`, OR
- any `(comp, ds)` has `D > 10` sentences, OR
- `count_components_with_J<0.5 > 25%` of components on TM or BBB.

Thresholds encoded as module constants in `scripts/ext01_diff_stage.py` (`HARD_TIER_MIN_J=0.3`, `MAX_SYM_DIFF=10`, `HARD_TIER_PCT_LOW_J=0.25`) — no per-call overrides (T-06-03-01 mitigation, carried over from Plan 06-03).

### Plan 06-07 addition — denominator-aware Jaccard skip

`DENOMINATOR_AWARE_J_SKIP = True` (module constant in `scripts/ext01_diff_stage.py`). When `|S_baseline[comp]| == 0` for a `(comp, ds)` cell, Jaccard collapses to 0 mechanically — this is a **denominator artefact**, not a variant defect (see Plan 06-03 BBB-`kurento` adjudication and `BENCHMARK_TABOO.md` §"Tailored Code Anti-Patterns"). In Plan 06-07 the J check on such cells is SKIPPED in the dataset rollup; the symmetric-difference (D) check is unchanged (D is well-defined when `|S_baseline|=0`). Toggling the flag requires editing the file (auditable via git diff). No per-call override surface (T-06-07-01 / T-06-07-02 mitigation).

The skip counts per `(variant, dataset)` are surfaced in the rollup field `n_components_J_skipped` and in the per-variant tables below as the **`n_C J skipped`** column.

## Per-variant roll-up vs regex baseline

The regex baseline is the **drop-decision-gating** comparison (Plan 06-03 protocol carried over). The four new alias-aware variants from Plan 06-06 are evaluated against the same regex predicate used in Plan 06-03.

Notation: `min J` = `min_jaccard_per_comp`; `mean J (wtd)` = `mean_jaccard_weighted` (weights = `|S_baseline[comp]|`, floor 1); `#C with J<0.5 / total` = `count_components_with_J<0.5 / n_components`; `max D` = `max_symmetric_diff`; `n_C J skipped` = `n_components_J_skipped` (denominator-aware skip count).

### s_linker13g_pre_alias (vs regex baseline)

| Dataset       | min J | mean J (wtd) | #C with J<0.5 / total | max D | n_C J skipped |
|---------------|------:|-------------:|----------------------:|------:|--------------:|
| mediastore    | 1.000 |        1.000 |                  0/14 |     0 |             7 |
| teastore      | 0.333 |        0.738 |                  2/11 |     3 |             6 |
| teammates     | 0.429 |        0.690 |                   1/8 |     7 |             0 |
| bigbluebutton | 0.600 |        0.961 |                  0/12 |     2 |             2 |
| jabref        | 1.000 |        1.000 |                   0/6 |     0 |             1 |

**Drop reasons:** *(none)*.

### s_linker13g_sem_alias (vs regex baseline)

| Dataset       | min J | mean J (wtd) | #C with J<0.5 / total | max D | n_C J skipped |
|---------------|------:|-------------:|----------------------:|------:|--------------:|
| mediastore    | 1.000 |        1.000 |                  0/14 |     0 |             7 |
| teastore      | 0.400 |        0.755 |                  1/11 |     3 |             6 |
| teammates     | 0.625 |        0.866 |                   0/8 |     6 |             0 |
| bigbluebutton | 0.600 |        0.961 |                  0/12 |     2 |             2 |
| jabref        | 1.000 |        1.000 |                   0/6 |     0 |             1 |

**Drop reasons:** *(none)*.

### s_linker13g_pre_full (vs regex baseline)

| Dataset       | min J | mean J (wtd) | #C with J<0.5 / total | max D | n_C J skipped |
|---------------|------:|-------------:|----------------------:|------:|--------------:|
| mediastore    | 1.000 |        1.000 |                  0/14 |     3 |             7 |
| teastore      | 0.333 |        0.738 |                  2/11 |     3 |             6 |
| teammates     | 0.429 |        0.649 |                   2/8 |     7 |             0 |
| bigbluebutton | 0.600 |        0.961 |                  0/12 |     2 |             2 |
| jabref        | 0.800 |        0.950 |                   0/6 |     1 |             1 |

**Drop reasons:** *(none)*.

### s_linker13g_sem_full (vs regex baseline)

| Dataset       | min J | mean J (wtd) | #C with J<0.5 / total | max D | n_C J skipped |
|---------------|------:|-------------:|----------------------:|------:|--------------:|
| mediastore    | 1.000 |        1.000 |                  0/14 |     0 |             7 |
| teastore      | 0.400 |        0.755 |                  1/11 |     3 |             6 |
| teammates     | 0.429 |        0.793 |                   1/8 |     6 |             0 |
| bigbluebutton | 0.600 |        0.961 |                  0/12 |     2 |             2 |
| jabref        | 0.800 |        0.950 |                   0/6 |     1 |             1 |

**Drop reasons:** *(none)*.

**BBB-kurento status:** The denominator-aware skip retires the Plan 06-03 BBB-`kurento` drop driver. All four new variants show `min_J = 0.600` on BBB (vs the rejected baselines' `0.000`) — not because BBB-`kurento` improved, but because the (`|S_baseline| = 0`) cell is now skipped from the J rollup. The remaining BBB minimum comes from a different component (see "Catastrophic per-(component, dataset) cells" below).

## Per-variant roll-up vs pure-LLM baseline (D-09 verification)

These tables compare each new variant's standalone-mention map to the cached pickle from the Plan 06-04 rejected baseline (zero LLM cost on the baseline side). Self-vs-self pairs are skipped by the harness. The pure-LLM-baseline tables are **informational only** — they are NOT drop-decision-gating (the regex baseline is the operative gate, per Plan 06-03 protocol).

Notation as above.

### vs pure-llm-pre baseline (`s_linker13g_pre`)

#### s_linker13g_pre_alias

| Dataset       | min J | mean J (wtd) | #C with J<0.5 / total | max D | n_C J skipped |
|---------------|------:|-------------:|----------------------:|------:|--------------:|
| mediastore    | 1.000 |        1.000 |                  0/14 |     0 |             7 |
| teastore      | 1.000 |        1.000 |                  0/11 |     0 |             6 |
| teammates     | 0.833 |        0.963 |                   0/8 |     2 |             0 |
| bigbluebutton | 1.000 |        1.000 |                  0/12 |     0 |             1 |
| jabref        | 0.800 |        0.958 |                   0/6 |     1 |             1 |

#### s_linker13g_sem_alias

| Dataset       | min J | mean J (wtd) | #C with J<0.5 / total | max D | n_C J skipped |
|---------------|------:|-------------:|----------------------:|------:|--------------:|
| mediastore    | 1.000 |        1.000 |                  0/14 |     0 |             7 |
| teastore      | 0.667 |        0.938 |                  0/11 |     1 |             6 |
| teammates     | 0.429 |        0.728 |                   1/8 |     5 |             0 |
| bigbluebutton | 1.000 |        1.000 |                  0/12 |     0 |             1 |
| jabref        | 0.800 |        0.958 |                   0/6 |     1 |             1 |

#### s_linker13g_pre_full

| Dataset       | min J | mean J (wtd) | #C with J<0.5 / total | max D | n_C J skipped |
|---------------|------:|-------------:|----------------------:|------:|--------------:|
| mediastore    | 1.000 |        1.000 |                  0/14 |     3 |             7 |
| teastore      | 1.000 |        1.000 |                  0/11 |     0 |             6 |
| teammates     | 0.455 |        0.935 |                   1/8 |     6 |             0 |
| bigbluebutton | 1.000 |        1.000 |                  0/12 |     0 |             1 |
| jabref        | 1.000 |        1.000 |                   0/6 |     0 |             1 |

#### s_linker13g_sem_full

| Dataset       | min J | mean J (wtd) | #C with J<0.5 / total | max D | n_C J skipped |
|---------------|------:|-------------:|----------------------:|------:|--------------:|
| mediastore    | 1.000 |        1.000 |                  0/14 |     0 |             7 |
| teastore      | 0.667 |        0.938 |                  0/11 |     1 |             6 |
| teammates     | 0.727 |        0.888 |                   0/8 |     4 |             0 |
| bigbluebutton | 1.000 |        1.000 |                  0/12 |     0 |             1 |
| jabref        | 1.000 |        1.000 |                   0/6 |     0 |             1 |

### vs pure-llm-sem baseline (`s_linker13g_sem`)

#### s_linker13g_pre_alias

| Dataset       | min J | mean J (wtd) | #C with J<0.5 / total | max D | n_C J skipped |
|---------------|------:|-------------:|----------------------:|------:|--------------:|
| mediastore    | 1.000 |        1.000 |                  0/14 |     0 |             7 |
| teastore      | 0.333 |        0.952 |                  1/11 |     2 |             6 |
| teammates     | 0.429 |        0.681 |                   1/8 |     7 |             0 |
| bigbluebutton | 1.000 |        1.000 |                  0/12 |     0 |             1 |
| jabref        | 1.000 |        1.000 |                   0/6 |     0 |             1 |

#### s_linker13g_sem_alias

| Dataset       | min J | mean J (wtd) | #C with J<0.5 / total | max D | n_C J skipped |
|---------------|------:|-------------:|----------------------:|------:|--------------:|
| mediastore    | 1.000 |        1.000 |                  0/14 |     0 |             7 |
| teastore      | 0.500 |        0.964 |                  0/11 |     1 |             6 |
| teammates     | 0.750 |        0.887 |                   0/8 |     4 |             0 |
| bigbluebutton | 1.000 |        1.000 |                  0/12 |     0 |             1 |
| jabref        | 1.000 |        1.000 |                   0/6 |     0 |             1 |

#### s_linker13g_pre_full

| Dataset       | min J | mean J (wtd) | #C with J<0.5 / total | max D | n_C J skipped |
|---------------|------:|-------------:|----------------------:|------:|--------------:|
| mediastore    | 1.000 |        1.000 |                  0/14 |     3 |             7 |
| teastore      | 0.333 |        0.952 |                  1/11 |     2 |             6 |
| teammates     | 0.429 |        0.640 |                   2/8 |     7 |             0 |
| bigbluebutton | 1.000 |        1.000 |                  0/12 |     0 |             1 |
| jabref        | 0.800 |        0.950 |                   0/6 |     1 |             1 |

#### s_linker13g_sem_full

| Dataset       | min J | mean J (wtd) | #C with J<0.5 / total | max D | n_C J skipped |
|---------------|------:|-------------:|----------------------:|------:|--------------:|
| mediastore    | 1.000 |        1.000 |                  0/14 |     0 |             7 |
| teastore      | 0.500 |        0.964 |                  0/11 |     1 |             6 |
| teammates     | 0.429 |        0.803 |                   1/8 |     6 |             0 |
| bigbluebutton | 1.000 |        1.000 |                  0/12 |     0 |             1 |
| jabref        | 0.800 |        0.950 |                   0/6 |     1 |             1 |

## Catastrophic per-(component, dataset) cells

All `(variant, dataset, component)` cells with `J < 0.5` OR `D > 5` across all three baselines. Sorted by baseline then variant then ds. Skipped cells are not listed (denominator-aware skip filter). Cause classification:

- `dotted-path-leak` — variant accepts a dotted-path mention the baseline rejects (13d-class). **None observed.**
- `regex-blind-spot` — regex misses mentions due to case-sensitivity; variant correctly identifies them. **Retired by the denominator-aware skip.**
- `over-restrictive` — baseline rejects mentions the variant accepts on semantic grounds (variant has more recall).
- `architecturally-different` — variant and baseline disagree on what counts as a standalone mention (semantic-scope difference, not a defect).
- `alias-recovery` (NEW for Plan 06-07) — the alias-aware variant ACCEPTS a mention that the pure-LLM rejected baseline did not. This is the success signal the new variants are designed to produce.

| Baseline       | Variant                | Dataset    | Component   |     J |  D | \|S_b\| | \|S_v\| | Likely cause |
|----------------|------------------------|------------|-------------|------:|---:|--------:|--------:|--------------|
| regex          | s_linker13g_pre_alias  | teastore   | Persistence | 0.333 |  2 |       1 |       3 | over-restrictive (alias-recovery candidate) |
| regex          | s_linker13g_pre_alias  | teastore   | Registry    | 0.400 |  3 |       2 |       5 | over-restrictive |
| regex          | s_linker13g_pre_alias  | teammates  | Client      | 0.429 |  4 |       3 |       7 | architecturally-different (variant accepts coref-style mentions) |
| regex          | s_linker13g_pre_alias  | teammates  | Storage     | 0.500 |  7 |       7 |      14 | architecturally-different (D below MAX_SYM_DIFF=10) |
| regex          | s_linker13g_pre_alias  | teammates  | Logic       | 0.625 |  6 |      10 |      16 | architecturally-different (mean_J still 0.690) |
| regex          | s_linker13g_sem_alias  | teastore   | Registry    | 0.400 |  3 |       2 |       5 | over-restrictive |
| regex          | s_linker13g_sem_alias  | teammates  | Logic       | 0.625 |  6 |      10 |      16 | architecturally-different |
| regex          | s_linker13g_pre_full   | teastore   | Persistence | 0.333 |  2 |       1 |       3 | over-restrictive (alias-recovery candidate) |
| regex          | s_linker13g_pre_full   | teastore   | Registry    | 0.400 |  3 |       2 |       5 | over-restrictive |
| regex          | s_linker13g_pre_full   | teammates  | Client      | 0.429 |  4 |       3 |       7 | architecturally-different |
| regex          | s_linker13g_pre_full   | teammates  | Common      | 0.455 |  6 |       5 |      11 | architecturally-different |
| regex          | s_linker13g_pre_full   | teammates  | Storage     | 0.500 |  7 |       7 |      14 | architecturally-different |
| regex          | s_linker13g_pre_full   | teammates  | Logic       | 0.625 |  6 |      10 |      16 | architecturally-different |
| regex          | s_linker13g_sem_full   | teastore   | Registry    | 0.400 |  3 |       2 |       5 | over-restrictive |
| regex          | s_linker13g_sem_full   | teammates  | Client      | 0.429 |  4 |       3 |       7 | architecturally-different |
| regex          | s_linker13g_sem_full   | teammates  | Storage     | 0.538 |  6 |       7 |      13 | architecturally-different |
| pure-llm-pre   | s_linker13g_sem_alias  | teammates  | Client      | 0.429 |  4 |       7 |       3 | architecturally-different (sem-alias REJECTS what pre accepts) |
| pure-llm-pre   | s_linker13g_pre_full   | teammates  | Common      | 0.455 |  6 |       5 |      11 | alias-recovery (full ACCEPTS 6 more than pure-pre) |
| pure-llm-sem   | s_linker13g_pre_alias  | teastore   | Persistence | 0.333 |  2 |       1 |       3 | alias-recovery (pre-alias accepts 2 more than pure-sem) |
| pure-llm-sem   | s_linker13g_pre_alias  | teammates  | Client      | 0.429 |  4 |       3 |       7 | alias-recovery (pre-alias accepts 4 more than pure-sem) |
| pure-llm-sem   | s_linker13g_pre_alias  | teammates  | Storage     | 0.500 |  7 |       7 |      14 | alias-recovery (pre-alias accepts 7 more) |
| pure-llm-sem   | s_linker13g_pre_full   | teastore   | Persistence | 0.333 |  2 |       1 |       3 | alias-recovery |
| pure-llm-sem   | s_linker13g_pre_full   | teammates  | Common      | 0.455 |  6 |       5 |      11 | alias-recovery |
| pure-llm-sem   | s_linker13g_pre_full   | teammates  | Storage     | 0.500 |  7 |       7 |      14 | alias-recovery |
| pure-llm-sem   | s_linker13g_pre_full   | teammates  | Client      | 0.429 |  4 |       3 |       7 | alias-recovery |
| pure-llm-sem   | s_linker13g_sem_full   | teammates  | Storage     | 0.538 |  6 |       7 |      13 | alias-recovery |
| pure-llm-sem   | s_linker13g_sem_full   | teammates  | Client      | 0.429 |  4 |       3 |       7 | alias-recovery |

**Observations:**

- **No 13d-class dotted-path catastrophes** in any variant against any baseline. The original Plan 06-03 catastrophe class is absent throughout the matrix.
- All hard-tier divergences are bounded: `max D ≤ 7` everywhere on TM, `max D ≤ 2` everywhere on BBB. None exceed the `MAX_SYM_DIFF=10` floor.
- The `pre`-family variants (`pre_alias`, `pre_full`) show MORE divergence vs the regex baseline than the `sem`-family (consistent with the regex pre-filter pruning sentences before the LLM judge sees them — `sem` lets the LLM see all sentences, which on TM happens to align better with the regex predicate).
- Against the pure-llm baselines, the `*_full` variants show MORE divergence than the `*_alias` variants (the additional linkmap injection nudges the judge to ACCEPT more TM mentions, particularly Storage / Client / Common — visible as `|S_v| > |S_b|`).

## Drop decisions

Mechanical drop verdicts computed via `apply_drop_rule` against the **regex** baseline rollups (the drop-decision-gating comparison). Pure-LLM-baseline rollups are informational only.

| Variant                | Drop? | Rationale |
|------------------------|-------|-----------|
| s_linker13g_pre_alias  | NO    | All hard-tier thresholds clear: TM `min_J=0.429>0.3`, `#C<0.5=1/8=12.5%<25%`, `max_D=7≤10`; BBB `min_J=0.600>0.3`, `#C<0.5=0/12`, `max_D=2≤10`. Non-hard-tier `max_D` all ≤3. |
| s_linker13g_sem_alias  | NO    | TM `min_J=0.625`, `#C<0.5=0/8`, `max_D=6`; BBB `min_J=0.600`, `#C<0.5=0/12`, `max_D=2`. All thresholds clear. |
| s_linker13g_pre_full   | NO    | TM `min_J=0.429`, `#C<0.5=2/8=25%` — at the `>25%` boundary but NOT exceeding (`2/8=0.25` is exactly the threshold; the rule fires only on strict `>`). `max_D=7`; BBB `min_J=0.600`, `max_D=2`. Cleared, but tightest of the four. |
| s_linker13g_sem_full   | NO    | TM `min_J=0.429`, `#C<0.5=1/8=12.5%`, `max_D=6`; BBB `min_J=0.600`, `max_D=2`. All thresholds clear. |

**All four new variants pass the mechanical drop rule.** Plan 06-03 had to override the rule because of the BBB-`kurento` denominator artefact; Plan 06-07 retires that artefact via the denominator-aware skip, so no override is required this time.

## D-09 verification

**Question:** Do the alias-aware variants demonstrably differ from the pure-LLM rejected baselines (`s_linker13g_pre`, `s_linker13g_sem`) on the standalone-mention map?

**Per-dataset evidence:**

| Dataset       | vs pure-llm-pre, divergence shape                          | vs pure-llm-sem, divergence shape                          |
|---------------|------------------------------------------------------------|------------------------------------------------------------|
| mediastore    | Identical (J=1.000 across all 4) — no divergence           | Identical (J=1.000 across all 4) — no divergence           |
| teastore      | `*_alias` slight divergence (0.667–1.000); `*_full` aligned with pre | `*_alias` 0.333–0.500; `*_full` 0.333–0.500 (alias-recovery on Persistence) |
| teammates     | `pre_alias` modest (0.833); `sem_alias`, `pre_full`, `sem_full` divergent (0.429–0.727) | All 4 substantially divergent (0.429–0.750) — alias-recovery on Storage / Client / Common |
| bigbluebutton | **Identical (J=1.000 across all 4)** — NO divergence       | **Identical (J=1.000 across all 4)** — NO divergence       |
| jabref        | `pre_alias`/`sem_alias` slight (0.800); `pre_full`/`sem_full` identical | `pre_alias`/`sem_alias` identical; `pre_full`/`sem_full` slight (0.800) |

**Substantive finding (BBB):** On BBB, all 4 new variants pick the **identical** anchor sentences as both pure-LLM baselines. Per-component J = 1.0 for every BBB component across all four `*_alias` and `*_full` variants when compared against either `pure-llm-pre` or `pure-llm-sem`. The standalone-mention map itself is NOT where the alias-aware track diverges from the rejected baseline on BBB. The map is `Recording Service: skip D=0 | kurento: J=1.00 D=0 | WebRTC-SFU: J=1.00 D=0 | HTML5 Server: J=1.00 D=0 | HTML5 Client: J=1.00 D=0 | ...` — every BBB component matches exactly.

**Implication for Plan 06-08:** If the alias-aware variants are to clear the GATE-05 BBB floor (≥ 0.8890) where the pure-LLM pair failed (0.8108), the improvement must come from **downstream tier behavior** — the alias map and linkmap are still computed and passed to the standalone-mention prompt as context, but the Tier-1b output is unchanged on BBB. The hypothesis to verify in Plan 06-08 is that the alias/linkmap context informs the downstream alias matching / coref tier rather than the standalone judgement itself. If Plan 06-08 BBB F1 is still 0.8108 across all 4 new variants, that confirms the negative finding — Plan 06-08 may then close empty (Pitfall-5 path).

**Substantive finding (TM):** On TM the new variants DO diverge substantially from both pure-LLM baselines (especially `pre_full` and `sem_full`). The divergence shape is consistently "alias-aware accepts more standalone mentions" (`|S_variant| > |S_baseline|` on Storage / Client / Common / E2E / UI). This is **alias-recovery** — the upstream knowledge injection nudges the judge to recognize more TM sentences as standalone mentions. Whether this translates to better TM F1 is for Plan 06-08 to determine; the diff stage confirms only that the variants ARE moving off the rejected-baseline floor on TM.

**Substantive finding (MS / JAB):** Near-identical anchors. Low-mention-density datasets where the regex baseline and all LLM variants converge — these will be the "easy passes" in Plan 06-08.

## Proposed finalist set for Plan 06-08

Based on the regex-baseline drop verdicts (mechanical, none drop) AND the D-09 vs-pure-LLM evidence (`*_full` shows more alias-recovery on TM; BBB is identical across all 4):

**Default proposal: all 4 new variants enter Plan 06-08's GATE-05 dev loop.**

- `s_linker13g_pre_alias` — alias-only context + regex pre-filter
- `s_linker13g_sem_alias` — alias-only context + LLM-only
- `s_linker13g_pre_full` — alias + linkmap context + regex pre-filter
- `s_linker13g_sem_full` — alias + linkmap context + LLM-only

Rationale: all four mechanically pass; the diff stage cannot empirically rank them on F1 alone (that is Plan 06-08's job). The dotted-path axis (`pre` vs `sem`) and the knowledge-richness axis (`alias` vs `full`) are both still open per CONTEXT.md D-04 / D-08.

**Alternative narrower proposal (lower LLM cost):** drop the two `*_full` variants — they show no BBB divergence from pure-LLM (per D-09) and they have the most TM divergence (suggesting they may also produce the most FPs in the full pipeline). Retain only `*_alias` for Plan 06-08. This halves Plan 06-08's full-sweep cost.

## Pitfall 5 escalation handling

Mechanical drop rule produced ZERO drops. Pitfall 5 (no-survivor) escalation does NOT trigger. Task 3 (`checkpoint:decision`) instead asks the user to confirm the finalist set size (all 4 vs the narrower 2-variant subset).

If the user picks `close-empty`, the rationale would be the D-09 BBB identity finding above — but note this is premature because the BBB GATE-05 improvement could still emerge in Plan 06-08 from downstream tier behavior with alias/linkmap context, not from the Tier-1b map itself. The diff stage cannot definitively answer the BBB question.

## LLM cost incurred (regex-baseline run only)

The pure-LLM-baseline runs are cache-only (zero LLM cost — they read the existing `s_linker13g_pre` / `s_linker13g_sem` pickles from Plan 06-04 plus the just-populated 4 new-variant pickles). Per-`(variant, dataset)` `(cname, snum)` pair counts (one entry per per-component LLM judgement returned; batched ≤50 sentences per call, so actual LLM-call count is substantially lower):

| Variant                | mediastore | teastore | teammates | bigbluebutton | jabref | Total |
|------------------------|-----------:|---------:|----------:|--------------:|-------:|------:|
| s_linker13g_pre_alias  |         20 |       16 |        84 |            35 |     20 |   175 |
| s_linker13g_sem_alias  |         20 |       16 |        87 |            35 |     20 |   178 |
| s_linker13g_pre_full   |         20 |       16 |        84 |            35 |     19 |   174 |
| s_linker13g_sem_full   |         20 |       15 |        70 |            35 |     20 |   160 |
| **Total**              |         80 |       63 |       325 |           140 |     79 | **687** |

Total ~687 pairs (vs Plan 06-03's 333 — ~2x as expected for 4 variants vs 2). Per CONTEXT.md D-06 this is acceptable under the relaxed-budget posture. All variant pickles persisted under `results/phase_cache/{s_linker13g_pre_alias,s_linker13g_sem_alias,s_linker13g_pre_full,s_linker13g_sem_full}/<ds>/standalone_map.pkl` for Plan 06-08 reuse (zero re-compute cost on the hard-tier GATE-05 leg).

## User adjudication

<!-- Task 3 will append a `## User adjudication` section here recording the option selected, UTC timestamp, and the user's verdict (verbatim). The "Proposed finalist set for Plan 06-08" section above will be renamed/extended into a "Final finalist set for Plan 06-08" section. -->

*Pending Task 3 — `checkpoint:decision` paused for user.*
