# Phase 6 — EXT-01 D-02 Diff Matrix

**Generated:** 2026-05-30T10:20:14
**Source:** `results/ablation_results/ablation_ext01_diff.json`
**Harness:** `scripts/ext01_diff_stage.py`
**Method:** RESEARCH.md §"Empirical Matrix Operationalization" — offline anchor-set Jaccard vs regex baseline. No full-pipeline invocation; only the variants' Tier-1 `_compute_standalone_mention_map` plus the `s_linker13._has_standalone_mention` static regex.

## Threshold rules (verbatim from RESEARCH.md lines 319-321)

Drop variant if (on TM or BBB):
- `min_jaccard_per_comp < 0.3`, OR
- any `(comp, ds)` has `D > 10` sentences, OR
- `count_components_with_J<0.5 > 25%` of components on TM or BBB.

Thresholds encoded as module constants in `scripts/ext01_diff_stage.py` (`HARD_TIER_MIN_J=0.3`, `MAX_SYM_DIFF=10`, `HARD_TIER_PCT_LOW_J=0.25`) — no per-call overrides (T-06-03-01 mitigation).

## Per-variant roll-up

Notation: `min J` = `min_jaccard_per_comp`; `mean J (wtd)` = `mean_jaccard_weighted` (weights = `|S_regex[comp]|`, floor 1); `#C with J<0.5 / total` = `count_components_with_J<0.5 / n_components`; `max D` = `max_symmetric_diff`.

### s_linker13g_pre

| Dataset       | min J | mean J (wtd) | #C with J<0.5 / total | max D |
|---------------|------:|-------------:|----------------------:|------:|
| mediastore    | 1.000 |        1.000 |                 0/14  |     0 |
| teastore      | 0.333 |        0.836 |                  2/11 |     3 |
| teammates     | 0.429 |        0.726 |                  1/8  |     7 |
| bigbluebutton | 0.000 |        0.933 |                 1/12  |     2 |
| jabref        | 0.800 |        0.952 |                  0/6  |     1 |

**Drop reasons:** `bigbluebutton: min_jaccard_per_comp=0.000 < 0.3 (hard-tier Jaccard floor)`.

### s_linker13g_sem

| Dataset       | min J | mean J (wtd) | #C with J<0.5 / total | max D |
|---------------|------:|-------------:|----------------------:|------:|
| mediastore    | 1.000 |        1.000 |                 0/14  |     0 |
| teastore      | 0.400 |        0.878 |                 1/11  |     3 |
| teammates     | 0.833 |        0.943 |                  0/8  |     2 |
| bigbluebutton | 0.000 |        0.933 |                 1/12  |     2 |
| jabref        | 1.000 |        1.000 |                  0/6  |     0 |

**Drop reasons:** `bigbluebutton: min_jaccard_per_comp=0.000 < 0.3 (hard-tier Jaccard floor)`.

## Catastrophic per-(component, dataset) cells

All `(variant, dataset, component)` cells with `J < 0.5 OR D > 5`. Sorted by variant then ascending J. Cause classification reflects the executor's inspection — see notes below the table.

| Variant         | Dataset       | Component   |     J | D | \|S_regex\| | \|S_variant\| | Likely cause |
|-----------------|---------------|-------------|------:|--:|------------:|--------------:|--------------|
| s_linker13g_pre | bigbluebutton | kurento     | 0.000 | 2 |           0 |             2 | regex-blind-spot (lowercase comp; doc capitalizes) |
| s_linker13g_pre | teastore      | Persistence | 0.333 | 2 |           1 |             3 | over-restrictive regex (variant accepts compatible mentions) |
| s_linker13g_pre | teastore      | Registry    | 0.400 | 3 |           2 |             5 | over-restrictive regex (architecturally-different sentences) |
| s_linker13g_pre | teammates     | Client      | 0.429 | 4 |           3 |             7 | architecturally-different (variant accepts coref-style mentions) |
| s_linker13g_pre | teammates     | Storage     | 0.500 | 7 |           7 |            14 | architecturally-different (max D=7 — still under MAX_SYM_DIFF=10) |
| s_linker13g_sem | bigbluebutton | kurento     | 0.000 | 2 |           0 |             2 | regex-blind-spot (same as pre) |
| s_linker13g_sem | teastore      | Registry    | 0.400 | 3 |           2 |             5 | over-restrictive regex (same as pre) |

Cause-classification key:

- `dotted-path-leak` — variant accepts a dotted-path mention that the regex rejects (13d-class). **None observed.**
- `regex-blind-spot` — regex misses mentions due to case-sensitivity rules in `_has_standalone_mention`; variant correctly identifies them. **Observed: BBB/`kurento`.**
- `over-restrictive` — regex rejects mentions the variant accepts on semantic grounds (the variant has more recall). Includes the "architecturally-different" subset where the variant simply finds different sentences for the same component.
- `architecturally-different` — variant finds different sentences for the same component (semantic-scope difference, NOT a defect).

Critical observation on the BBB/`kurento` cell (the sole driver of both drop decisions):

> The regex baseline returns `S_regex[bbb][kurento] = ∅`. This happens because `_has_standalone_mention` constructs the pattern `\bkurento\b` with `flags=0` for the lowercase-first-letter case, so any capitalized "Kurento" mention in the documentation is missed. Both variants find 2 such mentions. The symmetric difference is small (D=2) and well below the catastrophic threshold (`MAX_SYM_DIFF=10`). The Jaccard collapse to 0 is a denominator artefact: any non-empty `S_variant` paired with an empty `S_regex` produces J=0, regardless of how few mentions the variant found. This is a **gentle-divergence regex-blind-spot signature**, NOT a 13d-class dotted-path catastrophe.

## Drop decisions

| Variant         | Drop? | Rationale |
|-----------------|-------|-----------|
| s_linker13g_pre | yes   | BBB `min_jaccard_per_comp = 0.000` violates the `< 0.3` hard-tier floor. Single cell (`kurento`, D=2) — regex-blind-spot, not catastrophic in absolute terms. |
| s_linker13g_sem | yes   | Same single-cell driver as pre: BBB/`kurento`, J=0, D=2. All other hard-tier cells pass; TM `min_J=0.833`, BBB `mean_J_wtd=0.933`. |

Both variants are mechanically flagged for drop by the same single-cell signature.

## Finalists for Plan 04 full sweep

> **NO MECHANICAL SURVIVORS.** Both candidate variants are flagged by the catastrophic-diff drop rule on the same single cell (BBB/`kurento`, `min_J=0.000`). The Pitfall 5 escalation path is triggered (next section). The user adjudication recorded below names the actual finalist set Plan 04 will consume.

## Pitfall 5 escalation

NO SURVIVORS. The diff stage has dropped both candidate variants. Per RESEARCH.md Pitfall 5 and D-02 escalation policy, this outcome requires user adjudication before Plan 04 proceeds. Possible paths:

- **(a) Relax the catastrophic-diff threshold** (e.g., `min_jaccard < 0.2` on hard tier) and re-evaluate — only justified if the drop reasons inspect as gentle divergence, not 13d-class signatures. **Inspection here: the BBB/`kurento` cell is a regex-blind-spot (the regex returns ∅ due to case-sensitivity, while the variants correctly find 2 capitalized mentions; D=2 ≪ MAX_SYM_DIFF=10).** This is the textbook gentle-divergence case; relaxing the Jaccard threshold for cells where `|S_regex| = 0` (denominator artefacts) is defensible.
- **(b) Revise prompts** and replay Plans 01-03 — would only address the drop if the prompts were the cause; here the drop is a regex-baseline artefact, not a variant defect, so this path does not match the failure mode.
- **(c) Accept the negative result** and close Phase 6 empty (parallel to v1.0 VAR-04). The actual variant outputs match the regex everywhere it returns non-empty (BBB `mean_J_wtd=0.933` for both; TM `mean_J_wtd ≥ 0.726`). Closing empty would discard a likely-improvement-over-regex result.

The checkpoint task in this plan pauses for the user to pick.

## User adjudication

**Decision required.** See checkpoint context below for the offered options. Once the user resumes with a chosen option id, this section will be filled with:

- The selected option id.
- Any user-supplied parameters (e.g., relaxed threshold value).
- The final finalist set Plan 04 will sweep.
- A timestamp.

---

### Checkpoint context for the user

**Both variants flag drop on the same single cell**: BBB/`kurento`, where the regex baseline returns `∅` (because `kurento` is lowercase and the doc capitalizes it — `_has_standalone_mention`'s case-sensitive branch). The variants find 2 mentions. Symmetric difference is 2 (far below the catastrophic threshold of 10). This is the canonical "regex-blind-spot → Jaccard denominator artefact" signature, not a dotted-path leak.

**LLM cost incurred by the diff stage** (preliminary input for D-06):

| Variant         | Dataset       | LLM `(cname, snum)` pair count |
|-----------------|---------------|--------------------------------|
| s_linker13g_pre | mediastore    | 17 |
| s_linker13g_pre | teastore      | 16 |
| s_linker13g_pre | teammates     | 76 |
| s_linker13g_pre | bigbluebutton | 35 |
| s_linker13g_pre | jabref        | 20 |
| s_linker13g_sem | mediastore    | 20 |
| s_linker13g_sem | teastore      | 14 |
| s_linker13g_sem | teammates     | 80 |
| s_linker13g_sem | bigbluebutton | 35 |
| s_linker13g_sem | jabref        | 20 |
| **Total**       | —             | **333** |

(Each `(cname, snum)` pair is one entry in the per-component batched LLM response. Batches were 50 sentences per LLM call; total LLM calls were substantially fewer than 333 — exact count is in the `logs/ext01_diff_stage.log` `[cache write]` lines.)

**Options offered to the user** (mirrors Plan task 3 `<options>`):

1. `proceed-with-finalists` — Accept the mechanical drop verdict and have NO finalists. Honors the rule strictly. Closes Phase 6 empty.
2. `proceed-with-override` — Override the rule for one or both variants and send them to Plan 04. Requires written justification (the regex-blind-spot inspection above qualifies).
3. `relax-threshold` — Apply a relaxed Jaccard floor (e.g., 0.2) OR a denominator-aware variant (skip the J<0.3 check when `|S_regex|=0`) and recompute the drop decisions against the cached JSON (no LLM cost).
4. `revise-prompts` — Go back to Plan 01 / Plan 02. Unlikely to help — the drop is a regex artefact, not a prompt artefact.
5. `close-empty` — Accept the negative result and close Phase 6 empty.
