---
phase: 06-ext-01-project-agnostic-standalone-mention-llm-primitive
plan: 03
subsystem: llm-linker
tags: [llm-linker, sad-sam, standalone-mention, ext-01, d-02, diff-stage, jaccard, gate-05, gate-06, pitfall-5]

# Dependency graph
requires:
  - phase: 06
    plan: 02
    provides: "s_linker13g_pre.py + s_linker13g_sem.py (sub-variant siblings with Tier-1 _compute_standalone_mention_map)"
provides:
  - "scripts/ext01_diff_stage.py — offline anchor-set diff harness (regex baseline vs both variants, per-(comp, ds) Jaccard/sym-diff, mechanical drop rule)"
  - "results/ablation_results/ablation_ext01_diff.json — machine-readable D-02 diff matrix"
  - ".planning/phases/06-.../06-DIFF-MATRIX.md — human-readable report including drop decisions, Pitfall 5 escalation, and user adjudication"
  - "User-approved finalist set for Plan 04: {s_linker13g_pre, s_linker13g_sem}"
  - "BENCHMARK_TABOO.md §Tailored Code Anti-Patterns — new taboo class (case-mismatch regex baselines + per-cell threshold tuning)"
affects:
  - "06-04 (full 5-project sweep — consumes the user-approved finalist set; both variants enter D-03 macro-F1 winner pick)"
  - "GATE-06 universe (new banned tailoring pattern propagates to all future regex/threshold work in EXT-* phases)"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Offline anchor-set Jaccard diff as a pre-filter stage between sub-variant build (Plan 02) and full sweep (Plan 04) — kills 13d-class catastrophes cheaply (≤333 LLM calls vs ~thousands for full sweep)"
    - "Mechanical drop rule encoded as module constants (HARD_TIER_MIN_J=0.3, MAX_SYM_DIFF=10, HARD_TIER_PCT_LOW_J=0.25) with no per-call override surface — auditable via git diff (T-06-03-01 mitigation)"
    - "Per-(comp, ds, J, D, |S_regex|, |S_variant|) cells persisted in the JSON matrix so reviewers can re-run apply_drop_rule offline without the LLM (T-06-03-02 mitigation)"
    - "Pitfall 5 (no-survivor) escalation surfaced via checkpoint:decision task — user adjudication recorded inline in 06-DIFF-MATRIX.md before Plan 04 commits LLM budget"
    - "Baseline-blind-spot recognition pattern: when J=0 fires on a (|S_regex|=0, |S_variant|=small) cell, treat the baseline as suspect, not the variant — and ban any per-cell rescue logic in BENCHMARK_TABOO.md"

key-files:
  created:
    - "scripts/ext01_diff_stage.py — D-02 diff harness (≥200 lines, AST-validated)"
    - "results/ablation_results/ablation_ext01_diff.json — full per-(variant, ds, comp) matrix"
    - ".planning/phases/06-ext-01-project-agnostic-standalone-mention-llm-primitive/06-DIFF-MATRIX.md — D-02 report with user adjudication"
    - ".planning/phases/06-ext-01-project-agnostic-standalone-mention-llm-primitive/06-03-SUMMARY.md — this file"
  modified:
    - "BENCHMARK_TABOO.md — added §Tailored Code Anti-Patterns (case-mismatch regex baselines + per-cell threshold tuning)"

key-decisions:
  - "Override the mechanical drop rule for BOTH sub-variants — the single offending cell (BBB/kurento, J=0, D=2) is a regex-baseline blind spot (lowercase `\\bkurento\\b` flags=0 misses capitalized doc mentions), not a variant defect. All other cells pass the catastrophic-diff thresholds (max_D ≤ 7, mean_J_wtd ≥ 0.726 on TM, ≥ 0.933 on BBB across both variants)."
  - "Do NOT patch the regex with `re.IGNORECASE` or per-component casing tables — that bakes the benchmark casing convention into code. Instead, document the anti-pattern in BENCHMARK_TABOO.md and rely on the EXT-01 LLM primitive (which handles casing as a natural-language detail) for the full sweep."
  - "Do NOT relax the catastrophic-diff threshold per-cell — that shifts the leak from prompt to threshold logic (banned in the new taboo §Tailored Code Anti-Patterns). The rule logic in apply_drop_rule is unchanged."
  - "Finalist set for Plan 04: {s_linker13g_pre, s_linker13g_sem}. GATE-05 hard-tier-first dev loop remains in effect in Plan 04 as a second-pass safety net."

patterns-established:
  - "Diff-stage-before-full-sweep is a reusable pattern for EXT-* phases — cheap LLM cost, mechanical drop rule, checkpoint:decision escalation surface on no-survivor outcomes"
  - "Banning *tailored code* (not just tailored prompts) under GATE-06 — taboo coverage now extends to regex flags, casing tables, and per-cell threshold tuning"
  - "When a baseline blind spot drives a Jaccard denominator collapse, override is preferred over rule tuning, with the anti-pattern recorded in the taboo list"

requirements-completed: [EXT-01]

# Metrics
duration: ~25min (across Task 1 harness implementation, Task 2 report, Task 3 user adjudication)
completed: 2026-05-30
---

# Phase 06 Plan 03: EXT-01 D-02 Diff Stage + Pitfall 5 Adjudication Summary

**Offline anchor-set Jaccard diff harness built and run on all 5 benchmark datasets × 2 sub-variants (`s_linker13g_pre`, `s_linker13g_sem`) vs the `s_linker13._has_standalone_mention` regex baseline. The mechanical catastrophic-diff drop rule flagged BOTH variants on the same single cell (BBB/`kurento`, J=0, D=2). Inspection revealed this as a regex-baseline blind spot (`\bkurento\b` with `flags=0` misses capitalized doc occurrences). User adjudication: override the drop verdict, ban the anti-pattern in `BENCHMARK_TABOO.md`, and send both variants to Plan 04. Final finalist set: `{s_linker13g_pre, s_linker13g_sem}`.**

## Per-variant Roll-up

Notation: `min J` = `min_jaccard_per_comp`; `mean J (wtd)` = `mean_jaccard_weighted` over components with `|S_regex|`-weights (floor 1); `#C<0.5` = `count_components_with_J<0.5 / n_components`; `max D` = `max_symmetric_diff`.

### s_linker13g_pre (regex pre-filter + LLM judge)

| Dataset       | min J | mean J (wtd) | #C<0.5 / total | max D |
|---------------|------:|-------------:|---------------:|------:|
| mediastore    | 1.000 |        1.000 |          0/14  |     0 |
| teastore      | 0.333 |        0.836 |          2/11  |     3 |
| teammates     | 0.429 |        0.726 |           1/8  |     7 |
| bigbluebutton | 0.000 |        0.933 |          1/12  |     2 |
| jabref        | 0.800 |        0.952 |           0/6  |     1 |

Mechanical drop reason: `bigbluebutton: min_jaccard_per_comp=0.000 < 0.3 (hard-tier Jaccard floor)`.

### s_linker13g_sem (LLM-only, dotted-path in prompt)

| Dataset       | min J | mean J (wtd) | #C<0.5 / total | max D |
|---------------|------:|-------------:|---------------:|------:|
| mediastore    | 1.000 |        1.000 |          0/14  |     0 |
| teastore      | 0.400 |        0.878 |          1/11  |     3 |
| teammates     | 0.833 |        0.943 |           0/8  |     2 |
| bigbluebutton | 0.000 |        0.933 |          1/12  |     2 |
| jabref        | 1.000 |        1.000 |           0/6  |     0 |

Mechanical drop reason: `bigbluebutton: min_jaccard_per_comp=0.000 < 0.3 (hard-tier Jaccard floor)`.

Both variants exhibit very high agreement with the regex baseline everywhere except the BBB/`kurento` cell. `s_linker13g_sem` is slightly more aligned overall (TM `mean_J_wtd=0.943` vs `pre`'s 0.726; JAB perfect 1.000 vs `pre`'s 0.952), but the difference is well within "expected variant divergence" and is NOT a drop signal — Plan 04 will pick between them via macro F1 (D-03), not Jaccard.

## Mechanical Drop Decisions

| Variant         | Drop? | Rationale (mechanical) |
|-----------------|-------|------------------------|
| s_linker13g_pre | yes   | BBB `min_jaccard_per_comp = 0.000` violates the `< 0.3` hard-tier floor. Single cell (`kurento`, D=2). |
| s_linker13g_sem | yes   | Same single-cell driver: BBB/`kurento`, J=0, D=2. All other hard-tier cells pass. |

Both variants flagged. No mechanical survivors → Pitfall 5 escalation triggered → `checkpoint:decision` paused for user.

## User Adjudication

**Option selected:** `proceed-with-override: include both s_linker13g_pre and s_linker13g_sem`.

**User verdict (verbatim):**

> "that regex is too tailored, drop that, and add tailored examples to taboo list, never do that"

**Reasoning (recorded in `06-DIFF-MATRIX.md` §User adjudication):**

1. The sole offending cell — BBB/`kurento`, `min_J=0.000`, `D=2` — is driven by a regex-baseline blind spot, not a variant defect. The baseline (`s_linker13._has_standalone_mention`) constructs `re.compile(r"\bkurento\b", flags=0)` for the lowercase component string, so any capitalized `Kurento` mention in the BBB documentation is silently missed. Both EXT-01 variants correctly detect the 2 mentions in both casings.
2. The Jaccard collapse to 0 is a **denominator artefact**: a non-empty `S_variant` paired with an empty `S_regex` always yields J=0, regardless of how few mentions the variant actually found. The symmetric difference (`D=2`) is far below the catastrophic threshold (`MAX_SYM_DIFF=10`) — this is the canonical "gentle-divergence regex-blind-spot signature", not a 13d-class dotted-path catastrophe.
3. The fix is **not** to patch the regex (`re.IGNORECASE`, casing tables, per-component overrides) — those approaches bake the benchmark casing convention into code, which is the same class of leakage as putting benchmark terms in prompts. The fix is also **not** to relax the catastrophic-diff threshold (the rule logic must remain stable across variants).
4. The fix is to **document the anti-pattern in `BENCHMARK_TABOO.md`** so this class of tailoring never recurs, and to override the drop verdict for these two specific variants on the strength of the inspection. The taboo additions are now permanent guard-rails for all future EXT-* work.

**Anti-patterns added to `BENCHMARK_TABOO.md` §Tailored Code Anti-Patterns** (commit `71f65b0`):

- **Case-mismatch regex baselines** — constructing `re.compile(r"\b{name}\b", flags=0)` from a component string whose casing differs from the documentation. Detection: any per-component regex pattern, per-component flag override, or per-component synonym map. Fix path: replace the structural check with a project-agnostic LLM primitive (the EXT-01 pattern) — do NOT patch the regex.
- **Tailoring diff/comparison rules to specific (component, dataset) cells** — adjusting a Jaccard or symmetric-difference threshold to "rescue" one failing cell shifts the leak from prompt to threshold logic. Fix path: if a rule fires on what is provably a baseline blind spot, drop the baseline-as-ground-truth assumption for that cell and document the inspection.

## Final Finalist Set for Plan 04

```text
finalists = {
    "s_linker13g_pre",
    "s_linker13g_sem",
}
```

Plan 04 (D-03 macro-F1 winner pick) consumes this set. GATE-05 hard-tier-first dev loop (TM, BBB) remains in effect as a second-pass safety net before the full 5-project sweep.

## LLM Cost Incurred by the Diff Stage (preliminary D-06 input)

Per-(variant, dataset) batched-response `(cname, snum)` pair counts (one entry per per-component LLM judgement returned). Batches were ≤50 sentences per LLM call; actual call count is substantially lower than the pair count.

| Variant         | Dataset       | LLM `(cname, snum)` pair count |
|-----------------|---------------|-------------------------------:|
| s_linker13g_pre | mediastore    |                             17 |
| s_linker13g_pre | teastore      |                             16 |
| s_linker13g_pre | teammates     |                             76 |
| s_linker13g_pre | bigbluebutton |                             35 |
| s_linker13g_pre | jabref        |                             20 |
| s_linker13g_sem | mediastore    |                             20 |
| s_linker13g_sem | teastore      |                             14 |
| s_linker13g_sem | teammates     |                             80 |
| s_linker13g_sem | bigbluebutton |                             35 |
| s_linker13g_sem | jabref        |                             20 |
| **Total**       | —             |                        **333** |

Per-variant totals: `s_linker13g_pre = 164 pairs`, `s_linker13g_sem = 169 pairs`. Cache writes are persisted under `results/phase_cache/{variant}/{ds}/standalone_map.pkl` so reruns of the harness are pickle-loaded and incur zero LLM cost (T-06-03-04 mitigation).

## Deviations from Plan

### Auto-fixed Issues

None within Tasks 1 and 2 — the harness and the report were produced as specified.

### User Adjudication Outcome

The checkpoint (Task 3) was a `checkpoint:decision` and reached its expected pause point. The user's verdict triggered an out-of-plan deliverable: **modifying `BENCHMARK_TABOO.md`** to add the new §Tailored Code Anti-Patterns section. This is documented here as a deliberate, user-directed deviation:

- **[User-directed] Add §Tailored Code Anti-Patterns to `BENCHMARK_TABOO.md`**
  - Triggered by: user verdict on checkpoint option `proceed-with-override`
  - Two anti-patterns recorded: *Case-mismatch regex baselines*, *Tailoring diff/comparison rules to specific cells*
  - Files modified: `BENCHMARK_TABOO.md`
  - Commit: `71f65b0`

## Pointer to BENCHMARK_TABOO.md Addition

See `BENCHMARK_TABOO.md` §"Tailored Code Anti-Patterns (NEVER do this)" added in commit `71f65b0` (`docs(06-03): add tailored-code anti-patterns to benchmark taboo`). The two new sub-headings are:

- §"Anti-pattern: Case-mismatch regex baselines"
- §"Anti-pattern: Tailoring diff/comparison rules to specific (component, dataset) cells"

Both sub-headings include symptom, fix path, and detection guidance, and apply to all future EXT-* phases.

## Commits

| Task / Step                              | Commit  | Message |
|------------------------------------------|---------|---------|
| Task 1 — diff harness                    | `055b8f6` | `feat(06-03): add EXT-01 D-02 offline anchor-set diff harness` |
| Task 2 — diff matrix report              | `18ad875` | `docs(06-03): author EXT-01 D-02 diff matrix report` |
| Task 3 (user-directed) — taboo addition  | `71f65b0` | `docs(06-03): add tailored-code anti-patterns to benchmark taboo` |
| Task 3 — user adjudication recorded      | `598cb58` | `docs(06-03): record user adjudication — override drop, finalists = pre + sem` |

## Self-Check: PASSED

- Files verified present: `scripts/ext01_diff_stage.py`, `results/ablation_results/ablation_ext01_diff.json`, `.planning/phases/06-.../06-DIFF-MATRIX.md`, `.planning/phases/06-.../06-03-SUMMARY.md`, `BENCHMARK_TABOO.md`.
- Commits verified present in `git log --all`: `055b8f6`, `18ad875`, `71f65b0`, `598cb58`.
- `06-DIFF-MATRIX.md` contains the `## User adjudication` header (7 H2 headers total — all five plan-required sections plus internal subdivisions).
- No `<fill>` placeholders remain in `06-DIFF-MATRIX.md`.
