# Roadmap: llm-sad-sam-v45 Rule-to-LLM Ablation

## Overview

Starting from the frozen `s_linker12c` ICSE baseline, this milestone removes six structural helpers one-by-one and replaces each with an LLM primitive. Each removal lands as a standalone variant file (`s_linker13a` through `s_linker13f`), evaluated independently against the 5-project benchmark with a macro F1 floor of 93% and a per-dataset guard of no more than 2pp regression vs baseline. The chain ends with the winning variant promoted as `s_linker13` and an ablation table documenting every step. Process quality gates (GATE-01..06) apply to every variant phase.

## Quality Gates (Applied to Every Variant Phase)

The following gates apply in Phases 2, 3, 4, and 5. They are not a separate phase — they are the execution standard for variant creation and promotion.

| Gate | Requirement |
|------|-------------|
| GATE-01 | Dual floor: macro F1 ≥ 93% AND no dataset > 2pp below 12c per-dataset baseline |
| GATE-02 | Variant registered in `CANONICAL_VARIANTS` and `VARIANT_SPECS` in `run_ablation.py` |
| GATE-03 | Variant has structured docstring with `REMOVED_FROM:` and `RULES_REMOVED:` |
| GATE-04 | Benchmark-taboo audit on every new prompt constant before variant registration |
| GATE-05 | Hard-tier-first loop: regress >1pp on BBB or TM vs parent → no full sweep, rework |
| GATE-06 | Per-variant independent runs (no phase-checkpoint replay across variants) |

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Baseline and Infrastructure** - Baseline capture, per-variant checkpoint namespacing, diskcache migration, and first rule removal (13a) — completed 2026-05-28 (macro F1 0.9364; user-loosened BBB tolerance)
- [x] **Phase 2: Ambiguity Cleanup** - Remove structural unambiguity post-filter and its wrapper (13b + 13c) — completed 2026-05-29 (13b macro +0.0114; 13c macro 0.9314 under user-loosened BBB 6pp)
- [~] **Phase 3: Mention Classifier Migration** - Replace 4-regex `_classify_mention` with LLM enum (13d) — CLOSED EMPTY 2026-05-29; VAR-04 retired after 13d TM regression (-19pp from dotted-path FPs); milestone-level finding
- [x] **Phase 4: Alias Scope and Coref Fold** - Retire `_is_strong_alias` via scope field; fold alias signal into coref (13e + 13f) — completed 2026-05-29 (13e macro 0.9380; 13f macro 0.9509 — best in chain)
- [x] **Phase 5: Promote and Ablation Artifact** - Promote winning variant as s_linker13; generate ablation table and writeup — completed 2026-05-29 (s_linker13 macro 0.9509; ABLATION-TABLE.md/.tex; METHODOLOGY.md; PROMO-01..04 all satisfied)

## Phase Details

### Phase 1: Baseline and Infrastructure
**Goal**: A reproducible `s_linker12c` baseline is captured, per-variant checkpoint namespacing is in place, and `s_linker13a` (Spike 001 trailing-word LLM enrichment) passes the dual floor — giving a clean starting point for the entire ablation chain.
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01, INFRA-03, INFRA-05, VAR-01 (INFRA-02 and INFRA-04 struck per D-01 — see Phase 1 CONTEXT)
**Success Criteria** (what must be TRUE):
  1. `results/ablation_results/` contains a 12c baseline run with per-dataset F1, FP/FN counts, and a JSON result file
  2. `diskcache>=5.6.1` and `tabulate>=0.9.0` are in `pyproject.toml`
  3. `s_linker13a` registered in `run_ablation.py`; hard-tier run (teammates + BBB) completes with no regression >1pp vs 12c; full 5-project sweep confirms macro F1 ≥ 93% and no dataset >2pp below 12c baseline
  4. Each variant's `_checkpoint_dir` uses its own `_VARIANT_NAME` constant (no hardcoded `"s_linker12c"` string in 13a)
**Plans**: 5 plans
  - [x] 01-01-PLAN.md — Doc strike (D-01a): mark INFRA-02/04 STRUCK in REQUIREMENTS.md and ROADMAP.md
  - [x] 01-02-PLAN.md — Add diskcache/tabulate deps + migrate llm_client.py LLM-response cache to diskcache (INFRA-03)
  - [x] 01-03-PLAN.md — Add `_VARIANT_NAME` constant + D-07 assertion in s_linker12c.py (INFRA-05)
  - [x] 01-04-PLAN.md — Capture s_linker12c baseline on 5-project sweep (INFRA-01)
  - [!] 01-05-PLAN.md — Create s_linker13a.py (Spike 001 trailing-word LLM); hard-tier gate + full sweep (VAR-01) — **BLOCKED: GATE-05 hard reject 2026-05-15** (s_linker13a built and committed; hard-tier BBB F1=0.796 vs 12c 0.844, delta=-0.048pp << -0.02 rejection threshold; full sweep NOT executed). See 01-05-SUMMARY.md.

### Phase 2: Ambiguity Cleanup
**Goal**: The structural post-filter `_is_structurally_unambiguous` is retired (13b), and its now-trivial wrapper `_is_ambiguous_name_component` is inlined and removed (13c) — both passing the dual floor, confirming the LLM ambiguity classification from `_classify_components` can be trusted on its own.
**Depends on**: Phase 1
**Requirements**: VAR-02, VAR-03
**Quality gates applied**: GATE-01 through GATE-06
**Success Criteria** (what must be TRUE):
  1. `s_linker13b` has no call to `_is_structurally_unambiguous` anywhere in the file; full 5-project sweep macro F1 ≥ 93%, no dataset >2pp below 12c baseline
  2. `s_linker13c` has no call to `_is_ambiguous_name_component` or `_is_structurally_unambiguous`; full 5-project sweep passes dual floor
  3. Both variants are registered in `run_ablation.py` with correct docstrings (`REMOVED_FROM`, `RULES_REMOVED`); benchmark-taboo audit clean
  4. Ablation log has a row for 13b and 13c showing per-dataset F1 and ΔF1 vs parent
**Plans**: TBD

### Phase 3: Mention Classifier Migration
**Goal**: The 4-branch regex `_classify_mention` is fully replaced by an LLM-emitted enum field on the extracted candidate (Spike 003), with an exact-string contract enforced to prevent silent downstream mismatch — all while holding macro F1 ≥ 93%.
**Depends on**: Phase 2
**Requirements**: VAR-04
**Quality gates applied**: GATE-01 through GATE-06
**Success Criteria** (what must be TRUE):
  1. `s_linker13d` contains no call to `_classify_mention` and no regex branches for mention type; `mention_type` is read from the LLM-emitted candidate field
  2. An enum class (or constant set) with the exact 4 string values from the original `_classify_mention` is defined; coercion assertion is present so an out-of-enum LLM response raises immediately rather than silently degrading
  3. Full 5-project sweep macro F1 ≥ 93% and no dataset >2pp below 12c baseline; benchmark-taboo audit on `MENTION_TYPE_SCHEMA` prompt constant is clean
  4. Hard-tier run shows no change in seed-validation rejection rates vs 13c baseline (string coupling verified stable)
**Plans**: TBD

### Phase 4: Alias Scope and Coref Fold
**Goal**: The alias-discovery prompt emits a `scope: global|local` field replacing `_is_strong_alias` + `_get_strong_alias_mappings` (13e, widest blast radius — run twice on hard tier before full sweep); then `_has_strong_alias_mention` is folded into the coref prompt evidence schema (13f) — completing all six structural rule removals.
**Depends on**: Phase 3
**Requirements**: VAR-05, VAR-06
**Quality gates applied**: GATE-01 through GATE-06
**Success Criteria** (what must be TRUE):
  1. `s_linker13e` defines `AliasEntry` inline with `component` and `scope` fields; no call to `_is_strong_alias` or `_get_strong_alias_mappings`; side-by-side log of LLM scope vs structural classification on all aliases is recorded and shows no unexpected divergence; dual floor met on full 5-project sweep
  2. 13e is run twice on teammates + BBB before full sweep; both runs agree within Claude's normal run-to-run variance; no alias scope assignment inconsistency between runs (Phase 3 alias logs compared)
  3. `s_linker13f` contains no call to `_has_strong_alias_mention`; coref prompt output schema includes `antecedent_via_alias` field; coref TP/FP measured before and after fold on all 5 datasets; dual floor met
  4. Both variants registered in `run_ablation.py`; benchmark-taboo audit on `ALIAS_SCOPE_SCHEMA` prompt constant is clean; ablation log rows complete with per-dataset F1 and ΔF1
**UI hint**: no
**Plans**: TBD

### Phase 5: Promote and Ablation Artifact
**Goal**: The winning variant (the last 13x that holds the dual floor) is promoted as `s_linker13.py`; `_has_standalone_mention` keep-decision is formally logged; the ablation table (markdown + LaTeX) and a methodology writeup are produced — completing the deliverable.
**Depends on**: Phase 4
**Requirements**: PROMO-01, PROMO-02, PROMO-03, PROMO-04
**Success Criteria** (what must be TRUE):
  1. `s_linker13.py` exists with zero non-trivial rules (only `_has_standalone_mention` plus parsers/formatters surviving); registered in `run_ablation.py`
  2. PROJECT.md Key Decisions table contains a formal KEEP entry for `_has_standalone_mention` referencing Spike 002's O(N×M) classification
  3. Ablation table exists in both markdown and LaTeX (`tabulate` output); contains one row per variant (12c → 13a → 13b → 13c → 13d → 13e → 13f → 13) with per-dataset F1, ΔF1 vs parent, rules-removed list, and FP-by-phase breakdown
  4. Research writeup (markdown) documents methodology, the promotion chain, and the retained-primitive rationale for `_has_standalone_mention`
**Plans**: TBD

## Progress

**Execution Order:** Sequential: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Baseline and Infrastructure | 5/5 | Complete | 2026-05-28 |
| 2. Ambiguity Cleanup | 2/2 | Complete | 2026-05-29 |
| 3. Mention Classifier Migration | 1/1 (closed empty) | Complete (VAR-04 retired) | 2026-05-29 |
| 4. Alias Scope and Coref Fold | 2/2 | Complete | 2026-05-29 |
| 5. Promote and Ablation Artifact | 3/3 | Complete | 2026-05-29 |
