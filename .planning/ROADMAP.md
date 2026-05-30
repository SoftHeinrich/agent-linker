# Roadmap: llm-sad-sam-v45 — v2.0 Complete Rule Removal + Cross-Model

> **v1.0 archived.** Full v1.0 phase history is preserved at `.planning/milestones/v1.0-ROADMAP.md` (Phases 1–5, shipped 2026-05-29, re-audit `passed` 2026-05-30, final artifact `s_linker13.py` macro F1 0.9509). This file tracks the **active v2.0 milestone only**. Phase numbering continues from v1.0 — v2.0 starts at Phase 6.

## Milestones

- ✅ **v1.0 — Rule-to-LLM Ablation** (`s_linker12c` → `s_linker13`) — Phases 1–5 — shipped 2026-05-29. See `milestones/v1.0-ROADMAP.md`.
- 🚧 **v2.0 — Complete Rule Removal + Cross-Model — Generality First** — Phases 6–9 — active.

## v2.0 Goal

Finish the no-hand-crafted-rules thesis by replacing the last structural rule (`_has_standalone_mention`) with a project-agnostic LLM primitive, combine all retired-rule LLM primitives into `s_linker14`, and validate the combined result on GPT-5.2 — every step under hard generality constraint **GATE-06** (zero benchmark-derived values, reviewer-defensible across arbitrary projects).

## Phases

- [ ] **Phase 6: EXT-01 — Project-Agnostic Standalone-Mention LLM Primitive** — replace `_has_standalone_mention` with an LLM primitive that encodes no project-specific structure
- [ ] **Phase 7: EXT-02 — Drop Dotted-Path Guard** — remove the dotted-path regex inside the EXT-01 variant (gated on Phase 6 dual-floor pass + GATE-06 audit clean)
- [ ] **Phase 8: COMBINE — `s_linker14` Stack-or-Unify Combined LLM Primitives** — integrate Spike 001 trailing-words + scope-field + alias-coref-fold + EXT-01 into a single linker variant; stack-vs-unify decided at phase start from EXT-01 cost/quality signal
- [ ] **Phase 9: CROSS — GPT-5.2 Cross-Model Validation** — produce cross-model JSON results and comparison report for `s_linker13` and `s_linker14` on GPT-5.2 across all 5 datasets

## Phase Details

### Phase 6: EXT-01 — Project-Agnostic Standalone-Mention LLM Primitive
**Goal**: Ship a new linker variant (e.g., `s_linker13g.py`) in which `_has_standalone_mention` is replaced by an LLM primitive that encodes zero project-specific structure (no Java packages, no dotted paths, no BBB-style component names) and the variant still meets the dual floor.
**Depends on**: v1.0 deliverable `s_linker13.py` (baseline parent for the chain)
**Requirements**: EXT-01
**Success Criteria** (what must be TRUE):
  1. New linker variant exists as a standalone file with structured docstring (`REMOVED_FROM` / `RULES_REMOVED`) and is registered in `CANONICAL_VARIANTS` + `VARIANT_SPECS` of `run_ablation.py` (GATE-07).
  2. `_has_standalone_mention` (and its in-helper callees specific to this rule) is no longer reachable from the variant's call graph; only parsers/formatters and other v1.0-retained helpers remain on the structural side.
  3. The variant passes the dual floor on the full 5-project sweep: macro F1 ≥ 0.93 AND BBB ≤ 6pp below `s_linker12c` BBB AND each other dataset ≤ 2pp below the corresponding `s_linker12c` per-dataset baseline (GATE-01); per-dataset and macro F1 logged as JSON.
  4. **GATE-06 generality audit recorded in SUMMARY.md**: new prompt(s) and any new helper pass (a) `BENCHMARK_TABOO.md` scan AND (b) reviewer-defensibility check — no encoded project structure in prompt examples or code logic. Cost/quality signal (LLM-call count, latency vs `s_linker13`) captured and tagged for the Phase 8 stack-vs-unify decision.
**Plans**: 4 plans
- [ ] 06-01-PLAN.md — Prompt design (STANDALONE_MENTION_RULES_PRE_FILTERED + LLM_ONLY in prompts_v2.py) + GATE-06 pre-clearance audit
- [ ] 06-02-PLAN.md — Sub-variant scaffolding: s_linker13g_pre.py (regex pre-filter + LLM judge) and s_linker13g_sem.py (LLM-only, dotted-path in prompt) + run_ablation.py GATE-07 dual-registration
- [ ] 06-03-PLAN.md — D-02 offline anchor-set diff stage + catastrophic-diff drop rule + 06-DIFF-MATRIX.md + user-approved finalist set
- [ ] 06-04-PLAN.md — GATE-05 hard-tier dev loop + full 5-project sweep + D-03 winner pick + byte-copy promotion to canonical s_linker13g.py + GATE-06 final audit + 06-SUMMARY.md with tagged Phase-8 cost/quality block

### Phase 7: EXT-02 — Drop Dotted-Path Guard
**Goal**: Inside the EXT-01 variant, drop the dotted-path regex guard (the one that currently protects `ui.website`-style references) and confirm the variant still meets the dual floor.
**Depends on**: Phase 6 — Phase 7 is **only attempted if Phase 6 passes the dual floor AND its GATE-06 audit is clean**. If Phase 6 fails either gate, Phase 7 is skipped/deferred and the milestone proceeds without EXT-02.
**Requirements**: EXT-02
**Success Criteria** (what must be TRUE):
  1. Dotted-path regex guard is removed from the variant's `_has_standalone_mention`-replacement path; diff shows the deletion and its callers no longer reference it.
  2. Variant (with guard dropped) passes the dual floor on the full 5-project sweep (GATE-01); per-dataset and macro F1 logged as JSON, with explicit before/after comparison vs the Phase 6 variant on TeaMMates (the dotted-path-sensitive dataset, per v1.0 VAR-04 finding).
  3. **GATE-06 generality audit recorded in SUMMARY.md**: confirmation that no replacement project-specific logic was added to compensate for the dropped guard. If the LLM primitive cannot generalize without the guard, the phase closes empty with a documented negative result (parallel to v1.0 Phase 3 / VAR-04).
**Plans**: TBD (estimated 1–2 — guard removal + targeted TM/BBB probe; full sweep + audit)

### Phase 8: COMBINE — `s_linker14` Stack-or-Unify Combined LLM Primitives
**Goal**: Build `s_linker14.py` integrating all LLM rule-removal primitives accumulated through v1.0 + v2.0 (Spike 001 trailing-words, `scope: global|local` alias field, alias-coref fold, and EXT-01's standalone-mention replacement); decide stack-vs-unify at phase start using the EXT-01 cost/quality signal from Phase 6.
**Depends on**: Phase 6 (EXT-01 primitive + cost signal needed before stack-vs-unify can be chosen); Phase 7 outcome incorporated if Phase 7 landed (dotted-path-free variant), else falls back to Phase 6 variant baseline.
**Requirements**: COMBINE-01, COMBINE-02, COMBINE-03
**Success Criteria** (what must be TRUE):
  1. `s_linker14.py` exists as a standalone file (no inheritance), registered in `CANONICAL_VARIANTS` + `VARIANT_SPECS`; structured docstring records `RULES_REMOVED` (cumulative list) and a **stack-vs-unify provenance string** that names the chosen design and cites the EXT-01 cost/quality signal as rationale (COMBINE-01, GATE-07).
  2. `s_linker14` passes the dual floor on Claude Sonnet across all 5 datasets (GATE-01): macro F1 ≥ 0.93, BBB tolerance ≤ 6pp, other-dataset tolerance ≤ 2pp; per-dataset + macro F1 logged as JSON (COMBINE-02).
  3. New row added to `ABLATION-TABLE.md` and `ABLATION-TABLE.tex` for `s_linker14` with provenance string capturing the stack-vs-unify choice + rationale; previous v1.0 ablation rows preserved (COMBINE-03).
  4. **GATE-06 generality audit recorded in SUMMARY.md**: all integrated prompts re-audited as a unit (combination can surface leakage the per-phase audits missed); BENCHMARK_TABOO scan + reviewer-defensibility check both pass.
**Plans**: TBD (estimated 3 — stack-vs-unify decision memo + design; `s_linker14` implementation + full sweep; ablation-table update + GATE-06 unit re-audit)

### Phase 9: CROSS — GPT-5.2 Cross-Model Validation
**Goal**: Run `s_linker13` and `s_linker14` on GPT-5.2 across all 5 datasets via the cross-model evaluation harness, and produce a comparison report that frames any Claude-vs-GPT gap as a model-provider-property finding (not a failure to fix), per standing v2.0 policy.
**Depends on**: Phase 8 (`s_linker14` artifact required for CROSS-02). `s_linker13` arm (CROSS-01) is unblocked by v1.0 and may execute in parallel with later Phase 8 plans if the harness is ready.
**Requirements**: CROSS-01, CROSS-02, CROSS-03
**Success Criteria** (what must be TRUE):
  1. Cross-model evaluation harness produces JSON results for `s_linker13` on GPT-5.2 across all 5 datasets, with no backend-specific prompt tailoring beyond the existing model-adapter shim (CROSS-01).
  2. Cross-model evaluation harness produces JSON results for `s_linker14` on GPT-5.2 across all 5 datasets, under the same no-tailoring rule (CROSS-02).
  3. Cross-model markdown report compares Claude Sonnet vs GPT-5.2 per dataset for both variants, explicitly states whether macro F1 ≥ 0.93 holds cross-model for each, and frames any gap as a model-provider-property finding (CROSS-03).
  4. **GATE-06 generality audit recorded in SUMMARY.md**: harness, adapter shim, and any new code path added for the cross-model run contain no benchmark-derived values or per-project branching; the cross-model run is the strongest empirical evidence for the generality claim and must itself be clean.
**Plans**: TBD (estimated 2 — harness + `s_linker13` GPT-5.2 run; `s_linker14` GPT-5.2 run + comparison report + GATE-06 audit)

## Standing Gates (apply to every phase)

These four gates apply across all v2.0 phases and are not mapped to a single phase:

- **GATE-01** — Dual floor: macro F1 ≥ 0.93 AND BBB ≤ 6pp below `s_linker12c` BBB AND each other dataset ≤ 2pp below its `s_linker12c` per-dataset baseline.
- **GATE-05** — Hard-tier-first dev loop: regress >1pp on BBB or TM vs parent → no full sweep, re-work variant.
- **GATE-06** — Generality audit: every new prompt + helper passes (a) `BENCHMARK_TABOO.md` scan AND (b) reviewer-defensibility check; audit recorded per phase in SUMMARY.md.
- **GATE-07** — Every promoted variant registered in `CANONICAL_VARIANTS` + `VARIANT_SPECS` in `run_ablation.py`; standalone file (no inheritance); structured docstring with `REMOVED_FROM` / `RULES_REMOVED`.

## Coverage

All 8 active v2.0 requirements mapped exactly once:

| REQ-ID | Phase |
|--------|-------|
| EXT-01 | Phase 6 |
| EXT-02 | Phase 7 |
| COMBINE-01 | Phase 8 |
| COMBINE-02 | Phase 8 |
| COMBINE-03 | Phase 8 |
| CROSS-01 | Phase 9 |
| CROSS-02 | Phase 9 |
| CROSS-03 | Phase 9 |

GATE-01, GATE-05, GATE-06, GATE-07 are standing (apply to all phases — recorded in `REQUIREMENTS.md` Traceability as `all/standing`, not mapped to a single phase).

✓ 8/8 active requirements mapped — no orphans, no duplicates.

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 6. EXT-01 — Project-Agnostic Standalone-Mention LLM Primitive | v2.0 | 0/4 | Not started | — |
| 7. EXT-02 — Drop Dotted-Path Guard | v2.0 | 0/2 (est.) | Not started (gated on Phase 6) | — |
| 8. COMBINE — `s_linker14` Stack-or-Unify | v2.0 | 0/3 (est.) | Not started (depends on Phase 6) | — |
| 9. CROSS — GPT-5.2 Cross-Model Validation | v2.0 | 0/2 (est.) | Not started (depends on Phase 8) | — |
