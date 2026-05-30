# Requirements: llm-sad-sam-v45 v2.0 — Complete Rule Removal + Cross-Model

**Defined:** 2026-05-30
**Milestone:** v2.0 (Complete Rule Removal + Cross-Model — Generality First)
**Core Value:** Every rule removed from `s_linker13` and replaced by an LLM primitive must hold macro F1 ≥ 93% AND read as project-agnostic to a reviewer — or be rejected. Deliverable: defensible claim that traceability linking works without hand-crafted structural rules, validated across model providers.

**Hard Constraint (applies to every requirement):** Zero hardcoded benchmark-derived values or project-tailored rules in either prompts OR code logic. Only stopword-level English wordlists ("the", "a") and language-universal patterns (CamelCase) permitted. Every prompt example from safe SE/textbook domains (compiler, OS, e-commerce). Approach must defensibly generalize to any project a reviewer might apply it to. See PROJECT.md GATE-06.

## v2.0 Requirements

Requirements continue the EXT-NN numbering from v1.0's deferred items, plus new families (COMBINE, CROSS, GATE) for new scope.

### EXT — Last Structural Rule Replacement

- [ ] **EXT-01**: `_has_standalone_mention` replaced by a **project-agnostic** LLM primitive in a new linker variant (`s_linker13g.py` or similar). Relaxed cost budget; replacement must not encode project-specific structure (Java packages, dotted paths, BBB-style component names). Dual floor met (GATE-01).
- [ ] **EXT-02**: Dotted-path guard in `_has_standalone_mention` (the regex that protects `ui.website`-style references) DROPPED in the EXT-01 variant. **Gated:** only attempted if EXT-01 passes GATE-01 + GATE-06. Dual floor met after drop.

### COMBINE — Stacked / Unified Variant `s_linker14`

- [ ] **COMBINE-01**: New linker variant `s_linker14.py` integrates all LLM rule-removal primitives (Spike 001 trailing-words + scope-field + alias-coref-fold + EXT-01) — either as stacked separate calls or as a unified single-prompt design. Stack-vs-unify decision documented post-EXT-01 based on EXT-01 cost/quality signal.
- [ ] **COMBINE-02**: `s_linker14` passes dual floor (GATE-01) on Claude Sonnet across all 5 datasets. Per-dataset and macro F1 logged.
- [ ] **COMBINE-03**: `s_linker14` ablation row added to ABLATION-TABLE.md / .tex with provenance string capturing the stack-vs-unify choice + rationale.

### CROSS — GPT-5.2 Cross-Model Validation

- [ ] **CROSS-01**: Cross-model evaluation harness produces JSON results for `s_linker13` on GPT-5.2 (5 datasets). No backend-specific prompt tailoring beyond the existing model-adapter shim.
- [ ] **CROSS-02**: Cross-model evaluation harness produces JSON results for `s_linker14` on GPT-5.2 (5 datasets).
- [ ] **CROSS-03**: Cross-model report (markdown) compares Claude-vs-GPT-5.2 per dataset for both `s_linker13` and `s_linker14`; documents whether macro F1 ≥ 0.93 holds cross-model; reframes any gap as model-provider-property finding (not as failure to fix).

### GATE — Process & Generality

- [ ] **GATE-01** (carry from v1.0): Every variant passes dual floor — macro F1 ≥ 0.93 AND BBB ≤ 6pp below 12c BBB AND other datasets ≤ 2pp below 12c per-dataset baseline.
- [ ] **GATE-05** (carry from v1.0): Hard-tier-first dev loop enforced — regress >1pp on BBB or TM vs parent → no full sweep, re-work variant.
- [ ] **GATE-06** (new): Generality audit — every new prompt + helper passes (a) `BENCHMARK_TABOO.md` scan AND (b) explicit reviewer-defensibility check (would a reviewer believe this approach applies to a project they pick at random?). Audit recorded per phase in SUMMARY.md.
- [ ] **GATE-07** (new): Every promoted variant registered in `CANONICAL_VARIANTS` and `VARIANT_SPECS` in `run_ablation.py`; standalone file (no inheritance), structured docstring with `REMOVED_FROM` / `RULES_REMOVED`.

## Future Requirements

Deferred to v2.1+ or later milestones:

- **EXT-04** — Emit-biased boundary prompting on alias-discovery to shrink BBB borderline-4 variance band ~3pp → ~1pp. Variance work, not rule removal; out of v2.0 thesis scope.

## Out of Scope (v2.0)

- New seed/linker approaches (ILinker3+, ensembles beyond `s_linker14` combine) — v2.0 is finishing the 12c rule-removal chain, not exploration.
- Cost optimization — user has set "no LLM budget limit"; rule-replaceability and generality trump cost.
- SAM-Code, SAD-Code tasks — out of dataset scope.
- Re-opening VAR-04 (`_classify_mention`) — retired-as-rejection in v1.0; reviewers see this as the publishable negative result.
- Bench leakage — never permitted; enforced via GATE-04/GATE-06 + BENCHMARK_TABOO.md.
- Backend switch to Opus or other Claude tier — Claude Sonnet only per standing user preference; GPT-5.2 is evaluation-only (CROSS-01..03), not production target.

## Traceability

| REQ-ID | Phase | Status |
|--------|-------|--------|
| EXT-01 | TBD | pending |
| EXT-02 | TBD | pending |
| COMBINE-01 | TBD | pending |
| COMBINE-02 | TBD | pending |
| COMBINE-03 | TBD | pending |
| CROSS-01 | TBD | pending |
| CROSS-02 | TBD | pending |
| CROSS-03 | TBD | pending |
| GATE-01 | all | standing |
| GATE-05 | all | standing |
| GATE-06 | all | standing (NEW) |
| GATE-07 | all | standing |

*Phase column populated by roadmapper in Step 10.*
