# Requirements: llm-sad-sam-v45 — Milestone v2.6

**Defined:** 2026-06-02
**Milestone:** v2.6 — ILinker4 + LLM-Driven Training + Axiom Re-run
**Core Value:** Replace the mechanical training gate with an LLM Assessor that reasons over the actual error set (remaining FP + FN sentences); build ILinker4 as a Voyager-native seed extractor; fix axiom gaps that account for the remaining 27 systematic errors. Target: exceed canonical (s_linker13_min gpt-5.4 90.69%).

## Active v2.6 Requirements

### ILINKER — Voyager-Native Seed Extractor

- [ ] **REQ-V26-01** — `ilinker4.py` new standalone file. Voyager-native design: Pass A (extract) and Pass B (actor) prompts are structural scaffolding only — no inline rules. `SEED_EXTRACTION_RULES` and `SEED_ACTOR_RULES` are first-class bank slots injected at call time (empty string = baseline behavior identical to ILinker3). Does NOT inherit from `ilinker3.py`; standalone per user preference. `s_linker14_voyager.py` wires ILinker4 instead of ILinker3Injected.

- [ ] **REQ-V26-02** — Prompt hygiene audit: every static prompt in `s_linker14_voyager.py` and `ilinker4.py` audited and classified as (a) structural scaffolding (acceptable) or (b) inline rule (must migrate to bank slot). Audit report produced; all (b) items migrated. After migration: zero behavioral inline rules remain in static prompts.

### TRAINING LOOP — LLM-Driven Architecture (voyager_train v5)

- [ ] **REQ-V26-03** — O+D merge: collapse Oracle (O) and Distillator (D) into a single text-aware LLM role `OD`. `OD` receives: current bank, per-project FP sentence list, per-project FN sentence list, gold standard. In one call it (a) identifies failure modes, (b) proposes new bank patterns to address them. Prompt enforces abstract/general vocabulary — proposed pattern text must use discourse/syntactic/functional terminology, not project component names. Hard anti-superficiality instruction in prompt: proposed pattern must be semantically novel and cover an error class not already addressed by existing bank rules.

- [ ] **REQ-V26-04** — LLM Assessor role: replaces Gate A + Gate B entirely. One LLM call per proposal. Inputs: proposed pattern text + slot, full current bank (all existing patterns), remaining FP sentence list (with component context), remaining FN sentence list (with component context). Assessor decides: **accept / reject / revise** — with rationale identifying which specific FP or FN sentences the pattern would address. No hard vocabulary grep. Abstraction quality (abstract vs project-specific vocabulary in the proposed rule) is one of the Assessor's evaluation criteria. `revise` verdict returns a reformulated pattern for re-evaluation (max 1 revision cycle per proposal).

- [ ] **REQ-V26-05** — Cross-split redesign: each split trains independently, evaluates its trained bank on its own held-out test set, and reports against an axiom-only baseline on the same test set. Verdict question per split: "does training on projects {train_set} improve F1 on {test_set} beyond axiom-only?" No cross-split aggregation or dedup. Final deployed bank = mainline (MS+TS+TM train). Cross-split result is a stability/generalization check, not an input to consensus.

- [ ] **REQ-V26-06** — Log structure: every training log must report [TRAIN] and [TEST] project metrics separately per pass. Format: `[TRAIN] MS: F1=x TS: F1=x TM: F1=x macro=x` and `[TEST] BBB: F1=x JAB: F1=x macro=x` (or equivalent for non-mainline splits). The delta used for commit decision uses [TRAIN] macro only. [TEST] is reported for tracking but does not gate commits.

- [ ] **REQ-V26-07** — GATE-01 regression test unchanged: `s_linker13_min` Claude macro ≥ 0.9506 AND gpt-5.4 macro ≥ 0.9069 throughout v2.6. `s_linker14_voyager` bound to 0.87 floor only (experimental=True policy).

### AXIOM — Vocabulary Gap Fixes

- [ ] **REQ-V26-08** — Axiom Gap 1 (section-context naming / SCN): extend `COREF_RULES` or `DOC_KNOWLEDGE_EXTRACTION_RULES` axiom to cover role-referential definite NPs ("the server", "the client") where the component was established earlier in the section. Rule framing must be semantic/intent-level (not surface pattern). Empirical safety check: zero regression on MS/TS/JAB gold links before deployment. Targets 14 systematic FNs across BBB+TM.

- [ ] **REQ-V26-09** — Axiom Gap 2 (responsibility-list gerunds): extend `SEED_DISAMBIGUATION_RULES` axiom to reject bare gerund/nominal fragments describing a component's own capabilities without referencing an external participant. Must not reject legitimate cross-component gerund references. Empirical check: zero regression on MS/TS gold links. Targets 7 systematic FPs in TM.

- [ ] **REQ-V26-10** — Axiom Gap 3 (coref alias): extend `COREF_RULES` or `ANTECEDENT_ALIAS_RULES` axiom to instruct the LLM to set `antecedent_via_alias=True` when the antecedent sentence contains a known alias of the component (not just full canonical name). Uses existing code path at `s_linker14_voyager.py:1004`. Axiom change only, no code change. Safety check: BBB pronoun sentences must stay FP-free.

### TRAIN — Evaluation Tiers

- [ ] **REQ-V26-11** — Probe tier: 2-pass mainline run with new v5 loop. [TRAIN] and [TEST] F1 reported separately. Cheap-kill: [TEST] macro < 0.87 after pass 2 → KILL. Budget ≤ $10.

- [ ] **REQ-V26-12** — Range tier (conditional on Probe CONTINUE): convergence run, max 5 passes, mainline split. 5-dataset eval. 3-tier verdict. Budget ≤ $25.

- [ ] **REQ-V26-13** — Confirmation tier (conditional on Range ≥ 0.87): 3-split cross-validation with axiom-only baseline per held-out. Each split reports axiom-only vs trained-bank test F1. Final table vs v2.5 (89.1%) and canonical (90.69%). Budget ≤ $60.

### CARRY-FORWARD — Standing Gates

- [ ] **GATE-01** (carried) — `s_linker13_min` regression gate (Claude ≥ 0.9506, gpt-5.4 ≥ 0.9069) passes throughout v2.6. Code unmodified.
- [ ] **GATE-07** (carried) — `DEFAULT_BANK_PATH` updated to v2.6 trained bank; docstring updated with v2.6 results.
- [ ] **GATE-08** — Total training budget ≤ $80 (Phases 34–36). Infrastructure phases = $0.

## Future Requirements (deferred)

- Flex tier integration (`260601-flex-tier-integration.md`) — cost optimization, v2.7+
- AMBIGUITY_FEW_SHOT calibration — defer unless Assessor flags it

## Out of Scope for v2.6

- Cross-model Claude validation — gpt-5.4 only (per standing policy)
- `s_linker13_min` prompt changes — canonical frozen
- New benchmark datasets — 5-dataset benchmark unchanged
- GATE-06 mechanical vocab grep — replaced by Assessor abstraction-quality criterion + OD prompt enforcement

## Requirement Traceability

| REQ-ID | Phase |
|--------|-------|
| REQ-V26-01 | Phase 31 |
| REQ-V26-02 | Phase 31 |
| REQ-V26-03 | Phase 32 |
| REQ-V26-04 | Phase 32 |
| REQ-V26-05 | Phase 32 |
| REQ-V26-06 | Phase 32 |
| REQ-V26-07 | Throughout |
| REQ-V26-08 | Phase 33 |
| REQ-V26-09 | Phase 33 |
| REQ-V26-10 | Phase 33 |
| REQ-V26-11 | Phase 34 |
| REQ-V26-12 | Phase 35 |
| REQ-V26-13 | Phase 36 |
| GATE-01 | Throughout |
| GATE-07 | Phase 37 |
| GATE-08 | Phases 34–36 |
