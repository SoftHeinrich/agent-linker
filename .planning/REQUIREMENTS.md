# Requirements: llm-sad-sam-v45 — Milestone v2.3

**Defined:** 2026-06-01
**Milestone:** v2.3 — Trained Multi-Role Prompt Replacement (β architecture)
**Architecture spec:** `.planning/v2.3-prep/v2.3-ARCHITECTURE.md`
**Endpoint reasoning:** `.planning/notes/v23-architectural-endpoint-reasoning.md`
**Kickoff seed:** `.planning/v2.3-prep/v2.3-KICKOFF-SEED.md`
**Core Value:** Multi-role training (L + O + D-with-CoT-A + P) produces per-slot rich rules+examples that, composed with axiom skeletons, match or beat hand-authored `prompts_v3.py` on gpt-5.4 macro F1. Validates whether expert-hand-tuning of LLM-pipeline prompts can be automated. Static-prompt-elegance preserved; backend = gpt-5.4 only.

## Active v2.3 Requirements

### TRAIN — Multi-Role Training Harness

- [ ] **REQ-V23-01** — v4 β training harness produces a trained per-slot JSON bank from gpt-5.4 outer-loop training. Anchor on `scripts/voyager_train_tlr_v4_a_prime.py` (do NOT re-derive vocab-aligned R3); refactor outer loop to β role structure (L + O + D + P). Mainline split: train MS+TS+TM, test BBB+JAB.
- [ ] **REQ-V23-02** — `s_linker14_voyager.py` consumes trained bank via Voyager-mode runtime injection. Extends or clones `s_linker13_skill_learned_clean` pattern: axiom prompts (`prompts_v3_axiom.py`) wrapped at linker init with `LEARNED PATTERNS` header + per-slot bullet list of rule_text + example_block. Registered in `CANONICAL_VARIANTS` with `experimental=True` tag.
- [ ] **REQ-V23-03** — Training loop = L (linker, ~7 LLM calls/project) + O (text-aware Oracle, 1 LLM call/project) + D (text-blind Distillator with CoT-A inline, 1 LLM call/iter) + P (mechanical probation gate, 0 LLM cost). Per-iter delta-from-prior-iter tracked.
- [ ] **REQ-V23-04** — Oracle output schema = failure-mode-centric JSON. Top-level fields: `iter`, `split`, `L_predictions_summary` (macro_F1 + delta + per_dataset), `failure_modes[]` (with `id`, `title`, `affected_slot`, `symptom`, `apparent_cause`, `suggested_direction`, `evidence_count`, `abstract_example_pair`), `newly_introduced_errors[]` (with `introduced_by_bank_pattern` attribution). Full schema in `.planning/v2.3-prep/v2.3-ARCHITECTURE.md`.

### GATE — Promotion + Validation

- [ ] **REQ-V23-05** — Promotion verdict computed against 3-tier bar on gpt-5.4 macro F1, 5-dataset: STRONG ≥ 0.9173 (= trim1) / WEAK [0.87, 0.9173) / FAIL < 0.87. Reported in milestone audit.
- [ ] **REQ-V23-06** — Dual-artifact registration: `s_linker13_min` retains `canonical=True` (GATE-01 bound — Claude floor + cross-model floor 0.8977); `s_linker14_voyager` ships `experimental=True` (bound only to 0.87 floor). v2.3 ships v4 finding regardless of polarity without violating standing GATE-01.
- [ ] **REQ-V23-07** — Mainline single-split (Probe + Range tiers) on train MS+TS+TM, test BBB+JAB. Confirmation 3-split sweep (Voyager v2 splits 1+2+3) ONLY if mainline ≥ 0.87. Cheap-kill at each tier per budget cap.
- [ ] **REQ-V23-08** — Compact-B fallback (R345 single role with structured CoT, per `v2.2-SCOPE-DECISION.md`) auto-triggered as v2.3 mainline replacement if v4 FAIL (< 0.87). If Compact-B also FAILs, v2.3 ships negative-finding paper artifact.

### GENERALITY — Static-Prompt-Elegance + Leak Defense

- [ ] **REQ-V23-09** — GATE-06 BENCHMARK_TABOO grep + reviewer-defensibility critic LLM at bank-entry boundary. Every D pattern proposal grep'd before insertion. Failed-grep patterns rejected; D may be asked to reformulate. Per-pattern critic call gates final acceptance.
- [ ] **REQ-V23-10** — Per-(text_stem, comp_hash, backend, model) cache infrastructure applied uniformly across L (per-stage outputs), O (failure-mode JSON), D (pattern proposals), reviewer critic. Cache key formula per `s_linker14_probe_d_upstream_clean.py` (do NOT re-derive). Cache root: `results/voyager_v4_beta/cache/` (override via `VOYAGER4B_CACHE_ROOT` env var).
- [ ] **REQ-V23-11** — D's pattern proposals constrained to discourse-syntactic-functional vocabulary (Probe A' R3 vocab anchor). Allowed: subject-position, predicate, anaphora, parenthetical, namespace-prefix, section-heading, sentence-position, qualifier-clause, cross-reference + structural verbs (over-approved, under-rejected, propagated, missed, expansion, alias-of, container-of, sub-element-of). Forbidden: role nouns + architectural-style names.
- [ ] **REQ-V23-12** — Slot-uniform bank: trained patterns cover all 9 axiom slots (`AMBIGUITY_RULES`, `AMBIGUITY_FEW_SHOT`, `DOC_KNOWLEDGE_EXTRACTION_RULES`, `DOC_KNOWLEDGE_JUDGE_EXAMPLES`, `DOC_KNOWLEDGE_JUDGE_RULES`, `ENTITY_EXTRACTION_RULES`, `VALIDATION_RULES`, `COREF_RULES`, `SEED_DISAMBIGUATION_RULES`). No mixing static + bank within any slot.

### INFRA — Training Convergence + Budget

- [ ] **REQ-V23-13** — Convergence = macro F1 ≥ 0.90 on training projects, max 5 outer passes. Per-pass macros logged; converged-early result preserved (do not over-train).
- [ ] **REQ-V23-14** — Budget cap ~$100 gpt-5.4 total. Probe tier $5-10 (mainline split, 1-2 outer passes). Range tier $15-25 (mainline, run to convergence). Confirmation tier $40-60 (3-split sweep). Compact-B fallback budget $10-20 (if triggered). Per-tier cheap-kill on tier-floor miss.
- [ ] **REQ-V23-15** — Comparison reference: primary `s_linker14_voyager` (axiom + trained bank) vs `s_linker13_min` (hand-authored `prompts_v3.py`) — gpt-5.4 macro F1, 5-dataset. Secondary vs `prompts_v3_axiom.py` (axiom-only floor) to attribute lift between trained patterns and minimal-skeleton baseline. Both numbers in milestone audit.

### CARRY-FORWARD — Standing Gates

- [ ] **GATE-01** (carried) — `s_linker13_min` regression test passes throughout v2.3 (Claude macro ≥ 0.93 AND gpt-5.4 macro ≥ 0.8977). v4 NOT bound to this gate.
- [ ] **GATE-02** (carried) — frozen-compat regression test extended with `s_linker14_voyager` against locked-evaluation baseline JSON. New baseline allowed (single-run snapshot) since v4 is `experimental=True`.
- [ ] **GATE-06** (carried) — generality audit applies to all v4 artifacts: trained bank patterns + reviewer-critic prompts + Oracle prompts + Distillator prompts. Cross-dataset isolation methodology applies.
- [ ] **GATE-07** (carried) — `s_linker14_voyager` registered in `CANONICAL_VARIANTS` + `VARIANT_SPECS` with structured docstring documenting β architecture + trained-bank dependency.
- [ ] **GATE-08** (carried from v2.2) — cost-per-improvement audit on v4 mainline. v4 must justify ~$60-100 training cost via STRONG promotion OR document failure mode publishable as negative finding.

## Locked Decisions Reference

| Decision | Lock |
|---|---|
| Architectural endpoint | (A) Voyager-bank canonical |
| Promotion bar | 3-tier on gpt-5.4 macro F1 |
| Dual-artifact policy | s_linker13_min canonical=True + s_linker14_voyager experimental=True |
| Backend | gpt-5.4 only; Claude only if super necessary |
| Output artifact | Static JSON bank (Voyager-mode runtime injection) |
| Per-pattern content | Rich (rule_text + example_block + why_it_transfers + abstraction_check_cot) |
| Training architecture | β (L + O + D-with-CoT-A + P) |
| Oracle mode | (i) text-aware O, text-blind D |
| Oracle output | Failure-mode-centric (FM list + newly_introduced_errors) |
| Leak defense placement | Bank-entry boundary (not O/D handoff) |
| Slot scope | All 9 axiom slots, joint per iter |
| Probation gate | Mechanical, per-batch rollback on F1 delta < 0 |
| Convergence | macro ≥ 0.90, max 5 outer passes |
| Train/test split | Mainline MS+TS+TM → BBB+JAB; 3-split confirmation conditional |
| Budget | ~$100 cap, tiered cheap-kill |
| Comparison reference | vs prompts_v3.py primary, vs prompts_v3_axiom.py secondary |
| Cross-iter state | D + O both see current bank |
| Cache | Per-(text_stem, comp_hash, backend, model) uniform |
| D vocab constraint | Discourse-syntactic-functional only |
| R3 anchor | scripts/voyager_train_tlr_v4_a_prime.py |
| Fallback | Compact-B (R345 single CoT) on v4 FAIL |
| Linker name | s_linker14_voyager |
| Bank persistence | Per-project banks during training; aggregated global at end |

## Out of Scope (v2.3)

- Claude backend runs (per `[[feedback-prefer-gpt-backend]]` memory)
- Voyager v3 Claude splits 2+3 (parked per `.planning/todos/260601-skipped-experiments-v22.md`)
- Probe D Claude re-test (skipped per backend policy)
- Cross-model promotion of v4 to canonical (defer to v2.4 if reviewers require)
- (B) endpoint full runtime-rubric linker (contingency only — reconsidered if v4 FAIL)
- (B-new) runtime bank-builder (considered, rejected for v2.3, logged for future)
- Separate R5 abstraction-validator as standalone LLM role (folded into D's CoT-A)
- Changes to frozen artifacts (`s_linker13.py`, `prompts_v2.py`, `ilinker*`, `data_types_v2`, `document_loader_v2`, `pcm_parser_v2`)
- Modifications to `s_linker13_min` (canonical, must be preserved untouched)
- New seed/linker approaches beyond v4 β + Compact-B fallback

## Deferred to Future Milestones (v2.4+)

| Item | Why deferred |
|---|---|
| Claude cross-model verification of v4 | Per backend policy — only run if reviewers require; out-of-scope for v2.3 publish |
| (B-new) runtime bank-builder | Considered at v2.3 kickoff; rejected as more complex than (A) static bank with weaker A/B comparison story. Future milestone candidate if (A) shows per-doc adaptation gap |
| ADAPTER-01 backend-adaptive prompts | Re-opened by v2.1 trim4/5/6/7 evidence; not v2.3-scope |
| Extended Thinking on judge stages | Carried from v2.1; not v2.3-scope |
| Link provenance data structure | Carried from v2.1; not v2.3-scope |
| EXT-04 emit-biased boundary prompting | Carried from v2.0; not v2.3-scope |
| (C) Hybrid runtime + static per-slot | Ruled out by slot-asymmetry-is-ugly principle |
