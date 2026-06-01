---
gsd_state_version: 1.0
milestone: v2.4
milestone_name: Phase Summary
status: Phase 23 confirmation complete — verdict WEAK (90.5% macro). Phase 24 Milestone Close next.
stopped_at: context limit (2026-06-01)
last_updated: "2026-06-01T20:00:00.000Z"
last_activity: 2026-06-01 -- Phase 23 confirmation done; 3 splits (5 passes each, cap hit); cross-split bank 2 patterns; 5-dataset eval=90.5%; verdict=WEAK; REQ-V24-01 FAIL (split-2 0/5 committed, gate works but BBB patterns don't generalize)
progress:
  total_phases: 5
  completed_phases: 4
  total_plans: 6
  completed_plans: 6
  percent: 80
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-01 for v2.3 close)

**Core value:** Every rule removed and every prompt-rule trimmed must hold macro F1 ≥ 0.93 on Claude Sonnet AND gpt-5.4 macro within tolerance of the v2.0 baseline (0.9077) — or be rejected. Generality first (GATE-06).
**Current focus:** Phase 20 — Infrastructure Prep (gate redesign + axiom improvements)

## Current Position

Phase: 23 — COMPLETE. Phase 24 Milestone Close next (unconditional).
Status: Phase 23 confirmation ran 2026-06-01 (gpt-5.4, 3 splits × 5 passes each). Cross-split bank: 2 patterns (DOC_KNOWLEDGE_EXTRACTION_RULES + COREF_RULES). 5-dataset eval: 90.5% macro. Verdict: WEAK. REQ-V24-01 FAILED (split-2 0/5 committed — gate correct, BBB patterns don't generalize). Verdict doc at `.planning/phases/23-confirmation-tier/23-CONFIRMATION-VERDICT.md`.
Last activity: 2026-06-01 -- Phase 23 confirmation complete, verdict WEAK (90.5%), REQ-V24-01 FAIL documented

```
[Phase 20 ✅]──▶[Phase 21 ✅]──▶[Phase 22 ✅]──▶[Phase 23 ✅]──▶[Phase 24 ⬜]
  (infra)         (probe)         (range)        (confirm)       (close)
```

**Next action**: Phase 24 Milestone Close — audit, requirements close-out, archive, PROJECT.md update.

## v2.4 Design Decisions (Locked)

All locked in conversation from v2.3 post-mortem (2026-06-01). Full details in `.planning/v2.4-prep/v2.4-KICKOFF-SEED.md`.

### D-1 Fix: Traceability Gate (replaces broken F1-delta probation gate)

**Root cause of split2 empty bank**: 6 compounding bugs in `_probation_check()` — two independent stochastic L runs compared; LLM variance ±3.4–3.8pp per dataset swamps ≤1pp pattern signal.

**Gate A (deterministic, $0)**: D prompt must include `addresses_failure_modes` list (non-empty, all IDs exist in O's failure_modes for this pass). Reject if empty — not advisory.

**Gate B (LLM judge, ~$0.01/pass)**: Checks `fixes_cited_fm: true AND causes_new_error: false AND confidence: high|medium`. Uses O's failure_modes and newly_introduced_errors as inputs.

**Drop probation re-run entirely**: Gate A/B are the replacement.

**4 supporting bug fixes**: L cache (project + bank_content_hash + backend + model); `prior_f1s` from committed state; per-project candidate bank; `committed_macro >= CONVERGENCE_THRESHOLD` convergence condition.

### D-2 + D-3 Fix: Three Axiom Gaps

**Gap 1 — Section-context naming (SCN), 14 FNs in BBB+TM**:

- Correct fix: COREF_RULES extension (not DOC_KNOWLEDGE alias — entity extraction is batch-mode without window context; local aliases filtered at lines 788-791).
- Requires ~8-line code change in `_coref_cases_in_context`: build `role_ref_pat` from component terminal words; expand `pronoun_sents` → `anaphoric_sents`.
- COREF_RULES axiom change has zero effect without the code change (SCN sentences contain no pronoun).

**Gap 2 — Responsibility-list FPs, 7 FPs in TM**:

- Axiom only. SEED_DISAMBIGUATION_RULES: add "description of the component's own capabilities without referencing an external participant" to OTHER category.
- Empirically safe: 0/27 MS gold + 0/23 TS gold sentences start with gerund.

**Gap 3 — Alias coref**:

- One phrase in COREF_RULES: add "or aliased" alongside "named". Code path at line 1017 already checks `antecedent_via_alias`.
- Combined with Gap 1 into single COREF_RULES semantic reframing.

## Phase 20 Plans (all independent — any order)

| Plan | File Target | Change |
|------|-------------|--------|
| 20-P1 | `scripts/voyager_train_tlr_v4_beta.py` | Gate A + Gate B + drop probation + 4 bug fixes |
| 20-P2 | `prompts_v3_axiom.py` + `s_linker14_voyager.py` | COREF_RULES + SEED_DISAMBIGUATION_RULES + `_coref_cases_in_context` |
| 20-P3 | eval + DEFAULT_BANK_PATH | 5-dataset eval (gpt-5.4) as Phase 20 baseline |

## Standing Gates (carried forward)

- **GATE-01**: `s_linker13_min` macro F1 ≥ 0.93 Claude AND gpt-5.4 ≥ 0.8977. Applies to canonical only.
- **GATE-06**: BENCHMARK_TABOO grep + reviewer_critic. Applies at bank-entry AND at axiom-diff boundary (Phase 20-P2).
- **GATE-07**: `s_linker14_voyager` registered experimental=True; DEFAULT_BANK_PATH updated in Phase 20-P3.
- **GATE-08**: Total budget cap ~$80 (Phase 21–23). Lower than v2.3 because Phase 20 eliminates split2 empty-bank waste.

## Canonical Artifact (current)

- **`src/llm_sad_sam/linkers/experimental/s_linker13_min.py`** (v2.1 PROMOTED, `canonical=True`, unchanged)
- Claude Sonnet macro F1: 0.9506 | gpt-5.4 macro F1: 0.9069

## v2.3 Summary (for context)

**Verdict:** WEAK — cross-split macro F1 = 90.5% (gpt-5.4, 5-dataset). Gap to canonical: −0.19pp.
**Key finding:** Probation gate broken (6 bugs) → split2 0/5 passes committed. Axiom vocabulary ceiling (SCN + gerund FPs). 2 patterns survive cross-split filter; deliver +1.6pp lift.
**See:** `.planning/milestones/v2.3-MILESTONE-AUDIT.md`

## Accumulated Context

### Decisions (carried from v2.3)

- β architecture (L + O + D-with-CoT-A + P) — kept, same roles
- `s_linker14_voyager` standalone (no inheritance)
- Bank format: slot-uniform 9 slots, per-project during training, aggregated final_bank.json
- Cache: per-(text_stem, comp_hash, backend, model), VOYAGER4B_CACHE_ROOT override
- GATE-06 at bank-entry (helpers already built in Phase 14; apply in Phase 21+)
- gpt-5.4 default backend (Claude only if explicitly required)

### Pending Todos (resolved as v2.4 Phase 20)

3 todos filed during v2.3 — all resolved by Phase 20 plans:

1. `2026-06-01-design-better-axioms-section-context-responsibility.md` → Phase 20-P2
2. `2026-06-01-redesign-probation-gate-traceability.md` → Phase 20-P1
3. `2026-06-01-implement-refined-v3-axiom-diffs-feasibility-study.md` → Phase 20-P2

### Blockers/Concerns

None. All three v2.3 debts have locked designs ready for implementation.

## Deferred Items (v2.5+ candidates)

- Claude cross-model verification of v4 (per backend policy)
- Complete v4 re-architecture beyond infra fix
- Per-document or runtime-adaptive bank building
- (B-new) runtime bank-builder

## Session Continuity

Last session: 2026-06-01T15:09:14.993Z
Stopped at: context exhaustion at 75% (2026-06-01)
Resume file: None
Next action: Phase 20 — 3 independent plans (20-P1 gate, 20-P2 axioms, 20-P3 eval)
