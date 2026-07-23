# Phase 42: v2.7 Milestone Close

**Status**: not started
**Depends on**: Phase 41
**Budget**: $0

## Goal

v2.7 ships as a complete, auditable milestone covering Tier C axiom + s10 partial-injection port + recall-oracle training-loop redesign + cross-split semantics fix. GATE-01/06/07/08 verified. All v2.7 requirements resolved with PASS / FAIL / N/A. Debts explicitly carried forward to v2.8 or closed.

## Subtasks

1. **18a — Milestone audit**
   - Write `.planning/milestones/v2.7-MILESTONE-AUDIT.md` covering all v2.7 phases (38–42), with per-phase verdict, GATE statuses, and key numeric evidence (F1 per dataset, recall-oracle accept count, cross-split variance).
2. **18b — Requirements close-out**
   - Update `.planning/milestones/v2.7-REQUIREMENTS.md` (create if absent) with full traceability table.
3. **18c — Gate verifications**
   - GATE-01: `s_linker13_min` byte-equal to v2.6 close; Claude macro ≥ 0.9506; gpt-5.4 macro ≥ 0.9069 (re-run sanity).
   - GATE-06: all v2.7 new axiom slots, partial-injection prompts, recall-oracle prompts, committed bank patterns pass mechanical taboo regex + Assessor abstraction-quality check.
   - GATE-07: `s_linker14_voyager.py:DEFAULT_BANK_PATH` updated to v2.7 trained bank (or documented unchanged if Phase 41 = REGRESSION); docstring updated with verdict.
   - GATE-08: total Phase 41 spend ≤ $35 gpt-5.4; per-tier actuals recorded.
4. **18d — STATE.md + PROJECT.md update**
   - STATE: milestone = v2.7 complete; current focus rolls to v2.8 candidate or "project complete".
   - PROJECT.md Past Milestones entry for v2.7 added with outcome verdict + key numbers + GATE-01 confirmation.
5. **18e — Debt carry-forward**
   - Document v2.8 candidates from Phase 41 PARTIAL/REGRESSION findings (recall-oracle gap, cross-split variance, BBB structural FN remainder).
   - Update todo `.planning/todos/pending/260601-flex-tier-integration.md` if blocked status changed.

## Success Criteria

1. Milestone audit document exists with PASS/FAIL/N/A per requirement and per gate.
2. GATE-01 sanity re-run confirms `s_linker13_min` unchanged.
3. GATE-06 grep + audit clean on all v2.7 new prompt/slot/bank text.
4. GATE-07 DEFAULT_BANK_PATH state documented (changed or unchanged with reason).
5. GATE-08 cost ledger present.
6. STATE.md milestone field flips to v2.7 complete + current focus rolls.
7. PROJECT.md Past Milestones table has v2.7 entry.
8. Open debts (v2.8 candidates) listed in audit + STATE.

## Risk

- None — Phase 42 is documentation + verification.

## Out of Scope

- New code or prompt changes.
- Training re-runs.
- v2.8 kickoff.

## Plans

- TBD (audit)
- TBD (gate verifications)
