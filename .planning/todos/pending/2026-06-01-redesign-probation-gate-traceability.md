---
created: 2026-06-01T19:00:00Z
title: Redesign probation gate — traceability over F1 delta
area: tooling
files:
  - scripts/voyager_train_tlr_v4_beta.py:682-708
  - scripts/voyager_train_tlr_v4_beta.py:860-912
---

## Problem

The P-role probation gate (`_probation_check`) is broken for learning. It compares two
stochastic L-role runs (step-1 vs candidate bank), but LLM variance (±3.4pp MS, ±3.6pp
TS, ±3.8pp BBB) dwarfs the pattern effect signal (≤1pp). In split2 (MS+TS+BBB): all 5
passes rolled back → 0 patterns committed → empty bank → no learning.

Six compounding bugs:
1. L has no cache — two independent noisy runs compared
2. `prior_f1s` updated from stochastic `train_f1s_after_l`, not committed state
3. `candidate_banks[projects[0]]` used for all projects (wrong once banks diverge)
4. Convergence = `len(accepted)==0` — GATE-06 never rejects, so never converges
5. Binary `delta >= 0` threshold — coin flip under noise
6. Probation tests on training data, no holdout

## Solution

**Shift: replace F1 delta gate with Traceability Gate.**

Instead of "does adding this pattern increase F1?", ask:
"does this pattern address an observed failure mode, and does it not worsen known regressions?"

**Gate A — FM Coverage (deterministic)**
Require D to annotate each proposal with failure mode IDs it addresses:
```json
{ "addresses_failure_modes": ["FM-1", "FM-3"], "addresses_fm_rationale": "..." }
```
Gate checks: non-empty list + all cited IDs exist in O's output for this pass.
Zero LLM cost. Reject if no FM cited.

**Gate B — Regression guard (deterministic)**
O already produces `newly_introduced_errors[]` with slot + description.
If a D proposal's slot matches a `newly_introduced_error` and Jaccard(rule_text, why_suspected) >= 0.4 → flag for rejection.

**Supporting fixes (still worth doing):**
- Add L cache keyed on (project, bank_content_hash, backend, model) — eliminates noise for step-1 runs
- Fix `prior_f1s` to use committed state not stochastic L
- Per-project candidate bank in probation (not representative)
- Add `committed_macro >= CONVERGENCE_THRESHOLD` as convergence condition

**Q1:** FM citation enforced in D prompt (preferred) or post-hoc LLM micro-judge?
**Q2:** Fallback when D can't cite FM → reject (preferred) or advisory?
**Q3:** Keep L step-1 run for metrics but drop probation L re-run entirely?

Design discussion happened in session 2026-06-01. Context in MEMORY.md / conversation.

## Key data

- Split2 (MS+TS+BBB): 14 proposals, 14 rollbacks, 0 patterns. Probation deltas: -0.52, -0.94, -1.26, -1.51, -0.64pp
- Split1 (MS+TS+TM): 14 proposals, 1 commit (pass3, +0.0022pp lucky noise), 2 patterns
- Split3 (TS+TM+JAB): 8 patterns committed — easier datasets, less BBB drag
- Oracle structure: `failure_modes[].{id, title, affected_slot, symptom, suggested_direction}`
- D proposals have `why_it_transfers` and `abstraction_check_cot` but NO FM IDs cited
