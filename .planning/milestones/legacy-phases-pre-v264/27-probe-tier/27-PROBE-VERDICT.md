# Phase 27 Probe Verdict

**Date**: 2026-06-02
**Split**: mainline (MS+TS+TM train, BBB+JAB test)
**Backend**: OpenAI gpt-5.4
**Infrastructure**: v2.5 (oracle cache fix + 15-slot bank)
**Log**: `logs/voyager_v4_beta/probe_p27.log`
**Artifacts**: `results/voyager_v4b_v25/mainline/`

---

## Verdict: CONTINUE

**Final train macro F1: 0.9193** (threshold: ≥ 0.87 → CONTINUE)

---

## Per-Pass Results

| Pass | MS F1 | TS F1 | TM F1 | Macro F1 | Delta | Committed |
|------|-------|-------|-------|----------|-------|-----------|
| 1 | 0.9508 | 0.9818 | 0.8254 | **0.9193** | +0.9193 | ✅ 12 patterns |
| 2 | 0.9508 | 0.9818 | 0.8226 | 0.9184 | −0.0009 | ⬜ no-op (below MIN_COMMIT_DELTA) |

Pass 2 committed macro (held from Pass 1): **0.9193**

---

## Success Criteria Assessment (REQ-V25-09)

### SC-1: All 3 train projects complete Pass 1
✅ MS, TS, TM each ran L-role; F1 logged above. Oracle cache keys include `bank_content_hash` (confirmed in code, line 461–463).

### SC-2: Gate A fired at least once; Gate B fired at least once
✅ **Gate A**: Pass 1 — accepted=12, rejected=0 (all 12 proposals had FM citations).
✅ **Gate B**: Pass 1 — accepted=12, rejected=0 (dual-direction judge invoked on all 12).
Gate decisions logged in `logs/voyager_v4_beta/probe_p27.log`.

### SC-3: ≥1 pattern committed in one of the 6 new slots
✅ **5 of 6 new slots have committed patterns** (far exceeds criterion):

| Slot | New? | Patterns |
|------|------|----------|
| `GENERIC_WORD_USAGE_RULES` | ✅ new | 2 |
| `ALIAS_SCOPE_RULES` | ✅ new | 1 |
| `ANTECEDENT_ALIAS_RULES` | ✅ new | 2 |
| `COREF_TERMINAL_SPECIFICITY_RULES` | ✅ new | 1 |
| `DOC_KNOWLEDGE_JUDGE_EXAMPLES` | ✅ new | 1 |
| `SEED_EXTRACTION_RULES` | ✅ new | 0 |
| `SEED_ACTOR_RULES` | ✅ new | 0 |

Existing slots also populated: `VALIDATION_RULES` (2), `COREF_RULES` (2), `DOC_KNOWLEDGE_JUDGE_RULES` (1).

Total bank: 12 patterns across 8 slots.

### SC-4: Train macro F1 after Pass 2 assessed against cheap-kill
✅ Committed macro = 0.9193 >> 0.87 cheap-kill threshold. No kill.

### SC-5: Verdict documented
✅ This document. Verdict: **CONTINUE**. Numeric evidence: final_macro=0.9193, 12 patterns committed, 5/6 new slots populated.

### SC-6: Total cost ≤ $10 gpt-5.4
✅ Only Pass 1 ran O+D+Gates (Pass 2 skipped via MIN_COMMIT_DELTA=0.005 filter). 6 L-role linker runs (3 projects × 2 passes). Estimated cost: **~$4–6** (well within $10 budget). Token counts not logged at harness level; estimate from run duration and typical L+O+D call volumes.

---

## Infrastructure Validation

### REQ-V25-01: Oracle Cache Fix
Pass 2 computed L independently (no cross-split oracle reuse). Oracle JSON for Pass 1 saved (`pass1_{project}_oracle.json`); no Pass 2 oracle file (MIN_COMMIT_DELTA skipped O step). Oracle key includes `bank_content_hash` per code at line 461–463.

### REQ-V25-02: MIN_COMMIT_DELTA = 0.005
Pass 2 delta = −0.0009 < 0.005 → `[P] delta=-0.0009 < MIN_COMMIT_DELTA=0.005 — skipping O+D (variance filter)` logged. O+D+Gates did NOT run in Pass 2. ✅ Working correctly.

### REQ-V25-03: D Underfilled-Slot Steering
Pass 1 D had 9 empty slots at start (all new + existing). D prompt steered toward zero-pattern slots. Result: 5 new slots populated in first D run. ✅ Effective.

### REQ-V25-04/05/06: 15-Slot Bank
Linker reported `15 slots` in header. 5 new slots received committed patterns. `ILinker3Injected` subclass active. ✅ All wired.

---

## Next Phase

**Phase 28 (Range Tier)** — CONDITIONAL on this CONTINUE verdict. Proceed.

- Full convergence run (macro ≥ 0.90, max 5 passes)
- 5-dataset evaluation
- Budget: ≤ $25 gpt-5.4
