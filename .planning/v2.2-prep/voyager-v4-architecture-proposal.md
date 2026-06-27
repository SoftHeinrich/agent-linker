# Voyager v4 — Multi-Role Architecture Proposal

**Drafted:** 2026-06-01
**Status:** Proposal for v2.2 first plan
**Motivation:** Voyager v2 (gpt-5.4) and v3 (Claude, partial) confirmed single-role training is split-fragile. Mean held-out lift across 3 splits was −0.05pp; one split regressed −1.92pp. Root cause: the feedback LLM sees raw doc strings + the linker's blind spots, then the same LLM also distills + reviews — collapsing 4 distinct cognitive roles into one model produces distributionally-biased rules that pass lexical GATE-06 but fail at transfer.

## Current Voyager v2/v3 architecture (single-role baseline)

```
LINKER (gpt-5.4 / Claude) ─predictions─▶ GOLD COMPARE (mechanical) ─FP/FN─▶
  FEEDBACK LLM (same model) ─patterns─▶ GATE-06 grep + dedupe ─▶ SKILL BANK
                                                                      │
                                                                      ▼
DISTILL LLM (same model) ─candidates─▶ REVIEWER LLM (same model) ─verdicts─▶
  DISTILLED SKILLS ─frozen─▶ HELD-OUT TEST
```

**Problems exposed by Voyager v2/v3:**

1. **Surface-rule leakage channel**: feedback prompt includes raw FP/FN sentences as strings. Even with "no benchmark terms in output" constraint, learned rules calibrate to the surface-form distribution of the training set.
2. **GATE-06 grep is lexical**, not distributional. Patterns can pass strict taboo grep while still encoding the training-set vocabulary bias.
3. **Single-model conflation**: linker + feedback + distill + review are 4 different cognitive tasks. One model with one set of biases performs all four → biases compound rather than cancel.
4. **No iterative re-evaluation**: once a pattern enters the bank, it's never tested for whether it actually helped or hurt later iterations.
5. **Linker errors and validator errors are entangled**: s_linker13 internally has proposer + judge stages, but the feedback signal is "the FINAL output was wrong" — no attribution to which sub-stage failed.

## Proposed v4 architecture (multi-role + structural separation)

### Role table

| Role | What it does | Sees | Does NOT see | Output |
|---|---|---|---|---|
| **R1: Linker** (proposer) | Generate candidate sentence→component links | doc + axioms + linker skill bank | gold, validator skills, distilled patterns | raw candidate set |
| **R2: Validator** (judge) | Filter candidates with structured verdicts | doc + axioms + validator skill bank + R1 output | gold, linker skills, distilled patterns | `{candidate, verdict, weakness_class, evidence_span}` per link |
| **R3: Skill Distillator** | Extract abstract patterns separately for linker and validator | categorical error summaries (NOT raw doc strings) + current skill banks | gold, raw doc, raw FP/FN sentences | `{linker_skills: [...], validator_skills: [...]}` updates |
| **R4: Feedback Judge** | The only role with gold access. Categorize R1+R2 errors. Tell R3 what TO learn (not what TO say). | gold + R1 output + R2 output + skill banks | doc text (only sentence IDs from FP/FN, not contents) | per-role categorical signal: `{linker_missed_categories: [...], validator_overrejected_categories: [...]}` |
| **R5: Abstraction Validator** | Reject patterns that depend on a specific architectural style | proposed pattern + a small library of textbook architectural styles (microservice mesh, event-sourced, layered monolith, pipe-and-filter, etc.) | gold, doc, training set | per-pattern verdict: `{pattern, transferable: bool, reason}` |

### Data-flow diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   doc + axioms + linker_skills                                          │
│         │                                                                │
│         ▼                                                                │
│   ┌────────────┐                                                        │
│   │ R1: LINKER │──candidates──┐                                         │
│   └────────────┘              │                                         │
│                               ▼                                         │
│   doc + axioms + validator_skills + candidates                          │
│                               │                                         │
│                               ▼                                         │
│                       ┌────────────────┐                                │
│                       │ R2: VALIDATOR  │──verdicts──┐                   │
│                       │ (verdicts +    │            │                   │
│                       │  weakness_class)│           │                   │
│                       └────────────────┘            │                   │
│                                                     ▼                   │
│   gold + R1 candidates + R2 verdicts                                    │
│                                                     │                   │
│                                                     ▼                   │
│                                       ┌─────────────────────┐           │
│                                       │ R4: FEEDBACK JUDGE  │           │
│                                       │   (only oracle role)│           │
│                                       │   only sees IDs +   │           │
│                                       │   counts, not text  │           │
│                                       └──────────┬──────────┘           │
│                                                  │ categorical signal   │
│                                                  ▼                       │
│                                       ┌─────────────────────┐           │
│                                       │ R3: DISTILLATOR     │           │
│                                       │ (proposes patterns  │           │
│                                       │  per role: linker   │           │
│                                       │  or validator)      │           │
│                                       └──────────┬──────────┘           │
│                                                  │ candidate patterns   │
│                                                  ▼                       │
│                                       ┌─────────────────────┐           │
│                                       │ R5: ABSTRACTION     │           │
│                                       │     VALIDATOR       │           │
│                                       │  (tests against     │           │
│                                       │  multiple arch      │           │
│                                       │  style descriptions)│           │
│                                       └──────────┬──────────┘           │
│                                                  │ accepted patterns    │
│                                                  ▼                       │
│                                          linker_skills.json             │
│                                          validator_skills.json          │
│                                                                          │
└────────────────────────────────────────────────────────────────────────┘
```

### Key innovations vs v2/v3

#### 1. Categorical feedback channel (closes surface-rule leak)

R4 emits structured signal:
```json
{
  "linker_error_categories": [
    {
      "category": "abbrev_expansion_missed_no_iei",
      "count": 3,
      "advice": "Linker should accept abbreviation introduced parenthetically even without 'i.e.' marker"
    },
    {
      "category": "sub_element_promoted_to_container",
      "count": 2,
      "advice": "Linker should NOT promote a named sub-element to be an alias of its container component"
    }
  ],
  "validator_error_categories": [
    {
      "category": "tech_label_over_approved",
      "count": 4,
      "advice": "Validator should reject when sentence gives only package/technology label without role assignment"
    }
  ],
  "summary_stats": {
    "linker_fp_count": 7, "linker_fn_count": 5,
    "validator_overreject_count": 4, "validator_under_reject_count": 2
  }
}
```

R3 (distillator) receives this CATEGORICAL summary and proposes patterns. **R3 NEVER sees raw doc strings or raw FP/FN sentences.** This breaks the surface-rule leak channel that caused v2/v3 split-fragility.

#### 2. Linker vs validator skill banks (separate stores)

`results/voyager_v4/<split>/linker_skills.json` and `validator_skills.json` as distinct artifacts. Each contains patterns ONLY for that role.

At inference: R1 reads `linker_skills` only; R2 reads `validator_skills` only. Skill bleed-through is impossible.

#### 3. R5 abstraction validator (closes distributional bias)

Before any pattern enters its skill bank, R5 evaluates:
```
prompt = """Given this proposed pattern: "{pattern}"

Architectural styles to test:
  1. Microservice mesh (services communicate via async events, no shared DB)
  2. Event-sourced system (state derived from event stream, no current-state tables)
  3. Layered monolith (controller → service → repository → DB)
  4. Pipe-and-filter (data transforms in series, no central state)
  5. Hexagonal/ports-and-adapters (domain core surrounded by adapters)

For each style: would this pattern produce the same accept/reject decisions?
If pattern depends on style-specific vocabulary or assumptions → REJECT.
If pattern is style-neutral → ACCEPT.

Output: {{"verdict": "ACCEPT"|"REJECT", "reason": "...", "style_dependency": <style_name>|null}}
"""
```

R5 has NO access to gold, NO access to the doc, NO access to the training set. Pure abstraction check.

#### 4. R4's restricted view (oracle without surface contamination)

R4 sees:
- `(sentence_id, component_id, gold_verdict, linker_output, validator_output)` tuples
- Component ID list (which is abstract — `comp_3` not `MediaStore`)
- Aggregated counts

R4 does NOT see:
- Raw sentence text
- Component names
- Doc context

This forces R4's categorical output to be derived from STRUCTURAL information, not lexical patterns.

#### 5. Per-iter re-evaluation gate

After R3 proposes patterns and R5 accepts them, the patterns get a probation period: next iteration runs with the augmented skill banks and we measure delta-F1. If delta < 0 on a hold-out fold of the training set, the patterns are removed. This catches patterns that look good in isolation but hurt in composition.

## Comparison summary

| Property | v2/v3 (single-role) | v4 (multi-role) |
|---|---|---|
| Roles | 1 (LLM does everything) | 5 (R1-R5 separated) |
| Feedback channel | Raw FP/FN sentences | Categorical error summaries |
| Skill bank | Monolithic | Per-role (linker + validator) |
| Surface-rule leak | Yes (root cause of fragility) | Closed by R4 restricted view |
| Distributional bias check | None (only lexical grep) | R5 abstraction validator |
| Pattern re-evaluation | Never | Probation period each iter |

## Cost & complexity

Per training iteration, v2/v3 = ~2 LLM calls (linker + feedback).
v4 = ~5 LLM calls (R1 linker + R2 validator + R4 categorize + R3 distill + R5 abstraction).

**~2.5× more expensive per iteration** but:
- Probation gate may need fewer outer passes to converge
- Categorical feedback may produce stronger signal per call → fewer iters needed
- The split-fragility problem is the actual bottleneck; lower cost without solving it isn't progress

## Open design questions

1. **Should R1 and R2 use the same backend or different?** Same backend = cheaper, same biases. Different = more expensive, biases cancel. v4 default: same backend (Claude Sonnet); v4b experiment: R1 Claude + R2 gpt-5.4 (or vice versa).
2. **Should R4 use gold standard or LLM-as-oracle?** Gold = true supervised signal. LLM-as-oracle (a third model) = self-supervision (Reflexion-style). v4 default: gold; v4c experiment: LLM-as-oracle for the unsupervised regime.
3. **R5 abstraction validator: same backend or "smarter" model?** Smarter model = better abstraction check at higher cost. Default: same backend as R3.

## Implementation plan (for v2.2 Plan 14-XX onwards)

1. **Phase 14 Plan 14-01**: Build the 5-role harness. Refactor `voyager_train_tlr_v2.py` into `voyager_train_tlr_v4.py` with explicit R1-R5 calls.
2. **Phase 14 Plan 14-02**: Implement R4 categorical signal extraction + R5 abstraction validator.
3. **Phase 14 Plan 14-03**: Run v4 on Split 1 only (mediastore + teastore + teammates train, BBB + jabref test), Claude Sonnet, sampling-test mode (mediastore as smoke test before full split).
4. **Phase 14 Plan 14-04**: If Split 1 v4 beats v2 Split 1 by ≥ 1pp → run all 3 splits.
5. **Phase 14 Plan 14-05**: If 3-split mean held-out lift ≥ 1pp → write Voyager v4 rollup as v2.2 anchor finding.

## What v4 directly tests

| Hypothesis | Test |
|---|---|
| Surface-rule leak is the split-fragility root cause | R4 restricted view + R3 categorical-only input. If v4 closes the −1.92pp Split 3 regression, hypothesis confirmed. |
| Distributional bias is independent of lexical leak | R5 abstraction validator rejection rate. If R5 rejects many patterns that GATE-06 grep approved, hypothesis confirmed. |
| Linker vs validator skill separation produces stronger transfer | Compare `linker_skills.json` vs `validator_skills.json` content. If they differ meaningfully (not just rephrasing), separation is informative. |
| Multi-role iteration converges with fewer outer passes | Outer-pass count to convergence in v4 vs v2. |

## Falsification criteria

v4 fails if any of:
- 3-split mean held-out lift < 0.5pp (no real improvement over v2's ~0)
- R5 rejects 80%+ of R3's proposals (R3 can't produce abstract enough patterns)
- Per-iter cost > 4× v2 without proportional outer-pass reduction
- Split 3 still regresses (root cause not addressed by role separation)

If v4 fails, the **publishable negative result** is: "role separation alone does not solve distributional bias for many-decision classification — the limit is at the abstraction layer, not the architecture."
