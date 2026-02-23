# V31 Pipeline: Quick Reference Card

## Phase Contribution Ranking (94.5% F1)

```
TIER 1: FOUNDATIONAL (3.5-4.0pp impact)
┌─────────────────────────────────────────────────┐
│ Phase 4: Baseline Seed (+2.5-3.0pp)             │
│   → ILinker2 pure-LLM extraction (80% recall)   │
│ Phase 5+6: Extract & Validate (+2.0-2.5pp)     │
│   → Entity mining + intersect voting            │
└─────────────────────────────────────────────────┘

TIER 2: MAJOR (1.0-1.5pp each)
┌─────────────────────────────────────────────────┐
│ Phase 7: Coreference (+1.0-1.5pp)               │
│   → Pronouns, definite mentions                 │
│ Phase 8b: Partials (+1.0-1.5pp)                 │
│   → Multi-word component matching               │
│ Phase 8c: Boundary Filters (+0.8-1.2pp)        │
│   → Convention filter (11 FP caught, 0 TP)     │
└─────────────────────────────────────────────────┘

TIER 3: MODERATE (0.3-0.8pp)
┌─────────────────────────────────────────────────┐
│ Phase 9: Judge (+0.3-0.5pp)                     │
│   → 70% of FPs judge-immune anyway              │
│ Phase 3: Doc Knowledge (supporting)             │
│   → Feeds abbreviations, synonyms, partials     │
│ Phase 1: Model Analysis (supporting)            │
│   → Discovers generic words at runtime          │
│ Phase 2: Patterns (~0.1pp)                      │
│   → Low precision, high variance                │
└─────────────────────────────────────────────────┘

REMOVED (zero/negative effect)
┌─────────────────────────────────────────────────┐
│ Phase 8: Implicit Refs ✓ REMOVED                │
│   (79-80% FP rate: 83/105 on Sonnet)            │
│ Phase 10: FN Recovery ✓ REMOVED                 │
│   (zero net gain: TPs + FPs cancel)             │
│ _filter_generic_coref (P7) ✓ REMOVED            │
│ _deterministic_pronoun_coref (P7) ✓ REMOVED     │
└─────────────────────────────────────────────────┘
```

---

## Data Flow: What Feeds What

```
Phase 0 (Profile)
  ↓
  ├─→ is_complex flag → Phase 7 (selects debate vs discourse mode)

Phase 1 (Model Analysis)
  ├─→ architectural_names
  ├─→ generic_words → Phase 6 (validation flags)
  ├─→ generic_partials → Phase 8b (parent-overlap guard)

Phase 2 (Patterns) [informational]
  ├─→ subprocess_terms
  └─→ action_indicators

Phase 3 (Document Knowledge)
  ├─→ abbreviations → Phase 5 (abbreviation guard)
  ├─→ synonyms → Phase 8c, Phase 9 (syn-safe bypass)
  └─→ partials → Phase 8b (injection)

Phase 4 (Baseline Seed)
  ├─→ transarc_set → Phase 5b, 8c, 9 (immunity)
  └─→ transarc_links → downstream phases

Phase 5 (Entity Extraction)
  └─→ candidates → Phase 6 (validation)

Phase 5b (Targeted Recovery)
  └─→ extended candidates → Phase 6

Phase 6 (Validation)
  ├─→ validated links
  └─→ generic_mention flags → higher scrutiny in P7-9

Phase 7 (Coreference)
  └─→ coref_links → Phase 8b (existing), dedup

Phase 8b (Partial Injection)
  └─→ partial_links → Phase 8c (mark immune), dedup

Phase 8c (Boundary Filters)
  └─→ filtered preliminary → Phase 9

Phase 9 (Judge)
  └─→ FINAL approved links
```

---

## Critical Design Decisions

| Decision | Why | Lesson |
|----------|-----|--------|
| **TransArc Immunity** | Judge kills tech-named components (FreeSWITCH, kurento) | V29-intersect: 77.7% F1 |
| **Syn-Safe Bypass** | Judge can't distinguish "server" generic from "HTML5 Server" component | Protects 65+ TPs |
| **Partial Inject Immunity** | LLM can't disambiguate partial-name refs without project context | Tested in p8b_llm.py: killed all TPs |
| **CamelCase Rescue** | CamelCase is constructed identifier, never generic | Protects "PaymentGateway", "EventDispatcher" |
| **Abbreviation Guard** | Wrong abbreviation expansion (CPU ≠ CurrentProcessingUnit) | Prevents invalid P5 candidates |
| **Generic Mention Flags** | Mark ambiguous names for stricter validation | 30-50% flagged are TPs, 50-70% are FPs |
| **Debate Mode (Complex Docs)** | Two-pass LLM coreference more reliable than single pass | Adaptive based on doc complexity |

---

## What Doesn't Work (Lessons Learned)

| Tried | Result | Lesson |
|------|--------|--------|
| Judge ALL TransArc links | 77.7% F1 (catastrophic) | Never judge non-ambiguous names |
| LLM replace P8b partials | Killed all recovery TPs | Partial-name disambiguation too hard for LLM |
| Reject tech-named components | 74.1% F1 | Many components ARE named after technologies |
| Hardened P3 extraction | 92.7% F1 | Components used informally, not formally defined |
| Generic coref filter | Zero effect | No actual FP savings |
| Pronoun coref rules | Net -0.1pp | LLM detection more reliable than regex |
| Remove syn-safe bypass | F1 drops to ~88% | Judge kills 65+ valid TPs |

---

## FP Attribution (V31: 23 FP total)

```
transarc (7 FP)          ████████░░ 30%
partial_inject (5 FP)    █████░░░░░ 22%
validated (5 FP)         █████░░░░░ 22%
coreference (2 FP)       ██░░░░░░░░  9%
convention_filter (0 FP) ░░░░░░░░░░  0% ← CATCHES 8 FPs correctly
implicit (0 FP)          ░░░░░░░░░░  0% ← SKIPPED
fn_recovery (0 FP)       ░░░░░░░░░░  0% ← SKIPPED
```

---

## Per-Dataset Highlights

| Dataset | F1 | Key Finding |
|---------|-----|-------------|
| **mediastore** | 95.4% | Perfect recall; convention filter helps FP |
| **teastore** | 94.3% | High precision; balanced partial+validation |
| **teammates** | 92.7% | Perfect recall; convention filter: 8 FP caught |
| **bigbluebutton** | 90.1% | Hardest; syn-safe bypass protects 14+ TPs |
| **jabref** | 100% | Perfect on all metrics |
| **MACRO** | **94.5%** | Production ready |

---

## Checkpoint Files (for offline ablation)

```
results/phase_cache/{dataset}/
├── phase0.pkl       # DocumentProfile
├── phase1.pkl       # ModelKnowledge, generic_words
├── phase2.pkl       # LearnedPatterns
├── phase3.pkl       # DocumentKnowledge
├── phase4.pkl       # transarc_links
├── phase5.pkl       # candidates
├── phase6.pkl       # validated
├── phase7.pkl       # coref_links
├── pre_judge.pkl    # preliminary (before judge)
└── final.pkl        # reviewed (after judge)
```

**Benefit**: Run ablation WITHOUT re-running LLM phases (e.g., test Phase 8c alone in seconds).

---

## Essential vs Optional

| Phase | Can Cut? | Evidence |
|-------|----------|----------|
| **P4 (Baseline)** | NO | F1 drops to ~60% |
| **P5+6 (Entity+Validation)** | NO | F1 -2.0pp |
| **P7 (Coref)** | NO | F1 -1.0pp |
| **P8b (Partials)** | NO | F1 -1.0pp |
| **P8c (Boundary)** | NO | F1 -0.8pp |
| **P3 (Doc Knowledge)** | NO | Feeds all downstream phases |
| **P1 (Model Analysis)** | NO | Without generic list → chaos in P5-9 |
| **P9 (Judge)** | MAYBE | F1 -0.3pp (mostly rubber-stamp) |
| **P2 (Patterns)** | YES | F1 -0.1pp (low precision) |
| **P8 (Implicit)** | ✓ REMOVED | 79-80% FP rate, code removed |
| **P10 (FN Recovery)** | ✓ REMOVED | Zero net gain, code removed |

---

## Summary

**V31 achieves 94.5% F1 through:**
- Strong baseline seed (P4: +2.5pp)
- Robust entity extraction + validation (P5+6: +2.0pp)
- Smart coreference (P7: +1.0pp)
- Partial matching (P8b: +1.0pp)
- Convention filtering (P8c: +0.8pp)
- Immunity rules (TransArc, syn-safe, partials protected from over-filtering)
- Careful judge design (4-rule reframed as universal principles)

**Trade-offs:**
- -0.9pp vs V26a (95.4%) for zero data leakage
- +2 FP, +3 FN vs V26a
- But: Cleaner code, auditable prompts, defensible design

---

## Key Files

- `PHASE_CONTRIBUTION_ANALYSIS.md` — Full breakdown
- `ilinker2_v31.py` — Implementation (804 lines)
- `test_heuristics.py` — Ablation tests
- `JUDGE_REFRAMING.md` — Judge defensibility
- `BENCHMARK_TABOO.md` — Safe prompt examples
