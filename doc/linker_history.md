# Linker Version History

Macro F1 results across all 5 datasets (MS, TS, TM, BBB, JAB).

| Version | Macro F1 | Strategy Summary |
|---------|----------|-----------------|
| V5 | 83.8% | Base qualitative pipeline (no numeric thresholds) |
| V6 | 86.0% | V5 + dot-filter fix + generic judge examples |
| V6B | ~86% | V6 + abbreviation guard |
| V6C | ~87% | V6 + deterministic boundary filters (generic_word, package_path, weak_partial) |
| V9T | 83.6% | V5 + 6 error-driven features (A-F). Feature B (abbrev guard) best. |
| V11 | 84.9% | Generic-name-aware pipeline (7 changes from V6). Helped TM, hurt TS. |
| V12 | 82.0% | V11 + contextual stoplist. Over-filtered. |
| V13 | 86.1% | V6 + intersection generic filtering + TransArc-protected boundary filters |
| V14 | 86.7% | V13 + deterministic Phase 1 + structured judge (had Phase 6 bug) |
| V15 | 89.0% | V14 + root cause fix + enriched prompts. TM best-ever 91.7%. |
| V16 | 88.1% | V15 + alias-aware judge safety + CamelCase generic override |
| V23e | 94.1% | Hybrid batch/union judge. ALL-TIME BEST at time. |
| V24 | ~93% | V23e with hardcoded lists removed (benchmark-clean) |
| V25 | ~93% | SE textbook prompts (no leakage risk) |
| V25a/b | ~94% | V25 + deliberation TransArc judge variants |
| V26 | ~94% | Consolidated best approach |
| **V26a** | **95.4%** | V26 + synonym-safe judge bypass. **Current best.** |
| V26b | ~94% | V26 variant |
| V26c | ~95% | V26 + all-TransArc deliberation (only +2 FP killed) |
| V26d | ~95% | V26 + source-calibrated judge |
| V27b | ~94% | Deliberation Phase 3B generic judge |
| V27f | ~93% | Parallel branch pipeline + GoT aggregation |
| V27g | ~95% | GoT 3-branch sub-judges + source-aware weighting |
| V28 | ~95% | V26d + package-path TransArc filter |
| V29 (gate) | 94.2% | V26a + widened ambiguous gate + decomposed 2-question judge + intersection voting |

## What Does NOT Work
- **Judging ALL TransArc links** (V29-intersect 77.7%, V29-union 74.1%): kills TPs of technology-named components
- **Multi-run majority voting** (V9/v6_vote 87.5%): WORSE than single run
- **Phase-level voting** (v6_phase_vote 89.7%): noisy, kills borderline TPs
- **LLM-learned confusion patterns** (V7 89.2%): over-reject
- **Learned patterns as soft hints** (V8b 83.1%): destabilized judge
- **Union-accept + tiebreaker** (V8a 87.1%): union brings too many FPs
- **Proximity filter for auto-partials**: massive regression (82%)

## ILinkers (pure LLM, no TransArc baseline)
All clean, no data leakage (audited Feb 21). Use safe SE textbook prompt examples.

| Version | Macro F1 | Strategy |
|---------|----------|----------|
| I1 | 73.4% | Pure LLM extraction, high recall but very low precision (TM 42.5%, 143 FP) |
| I2 (run1/run2) | 86.5% / 87.9% | Precision-focused extraction, no contextual. High P (~95%) but lower R (~76%) |
| **V26a+I2** (run1/run2) | **92.7% / 95.7%** | V26a pipeline with I2 replacing TransArc as Phase 4 seed. Best pure-LLM. |

### V26a+I2 Per-Dataset (2 runs)
| Dataset | Run 1 | Run 2 |
|---------|-------|-------|
| MS | 98.4% (FP=0, FN=1) | 100.0% (FP=0, FN=0) |
| TS | 87.1% (FP=8, FN=0) | 96.4% (FP=2, FN=0) |
| TM | 90.6% (FP=7, FN=4) | 91.5% (FP=7, FN=3) |
| BBB | 90.1% (FP=10, FN=3) | 90.8% (FP=9, FN=3) |
| JAB | 97.3% (FP=1, FN=0) | 100.0% (FP=0, FN=0) |

### I2 Standalone Per-Dataset (2 runs)
| Dataset | Run 1 | Run 2 |
|---------|-------|-------|
| MS | 85.2% (FP=0, FN=8) | 89.3% (FP=0, FN=6) |
| TS | 84.0% (FP=2, FN=6) | 83.3% (FP=1, FN=7) |
| TM | 86.8% (FP=3, FN=11) | 87.0% (FP=4, FN=10) |
| BBB | 79.2% (FP=2, FN=20) | 80.0% (FP=1, FN=20) |
| JAB | 97.3% (FP=1, FN=0) | 100.0% (FP=0, FN=0) |

Key insight: I2 has very high precision (~95%) but misses ~24% of links (especially BBB). V26a's remaining phases (coref, entity extraction, validation) recover most of the recall, making V26a+I2 competitive with TransArc-based V26a.
