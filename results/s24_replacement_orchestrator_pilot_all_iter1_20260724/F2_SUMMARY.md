# S24 replacement orchestrator — fixed-phase replay summary

The controller assembled its result from recorded knowledge/entity/coreference
phase-tool outputs plus a fresh semantic coverage audit. The S21 final checkpoint
was loaded only after finalization for comparison.

| System | TP | FP | FN | Macro F1 | Macro F2 | Pooled F1 | Pooled F2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S21 | 168 | 5 | 27 | 93.34% | 90.83% | 91.30% | 88.14% |
| Current dynamic augmentation | 173 | 5 | 22 | 94.68% | 92.80% | 92.76% | 90.29% |
| Replacement orchestrator | 174 | 9 | 21 | 94.01% | 92.96% | 92.06% | 90.34% |

The replacement improves S21 by 2.14pp macro F2 and 2.20pp pooled F2. It also
improves macro and pooled F1. Relative to the prior dynamic augmentation it gains
one TP and leads by 0.16pp macro F2 and 0.05pp pooled F2, while trading four
additional false positives and lower F1.

All five projects selected entity → coreference → coverage audit. Thus the
replacement and F2 hypotheses passed, but route diversity was not observed.
Controller assessments, evidence, and tool outputs remained project-specific.
