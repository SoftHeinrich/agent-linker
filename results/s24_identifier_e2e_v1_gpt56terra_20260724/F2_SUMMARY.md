# S24 exact-identifier fresh paired E2E

All predictions were obtained from fresh OpenAI API calls using
`gpt-5.6-terra` with reasoning effort `none`. Gold was consulted only after
inference for evaluation.

| Variant | TP / FP / FN | Macro F1 | Pooled F1 | Macro F2 | Pooled F2 |
| --- | --- | ---: | ---: | ---: | ---: |
| S21 | 176 / 24 / 19 | 91.44% | 89.11% | 92.31% | 89.80% |
| S24 | 178 / 7 / 17 | 95.36% | 93.68% | 94.12% | 92.23% |
| Delta | +2 / -17 / -2 | +3.92 pp | +4.57 pp | +1.81 pp | +2.43 pp |

The production exact-identifier capability ran only on BigBlueButton. It
accepted two `bbb-web` occurrences as the runtime catalog component `BBB web`;
both are gold links and neither introduced a false positive.

Controller paths:

- MediaStore: entity -> coreference -> finalize
- TeaStore: entity -> coreference -> finalize
- TEAMMATES: entity -> role -> coreference -> finalize
- BigBlueButton: entity -> coreference -> role -> identifier -> finalize
- JabRef: entity -> coreference -> finalize

Both macro and pooled F1 non-regression gates pass, and both macro and pooled
F2 improve over the fresh paired S21 run.
