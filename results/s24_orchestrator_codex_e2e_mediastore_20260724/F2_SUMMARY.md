# F2 summary

This end-to-end smoke was generated before `run_ablation.py` began writing F2
into its JSON result object. The score below is computed from the preserved
confusion counts using:

```text
F2 = 5 TP / (5 TP + 4 FN + FP)
```

| Project | TP | FP | FN | F1 | F2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| mediastore | 31 | 1 | 0 | 98.41% | 99.36% |

This is a fresh execution from raw document and model inputs. It is a
single-project smoke test, not the all-project fixed-phase replay.
