# Completed but rejected E2E design

This fresh five-project run completed but did not beat S21.

- S21: macro F2 92.9%, pooled F2 89.9%, 14 FP.
- S24: macro F2 90.2%, pooled F2 89.6%, 86 FP.

The controller selected `coverage_audit` on every project. That overlapping
capability contributed 56 false positives, erasing the recall benefit of the
catalog-handle route. The final design removes coverage audit from this
variant's capability registry; it does not apply a score or count gate.
