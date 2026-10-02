# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **terra NoValidator vs Full:** file-F1 -0.002108, file-F2 +0.096444, worst-component F1 +0.020674, harmonic-component F1 +0.125696.
- **terra NoNameValid (judge off) vs Full:** file-F1 -0.038376, file-F2 +0.009278.
- **terra NoCitation (judge off) vs Full:** file-F1 +0.020714, file-F2 +0.090648.

## RQ4 Linker Sets

- **terra NameOnly vs Full:** file-F1 -0.051685, file-F2 -0.056150, worst-component F1 +0.000000.
- **terra CorefOnly vs Full:** file-F1 -0.608826, file-F2 -0.619087, worst-component F1 -0.448383.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
