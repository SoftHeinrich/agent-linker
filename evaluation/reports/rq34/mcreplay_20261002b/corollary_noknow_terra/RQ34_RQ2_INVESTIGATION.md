# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **terra NoValidator vs Full:** file-F1 +0.004211, file-F2 +0.102514, worst-component F1 +0.010624, harmonic-component F1 +0.083421.
- **terra NoNameValid (judge off) vs Full:** file-F1 -0.033543, file-F2 +0.011197.
- **terra NoCitation (judge off) vs Full:** file-F1 +0.026575, file-F2 +0.097376.

## RQ4 Linker Sets

- **terra NameOnly vs Full:** file-F1 -0.051097, file-F2 -0.054847, worst-component F1 -0.000441.
- **terra CorefOnly vs Full:** file-F1 -0.617926, file-F2 -0.626349, worst-component F1 -0.437272.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
