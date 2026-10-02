# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **terra NoValidator vs Full:** file-F1 +0.008028, file-F2 +0.100994, worst-component F1 +0.048402, harmonic-component F1 +0.091170.
- **terra NoNameValid (judge off) vs Full:** file-F1 -0.029726, file-F2 +0.009677.
- **terra NoCitation (judge off) vs Full:** file-F1 +0.027421, file-F2 +0.096939.

## RQ4 Linker Sets

- **terra NameOnly vs Full:** file-F1 -0.050948, file-F2 -0.054748, worst-component F1 -0.000441.
- **terra CorefOnly vs Full:** file-F1 -0.614108, file-F2 -0.627869, worst-component F1 -0.399494.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
