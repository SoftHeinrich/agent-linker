# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **luna NoValidator vs Full:** file-F1 -0.087108, file-F2 +0.031738, worst-component F1 -0.069234, harmonic-component F1 +0.026300.
- **luna NoNameValid (judge off) vs Full:** file-F1 -0.043692, file-F2 +0.006389.
- **luna NoCitation (judge off) vs Full:** file-F1 -0.060764, file-F2 +0.031822.

## RQ4 Linker Sets

- **luna NameOnly vs Full:** file-F1 -0.046182, file-F2 -0.048488, worst-component F1 +0.000000.
- **luna CorefOnly vs Full:** file-F1 -0.588851, file-F2 -0.598395, worst-component F1 -0.407407.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
