# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **luna NoValidator vs Full:** file-F1 -0.078312, file-F2 +0.045532, worst-component F1 -0.102541, harmonic-component F1 +0.018797.
- **luna NoNameValid (judge off) vs Full:** file-F1 -0.034895, file-F2 +0.020183.
- **luna NoCitation (judge off) vs Full:** file-F1 -0.061178, file-F2 +0.033280.

## RQ4 Linker Sets

- **luna NameOnly vs Full:** file-F1 -0.049699, file-F2 -0.053022, worst-component F1 -0.003288.
- **luna CorefOnly vs Full:** file-F1 -0.580054, file-F2 -0.584602, worst-component F1 -0.440715.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
