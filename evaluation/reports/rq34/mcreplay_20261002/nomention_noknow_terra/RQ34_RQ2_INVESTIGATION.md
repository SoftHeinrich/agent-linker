# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **terra NoValidator vs Full:** file-F1 +0.016340, file-F2 +0.114951, worst-component F1 +0.046921, harmonic-component F1 +0.098696.
- **terra NoNameValid (judge off) vs Full:** file-F1 -0.021414, file-F2 +0.023634.
- **terra NoCitation (judge off) vs Full:** file-F1 +0.034165, file-F2 +0.107031.

## RQ4 Linker Sets

- **terra NameOnly vs Full:** file-F1 -0.051155, file-F2 -0.054866, worst-component F1 +0.000000.
- **terra CorefOnly vs Full:** file-F1 -0.605796, file-F2 -0.613912, worst-component F1 -0.400976.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
