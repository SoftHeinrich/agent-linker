# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **luna NoValidator vs Full:** file-F1 -0.154999, file-F2 -0.058453, worst-component F1 -0.218511, harmonic-component F1 -0.172229.
- **luna NoNameValid (judge off) vs Full:** file-F1 -0.044395, file-F2 -0.003109.
- **luna NoCitation (judge off) vs Full:** file-F1 -0.126107, file-F2 -0.051043.
- **terra NoValidator vs Full:** file-F1 -0.079508, file-F2 -0.001940, worst-component F1 -0.209384, harmonic-component F1 -0.109234.
- **terra NoNameValid (judge off) vs Full:** file-F1 -0.043480, file-F2 +0.015259.
- **terra NoCitation (judge off) vs Full:** file-F1 -0.043740, file-F2 -0.012289.

## RQ4 Linker Sets

- **luna NameOnly vs Full:** file-F1 -0.043017, file-F2 -0.065213, worst-component F1 -0.064200.
- **luna CorefOnly vs Full:** file-F1 -0.555526, file-F2 -0.653182, worst-component F1 -0.670686.
- **terra NameOnly vs Full:** file-F1 -0.047004, file-F2 -0.070501, worst-component F1 -0.091303.
- **terra CorefOnly vs Full:** file-F1 -0.614326, file-F2 -0.692941, worst-component F1 -0.730333.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
