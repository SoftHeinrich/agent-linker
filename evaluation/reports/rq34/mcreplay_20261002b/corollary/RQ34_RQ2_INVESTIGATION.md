# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **luna NoValidator vs Full:** file-F1 -0.157464, file-F2 -0.054456, worst-component F1 -0.234316, harmonic-component F1 -0.177756.
- **luna NoNameValid (judge off) vs Full:** file-F1 -0.046861, file-F2 +0.000889.
- **luna NoCitation (judge off) vs Full:** file-F1 -0.128945, file-F2 -0.052402.
- **terra NoValidator vs Full:** file-F1 -0.075683, file-F2 -0.005835, worst-component F1 -0.245004, harmonic-component F1 -0.128579.
- **terra NoNameValid (judge off) vs Full:** file-F1 -0.039655, file-F2 +0.011363.
- **terra NoCitation (judge off) vs Full:** file-F1 -0.042013, file-F2 -0.012768.

## RQ4 Linker Sets

- **luna NameOnly vs Full:** file-F1 -0.043686, file-F2 -0.065743, worst-component F1 -0.065917.
- **luna CorefOnly vs Full:** file-F1 -0.557992, file-F2 -0.649184, worst-component F1 -0.686490.
- **terra NameOnly vs Full:** file-F1 -0.046980, file-F2 -0.070423, worst-component F1 -0.081982.
- **terra CorefOnly vs Full:** file-F1 -0.610500, file-F2 -0.696836, worst-component F1 -0.765954.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
