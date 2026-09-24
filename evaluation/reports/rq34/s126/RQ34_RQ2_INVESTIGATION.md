# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **luna NoValidator vs Full:** file-F1 -0.145613, file-F2 -0.041666, worst-component F1 -0.184810, harmonic-component F1 -0.162490.
- **luna NoNameValid (judge off) vs Full:** file-F1 -0.035010, file-F2 +0.013678.
- **luna NoCitation (judge off) vs Full:** file-F1 -0.128216, file-F2 -0.050597.
- **terra NoValidator vs Full:** file-F1 -0.073062, file-F2 -0.009537, worst-component F1 -0.230051, harmonic-component F1 -0.130200.
- **terra NoNameValid (judge off) vs Full:** file-F1 -0.037034, file-F2 +0.007661.
- **terra NoCitation (judge off) vs Full:** file-F1 -0.046774, file-F2 -0.016435.

## RQ4 Linker Sets

- **luna NameOnly vs Full:** file-F1 -0.045053, file-F2 -0.067259, worst-component F1 -0.066275.
- **luna CorefOnly vs Full:** file-F1 -0.546140, file-F2 -0.636395, worst-component F1 -0.636984.
- **terra NameOnly vs Full:** file-F1 -0.046399, file-F2 -0.069762, worst-component F1 -0.075457.
- **terra CorefOnly vs Full:** file-F1 -0.607879, file-F2 -0.700539, worst-component F1 -0.751001.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
