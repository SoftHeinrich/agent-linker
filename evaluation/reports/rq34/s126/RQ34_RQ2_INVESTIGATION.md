# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **luna NoValidator vs Full:** file-F1 -0.131081, file-F2 -0.043325, worst-component F1 -0.232000, harmonic-component F1 -0.169309.
- **luna NoNameValid (judge off) vs Full:** file-F1 -0.043674, file-F2 +0.000920.
- **luna NoCitation (judge off) vs Full:** file-F1 -0.105323, file-F2 -0.039783.
- **terra NoValidator vs Full:** file-F1 -0.095270, file-F2 -0.017345, worst-component F1 -0.240299, harmonic-component F1 -0.136768.
- **terra NoNameValid (judge off) vs Full:** file-F1 -0.048089, file-F2 +0.005049.
- **terra NoCitation (judge off) vs Full:** file-F1 -0.058252, file-F2 -0.019287.

## RQ4 Linker Sets

- **luna NameOnly vs Full:** file-F1 -0.049506, file-F2 -0.070771, worst-component F1 -0.099913.
- **luna CorefOnly vs Full:** file-F1 -0.527621, file-F2 -0.624305, worst-component F1 -0.718110.
- **terra NameOnly vs Full:** file-F1 -0.045590, file-F2 -0.069436, worst-component F1 -0.081794.
- **terra CorefOnly vs Full:** file-F1 -0.627338, file-F2 -0.707206, worst-component F1 -0.781293.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
