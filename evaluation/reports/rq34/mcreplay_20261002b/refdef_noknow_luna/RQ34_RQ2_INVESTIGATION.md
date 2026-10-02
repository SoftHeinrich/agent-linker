# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **luna NoValidator vs Full:** file-F1 -0.072756, file-F2 +0.042565, worst-component F1 -0.089881, harmonic-component F1 +0.019943.
- **luna NoNameValid (judge off) vs Full:** file-F1 -0.029340, file-F2 +0.017216.
- **luna NoCitation (judge off) vs Full:** file-F1 -0.057801, file-F2 +0.033990.

## RQ4 Linker Sets

- **luna NameOnly vs Full:** file-F1 -0.048112, file-F2 -0.051200, worst-component F1 -0.000441.
- **luna CorefOnly vs Full:** file-F1 -0.574498, file-F2 -0.587569, worst-component F1 -0.428055.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
