# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **luna NoValidator vs Full:** file-F1 -0.076249, file-F2 +0.041522, worst-component F1 -0.066818, harmonic-component F1 +0.032169.
- **luna NoNameValid (judge off) vs Full:** file-F1 -0.032833, file-F2 +0.016173.
- **luna NoCitation (judge off) vs Full:** file-F1 -0.055717, file-F2 +0.036001.

## RQ4 Linker Sets

- **luna NameOnly vs Full:** file-F1 -0.044621, file-F2 -0.046400, worst-component F1 +0.000000.
- **luna CorefOnly vs Full:** file-F1 -0.577992, file-F2 -0.588612, worst-component F1 -0.404991.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
