# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **luna NoValidator vs Full:** file-F1 -0.158062, file-F2 -0.071803, worst-component F1 -0.299826, harmonic-component F1 -0.209891.
- **luna NoFullNameValid (judge off) vs Full:** file-F1 -0.018855, file-F2 -0.004035.
- **luna NoPartialNameValid (judge off) vs Full:** file-F1 -0.031942, file-F2 -0.013546.
- **luna NoCitation (judge off) vs Full:** file-F1 -0.130747, file-F2 -0.060029.
- **terra NoValidator vs Full:** file-F1 -0.109635, file-F2 -0.027994, worst-component F1 -0.262580, harmonic-component F1 -0.160371.
- **terra NoFullNameValid (judge off) vs Full:** file-F1 -0.028402, file-F2 +0.007076.
- **terra NoPartialNameValid (judge off) vs Full:** file-F1 -0.037397, file-F2 -0.016978.
- **terra NoCitation (judge off) vs Full:** file-F1 -0.061240, file-F2 -0.019219.

## RQ4 Linker Sets

- **luna FullNameOnly vs Full:** file-F1 -0.045279, file-F2 -0.074050, worst-component F1 -0.100430.
- **luna PartialNameOnly vs Full:** file-F1 -0.860693, file-F2 -0.898723, worst-component F1 -0.785496.
- **luna CorefOnly vs Full:** file-F1 -0.562100, file-F2 -0.676352, worst-component F1 -0.738493.
- **terra FullNameOnly vs Full:** file-F1 -0.050590, file-F2 -0.082511, worst-component F1 -0.107704.
- **terra PartialNameOnly vs Full:** file-F1 -0.863330, file-F2 -0.877882, worst-component F1 -0.805862.
- **terra CorefOnly vs Full:** file-F1 -0.633310, file-F2 -0.714977, worst-component F1 -0.782486.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
