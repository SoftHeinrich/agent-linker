# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **terra NoValidator vs Full:** file-F1 -0.037611, file-F2 +0.060343, worst-component F1 +0.068433, harmonic-component F1 -0.006425, coverage +0.095111, noise +0.218091.
- **terra NoFullNameValid (judge off) vs Full:** file-F1 -0.007138, file-F2 +0.025111, noise +0.050710.
- **terra NoPartialNameValid (judge off) vs Full:** file-F1 -0.035444, file-F2 -0.012337, noise +0.091933.
- **terra NoCitation (judge off) vs Full:** file-F1 -0.009864, file-F2 +0.056660, noise +0.155483.

## RQ4 Linker Sets

- **terra FullNameOnly vs Full:** file-F1 -0.085365, file-F2 -0.104073, worst-component F1 -0.080000, coverage -0.149169.
- **terra PartialNameOnly vs Full:** file-F1 -0.763254, file-F2 -0.728947, worst-component F1 -0.373333, coverage -0.732048.
- **terra CorefOnly vs Full:** file-F1 -0.517307, file-F2 -0.551893, worst-component F1 -0.368205, coverage -0.618419.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full; positive noise deltas mean more false-positive mass per predicted sentence.
