# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **terra NoValidator vs Full:** file-F1 -0.071799, file-F2 -0.006572, worst-component F1 -0.204617, harmonic-component F1 -0.119009.
- **terra NoNameValid (judge off) vs Full:** file-F1 -0.054707, file-F2 -0.005622.
- **terra NoCitation (judge off) vs Full:** file-F1 -0.034183, file-F2 -0.002130.

## RQ4 Linker Sets

- **terra FullNameOnly vs Full:** file-F1 -0.061772, file-F2 -0.083739, worst-component F1 -0.083573.
- **terra PartialNameOnly vs Full:** file-F1 -0.745406, file-F2 -0.714254, worst-component F1 -0.473389.
- **terra CorefOnly vs Full:** file-F1 -0.527187, file-F2 -0.558426, worst-component F1 -0.473389.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
