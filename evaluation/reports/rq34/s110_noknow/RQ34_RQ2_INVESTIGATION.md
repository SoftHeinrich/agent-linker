# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **terra NoValidator vs Full:** file-F1 -0.075971, file-F2 +0.005654, worst-component F1 -0.162458, harmonic-component F1 -0.102149.
- **terra NoFullNameValid (judge off) vs Full:** file-F1 -0.004392, file-F2 +0.028459.
- **terra NoPartialNameValid (judge off) vs Full:** file-F1 -0.034220, file-F2 -0.010661.
- **terra NoCitation (judge off) vs Full:** file-F1 -0.052421, file-F2 -0.004542.

## RQ4 Linker Sets

- **terra FullNameOnly vs Full:** file-F1 -0.066717, file-F2 -0.082542, worst-component F1 -0.038216.
- **terra PartialNameOnly vs Full:** file-F1 -0.747885, file-F2 -0.708824, worst-component F1 -0.427811.
- **terra CorefOnly vs Full:** file-F1 -0.575759, file-F2 -0.592317, worst-component F1 -0.416124.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
