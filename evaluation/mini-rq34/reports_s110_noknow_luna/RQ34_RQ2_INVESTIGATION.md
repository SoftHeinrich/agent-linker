# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **luna NoValidator vs Full:** file-F1 -0.126827, file-F2 -0.036632, worst-component F1 -0.206847, harmonic-component F1 -0.139936.
- **luna NoFullNameValid (judge off) vs Full:** file-F1 -0.010971, file-F2 +0.007197.
- **luna NoPartialNameValid (judge off) vs Full:** file-F1 -0.036667, file-F2 -0.017883.
- **luna NoCitation (judge off) vs Full:** file-F1 -0.097179, file-F2 -0.026887.

## RQ4 Linker Sets

- **luna FullNameOnly vs Full:** file-F1 -0.061797, file-F2 -0.085767, worst-component F1 -0.137762.
- **luna PartialNameOnly vs Full:** file-F1 -0.738484, file-F2 -0.721199, worst-component F1 -0.486031.
- **luna CorefOnly vs Full:** file-F1 -0.554769, file-F2 -0.594056, worst-component F1 -0.452121.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
