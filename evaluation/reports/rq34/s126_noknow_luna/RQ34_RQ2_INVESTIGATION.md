# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **luna NoValidator vs Full:** file-F1 -0.092887, file-F2 +0.031690, worst-component F1 -0.034092, harmonic-component F1 +0.123456.
- **luna NoNameValid (judge off) vs Full:** file-F1 -0.039065, file-F2 +0.014598.
- **luna NoCitation (judge off) vs Full:** file-F1 -0.064388, file-F2 +0.034096.

## RQ4 Linker Sets

- **luna NameOnly vs Full:** file-F1 -0.055516, file-F2 -0.059917, worst-component F1 +0.000000.
- **luna CorefOnly vs Full:** file-F1 -0.545872, file-F2 -0.559702, worst-component F1 -0.411740.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
