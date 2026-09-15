# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **luna NoValidator vs Full:** file-F1 -0.139229, file-F2 -0.039699, worst-component F1 -0.154303, harmonic-component F1 -0.120449.
- **luna NoNameValid (judge off) vs Full:** file-F1 -0.043965, file-F2 -0.000957.
- **luna NoCitation (judge off) vs Full:** file-F1 -0.112272, file-F2 -0.036791.

## RQ4 Linker Sets

- **luna FullNameOnly vs Full:** file-F1 -0.055243, file-F2 -0.081645, worst-component F1 -0.098815.
- **luna PartialNameOnly vs Full:** file-F1 -0.746834, file-F2 -0.721202, worst-component F1 -0.452089.
- **luna CorefOnly vs Full:** file-F1 -0.515396, file-F2 -0.556466, worst-component F1 -0.452089.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
