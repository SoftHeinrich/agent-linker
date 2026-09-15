# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **luna NoValidator vs Full:** file-F1 -0.162554, file-F2 -0.072911, worst-component F1 -0.202037, harmonic-component F1 -0.165015.
- **luna NoNameValid (judge off) vs Full:** file-F1 -0.058782, file-F2 -0.019988.
- **luna NoCitation (judge off) vs Full:** file-F1 -0.124762, file-F2 -0.054897.
- **terra NoValidator vs Full:** file-F1 -0.112705, file-F2 -0.030088, worst-component F1 -0.257689, harmonic-component F1 -0.158759.
- **terra NoNameValid (judge off) vs Full:** file-F1 -0.066723, file-F2 -0.011123.
- **terra NoCitation (judge off) vs Full:** file-F1 -0.061690, file-F2 -0.020078.

## RQ4 Linker Sets

- **luna FullNameOnly vs Full:** file-F1 -0.041005, file-F2 -0.075982, worst-component F1 -0.008872.
- **luna PartialNameOnly vs Full:** file-F1 -0.858167, file-F2 -0.889248, worst-component F1 -0.726916.
- **luna CorefOnly vs Full:** file-F1 -0.595251, file-F2 -0.695061, worst-component F1 -0.726916.
- **terra FullNameOnly vs Full:** file-F1 -0.048865, file-F2 -0.081979, worst-component F1 -0.087911.
- **terra PartialNameOnly vs Full:** file-F1 -0.871393, file-F2 -0.879456, worst-component F1 -0.821108.
- **terra CorefOnly vs Full:** file-F1 -0.607480, file-F2 -0.688147, worst-component F1 -0.821108.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
