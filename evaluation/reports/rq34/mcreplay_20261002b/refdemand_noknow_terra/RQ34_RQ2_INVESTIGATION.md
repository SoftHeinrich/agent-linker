# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **terra NoValidator vs Full:** file-F1 +0.016305, file-F2 +0.111893, worst-component F1 +0.040254, harmonic-component F1 +0.094094.
- **terra NoNameValid (judge off) vs Full:** file-F1 -0.021449, file-F2 +0.020575.
- **terra NoCitation (judge off) vs Full:** file-F1 +0.029312, file-F2 +0.099190.

## RQ4 Linker Sets

- **terra NameOnly vs Full:** file-F1 -0.051170, file-F2 -0.054865, worst-component F1 -0.000220.
- **terra CorefOnly vs Full:** file-F1 -0.605831, file-F2 -0.616971, worst-component F1 -0.407643.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
