# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **luna NoValidator vs Full:** file-F1 -0.145613, file-F2 -0.041666, worst-component F1 -0.184810, harmonic-component F1 -0.162490.
- **luna NoNameValid (judge off) vs Full:** file-F1 -0.035010, file-F2 +0.013678.
- **luna NoCitation (judge off) vs Full:** file-F1 -0.128216, file-F2 -0.050597.
- **terra NoValidator vs Full:** file-F1 -0.078211, file-F2 -0.011546, worst-component F1 -0.221141, harmonic-component F1 -0.127912.
- **terra NoNameValid (judge off) vs Full:** file-F1 -0.041719, file-F2 +0.005837.
- **terra NoCitation (judge off) vs Full:** file-F1 -0.047547, file-F2 -0.016668.

## RQ4 Linker Sets

- **luna FullNameOnly vs Full:** file-F1 -0.042176, file-F2 -0.072424, worst-component F1 -0.004410.
- **luna PartialNameOnly vs Full:** file-F1 -0.849199, file-F2 -0.868738, worst-component F1 -0.681067.
- **luna CorefOnly vs Full:** file-F1 -0.546140, file-F2 -0.636395, worst-component F1 -0.636984.
- **terra FullNameOnly vs Full:** file-F1 -0.035286, file-F2 -0.076347, worst-component F1 -0.080465.
- **terra PartialNameOnly vs Full:** file-F1 -0.848050, file-F2 -0.869064, worst-component F1 -0.773246.
- **terra CorefOnly vs Full:** file-F1 -0.611283, file-F2 -0.700499, worst-component F1 -0.759868.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
