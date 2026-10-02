# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **luna NoValidator vs Full:** file-F1 -0.161618, file-F2 -0.056233, worst-component F1 -0.260676, harmonic-component F1 -0.185648.
- **luna NoNameValid (judge off) vs Full:** file-F1 -0.051014, file-F2 -0.000889.
- **luna NoCitation (judge off) vs Full:** file-F1 -0.131405, file-F2 -0.052994.
- **terra NoValidator vs Full:** file-F1 -0.051269, file-F2 +0.015720, worst-component F1 -0.188207, harmonic-component F1 -0.100345.
- **terra NoNameValid (judge off) vs Full:** file-F1 -0.015242, file-F2 +0.032918.
- **terra NoCitation (judge off) vs Full:** file-F1 -0.036621, file-F2 -0.007217.

## RQ4 Linker Sets

- **luna NameOnly vs Full:** file-F1 -0.043020, file-F2 -0.065253, worst-component F1 -0.063336.
- **luna CorefOnly vs Full:** file-F1 -0.562145, file-F2 -0.650962, worst-component F1 -0.712851.
- **terra NameOnly vs Full:** file-F1 -0.049616, file-F2 -0.073038, worst-component F1 -0.084781.
- **terra CorefOnly vs Full:** file-F1 -0.586087, file-F2 -0.675281, worst-component F1 -0.709156.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
