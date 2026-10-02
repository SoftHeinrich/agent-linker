# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **luna NoValidator vs Full:** file-F1 -0.154333, file-F2 -0.048757, worst-component F1 -0.234930, harmonic-component F1 -0.174160.
- **luna NoNameValid (judge off) vs Full:** file-F1 -0.043730, file-F2 +0.006587.
- **luna NoCitation (judge off) vs Full:** file-F1 -0.124689, file-F2 -0.046376.
- **terra NoValidator vs Full:** file-F1 -0.060553, file-F2 +0.019087, worst-component F1 -0.179608, harmonic-component F1 -0.043854.
- **terra NoNameValid (judge off) vs Full:** file-F1 -0.024526, file-F2 +0.036286.
- **terra NoCitation (judge off) vs Full:** file-F1 -0.038892, file-F2 -0.008186.

## RQ4 Linker Sets

- **luna NameOnly vs Full:** file-F1 -0.043946, file-F2 -0.066200, worst-component F1 -0.067567.
- **luna CorefOnly vs Full:** file-F1 -0.554860, file-F2 -0.643486, worst-component F1 -0.687105.
- **terra NameOnly vs Full:** file-F1 -0.047546, file-F2 -0.070761, worst-component F1 -0.084609.
- **terra CorefOnly vs Full:** file-F1 -0.595371, file-F2 -0.671914, worst-component F1 -0.700557.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
