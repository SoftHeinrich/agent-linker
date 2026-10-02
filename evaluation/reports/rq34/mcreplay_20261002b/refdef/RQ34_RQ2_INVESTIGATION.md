# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **luna NoValidator vs Full:** file-F1 -0.151379, file-F2 -0.050369, worst-component F1 -0.202566, harmonic-component F1 -0.172967.
- **luna NoNameValid (judge off) vs Full:** file-F1 -0.040775, file-F2 +0.004975.
- **luna NoCitation (judge off) vs Full:** file-F1 -0.129579, file-F2 -0.051506.
- **terra NoValidator vs Full:** file-F1 -0.048670, file-F2 +0.015700, worst-component F1 -0.194489, harmonic-component F1 -0.103990.
- **terra NoNameValid (judge off) vs Full:** file-F1 -0.012643, file-F2 +0.032898.
- **terra NoCitation (judge off) vs Full:** file-F1 -0.037113, file-F2 -0.008130.

## RQ4 Linker Sets

- **luna NameOnly vs Full:** file-F1 -0.043736, file-F2 -0.065677, worst-component F1 -0.068732.
- **luna CorefOnly vs Full:** file-F1 -0.551906, file-F2 -0.645098, worst-component F1 -0.654741.
- **terra NameOnly vs Full:** file-F1 -0.055576, file-F2 -0.080133, worst-component F1 -0.087084.
- **terra CorefOnly vs Full:** file-F1 -0.583488, file-F2 -0.675301, worst-component F1 -0.715438.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
