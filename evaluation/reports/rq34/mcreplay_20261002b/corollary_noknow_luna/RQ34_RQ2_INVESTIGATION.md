# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **luna NoValidator vs Full:** file-F1 -0.084039, file-F2 +0.037217, worst-component F1 -0.105932, harmonic-component F1 +0.018633.
- **luna NoNameValid (judge off) vs Full:** file-F1 -0.040623, file-F2 +0.011868.
- **luna NoCitation (judge off) vs Full:** file-F1 -0.060276, file-F2 +0.031480.

## RQ4 Linker Sets

- **luna NameOnly vs Full:** file-F1 -0.047473, file-F2 -0.049503, worst-component F1 -0.003294.
- **luna CorefOnly vs Full:** file-F1 -0.585781, file-F2 -0.592917, worst-component F1 -0.444106.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
