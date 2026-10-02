# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **terra NoValidator vs Full:** file-F1 +0.003803, file-F2 +0.098976, worst-component F1 +0.028345, harmonic-component F1 +0.084836.
- **terra NoNameValid (judge off) vs Full:** file-F1 -0.033950, file-F2 +0.007659.
- **terra NoCitation (judge off) vs Full:** file-F1 +0.027292, file-F2 +0.096984.

## RQ4 Linker Sets

- **terra NameOnly vs Full:** file-F1 -0.050818, file-F2 -0.054718, worst-component F1 +0.005128.
- **terra CorefOnly vs Full:** file-F1 -0.618333, file-F2 -0.629887, worst-component F1 -0.419551.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
