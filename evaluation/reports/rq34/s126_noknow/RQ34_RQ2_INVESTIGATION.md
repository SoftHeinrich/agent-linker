# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **terra NoValidator vs Full:** file-F1 +0.011532, file-F2 +0.114004, worst-component F1 +0.006286, harmonic-component F1 +0.084336.
- **terra NoNameValid (judge off) vs Full:** file-F1 -0.026221, file-F2 +0.022686.
- **terra NoCitation (judge off) vs Full:** file-F1 +0.029854, file-F2 +0.102015.

## RQ4 Linker Sets

- **terra NameOnly vs Full:** file-F1 -0.051769, file-F2 -0.055460, worst-component F1 -0.000661.
- **terra CorefOnly vs Full:** file-F1 -0.610604, file-F2 -0.614860, worst-component F1 -0.441611.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
