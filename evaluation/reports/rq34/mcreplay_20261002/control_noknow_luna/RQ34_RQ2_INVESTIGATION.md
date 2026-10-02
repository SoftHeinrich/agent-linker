# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **luna NoValidator vs Full:** file-F1 -0.075962, file-F2 +0.040539, worst-component F1 -0.062852, harmonic-component F1 +0.034950.
- **luna NoNameValid (judge off) vs Full:** file-F1 -0.032546, file-F2 +0.015190.
- **luna NoCitation (judge off) vs Full:** file-F1 -0.058646, file-F2 +0.032663.

## RQ4 Linker Sets

- **luna NameOnly vs Full:** file-F1 -0.047869, file-F2 -0.051403, worst-component F1 +0.000000.
- **luna CorefOnly vs Full:** file-F1 -0.577704, file-F2 -0.589594, worst-component F1 -0.401026.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full.
