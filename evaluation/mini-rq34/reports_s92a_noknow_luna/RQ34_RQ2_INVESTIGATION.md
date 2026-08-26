# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **luna NoValidator vs Full:** file-F1 -0.155242, file-F2 -0.053857, worst-component F1 -0.059299, harmonic-component F1 +0.016007, coverage +0.054670, noise +0.275380.
- **luna NoFullNameValid (judge off) vs Full:** file-F1 -0.005718, file-F2 +0.009833, noise +0.019943.
- **luna NoPartialNameValid (judge off) vs Full:** file-F1 -0.032928, file-F2 -0.015211, noise +0.072187.
- **luna NoCitation (judge off) vs Full:** file-F1 -0.135332, file-F2 -0.048995, noise +0.244995.

## RQ4 Linker Sets

- **luna FullNameOnly vs Full:** file-F1 -0.093344, file-F2 -0.125077, worst-component F1 -0.147619, coverage -0.170934.
- **luna PartialNameOnly vs Full:** file-F1 -0.755896, file-F2 -0.750681, worst-component F1 -0.451705, coverage -0.758306.
- **luna CorefOnly vs Full:** file-F1 -0.417782, file-F2 -0.493014, worst-component F1 -0.397737, coverage -0.517961.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full; positive noise deltas mean more false-positive mass per predicted sentence.
