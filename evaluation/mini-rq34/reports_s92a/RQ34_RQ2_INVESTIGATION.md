# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **luna NoValidator vs Full:** file-F1 -0.155019, file-F2 -0.069151, worst-component F1 -0.225604, harmonic-component F1 -0.159512, coverage +0.032770, noise +0.255855.
- **luna NoFullNameValid (judge off) vs Full:** file-F1 -0.021444, file-F2 -0.003514, noise +0.038757.
- **luna NoPartialNameValid (judge off) vs Full:** file-F1 -0.025859, file-F2 -0.010628, noise +0.057094.
- **luna NoCitation (judge off) vs Full:** file-F1 -0.127052, file-F2 -0.056283, noise +0.217051.
- **terra NoValidator vs Full:** file-F1 -0.128307, file-F2 -0.039922, worst-component F1 -0.099624, harmonic-component F1 -0.092957, coverage +0.062576, noise +0.231975.
- **terra NoFullNameValid (judge off) vs Full:** file-F1 -0.019806, file-F2 +0.011227, noise +0.050819.
- **terra NoPartialNameValid (judge off) vs Full:** file-F1 -0.035969, file-F2 -0.014284, noise +0.083037.
- **terra NoCitation (judge off) vs Full:** file-F1 -0.095654, file-F2 -0.034710, noise +0.173792.

## RQ4 Linker Sets

- **luna FullNameOnly vs Full:** file-F1 -0.036767, file-F2 -0.072944, worst-component F1 -0.051918, coverage -0.121382.
- **luna PartialNameOnly vs Full:** file-F1 -0.833852, file-F2 -0.875356, worst-component F1 -0.705630, coverage -0.836963.
- **luna CorefOnly vs Full:** file-F1 -0.489838, file-F2 -0.607792, worst-component F1 -0.700501, coverage -0.596148.
- **terra FullNameOnly vs Full:** file-F1 -0.052279, file-F2 -0.087015, worst-component F1 -0.081373, coverage -0.137978.
- **terra PartialNameOnly vs Full:** file-F1 -0.870713, file-F2 -0.884366, worst-component F1 -0.705658, coverage -0.830532.
- **terra CorefOnly vs Full:** file-F1 -0.647755, file-F2 -0.723911, worst-component F1 -0.700896, coverage -0.709414.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full; positive noise deltas mean more false-positive mass per predicted sentence.
