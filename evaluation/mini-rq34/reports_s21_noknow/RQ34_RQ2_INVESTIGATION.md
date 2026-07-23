# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **openai NoValidator vs Full:** file-F1 -0.001583, worst-component F1 +0.097124, harmonic-component F1 +0.157786, coverage +0.059897, noise +0.151438.
- **openai validator split:** NoEntityValid file-F1 +0.025658 vs NoCitation file-F1 -0.028143; NoEntityValid noise +0.080913 vs NoCitation noise +0.087180.

## RQ4 Linker Sets

- **openai EntityOnly vs Full:** file-F1 -0.038485, worst-component F1 -0.053968, coverage -0.083842.
- **openai CorefOnly vs Full:** file-F1 -0.650029, worst-component F1 -0.533333, coverage -0.638167.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full; positive noise deltas mean more false-positive mass per predicted sentence.
