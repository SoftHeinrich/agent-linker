# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **claude NoValidator vs Full:** file-F1 -0.006847, worst-component F1 -0.046354, harmonic-component F1 -0.013229, coverage +0.030196, noise +0.075367.
- **claude validator split:** NoEntityValid file-F1 -0.008633 vs NoCitation file-F1 +0.002799; NoEntityValid noise +0.041241 vs NoCitation noise +0.052087.
- **openai NoValidator vs Full:** file-F1 -0.069333, worst-component F1 -0.114886, harmonic-component F1 -0.062134, coverage +0.030960, noise +0.162420.
- **openai validator split:** NoEntityValid file-F1 -0.045538 vs NoCitation file-F1 -0.029739; NoEntityValid noise +0.105300 vs NoCitation noise +0.088245.

## RQ4 Linker Sets

- **claude EntityOnly vs Full:** file-F1 -0.059890, worst-component F1 -0.047138, coverage -0.118718.
- **claude CorefOnly vs Full:** file-F1 -0.312612, worst-component F1 -0.294184, coverage -0.353903.
- **openai EntityOnly vs Full:** file-F1 -0.046132, worst-component F1 -0.058599, coverage -0.103501.
- **openai CorefOnly vs Full:** file-F1 -0.668630, worst-component F1 -0.753043, coverage -0.696283.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full; positive noise deltas mean more false-positive mass per predicted sentence.
