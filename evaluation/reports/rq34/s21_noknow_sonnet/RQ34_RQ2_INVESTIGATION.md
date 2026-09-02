# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **claude NoValidator vs Full:** file-F1 +0.008172, worst-component F1 +0.105697, harmonic-component F1 +0.118263, coverage +0.058052, noise +0.099522.
- **claude validator split:** NoEntityValid file-F1 +0.014890 vs NoCitation file-F1 -0.004887; NoEntityValid noise +0.068780 vs NoCitation noise +0.054297.

## RQ4 Linker Sets

- **claude EntityOnly vs Full:** file-F1 -0.037228, worst-component F1 -0.050000, coverage -0.077980.
- **claude CorefOnly vs Full:** file-F1 -0.468574, worst-component F1 -0.473990, coverage -0.480583.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full; positive noise deltas mean more false-positive mass per predicted sentence.
