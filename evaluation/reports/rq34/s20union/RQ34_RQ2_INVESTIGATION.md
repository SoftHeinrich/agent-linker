# RQ3/RQ4 Through the RQ2 Doc-to-Code Lens

Method: SAD-SAM phase-cache link sets are composed through recovered SAM-CODE links, then scored with the RQ2 doc-to-code panel. Rows below use the run-average values.

## RQ3 Validator Counterfactuals

- **claude NoValidator vs Full:** file-F1 -0.014617, worst-component F1 -0.077049, harmonic-component F1 +0.024938, coverage +0.027788, noise +0.099834.
- **claude validator split:** NoEntityValid file-F1 -0.014799 vs NoCitation file-F1 +0.005560; NoEntityValid noise +0.087694 vs NoCitation noise +0.018671.
- **openai NoValidator vs Full:** file-F1 -0.038913, worst-component F1 +0.010818, harmonic-component F1 +0.036401, coverage +0.052238, noise +0.137721.
- **openai validator split:** NoEntityValid file-F1 -0.012698 vs NoCitation file-F1 -0.031399; NoEntityValid noise +0.084015 vs NoCitation noise +0.072127.

## RQ4 Linker Sets

- **claude EntityOnly vs Full:** file-F1 -0.049511, worst-component F1 -0.125336, coverage -0.095498.
- **claude CorefOnly vs Full:** file-F1 -0.415297, worst-component F1 -0.626154, coverage -0.483482.
- **openai EntityOnly vs Full:** file-F1 -0.045832, worst-component F1 -0.120137, coverage -0.106931.
- **openai CorefOnly vs Full:** file-F1 -0.443616, worst-component F1 -0.439777, coverage -0.495027.

Reading rule: negative deltas mean the counterfactual/linker-only set is worse than Full; positive noise deltas mean more false-positive mass per predicted sentence.
