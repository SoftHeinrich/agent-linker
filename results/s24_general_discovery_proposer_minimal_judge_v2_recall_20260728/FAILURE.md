# Generic proposer v2 — invalidated

Command configuration:

- model: `gpt-5.6-terra`
- reasoning effort: `none`
- datasets: BigBlueButton, TeamMates
- discovery: exhaustive generic LLM proposer
- judge: minimal semantic judge
- saved floor: spike-015 production checkpoint

Result: 18 candidates, 2/12 checkpoint participant TPs reached, 3 TP / 11 FP,
13,506 prompt tokens. Asking for exhaustive discovery increased noise but did
not solve recall.

The raw structured result is preserved in `pilot_results.json`.
