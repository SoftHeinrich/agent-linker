# Generic proposer v1 — invalidated

Command configuration:

- model: `gpt-5.6-terra`
- reasoning effort: `none`
- datasets: BigBlueButton, TeamMates
- discovery: generic LLM proposer
- judge: prior participant judge
- saved floor: spike-015 production checkpoint

Result: 7 candidates, 1/12 checkpoint participant TPs reached, 1 TP / 4 FP,
11,348 prompt tokens. The proposer missed nearly every shortened participant
mention and did not provide a safe replacement for lexical discovery.

The raw structured result is preserved in `pilot_results.json`.
