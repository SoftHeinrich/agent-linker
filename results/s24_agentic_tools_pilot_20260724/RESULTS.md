# S24 agentic tools — pilot iteration 1

## Verdict

**FAILED the predeclared precision gate.**

The controller selected four distinct project-dependent plans and improved the
same fixed S21 floor from macro F1 93.34% to 94.75%, but the additions contained
8 TP and 4 FP: 66.67% marginal precision versus the required 95%.

The failures were acceptance defects, not unknown tool calls or controller
budget violations:

- weak lexical aliases of Phase-1-ambiguous component names were passed to a
  validator that treats canonical names leniently;
- a short anchored target was accepted inside a longer approved alias for a
  different component.

Complete controller prompts, tool traces, per-project metrics, and marginal links
are preserved in `pilot_results.json`.
