# S24 agentic tools — pilot iteration 2

## Verdict

**PASSED.**

Two runtime-only grounding corrections were applied: weak lexical aliases do not
become exact identifiers for Phase-1-ambiguous targets, and a longer approved
competing alias defeats a short anchored target.

| Measure | Result |
| --- | ---: |
| Marginal TP / FP | 6 / 0 |
| Marginal precision | 100% |
| Fixed-floor macro F1 | 93.34% |
| Final macro F1 | 94.88% |
| Delta | +1.54pp |
| Distinct tool plans | 4 |

The controller and recovery calls used the Codex CLI backend. S21 was not
resampled: every addition was scored against the saved floor that generated it.
Complete traces are in `pilot_results.json`.
