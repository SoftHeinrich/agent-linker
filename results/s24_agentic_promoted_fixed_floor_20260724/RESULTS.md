# SLinker24Agentic — promoted fixed-floor replay

## Verdict

**PASSED with the production class.**

This run imports `SLinker24Agentic` from the runtime package and calls its
production `_augment_floor()` method over all five saved S21 floors.

| Project | Plan | TP / FP added | Floor F1 | Final F1 |
| --- | --- | ---: | ---: | ---: |
| mediastore | alias | 1 / 0 | 94.92% | 96.67% |
| teastore | alias + anchor | 1 / 0 | 96.15% | 98.11% |
| teammates | alias + anchor | 1 / 0 | 89.91% | 90.91% |
| bigbluebutton | anchor | 2 / 0 | 85.71% | 87.72% |
| jabref | none | 0 / 0 | 100.00% | 100.00% |
| **Macro** | 4 distinct plans | **5 / 0** | **93.34%** | **94.68%** |

Marginal precision is 100%; macro F1 improves by 1.34pp. The controller and
recovery calls used the Codex CLI backend. Full prompts, responses, plans,
per-link gold labels, and metrics are in `pilot_results.json`.
