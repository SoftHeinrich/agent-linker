# S24 dedicated anchored-validator — valid composite N=1 result

## Configuration

- Variant: `s_linker24`
- Backend: OpenAI `gpt-5.4`; Flex enforced; `reasoning_effort=none`
- Run protocol: fail-closed on an exhausted LLM request
- Run date: 2026-07-24

## Valid constituent results

The prior dedicated-validator run is used only where its required calls completed:

| Project | Source result |
| --- | --- |
| mediastore | original 2026-07-23 run, no failed calls |
| teastore | original 2026-07-23 run, no failed calls |
| jabref | original 2026-07-23 run; both failed requests were retried successfully with the identical prompts |
| teammates | this 2026-07-24 fail-closed recovery, 0 failed calls |
| bigbluebutton | this 2026-07-24 fail-closed recovery, 0 failed calls |

## Independent mini-src score

| Project | P | R | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mediastore | 1.0000 | 0.9677 | 0.9836 | 30 | 0 | 1 |
| teastore | 1.0000 | 0.9630 | 0.9811 | 26 | 0 | 1 |
| teammates | 0.9434 | 0.8772 | 0.9091 | 50 | 3 | 7 |
| bigbluebutton | 0.9423 | 0.7903 | 0.8596 | 49 | 3 | 13 |
| jabref | 1.0000 | 1.0000 | 1.0000 | 18 | 0 | 0 |
| macro | 0.9771 | 0.9196 | 0.9467 | 173 | 6 | 22 |

Pooled F2 is 0.9020. This supersedes the invalid full-score comparison from the
2026-07-23 S24 validator run, whose BBB and Teammates outputs contained unrecovered
capacity failures.
