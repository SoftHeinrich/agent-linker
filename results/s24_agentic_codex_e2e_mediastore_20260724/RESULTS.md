# SLinker24Agentic — end-to-end Mediastore smoke

## Configuration

- Variant: `s_linker24_agentic`
- Backend: Codex CLI (`LLM_BACKEND=codex`)
- Dataset: mediastore
- Run date: 2026-07-24
- Execution: normal `run_ablation.py` path; no checkpoint replay

## Result

| Path | P | R | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Same-run S21 floor | 96.77% | 96.77% | 96.77% | 30 | 1 | 1 |
| S24 agentic final | 96.88% | 100.00% | 98.41% | 31 | 1 | 0 |

The controller selected only `alias_phase4`. It added one TP and zero FP, for a
same-run F1 gain of 1.64pp.

The final runner output is `ablation_20260724_010443.json`; emitted links are in
`s_linker24_agentic_mediastore_links.csv`. This validates the complete live
pipeline but is not directly comparable to the GPT-5.4/Flex paper runs.
