# S21 GPT-5.6-terra — live N=1 control

## Configuration

- Variant: canonical `s_linker21`
- Backend: OpenAI `gpt-5.6-terra`
- Service tier: `flex`, enforced with `OPENAI_ENFORCE_FLEX=1`
- Reasoning: explicitly disabled with `OPENAI_REASONING_EFFORT=none`
- Datasets: mediastore, teastore, teammates, bigbluebutton, jabref
- Run date: 2026-07-23

## Independent mini-src score

| Project | P | R | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mediastore | 1.0000 | 0.9677 | 0.9836 | 30 | 0 | 1 |
| teastore | 1.0000 | 0.9630 | 0.9811 | 26 | 0 | 1 |
| teammates | 0.8125 | 0.9123 | 0.8595 | 52 | 12 | 5 |
| bigbluebutton | 0.9020 | 0.7419 | 0.8142 | 46 | 5 | 16 |
| jabref | 1.0000 | 1.0000 | 1.0000 | 18 | 0 | 0 |
| macro | 0.9429 | 0.9170 | 0.9277 | 172 | 17 | 23 |

Pooled F2 is 0.8875 (172 TP, 17 FP, 23 FN). The directory includes the raw
CSV predictions, evaluator JSON, and full text request provenance.
