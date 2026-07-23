# S24 anchored-recovery — live N=1 result

## Configuration

- Variant: `s_linker24` (S21 floor plus anchored sibling/prefix recovery)
- Backend: OpenAI `gpt-5.4` (responses identify `gpt-5.4-2026-03-05`)
- Service tier: `flex`, enforced with `OPENAI_ENFORCE_FLEX=1`
- Reasoning: explicitly disabled with `OPENAI_REASONING_EFFORT=none`
- Datasets: mediastore, teastore, teammates, bigbluebutton, jabref
- Run date: 2026-07-23

The first request and one later request received a temporary `flex_unavailable`
429; the client's retry completed the run without changing the enforced tier.

## Independent mini-src score

| Project | P | R | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mediastore | 1.0000 | 0.9677 | 0.9836 | 30 | 0 | 1 |
| teastore | 1.0000 | 0.9259 | 0.9615 | 25 | 0 | 2 |
| teammates | 0.9423 | 0.8596 | 0.8991 | 49 | 3 | 8 |
| bigbluebutton | 0.9231 | 0.7742 | 0.8421 | 48 | 4 | 14 |
| jabref | 1.0000 | 1.0000 | 1.0000 | 18 | 0 | 0 |
| macro | 0.9731 | 0.9055 | 0.9373 | 170 | 7 | 25 |

Pooled F2 is 0.8882 (170 TP, 7 FP, 25 FN). The complete raw result and all
request/response text traces are in this directory.

## Marginal-addition finding

This first S24 implementation made **zero** accepted additions. Its resolver
did approve some locally anchored candidate cases in teammates and BigBlueButton,
but S21's inherited strict coreference gate rejected all of them. Therefore the
table above is a fresh stochastic S21-floor draw, not evidence that S24 improved
or degraded recall. It fails the S24 promotion criterion (no clean marginal TP)
and should remain an experimental negative result until the anchored gate is
redesigned and separately evaluated.
