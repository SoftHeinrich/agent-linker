# S24 dedicated anchored-validator — live N=1 result

## Configuration

- Variant: `s_linker24` (unchanged S21 floor plus anchored recovery)
- Backend: OpenAI `gpt-5.4`
- Service tier: `flex`, enforced with `OPENAI_ENFORCE_FLEX=1`
- Reasoning: explicitly disabled with `OPENAI_REASONING_EFFORT=none`
- Datasets: mediastore, teastore, teammates, bigbluebutton, jabref
- Run date: 2026-07-23

Flex intermittently returned `flex_unavailable` 429s. The client retried and
completed every project without changing the required service tier.

## Independent mini-src score

| Project | P | R | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mediastore | 1.0000 | 0.9677 | 0.9836 | 30 | 0 | 1 |
| teastore | 1.0000 | 0.9630 | 0.9811 | 26 | 0 | 1 |
| teammates | 0.8475 | 0.8772 | 0.8621 | 50 | 9 | 7 |
| bigbluebutton | 0.9310 | 0.4355 | 0.5934 | 27 | 2 | 35 |
| jabref | 1.0000 | 1.0000 | 1.0000 | 18 | 0 | 0 |
| macro | 0.9557 | 0.8487 | 0.8840 | 151 | 11 | 44 |

Pooled F2 is 0.8015 (151 TP, 11 FP, 44 FN). This full score is not a direct
S24 delta: every S21 floor phase was freshly sampled and, notably, the
BigBlueButton floor was unusually low in this run.

## Marginal S24 result

The dedicated path considered 40 eligible candidates: 6 were resolver-approved
and 3 passed its anchored validator. All 3 are gold links, with 0 marginal FP:

| Project | Sentence | Addition | Gold |
| --- | ---: | --- | --- |
| bigbluebutton | S52 | `Apps` | yes |
| bigbluebutton | S66 | `FreeSWITCH` | yes |
| teammates | S88 | `Logic` | yes |

This clears the N=1 marginal precision requirement and demonstrates that the
separate anchored validator fixes the earlier gate mismatch. It does **not** yet
meet the N=3 promotion criterion; repeat it twice before promoting the variant.
