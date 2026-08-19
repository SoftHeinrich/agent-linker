# s25: everything else, ablated — 2026-08-11

Eleven arms for the decisions no earlier round had priced. Off checkpoints,
upstream frozen from `results/s25_cleanup_verify_20260810`, one thing changed per
arm, five runs per side on all five projects, permutation-tested, scored on TP /
FP / F1 / F2. Script: `approach/pilot/ablate_all.py`.

Two arms overturned a judgment I had made without measuring, which is the point
of running them.

## Full-name stage

| Arm | TP | FP | F1 | F2 | Verdict |
|---|---|---|---|---|---|
Drop the spelling-variant proposer | −2.4 (p=0.05) | +0.4 (p=0.44) | **−0.9 (p=0.01)** | **−1.1 (p=0.01)** | **keep** — I had recommended cutting this as poor value; it is worth 2.4 gold links, not 2 |
Drop the uniqueness pass (p1 only) | +2.4 (p=0.01) | **+10.0 (p=0.01)** | −1.6 (p=0.01) | +0.2 (p=0.44) | **keep** — the second pass trades 2.4 recall for 10 precision; F1 says keep, F2 is indifferent |
Drop the anchor sentences | −2.2 (p=0.01) | +0.6 (p=0.17) | −0.8 (p=0.01) | −1.0 (p=0.01) | **keep** — load-bearing, consistent with the jabref result where more anchors removed the ordinary-English false positives |
Drop the matched span from the bundle | +0.8 (p=0.44) | +0.2 (p=1.00) | +0.2 (p=0.44) | +0.3 (p=0.44) | neutral on the stage, **reverted end-to-end** (below) |
Drop the preceding sentence from the bundle | −0.4 (p=1.00) | +0.6 (p=0.17) | −0.3 (p=0.30) | −0.2 (p=0.30) | neutral on the stage, **reverted end-to-end** (below) |
Drop the source tag from the bundle | +0.6 (p=0.52) | +1.4 (p=0.05) | −0.1 (p=0.60) | +0.2 (p=0.54) | keep — one word, and dropping it costs precision |

## Partial-name stage

| Arm | TP | FP | F1 | F2 | Verdict |
|---|---|---|---|---|---|
Skip the target-blind denotation step | +0.6 (p=1.00) | **+12.0 (p=0.01)** | **−2.6 (p=0.01)** | −0.8 (p=0.05) | **keep** — the workflow's best idea, now measured: withholding the target is worth 12 false positives |
Exact word match instead of the prefix rule | −1.0 (p=0.01) | −0.2 (p=1.00) | −0.3 (p=0.01) | −0.4 (p=0.01) | **keep** — the "morphology approximation" earns 1 gold link; small, but significant and cheap to state |
Drop the unique-owner requirement | +0.6 (p=0.84) | +2.4 (p=0.02) | −0.4 (p=0.23) | ±0.0 (p=1.00) | keep — buys 2.4 precision at no recall cost |
Drop the qualified-identifier skip | ±0.0 (p=1.00) | +0.6 (p=0.21) | −0.1 (p=0.21) | −0.1 (p=0.21) | keep — neutral, and the test is shared with the variant generator, so cutting the call deletes no code |
Drop the whole-name exclusion | **+3.2 (p=0.01)** | **+6.2 (p=0.01)** | −0.5 (p=0.13) | **+0.8 (p=0.01)** | **open** — a real F1/F2 trade, see below |

## The two stage-neutral cuts that failed end-to-end

The matched span and the preceding sentence appear twice in every full-name judge
case: once in the case header (`Case n: "span" -> Component`, then the sentence
with its `[prev: ...]` prefix) and once inside the evidence block. Removing
either duplicate is neutral on the judging stage. Three five-project runs with
both removed:

| | with the repetition | without |
|---|---|---|
| TP | 178–183 (six-run band) | 181 / 182 / 183 |
| FP | 4–6 | **8 / 9 / 8** |
| macro F1 | 96.42 ± 0.42 | **95.23** |
| macro F2 | 95.38 ± 0.58 | 95.23 |

Recall holds; precision does not. Reverted. **Repeating the evidence next to the
rubric is not redundant for the model** — that is now the third time a
stage-neutral simplification turned out pipeline-negative, always on precision,
always because this stage's output is locked in by the earlier-wins union and
withheld from the two stricter linkers.

## Open decision: the whole-name exclusion

`_name_word_candidates` skips any sentence that states a whole name, on the
grounds that those sentences belong to \linkerB. Dropping that condition:

- **+3.2 TP, +6.2 FP, F1 −0.5 (p=0.13), F2 +0.8 (p=0.01)**

It is the only remaining knob where F1 and F2 disagree with significance on the
F2 side. Adopting it would need an end-to-end confirmation and a decision about
which measure leads the paper. Not adopted.

## Net

Of eleven arms: **nine confirmed the existing design** (six of them with a
number it did not have before), two were stage-neutral and reverted after the
end-to-end check, and one is an open F1/F2 trade. Nothing in the workflow now
lacks a measurement, and the two things I had judged as poor value by inspection
— the spelling-variant proposer and the prefix rule — both turned out to earn
their place.
