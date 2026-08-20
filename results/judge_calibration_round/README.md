# The judge calibration round (s83, s85) — tuning a lax model without touching a strict one

`s_linker82` beat `s_linker81` on two models by very different amounts (macro F2 +1.76 on
gpt-5.6-terra, +3.80 on gpt-5.6-luna). This round asked whether the gap can be closed from
the judging prompts. It can, but only after fixing how a judging stage is scored.

## The gap is precision, and it is entirely in the judges

Three paired runs per model, `s_linker82` (`../audit_e2e_s82hy_r{1,2,3}_20260820`,
`../audit_e2e_s82luna_r{1,2,3}_20260820`):

| | terra | luna |
|---|---|---|
| TP / FP per five-project run | 183.0 / 30.0 | 184.0 / 77.0 |
| macro F1 / F2 | 92.25 / 94.08 | 85.38 / 91.16 |
| gold proposals, full / partial / coref | 149.7 / 23.3 / 83.7 | 149.3 / 22.0 / 102.0 |
| judge keeps **gold** candidates | 98.9% / 78.6% / 49.8% | 99.3% / 80.3% / 72.5% |
| judge keeps **spurious** candidates | 37.5% / 17.8% / 19.9% | 47.0% / 38.6% / 28.8% |

Recall is model-independent at every proposer and every judge. What differs is how much
junk each judge admits.

## The scoring error that cost seven arms

Seven settings were built and refused on the coreference stage's own gold count. That
count is the wrong measure, and the correction is deterministic (free, off the recorded
runs):

| | terra | luna |
|---|---|---|
| approved coref **gold** already produced by an earlier linker | 24.0 (57.6%) | 54.7 (73.9%) |
| approved coref **gold** genuinely new | 17.7 | 19.3 |
| approved coref **spurious** genuinely new | 11.3 (97%) | 33.7 (89%) |

`_union` merges by pair, so the duplicated gold is free and the spurious is not. Luna's
coreference stage contributes **19.3 new gold against 33.7 new spurious per run — 44% of
that model's entire false-positive mass**. Every strict arm had been charged for losing
gold the pipeline already had.

**Method fix, reusable:** score a judging arm by substituting its kept pairs into the same
run's recorded links from the other stages (`pipeline_exact.py`). Upstream sampling is held
fixed, the only variance is the judge, and replaying the control's own pairs through it
reproduces the recorded end-to-end numbers to the decimal. This is an exact score, not a
projection, and it costs no extra API calls beyond the one stage.

## What was adopted (`s_linker83`, composed with s84's morphology into `s_linker85`)

Three changes at the coreference judge, exact pipeline scoring, three runs a side:

| arm | terra F1 / F2 | luna F1 / F2 |
|---|---|---|
| `s_linker82` (control) | 92.25 / 94.08 | 85.38 / 91.16 |
| + judge shown the resolution it judges, + actor/artifact clause | 92.44 / 94.31 | 86.36 / 91.59 |
| **+ ground for rejecting stated first** | **93.69 / 94.51** | **89.20 / 92.43** |

TP/FP: terra 182.0 / 22.3 and luna 181.7 / 51.7, against 183.0 / 30.0 and 184.0 / 77.0.
**Luna F1 +3.82 / F2 +1.27, terra F1 +1.44 / F2 +0.43.**

1. **The judge is shown the resolution.** It had a sentence and a component name, so it
   could not check an antecedent it was never shown, and rejected half the gold put to it.
2. **The actor/artifact distinction**, which the rubric never stated: an expression
   denoting what a component acts on or produces refers to that thing, not the component.
   Found by reading luna's spurious approvals — `The PDF document` → Presentation
   Conversion, `The scaled image` / `a cache` → ImageProvider, `The status view` → WebUI.
3. **The ground for rejecting, stated before the verdict.** The mirror of this branch's
   oldest measured rule (claim-before-verdict, worth 35.2 TP). Asked of the strict gate
   only: "approve by default" and "state the strongest ground for rejecting" are
   contradictory standards to put in one prompt.

## What stayed refused

| setting | result |
|---|---|
| partial name: name which catalog component the phrase refers to | terra 7.0 gold/run against 18.3 — a model told to pick from a list picks |
| partial name: quote the whole phrase, answer about it | terra 22.0 g / 28.3 n against 18.3 / 12.7 |
| judges report their own certainty | luna's spurious approvals are confidently wrong: 17.7 gold → 11.3 to save 5.3 spurious |
| verify the judge's claim quote against the sentence | 0.0 ungrounded spurious approvals on **either** model — s48's "verifying is worth nothing" holds for the lax model too |
| enforce the coref contract in code (drop a reference that writes the name) | terra −17.7 gold / −0.0 spurious; luna −40.3 / −5.0 |
| coref: drop links whose antecedent does not name the component | F1-positive on both (+0.43, +0.92), F2-negative on both; an F1-led budget would take it |

## The finding that survives the correction

The two models do not differ in what they understand by a question — they differ in which
way they lean when the case is close. Every *rewording* of a judge's question slides both
models along one precision/recall dial. What moved them apart was not a sharper question
but **a second commitment in the same contract** (the ground against, beside the ground
for) and **a distinction the rubric had never carried**. The first is asymmetric because a
strict judge already raises the objection silently; the second is asymmetric because a
strict judge already rejects that class.

## Cost discipline

No end-to-end batch was spent deciding any arm: coreference is the last linker, so nothing
downstream can be starved and `composition_check`'s precondition is structurally vacuous.
Two settings were refused for zero API calls by replaying recorded runs. The confirming
paired E2E (`../s85_e2e_{terra,luna}_r{1,2,3}_20260820`) was run once, on the composed
head only.
