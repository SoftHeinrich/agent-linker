# s_linker26 — one reading of the document — 2026-08-12

s26 merges the two questions s25 asks in two stages: one prompt per sentence
batch returns the references in that passage **and** any name the passage
establishes, with the table accumulated and fed forward to the next batch. No
knowledge stage, no alias judge: two prompts and two LLM calls per project fewer.

Three five-project runs, same model and settings as every other run here
(`gpt-5.6-terra`, OpenAI, reasoning effort `none`).

| | s26 (one reading, N=3) | s25 (N=6 reference band) |
|---|---|---|
| macro F1 | **94.27** (94.1 / 95.1 / 93.6) | 96.42 ± 0.42 |
| macro F2 | **93.47** (93.2 / 94.2 / 93.0) | 95.38 ± 0.58 |
| TP | 175.7 (175 / 177 / 175) | 180.8 |
| FP | 11.0 (11 / 8 / 14) | 4.8 |

**Verdict: rejected.** −5.1 TP and +6.2 FP; both F1 and F2 land clearly outside
the s25 band. `experimental=True`; not promoted.

## Why it fails — diagnosed, not assumed

`approach/pilot/s26_diagnosis.py` compares the two variants' own checkpoints.
The result corrects the obvious guess.

**1. The merged reading builds a bigger table, not a worse one in the simple sense.**
49 terms against 27, sharing 20. What each finds alone is systematic:

| | terms |
|---|---|
only the document-wide pass | `ui`, `webui`, `e2e`, `gae`, `test driver`, `akka-apps`, `presentation conversion flow` |
only the per-passage reading | `logic component`, `storage layer`, `back-end logic`, `web browser`, `client`, `core`, `outer layers`, `conversion process`, `svg conversion`, `logic.api`, `logic.core`, … (29 in all) |

The seven the global pass finds are **short abbreviations** — defined once,
usually far from most of their uses, so a 50-sentence window sees the use and not
the definition. The 29 the reading adds are **descriptive phrases and generic
words**, plus `logic.api` and `logic.core`, which the alias rubric explicitly
forbids: the same rule s25's dedicated prompt follows gets violated when it is
appended to an extraction prompt. Nothing filters any of it, because the judge is
gone.

**2. The damage is not where it was expected.** Per linker, averaged over three
runs each:

| linker | s25 TP / FP | s26 TP / FP | delta |
|---|---|---|---|
full-name | 152.3 / 3.0 | 150.0 / 8.3 | −2.3 / +5.3 |
**partial-name** | **14.0 / 0.0** | **7.7 / 1.3** | **−6.3 / +1.3** |
coreference | 26.4 / 1.7 | 27.0 / 3.7 | +0.6 / +2.0 |

The largest single loss is the **partial-name linker losing nearly half its
links** — and it is a stage the merge does not touch.

**3. The reason: the alias table has two opposite jobs.** It *admits* full-name
candidates, and it *suppresses* partial-name candidates —
`_name_word_candidates` skips any sentence that states a whole name in N(c). Add
`logic component`, `storage layer`, `back-end logic`, `client` to N(c) and
sentences the partial-name linker would have proposed are now classed as stating
a whole name. So **growing the table moves work from the strict linker to the
lenient one**: s26's full-name links go *up* (155.3 → 158.3) while the two
stricter linkers contribute less and less correctly (TP 38.3 → 33.7).

**4. Direct admission explains under half of it.** Of the 9 extra false positives
only 4 are admitted by a name only s26 has; of the 10 missing true positives only
1 needs a name only s25 has. The rest is this indirect, cross-stage effect.

## The methodological point, sharpened

On the full-name stage this merge was **exactly neutral** — TP +0.2 (p=1.00),
F2 ±0.0 (p=0.98). It had to be: the effect that dominates lives in a *different
stage*. The alias table is read by five consumers across all three linkers, and
one of those uses is a suppressor, so no single-stage arm can see the table's
real cost. That is the fifth case here of a stage arm disagreeing with the
pipeline, and the first one whose mechanism is fully attributed.

## What it buys the paper

Before this run, the strongest thing that could be said about the two-stage
separation was that merging it away was neutral on one stage — a weak defence of
an architecture. Now: **the separation is worth 2.2 F1 and 1.9 F2, measured.**
The knowledge module is not a stylistic choice, and `s_linker26.py` is the
artifact that shows what the alternative costs.
