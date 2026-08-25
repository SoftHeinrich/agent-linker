# The judge round (s111–s113) — one law at every judging site

The consolidation round left the pipeline with three judges and no single account of
them. This round asks whether one exists, tests the two arms it predicts, and answers
everything it can at **level 1** first. Extraction is frozen throughout: both gates
measured here sit behind deterministic scans, so a candidate set is byte-identical
across arms and only the judging prompt varies.

Tooling, all of it callable with no LLM spend:
`pilot/judge_census.py` (what each gate is handed and what it does with it),
`pilot/lenient_audit.py` (the lenient gate's errors, split by facts a runtime could
compute), `pilot/chooser_audit.py` (the consolidation round's unbuilt arm, repriced on
the head), `pilot/test_s112_order.py` (24 prompt invariants against recorded calls).
Level 2: `pilot/nextgen_pilots.py`, driven by `pilot/run_judge_round.sh`.

## The law, stated once

> **Every judging site enumerates its alternatives before it commits. Code enumerates
> the alternatives the case already contains; the model enumerates the ones it does
> not; the default polarity is the base rate of the stream that routes cases there.**

The first two clauses are the consolidation round's transferable result, which it drew
from four arms at two sites. The third is what makes the three judges one design rather
than three, and it is measurable: `pilot/judge_census.py` reads each gate's own
decisions off six recorded runs of the head.

| gate | base rate of gold in its cases | default | TP | FP | net gold lost | F2-weighted headroom |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| full name, lenient | 0.70 / 0.74 | approve unless a ground fires | 149.7 / 162.3 | **17.3 / 41.3** | 3.3 / 0.7 | **27.3 / 43.3** |
| partial name, sortal | 0.31 / 0.19 | classify, then keep participants | 19.7 / 11.0 | 7.0 / 14.7 | 4.3 / 4.3 | 20.0 / 27.7 |
| coreference, strict | 0.57 / 0.46 | reject when uncertain | 39.0 / 40.7 | 1.7 / 4.0 | 4.7 / 3.0 | 15.7 / 13.0 |

terra / luna, mean per five-project run. Headroom weights one lost gold link at three
false positives, the F2 derivative at the head's operating point.

**The polarity clause holds where it can be checked.** The gate handed the cleanest
stream is the lenient one and it approves by default; the gate handed the dirtiest is
the sortal one and it demands a positive classification; the strict gate sits between
them and rejects when uncertain. Two of the three thresholds were set by pilot years of
this branch apart, and they landed in base-rate order.

**Where the law is descriptive and not yet predictive:** three streams, one task, two
models. It orders the defaults correctly; it does not tell you where to put a fourth.

## What level 1 refused, for free

### The contrastive chooser: its own ceiling was eaten by the repair beside it

The consolidation round priced a chooser over the sortal gate's sibling groups at
**−8.3 FP / +2.3 TP** and left it unbuilt, owing a stage pilot. That price was read off
`s_linker92a`. `s_linker109` then landed, refusing 12 candidates a run whose word is
written only inside another component's name — the same population. Repriced on the
head's own runs (`pilot/chooser_audit.py`, six runs):

| model | questions a run | options a question | with no gold answer | **removable FP** | **recoverable FN** | TP the chooser puts at risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| terra | 7.0 | 2.00 | 3.0 | **2.3** | **0.0** | 8.0 |
| luna | 5.7 | 2.00 | 3.0 | **0.7** | 1.7 | 3.7 |

The whole ceiling is now 2.3 and 0.7 false positives a run against a recorded null
floor of 10.7, and the gold standing inside those same groups is 3.5× the ceiling on
terra. **A perfect chooser is not worth a call and an imperfect one is a liability.**
Refused, and the owed item is closed without spending anything.

### Supplying the surface as a fact: already there

`pilot/lenient_audit.py` isolates the lenient gate's false positives with a fact:
whether every writing of the name in that sentence is lowercased while the catalog
capitalises it.

| model | FP share in that bucket | precision inside it | precision overall |
| --- | ---: | ---: | ---: |
| terra | **60%** | 0.55 | 0.90 |
| luna | **78%** | 0.43 | 0.80 |

So the use/mention call is where the gate leaks, on both models, and the fact that
marks it is code-computable. It is also **already in the prompt**: every case line
reads `Case 1: "<surface>" -> <Name>`, so the sentence's writing and the catalog's
stand side by side. Adding it again is what `s_linker92e` did and it was refused. What
the case lacks is not the fact but the weighing of it — which is the law's second
clause, and the only arm it licenses here.

## What level 2 says, terra

`pilot/run_judge_round.sh terra`, three samples, every arm of a gate in the **same
invocation**, candidate sets asserted identical across arms by the harness (they are:
both proposers are scans). Scored by `pilot/judge_round_stats.py`, which orders arms by
`3*gold - spurious` -- the F2 derivative at the head's operating point written out, so a
precision win that costs more than one gold per three false positives is visible as the
loss it is.

### The sortal gate

| arm | gold | spurious | precision | 3·gold − spurious | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| control (head) | 21.3 | 8.0 | 0.727 | 56.0 | |
| **`s112` quote before verdict** | **23.0** | 8.3 | 0.734 | **60.7** | QUALITY-NEUTRAL, every point estimate in the arm's favour (gold p 0.30, floor 0.10) |
| `s113` readings enumerated | 14.3 | 8.0 | 0.642 | 35.0 | **QUALITY-CHANGING against** (gold p 0.10 = floor) |

`s113` is refuted outright and in the least ambiguous way available: **−7.0 gold at
±0.0 spurious**. It is not a precision instrument at this gate; it is a loss with no
compensating side. Its run-to-run spread is the control's (within-arm set distance
[7, 0, 7] against [3, 4, 7]), so this is not noise about a neutral point.

`s112` moves two strings and gains 1.7 gold a run at 0.3 spurious. At the n = 3 floor
that is neutral, not proven -- but it is the only arm of this round whose every
statistic points the same way, and it is the one that adds no token to the reply.

### The lenient gate

| arm | gold | spurious | precision | 3·gold − spurious | within-arm spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| control (head) | 150.0 | 13.0 | 0.920 | 437.0 | [3, 1, 2] |
| `s111` readings enumerated | 142.0 | 8.7 | **0.942** | 417.3 | **[14, 14, 4]** |

**QUALITY-CHANGING against, under F2**: −8.0 gold to save 4.3 spurious is 0.54 false
positives per gold link, against the three this budget demands. Under F1 the same trade
is close to neutral, which is what the regex round measured end to end and read as a
win. The metric is doing the deciding, and the round that adopted F2 has to refuse it.

**And the arm is five times less stable.** On a *fixed* candidate set the control's two
runs differ by 2.0 links on average and the arm's by 10.7. Every call succeeded and the
longest reply used 1642 completion tokens against a 4096 cap, so this is not truncation
-- the template genuinely makes the verdict less repeatable.

### Where the loss sits, and why the law predicts it

Per project, at the lenient gate, with the base rate of gold among the scan's own
candidates:

| project | candidates | base rate | control gold / spurious | `s111` gold / spurious |
| --- | ---: | ---: | ---: | ---: |
| bigbluebutton | 42 | 0.90 | 37.3 / 4.0 | 37.7 / 4.0 |
| jabref | 20 | 0.90 | 18.0 / 1.0 | 18.0 / **0.0** |
| mediastore | 31 | 0.90 | 28.0 / 3.0 | 28.0 / **0.0** |
| teastore | 25 | 0.84 | 21.0 / 0.0 | 19.7 / 0.0 |
| **teammates** | 97 | **0.51** | **45.7 / 5.0** | **38.7 / 4.7** |

On the four projects whose stream is 84–90% gold the arm is free or better: it removes
four spurious links a run and costs 1.3 gold. **The entire loss is the one project whose
stream is half spurious** -- and there the arm's spurious count is 10, 1, 3 across three
identical inputs, which is the instability, not a shift.

That is the law's third clause doing work it was not fitted on. The lenient default is
correct for a 0.90 stream and is carrying the whole burden on a 0.51 one, and the
template that changes *how* the verdict is reached breaks exactly where the verdict
matters most.

## What level 2 says, luna — and it is not what terra says

`pilot/run_judge_round.sh luna`, same three samples, same pinned alias tables, same
fixed candidate sets.

### The sortal gate, both models side by side

| arm | terra gold / spurious | luna gold / spurious | terra net | luna net |
| --- | ---: | ---: | ---: | ---: |
| control | 21.3 / 8.0 | 21.3 / 9.7 | 56.0 | 54.3 |
| `s112` quote first | **23.0** / 8.3 | **18.0** / 6.3 | **+4.7** | **−6.7** |
| `s113` + readings | **14.3** / 8.0 | **21.7** / 3.3 | **−21.0** | **+7.3** |

**Both arms change sign between models, and they change it in opposite directions.**
`s112` gains 1.7 gold on terra and loses 3.3 on luna; `s113` loses 7.0 on terra and is
the best arm of either model on luna (+0.3 gold at −6.3 spurious). Both are
QUALITY-CHANGING on luna at the n = 3 floor. Under this branch's standing rule -- an arm
the second model refuses is refused -- **both are out**.

**And this gate cannot carry the question anyway.** Of the five projects, only
bigbluebutton contributes gold at the sortal gate on either model (terra adds 3.0 from
teammates, luna 0.0). Every number in the table above is one project's 48 candidates
measured three times. That is not a population a reply template can be settled on, and
the honest reading of the sign flip is that neither arm was ever separated from noise
rather than that the models disagree about something.

### The lenient gate replicates, in the same direction

| model | arm | gold | spurious | net | within-arm spread |
| --- | --- | ---: | ---: | ---: | ---: |
| terra | control | 150.0 | 13.0 | 437.0 | 2.0 |
| terra | `s111` | **142.0** | 8.7 | 417.3 | **10.7** |
| luna | control | 153.3 | 32.0 | 428.0 | 10.0 |
| luna | `s111` | **148.0** | 22.0 | 422.0 | **26.0** |

**Same trade on both models: gold down (−8.0, −5.3), spurious down further (−4.3,
−10.0), net negative on both (−19.7, −6.0), and the arm two to five times less stable
than the control on a candidate set that does not move.** On luna as on terra the
largest single loss is teammates, the one project whose stream is half spurious.

This closes the regex round's outstanding level-4 item at level 2, for about 240 calls
instead of an end-to-end batch: `s_linker92f`'s template is a **precision instrument**,
it is priced correctly at F1 and refused at F2, and it costs decision stability
everywhere it is applied.

## The round's result

| | decided at | verdict |
| --- | --- | --- |
| the contrastive chooser | level 1 | refused -- `s_linker109` ate its ceiling (2.3 / 0.7 FP a run) |
| supplying the surface as a fact | level 1 | refused -- already in every case line, which is why `s92e` failed |
| `s_linker111` readings at the lenient gate | level 2, both models | **refused under F2**; correct at F1; 2–5× less stable |
| `s_linker112` quote before verdict, sortal gate | level 2, both models | **refused** -- sign flip, and the gate's population is one project |
| `s_linker113` readings at the sortal gate | level 2, both models | **refused** -- sign flip, opposite to `s112`'s |

**The head does not move. `s_linker110` stands.** Nothing here is composed and nothing is
owed an E2E, because no arm reached a verdict that would justify one.

### What the round establishes anyway

The law survives its own test and gains a mechanism it did not have:

> **Code enumerating a fact is stable; the model enumerating its own alternatives is
> not, and at a judging site the variance costs more than the precision buys.**

Both halves are now measured on the same branch, at the same operating point:

| | who enumerates | site | gold | spurious | run-to-run spread |
| --- | --- | --- | ---: | ---: | --- |
| `s_linker110` | code | proposal (resolver) | −0.2 / −0.5 | **−4.7 / −15.3** | refusal count identical every run |
| `s_linker111` | the model | judging (lenient) | **−8.0 / −5.3** | −4.3 / −10.0 | 2–5× the control's |
| `s_linker113` | the model | judging (sortal) | −7.0 / +0.3 | ±0.0 / −6.3 | sign flips between models |

`s_linker106` said the same thing at the resolver a round earlier (spurious +6.6 when
the model enumerated, −10.0 when code did) and was read then as a fact-versus-weighing
distinction. It is that, and it is also a variance result: the enumeration the model
supplies is *resampled every call*, so it moves the verdict around; the enumeration code
supplies is the same list every time.

**What this predicts, and has not been tested:** a model-enumerated alternative set
should stop costing anything as soon as it is sampled more than once and reduced -- the
same self-consistency the branch already measured as worth +1.2 F1 for free over three
runs. That is the arm this round would build next, and it is a level-2 question at the
lenient gate on fixed candidates, exactly like everything above.

## The architecture the round leaves: one structure, three skills

The three judges were three code paths — `_validate_with_evidence`,
`_classify_denotations`, `_validate_coref_links` — each with its own batching loop, its
own reply parser, its own bounds check and its own decision record. They do the same
thing, and the branch had never written that thing down. `s_linker114` does:

> Take candidates. Batch them at `JUDGE_BATCH`. Render the batch's shared evidence once
> and each case as a numbered block. Compose one prompt — question, rubric, cases, reply
> schema. Ask once. Parse by case index, bounds-checked. Record one decision per
> candidate, defaulting to the skill's own polarity. Return what was kept.

A `JudgeSkill` carries only what differs, and every field of it is a measured decision:

| skill | question | withheld | rubric | reply fields, in order | polarity | why that polarity |
| --- | --- | --- | --- | --- | --- | ---: |
| `entity` | is this written name doing naming work here? | nothing | layered entity + qualified + stricter | claim, approve | approve unless a ground fires | base rate 0.70 / 0.74 |
| `denotation` | what does this expression denote? | **target and catalog** | qualified | denotation, claim | keep `participant` | 0.31 / 0.19 |
| `coref` | does this expression point to this component? | nothing | layered coref | claim, objection, approve | reject when uncertain | 0.57 / 0.46 |

Every difference in that table has a measurement behind it and nothing else differs.
The target is withheld from `denotation` because showing it is `s_linker25` at −5.5 gold
and `s_linker108` at −0.40 macro F2. `objection` belongs to the strict skill alone
because "approve by default" and "state the strongest ground to reject" are
contradictory in one prompt. `coref` is shown the resolver's committed reference and
antecedent because `s_linker82` withheld them and the judge then rejected half the gold
put to it.

**It is byte-identical to the head, and that is the deliverable.**
`pilot/test_s114_skills.py` runs the head's own methods and the variant's own methods
side by side over six recorded runs with `_ask` stubbed to record and answer nothing:
**142/142 judging batches send the same prompt, record the same decision, and keep the
same set.** A refactor that moves a measured number is not a refactor.

**Honesty about the size:** this is not a code reduction. 133 statements against the
head's 136. What changes is that the loop exists **once** instead of three times and the
six ways the judges differ are five declared fields instead of three divergent
implementations — so the next judging arm is an edit to one field, not a fourth copy of
the loop, and an arm that forgets to differ where the measurements say it must is now
visible in a table rather than buried in a method.

One deliberate difference, priced first: the head writes **no decision row at all** for a
case its reply never answered, so a silent omission leaves no trace. The variant records
it, rejected. Over the six recorded runs that case occurs **0.0 times a run** against
79.3–84.7 candidates, so it changes nothing measured and stands as a tripwire.
