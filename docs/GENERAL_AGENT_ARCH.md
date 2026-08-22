# Blind-Ensemble Proposal with Evidence-Grounded Verification

A task-agnostic agent architecture, and the measurements that force each part of
it. Nothing below is specific to traceability, entity linking, or software
engineering; the linking task is used only as the worked instantiation.

## The shape

```
  ground     one pass that discovers the vocabulary the input uses
                 |
  propose    k proposers, each a different VIEW of the same input,
             none of which can see any other's output          <-- recall lives here
                 |
  accrete    union (monotone; a proposer may only add)
                 |
  evidence   deterministic, per candidate: what the input actually says
                 |
  verify     judge(s) over evidence, default polarity set by the metric budget
                                                                <-- precision lives here
```

Four roles, two of which are LLM roles (`propose`, `verify`) and two of which are
code (`accrete`, `evidence`). `ground` is optional and is an LLM role when the
input's vocabulary is not known in advance.

## Why the split is where it is

**Recall is bounded at `propose` and cannot be recovered downstream.** A verifier
may only remove. In our instance ~95% of residual false negatives are pairs that
never reached a judge at all. So every recall-led design decision belongs to
`propose`, and the metric budget decides how many proposers to run.

**Precision is cheap at `verify` and expensive at `propose`.** Asking a proposer
to be careful costs recall; asking a verifier to be careful costs nothing but
calls. This is why the two roles must not be merged, and it is the general form of
a result our ablation ledger recorded twelve times: every consolidation of a
proposing decision and a judging decision into one call raised recall and lowered
precision.

## Hypothesis 1 (REFUTED on our instance): enumerate externally, judge membership

Enumerate the candidate set in code, and give the model a membership question over
it. A model asked to *author* a set omits silently; the same model asked to *judge
membership* is far more accurate, and the gap is large, stable and not a capacity
problem:

| | judging F1 | authoring F1 | gap |
| --- | ---: | ---: | ---: |
| complete-ground-truth construction (arXiv 2608.01000, 2026) | 0.60-0.77 | 0.26-0.48 | **+0.25 to +0.34** |

The gap does not close over a 24x parameter range, and planted over-inclusions are
detected **6-7x more often** than planted omissions. That asymmetry is the reason
a verifier tier cannot repair a proposer's miss: an omission leaves no trace to
review. Independently, of answer entities *already present in the prompt*, one
2026 study extracts only 57.4% (arXiv 2606.25656).

Phrasing does not fix it either: the literature's prompt rewrites move F1 by -0.06
to +0.10 with the gap intact, and our own attempt ("report every one, do not stop
at the first") moved the deficit metric 14.0 to 14.3 of 26.

**We built the membership form and it lost badly.** The proposer's free-form reply
was replaced by one accounting for every candidate -- referenced or
not-referenced, so an omission is not expressible:

| arm | proposals | gold | spurious | precision |
| --- | ---: | ---: | ---: | ---: |
| control | 52.5 | 37.8 | 14.7 | 0.721 |
| membership | 34.5 | **31.3** | 3.2 | **0.908** |

Gold fell 6.5 a project-run, the largest recall loss of any arm measured. Forcing
a decision per candidate did not surface omissions; it made the model
**conservative**, converting recall into precision at the worst exchange rate an
F2 budget can face.

The honest statement is therefore narrower than the literature's: an LLM judges
membership better than it authors a set **when the judgement is what is scored**,
but converting a *proposer* into a membership form can shift its operating point
toward precision. Where the enumerated set is large and mostly negative, asking
for the negatives appears to prime rejection. Record this as an open conflict
between a strong external result and a local measurement, not a settled law.

## Law 2: proposers must be blind

Two proposers that ask different questions about the same input and **cannot see
each other's output** recover substantially more than any single pass that asks
both questions. We measured this six ways on the same data, on two models:

| variant | mechanism | recall-critical population |
| --- | --- | ---: |
| two blind proposers | the architecture above | **19.6 / 26** |
| one pass, both questions | merge the questions | 14.0 |
| one pass, larger/smaller window | + granularity | 14.0 |
| one pass, per-item obligation | + forced per-item answer | 14.0 |
| one pass, + context table | + retrieval context | 14.0 |
| one pass, told to report several | + explicit instruction | 14.3 |
| two passes, second conditioned on first | gleaning | 14.0 (added **zero**) |

The population is items that require more than one answer. A single pass returns
about one answer per item and stops, however it is instructed, windowed or
obliged. Two blind passes each return their own answer and the union carries both.

**Conditioning destroys the effect.** The gleaning variant — show the second pass
what the first found, ask only for the remainder, which is GraphRAG's entity
gleaning — added zero items in two of three samples and was the *worst* arm
overall. Telling a generator what has already been found marks the item as
handled. Independence is not a nicety of the ensemble; it is the mechanism.

**This law is not ours; it is replicated independently.** *Diversity of Thought*
(arXiv:2310.07088) runs identical diverse prompts two ways — one call per approach
(blind) versus all approaches concatenated into one chained prompt (conditioned):

| GPT-4 | Graph Coloring | Blocksworld-3 | Blocksworld-4/5 |
| --- | ---: | ---: | ---: |
| chained | 74.0 | 82.0 | 57.0 |
| **blind** | **97.0** | **94.0** | **69.6** |

+23.0 / +12.0 / +12.6 points from execution topology alone, with the cause
measured as **error propagation of 6.2% (GPT-4) and 5.5% (GPT-3.5)** — the chained
arm inherits the first pass's errors at about the rate it loses accuracy. Their
self-consistency baseline reaches 23.0 on Graph Coloring where three blind
framings reach 97.0, which is the same separation between framing diversity and
sampling diversity that we measured.

Consequences, all of which we measured rather than assumed:

* **Resampling is not blindness.** Sampling one proposer k times and unioning
  recovers little, because the samples are correlated: union over three samples
  reached 43 where two blind proposers reached 51.
* **Instruction is not a second look.** Telling one pass that an item may have
  several answers moved the metric 14.0 → 14.3.
* **Adding a blind proposer adds recall.** Running the merged pass as a *third*
  blind proposer alongside the two raised proposal recall on every measurement:
  94.9 → 96.1 (terra) and 90.4 → 94.4 (luna) macro over projects.

## Law 3 (diagnosis confirmed, remedy did not pay): route by evidence, not by producer

A verifier specialised to one question will refuse a correct item for being the
*wrong kind* of item, and that refusal is silent. Measured in our instance: the
coreference judge rejected true links with the objection *"named directly in the
sentence rather than being a referring expression"* -- one component 13 times and
another 6 times in a single recorded project -- while the named-mention judge
approved those same components 33 and 24 times. Whenever the other proposer had
not independently produced the same item, a correct item died on a type
technicality.

The failure is structural, not a prompt defect: each judge answered its own
question correctly. The pipeline needed "is this item correct?" and the judge was
asked "is this item of my kind?".

The rule that follows is general. **Proposers decide what the candidates are;
evidence decides who judges them.** Compute the routing predicate in code, apply it
after all proposers have spoken (so blindness is preserved), and let a candidate
reach the verifier whose question matches the evidence it carries.

**We implemented exactly that, and on F2 it was a wash.** Composed, five projects:

| variant | P | R | F1 | F2 |
| --- | ---: | ---: | ---: | ---: |
| head | 87.4 | 96.0 | 91.0 | 93.8 |
| routed by evidence | 91.6 | **94.5** | **92.9** | 93.8 |

Precision rose 4.2 and recall *fell* 1.5 -- the opposite of the predicted
direction. Moving a candidate out of the specialised judge's stream also removes
it from the cases that judge would have approved, and the receiving judge applies
its own, differently-shaped scepticism. So the diagnosis is confirmed (a
specialised verifier does refuse correct items for being the wrong kind, and does
so silently) while this remedy only converts the loss into a different loss.

The transferable part is the diagnosis and the instrumentation: **inspect what your
verifiers reject, by objection type.** A verifier refusing correct items for
being out of scope is invisible in aggregate metrics and cost us six of eight
residual misses. Fixing it by re-routing is one option and was not the winning
one here; widening the receiving judge's remit, or letting a candidate be judged
by both, are untested alternatives.

## Choosing where to spend: differentiate the metric

Before adding a stage, price a point of each quantity at the current operating
point. For F-beta at precision p and recall r, the marginal value of recall
exceeds precision by a factor that grows with beta and with p/r. At our operating
point (p 87.1, r 97.0, beta 2):

| change | F2 |
| --- | ---: |
| +1.0 precision | +0.24 |
| +1.0 recall | **+0.76** |

Recall was worth three times precision per point, which is why every arm that
bought precision by conceding recall lost, and why the arm that bought recall at
flat precision won. This calculation costs nothing and should precede the
experiment, not follow it.

## What each role is parameterised by, and by what

| role | knob | set by |
| --- | --- | --- |
| `propose` | k, and the *view* each proposer takes | the recall budget; views should differ in the QUESTION, not the sampling temperature |
| `accrete` | none | it is a union; keep it monotone |
| `evidence` | what the bundle shows | the verifier's failure modes, not the task |
| `verify` | default polarity, how many judges, and **which judge sees which candidate** | the metric budget for the polarity; the *evidence* for the routing (Law 3) |

The asymmetry in `verify` is the metric expressed as code. Under an F2 budget a
judge that abstains or approves when uncertain is correct; under an F1 budget it
is not. This is the only place the metric should appear.

## Instantiations

The architecture is indifferent to the task. The same four roles, different views:

| task | `ground` | proposer views (blind) | `evidence` | `verify` |
| --- | --- | --- | --- | --- |
| document → entity linking | alias discovery | (a) items naming the entity, (b) items referring back to it, (c) deterministic lexical scan | the item, its neighbours, prior mentions | approve/reject per candidate |
| multi-hop QA | — | (a) decompose into sub-questions, (b) retrieve-then-read, (c) direct answer | retrieved passages | answer supported? |
| claim verification | — | (a) atomic claim split, (b) whole-claim retrieval | retrieved evidence | entailed / contradicted / neither |
| long-document summarisation factuality | — | (a) per-sentence check, (b) per-claim check | source spans | supported? |

In each row the proposers differ by *question*, not by seed, and none is shown
another's output.

## Where the credit belongs

Two of the three laws are established elsewhere and are cited, not claimed:
blindness by Chain-of-Verification and *Diversity of Thought*; membership-over-
authoring by the silent-omission line. What appears not to be occupied is the
combination this instance needs — **unioning differently-framed proposers for
recall and then filtering with a verifier tier**. The surveyed systems that union
(L3X, EVAPORATE) do not vary the question asked; the systems that vary the
question (Div-Se) aggregate by vote rather than union. Treat that gap as
suggestive rather than settled: it rests on a literature sweep, not a proof.

Independent in-field precedent for the recall/precision shape: decomposing one
prompt into independent sub-prompts raises recall 0.419 → 0.501, 0.179 → 0.259 and
0.072 → 0.142 across three element types with precision flat to slightly down
(arXiv:2410.09854) — the trade an F-beta>1 objective rewards.

## Threats to carry

* **Diverse framings decorrelate only partly.** AMA's error-set Jaccard is 42.2
  against 39.9 for its own i.i.d. comparison. Our 12x observed-to-expected ratio
  is computed against a different (fully independent) baseline and the two numbers
  should not be read as directly comparable.
* **Panels saturate.** A nine-judge panel has been measured at an effective size
  of 2.18; proposal coverage keeps scaling while *selection* plateaus.
* **Selection is the harder half.** Our own error analysis puts an oracle
  discriminator at +2.4 points of macro F1, so the recall side being tractable is
  not evidence that the verifier side is.

## What this does not license

* It does not say more proposers is always better. It says a proposer's marginal
  value is its **non-overlap**, and non-overlap comes from asking a different
  question. Two views of the same question are worth roughly one view.
* It does not say merging stages is always wrong. Merging two *proposers* costs
  recall; merging a proposer into a judge costs precision. Both are measured, and
  they are different failures.
* It says nothing about which judge default is right in general — only that the
  default is where the metric belongs.

## The metric decides the verdict, and it must be stated first

The same arm is accepted or refused depending on the budget. Composed end-to-end,
five projects, macro over projects:

| variant | P | R | F1 | F2 | calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| two blind proposers (head) | 87.1 | **97.0** | 91.5 | **94.6** | 16.4 |
| one merged proposer | 91.5 | 94.2 | **92.7** | 93.6 | 14.0 |
| | | | **+1.29** | **-1.02** | -15% |

Under F1 the merged arm wins and is 15% cheaper. Under F2 it loses. Recall fell
97.0 to 94.2 at the proposal stage and the judges could not recover it, because
they only remove. **Report both, name the primary, and do not let a cheaper
architecture be adopted on the secondary metric.**

## Status

Proposal stage: two models, five documents, three samples an arm, every arm
against control in its own invocation. Composed: one paired run, five projects,
above. Open: the third-blind-proposer arm and the membership-shaped proposer,
both composed, both recall-led.
