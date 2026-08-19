# Can the workflow be simpler than s25? Twenty-three attempts — 2026-08-12/13

s25 reads the document twice, links with three linkers in a fixed order, and judges
each linker's candidates in a dedicated call. Every part of that looks like it could
be accreted structure, so every part was removed, merged or replaced in a variant that
is implemented, registered and runnable.

Of the first twenty, nineteen lose, and the one that holds removes no stage at all — it
retires a tuned constant (`s_linker45`: coreference resolution reads the judges' batch
size, two batch constants instead of three, **26% fewer calls, F1 p = 0.52, F2
p = 0.91**). Three more then went after *mechanism* rather than cost, and **all three
hold**:

* **`s_linker47`** removes an LLM stage — the partial-name linker's grounded identity
  review — for **F1 +0.2 (p = 0.53) and F2 +1.3 (p = 0.01)**;
* **`s_linker48`** removes no stage and merges eight copies of three conditions, in five
  shapes, into three named predicates, deleting three conjuncts that never fired:
  **F1 +0.3 (p = 0.50)** with a composition statistic of **−0.2 (p = 0.59)**;
* **`s_linker49`** composes the two: **F1 −0.2 (p = 0.50), F2 +0.9 (p = 0.03)**.

Every arm from `s_linker43` on is measured **paired**, both variants in the same
invocation, because the model's absolute level moves between invocations by more than
the effects being measured.

| Variant | Architecture | calls | TP | FP | macro F1 | macro F2 |
|---|---|---|---|---|---|---|
**s25** | alias pass + judge + batched extraction | 2 + N | **180.8** | **4.8** | **96.42 ± 0.42** (N=6) | **95.37 ± 0.57** |
s26 | one merged prompt per batch, table fed forward | N | 175.7 | 11.0 | 94.27 (N=3) | 93.47 |
s27 | one merged prompt, whole document, no batching | **1** | 168.0 | 18.0 | 91.70 (N=1) | 90.78 |
s28 | s26, and the table no longer suppresses partial names | N | 174.7 | 11.0 | 93.89 (N=3) | 93.02 |
s29 | alias judge replaced by a lexical grounding check | 1 + N | 162.0 | 4.3 | 90.07 (N=3) | 86.51 |
s30 | alias judging folded into the extraction pass | 1 + N | 166.0 | 8.0 | 90.40 (N=1) | 87.50 |
s31 | alias pass proposes **and reviews itself** in one call | 1 + N | **178.7** | 9.7 | 94.09 (N=3) | 93.94 |
s32 | judge's rubric **carried by the extraction pass**, any batch | 1 + N | **181.3** | 13.0 | 94.86 (N=3) | **95.01 (in band)** |
s33 | the same, majority of batches | 1 + N | **181.3** | 13.0 | 94.75 (N=3) | **94.97 (in band)** |
s34 | the same, **unanimous** | 1 + N | **181.7** | 10.7 | 95.11 (N=3) | **95.20 (in band)** |
s35 | the same, review asked **before** the document | 1 + N | 162.0 | 8.3 | 89.66 (N=3) | 86.38 |
s36 | the full-name judge's two criteria in **one call** | 2 + N | **181.7** | 8.3 | 95.70 (N=6, **p = 0.01**) | **95.38 (identical, p = 1.00)** |
s37 | s36 plus a committed quote per criterion | 2 + N | **182.2** | 8.8 | 95.65 (N=6, **p = 0.017**) | **in band (p = 0.810)** |
s38 | s36's one prompt, **sampled twice and ANDed** | 2 + N | **182.0** | 7.2 | 95.95 (N=6, p = 0.071) | **95.58 (p = 0.457)** |
s42 | s36 plus a **three-value** mention label | 2 + N | 181.7 | 9.0 | 95.75 (N=3 vs s36: p = 0.50) | 95.29 (p = 0.70) |
s43 | **s25** plus the three-value mention label | 2 + N | 179.0 | 8.3 | 95.06 (N=3, **p = 0.10 = floor**) | 94.18 (**p = 0.10 = floor**) |
s44 | **s25** with only the label's case grading merged | 2 + N | 181.7 | 10.0 | −0.9 vs paired s25 (N=6, **p = 0.05**) | −0.5 (p = 0.21) |
**s45** | **s25 with the coreference batch set to the judges' (25, not 10)** | **2 + N, 26% fewer calls** | **182.2** | 9.2 | **−0.2 vs paired s25 (N=6, p = 0.52)** | **−0.0 (p = 0.91)** |
s46 | s25 with the alias table no longer suppressing partial names | 2 + N | 179.3 | 13.5 | −1.5 (N=6, **p = 0.00**) | −1.0 (**p = 0.02**) |
**s47** | **s25 with the grounded identity review removed (one LLM stage fewer)** | 2 + N | **187.7** | 17.0 | **+0.2 vs paired s25 (N=6, p = 0.53)** | **+1.3 (p = 0.01)** |
**s48** | **s25 with 8 condition copies merged into 3 predicates; no stage removed** | 2 + N | 182.2 | 8.8 | **+0.3 (N=6, p = 0.50)**; composition −0.2 (p = 0.59) | +0.2 (p = 0.57) |
**s49** | **s47 and s48 composed** | 2 + N | **187.5** | 19.2 | **−0.2 (N=6, p = 0.50)** | **+0.9 (p = 0.03)** |

## Result: s38 reaches parity, and is one judging prompt smaller

`s_linker38` stops trying to spend fewer calls and uses the law constructively.
Precision comes from *independence* between two verdicts -- and independence does
not require two different prompts. s_linker25 runs two judging passes whose prompts
differ only in a focus sentence (relevance, then uniqueness); s38 runs **one** prompt
carrying both criteria, samples it **twice**, and keeps a link only when both samples
approve on both criteria.

Six runs each side, exact two-sided permutation over all 924 splits:

| | s25 | **s38** | difference |
|---|---|---|---|
| link-judging prompts | 2 | **1** | |
| LLM calls / run | 89 | 88 | |
| TP | 180.8 | **182.0** | +1.17, p = 0.429 |
| FP | **4.8** | 7.2 | +2.33, p = 0.175 |
| macro F1 | 96.42 ± 0.43 | 95.95 ± 0.29 | −0.47, **p = 0.071** |
| macro F2 | 95.37 ± 0.57 | **95.58 ± 0.29** | +0.20, p = 0.457 |

**No difference reaches significance on any measure, F1 included**, and the
relevance/uniqueness *pass* distinction leaves the architecture: one judge, one
rubric, one prompt, and a self-agreement gate.

### RETRACTED — s38's mechanism is not what holds that parity

Auditing s38's own six runs (`approach/pilot/s38_audit.py`,
`results/s38_audit/README.md`) shows the self-agreement gate is nearly inert: the
two samples split on **1.0 of 174.7 judged candidates per run (0.6%)**, and ANDing
rather than ORing those splits is worth 0.7 false positives against 0.3 true
positives. Taking `s_linker36` — the same merged prompt asked **once** — to six
runs then closes the direction: **macro F1 −0.7 (p = 0.01), FP +3.5 (p = 0.01)**,
TP +0.8 (p = 0.44), macro F2 +0.0 (p = 1.00), 79 calls against 89. s38 sits between
the two, and its F1 p = 0.071 is a smaller version of the same loss, not parity in
a different design.

The audit also explains *why*, from the judges' own verdicts:

| | judging arrangement | FP/run | verdicts that disagree | unanimous rejections |
|---|---|---|---|---|
| s25 | two prompts, one focus each | **4.8** | **4.7 of 172.3 (2.7%)** — 3.7 not gold | 11.3 |
| s38 | one merged prompt, two samples | 7.2 | 1.0 of 174.7 (0.6%) | 11.3 |
| s36 | one merged prompt, one sample | 8.3 | — | — |

The merge preserves the unanimous rejections exactly and loses the disagreements,
and the 3.7 false positives s25's disagreements remove are the 3.5 by which it
leads s36. **Independence has to come from asking a different question, not from
resampling the same one** — two focuses are 4.5× more independent than two samples
of one prompt. That is the sixth measurement of independence in this workflow, and
the sharpest: it retires the entire s32–s38 line in one number.

### The one simplification that looked free, and was not

The audit also found the **mention label over-specified**: three of its five values
are approved at 96.9 / 100.0 / 100.0%, so grading the *case* of a stated name
changed no verdict. Collapsing five values to three is genuinely free on the
merged-judging base — `s_linker42` = s36 + the label reads TP ±0.0 (p = 1.00),
F1 −0.1 (p = 0.50) over three runs a side — so it was lifted onto s25 as
`s_linker43`, and **there it costs 1.3 F1 and 1.3 F2, both at the n=3 p-floor of
0.10** (the most extreme of all ten labellings on both scores).

That is the sixth time an arm measured neutral in one composition comes out
negative in another, and the first where *both* compositions were end-to-end. Equal
approval rates per label value are a **screen, not a proof**: they aggregate over
cases, and rewriting the field changes the prompt for 132 cases per run.
`s_linker44` splits the failed pair — it merges only the case grading (five values
become four) and keeps the field for every candidate, which is the half the traces
actually support — and **that half fails too**, at six paired runs: TP +0.3
(p = 0.87), FP +1.2 (p = 0.55), **macro F1 −0.9 (p = 0.05)**, macro F2 −0.5
(p = 0.21). The loss is jabref −3.94 F1 (FP 0.2 → 1.7 on a 13-sentence document that
carries the highest share of merged-value pairs, 20 of 78) and teastore −1.57,
against teammates +1.89.

**And the first three of those six runs read the opposite**: TP +2.0, FP −0.7, F1
−0.0 (p = 1.00), F2 +0.3, with s44 holding the tighter within-arm spread. Runs 4–6
inverted it. Three runs of this pipeline manufacture a neutral as easily as a
regression; **six paired runs is the bar** — the same lesson as the earlier
over-tight ±0.1 band, in the other direction.

So the label audit produced no adoptable change. Its *readings* were correct — the
second sample really splits on 0.6% of cases, the three values really are approved at
96.9 / 100.0 / 100.0% — and neither licensed a removal, because rewriting the field
changes the prompt for every case that carries it (132 per run) and this model's
verdicts move with prompt text that carries no new information. Seventh instance of
an arm passing a screen and failing composition, and the first where the screen was
trace-based rather than stage-based.

## Two alias judges with distinct purposes: tested, and not supported

The alias judging invites a two-judge design whose purposes are separately
defensible: **validity** (can this phrase name this component -- context-free,
lenient) and **usage** (does a passage use it as one -- in context, strict).
`s_linker39` and `s_linker40` implement it, the usage judge riding inside the reading
at no extra call, the two verdicts unioned so each admits what it is competent to
see.

It does not work, and the reason is specific: **the usage judge is dominated.** On
four of the five projects it approves nothing the validity judge had not already
approved, so the union is inert. On the fifth it contributes exactly two terms of its
own -- `core` and `outer shell` on JabRef -- both generic phrases the validity judge
correctly rejected, and both become false positives (JabRef precision 90.0 against
s25's 100.0). s40 reads macro F1 93.5.

So the usage question, asked four different ways -- confirmation of use
(`s_linker30`), review before the document (`s_linker35`), and union with a validity
judge (`s_linker39`, `s_linker40`) -- never once admits a real alias the validity
judge missed. Intersecting instead over-rejects, costing a third of MediaStore. The
alias judging is **one** judgment, and it is the lenient context-free one.

## Definitive comparison: s37 against s25, six runs each, exact permutation test

The band comparisons used elsewhere in this file compare a three-run mean against a
six-run band. For the closest candidate that is not good enough, so `s_linker37`
was run six times and tested against s_linker25's six runs directly -- exact
two-sided permutation over all 924 splits of the twelve per-run macro scores.

| | s25 | s37 | difference |
|---|---|---|---|
| judging calls | 2 per batch | **1 per batch** | |
| LLM calls / 5-project run | 89 | **79** | −11% |
| prompts | 3 | 3 | |
| TP | 180.8 | **182.2** | +1.4 |
| FP | **4.8** | 8.8 | +4.0 |
| macro F1 | **96.42 ± 0.43** | 95.65 ± 0.49 | **−0.77, p = 0.017** |
| macro F2 | 95.37 ± 0.57 | **95.46 ± 0.54** | **+0.08, p = 0.810** |

**On F2 the simpler architecture is indistinguishable from s25** (p = 0.81, nominally
higher) **with better recall and 11% fewer calls. On F1 it is significantly worse**
(p = 0.017). That is the exploration's final answer, and it is a measurement rather
than a band judgement: F1 parity at lower cost is not available in this design
space, and F2 parity is.

`s_linker37` is `s_linker36` plus a committed quote per criterion -- the device worth
35 links elsewhere in this workflow, applied inside the merged call to make its two
answers as expensive as two dedicated passes. It did not recover the precision
(FP 8.8 against s36's 9.3, within noise), which is what pins the law below.

## The general law the twelve variants establish

| | consolidation | TP | FP | F1 | F2 |
|---|---|---|---|---|---|
s25 | — (every decision its own call) | 180.8 | **4.8** | **96.42 ± 0.42** | 95.37 ± 0.57 |
**s36** | full-name judge: 2 criteria in **1 call** (M calls, not 2M) | **181.7** | 9.3 | 95.52 | **95.35 (in band)** |
s34 | alias judging carried by extraction | **181.7** | 10.7 | 95.11 | **95.20 (in band)** |
s31 | alias pass reviews itself | 178.7 | 9.7 | 94.09 | 93.94 |

Every consolidation of two LLM decisions into one call moves the same way:
**recall up, precision down.** Five independent instances -- self-review (s31),
carried judging under three thresholds (s32-s34), and merged judging criteria (s36)
-- and not one exception. Splitting a decision across calls buys precision; merging
buys recall. s25 sits at the precision corner of that frontier, and F1 rewards it
there; F2 does not distinguish it from the two best simplifications.

s36 is the cheapest point found: **79 LLM calls per five-project run against s25's
89**, F2 statistically identical (95.35 against 95.37), recall higher (181.7 against
180.8), F1 short by 0.90.

## The frontier, and why s25 sits on its corner

s35 was the last lever: ask the carried review first, before the document, so it is
not answered by a model that has already spent itself extracting. It works as
intended on precision -- 8.3 false positives, the best of the carried line -- and
destroys recall, MediaStore falling to 61.3% again, because a review answered before
the document is context-blind and rejects the interchangeable aliases (`Database`
for `DB`) that carry a third of that project.

That completes a clean trade-off surface. Every arrangement of the carried review
lands on one side or the other:

| arrangement | FP | TP | failure |
|---|---|---|---|
review after the document, any batch (s32) | 13.0 | 181.3 | over-approves |
review after, majority (s33) | 13.0 | 181.3 | over-approves |
review after, unanimous (s34) | 10.7 | **181.7** | over-approves, less |
review before the document (s35) | **8.3** | 162.0 | rejects real aliases |
**dedicated call (s25)** | **4.8** | 180.8 | — |

s25's judge is the only arrangement that is strict *and* keeps the interchangeable
aliases, and it manages that because it combines three things no single call
reproduced across eleven attempts: it is answered with **undivided attention**, it
is **context-free** (its prompt carries the component list and the proposed mappings,
not the document, so it judges the phrase rather than the passage), and its rubric is
**lenient** ("When uncertain, prefer APPROVE"). Carrying it after the document buys
leniency and loses strictness; carrying it before buys strictness and loses the
leniency that context-free judging needs to keep real names.

**Conclusion.** Eleven variants map the space. The simplification is available and
real -- s34 is one prompt and one LLM call cheaper, matches s25 on F2, and beats it
on recall -- but no simpler architecture matched s25 on F1. Further permutation of
thresholds and orderings was stopped deliberately: those are small integers and word
orders fitted against five projects, and tuning them until F1 lands in band is the
benchmark-fitting this investigation flags elsewhere as a validity threat.

## Result: a simpler architecture that matches F2 and beats recall, and is still short on F1

The s32-s34 line is **strictly simpler than s25** -- one prompt and one LLM call
fewer (1 + N against 2 + N) -- and it keeps all four load-bearing properties:
whole-document proposal, semantic judgment, lenient rubric, and independence from
the proposer. The judging is not removed; it is *carried* by the extraction calls,
which already run, already receive the candidate list, and did not produce it.

| | s25 | s34 |
|---|---|---|
| calls / prompts | 2 + N / 3 | **1 + N / 2** |
| TP | 180.8 | **181.7** |
| FP | **4.8** | 10.7 |
| macro F1 | **96.42 ± 0.42** | 95.11 (one run 96.04, inside) |
| macro F2 | 95.37 ± 0.57 | **95.20 — inside the band** |

So on the recall-weighted measure the simpler architecture matches, and on recall
it wins. The whole remaining gap is precision: ~6 extra false positives.

**Why the precision gap resists.** A review carried by a pass whose main job is
extraction over-approves -- it keeps what it is offered. The approved lists show it
directly: `back end`, `front-end`, `datastore`, phrases s25's dedicated judge
rejects. Three thresholds were tried on the same architecture, and they move the
gap without closing it: any-batch 13.0 FP, majority 13.0, unanimous 10.7. What a
dedicated call buys is not independence -- s32-s34 have that -- but **undivided
attention**: a model asked only to judge a list judges it more strictly than one
also asked to extract from a document.

Further threshold tuning was stopped deliberately. The knob is a small integer
fitted against five projects, and moving it until F1 lands in band is exactly the
benchmark-fitting this investigation flags elsewhere as a validity threat.

## Six earlier attempts, and why each failed The exploration is closed: **no simpler
architecture reaches s25 on this benchmark.** But the six failures have one
explanation, not six.

## The unifying principle: separate, semantic, lenient, independent — and undivided

Each attempt removes one property of s25's alias judging, and each loses in the way
that property predicts:

| Attempt | Property removed | Consequence |
|---|---|---|
s26, s27, s28 | the **global view** the proposal needs | recall: abbreviations defined once and used far away are missed |
s29 | **semantics** (a lexical check instead) | recall collapse: `Database` for a component named `DB` is never defined in one sentence, so there is nothing to check |
s30 | **leniency** (in-context confirmation instead) | same collapse: the passage uses those aliases as ordinary noun phrases |
s31 | **independence** (the proposer reviews itself) | recall is nearly matched (178.7 vs 180.8) and precision halves (FP 9.7 vs 4.8): a proposer approves its own list |

s31 is the informative one. It keeps the global view, the rubric verbatim, the same
model and the leniency, and it recovers almost all the recall the other attempts
lost — and it still loses 2.3 F1, entirely on precision. **What a separate judge
buys is not a better rule but a second, independent look.**

That is not a new principle in this workflow; it is the third measurement of the
same one. The partial-name judge's first step withholds the target from the model
precisely so the step tests identity instead of confirming it — worth 12 false
positives. Requiring the link judges to quote before ruling is worth 35 links. Now
the alias judge: removing its independence costs 4.9 false positives.

**Independence of a judging step from the step it judges is load-bearing, measured
three ways.** That is the paper-level finding of this exploration, and it is worth
more than the call it would have saved.

## Why none of them reaches s25

**The two questions have opposite optimal granularities.** That is the finding, and
each variant demonstrates one half of it.

**References want a small window.** s27 puts the whole document in one call and its
accuracy tracks document length almost monotonically:

| project | sentences | s27 macro F1 |
|---|---|---|
jabref | 13 | 100.0 |
mediastore | 37 | 98.4 |
teastore | 43 | 96.3 |
bigbluebutton | 87 | **79.7** |
teammates | 198 | **84.1** |

On teammates the single call reported **50 references where four 50-sentence
batches report about 89**. Batching is not a prompt-size workaround; it buys
thoroughness, and its value grows with the document.

**Names want the whole document.** s26 keeps the batching and loses exactly the
short forms that a document defines once and uses far away — `ui`, `webui`, `e2e`,
`gae`, `test driver`, `akka-apps` are found only by the global pass. A 50-sentence
window sees the use and not the definition.

So one pass cannot serve both. Give it a small window and it misses the
definitions; give it the whole document and it misses references. **s25's two
stages are the two granularities**, and that is why the separation survives.

## What s28 rules out

The s26 diagnosis (`pilot/s26_diagnosis.py`) found the largest single effect in a
stage the merge does not touch: the partial-name linker lost nearly half its links
(TP 14.0 → 7.7), because the alias table both *admits* full-name candidates and
*suppresses* partial-name ones, so a larger table narrows the strict linker. s28
removes that second role — one condition fewer — and recovers nothing
(F1 93.89 against s26's 94.27). The dual-role interaction is real and worth
stating, but it is not what the merge costs.

## The one simplification that holds: retire a constant, not a stage

Every attempt above removes something the workflow *does*. The change that survives
removes something it *assumes*. s25 carries three batching constants — 50 sentences per
extraction call, 25 candidates per judging call, 10 sentences per coreference-resolution
call — and the third has no counterpart anywhere and makes coreference resolution 46% of
all calls. `s_linker45` sets it to `JUDGE_BATCH`, by unification rather than search.

Six paired runs, s25 and s45 in the same invocations:

| | s25 | **s45** |
|---|---|---|
| batch constants the paper must justify | 3 | **2** |
| calls / run | 88.8 | **65.3 (−26%)** |
| coreference resolve calls | 40.0 | **17.0** |
| TP | 181.3 | **182.2** (p = 0.56) |
| FP | 7.0 | 9.2 (p = 0.34) |
| macro F1 | 96.11 | 95.91 (**p = 0.52**) |
| macro F2 | 95.47 | 95.44 (**p = 0.91**) |

No project collapses (mediastore +0.26 F1, jabref +0.43, teammates −0.26,
bigbluebutton −0.46, teastore −0.96), and recall is the higher of the two. `s_linker27`'s
passage-length effect does not reach from 10 to 25 sentences for this question:
resolving a back-reference needs the sentences either side of the target, not a short
window.

**So the design space has exactly one free move left in it, and it is not architectural.**
Every stage, every judge, every rubric and every field is priced; what was not priced was
a magic number, and it was worth a quarter of the cost.

## The dual role of the alias table is load-bearing in both directions

`s_linker46` gives the table one role — it still admits full-name candidates but no
longer suppresses partial-name ones. Sized first off six real tables: 59 → 75 candidates
over the five projects, 3.8 gold per run. Measured: TP −2.0 (p = 0.39), **FP +6.5
(p = 0.01)**, **macro F1 −1.5 (p = 0.00)**, macro F2 −1.0 (p = 0.02).

**Freeing 16 candidates cost 2.0 true positives.** Adding candidates cannot remove a
link directly, so the loss is batch composition in the two-step partial-name judge — the
same mechanism the `_unlinked` arm measured in the other direction (−6.8 FP purely from
changing which cases share a batch). The table opens one gate and closes another, and
removing the second role costs 1.5 F1; state it as a property of the design, not an
accident of implementation.

## The third constraint: curation must be lenient, and it is load-bearing

s29 and s30 both keep the two granularities and only change *how* the alias table
is curated — s29 checks the model's evidence sentence lexically, s30 asks the
extraction pass to confirm which aliases it saw used. Both collapse on the same
project:

| | mediastore recall | aliases kept |
|---|---|---|
s25 (judge) | 96.8% | `Database`, `DataStorage`, `AudioAccess`, `ReEncoder` |
s29 (grounding check) | **61.3%** | none survive the check |
s30 (folded confirmation) | **61.3%** | **0 of 4 confirmed** |

MediaStore's four aliases carry **10 of its 30 links, all gold** — the highest
alias dependence of any project (29 alias-only links across the five, 23 gold).
And they are the hard kind: the document says `Database` where the model says `DB`,
interchangeably, **never defining the equivalence in any sentence**. So:

- a check for an establishing sentence rejects them (s29) — the definition does not
  exist to be quoted;
- asking the reader to confirm "aliases it saw used as a name" rejects them (s30) —
  the passage uses them as ordinary noun phrases.

What the judge actually contributes is a *lenient semantic* decision — its rubric
ends "When uncertain, prefer APPROVE" — and that leniency is what keeps a third of
MediaStore's links. Neither a lexical test nor an in-context confirmation
reproduces it.

## What this buys the paper

The knowledge module can now be defended as **necessary rather than chosen**:

> A single reading cannot serve both questions. Extraction accuracy degrades with
> the length of the passage read (F1 98.4 at 37 sentences, 79.7 at 87, 84.1 at
> 198), so references must be read in small windows; alias definitions are stated
> once and used far away, so names must be read from the whole document. The two
> stages are those two granularities, and merging them costs 2.2 F1 (s26), or 4.7
> F1 if the merge also drops the batching (s27).

Three implemented alternatives is a stronger justification than any ablation of
the existing design, and it answers the reviewer question — "why two stages that
ask the same thing?" — with a measurement instead of a rationale.

## Artifacts

- `approach/src/llm_sad_sam/linkers/experimental/s_linker2{6,7,8}.py`, all
  `experimental=True`, none promoted
- `approach/pilot/test_s26_unified.py` — asserts every stage after the reading is
  s25's byte for byte (31 methods, 9 rubrics, 7 bounds, both deterministic
  generators), so the comparisons measure the architecture and nothing else
- `approach/pilot/s26_diagnosis.py` — the per-linker attribution
- runs: `results/s26_unified_e2e_r{1,2,3}_20260812`,
  `results/s27_singlecall_e2e_20260812`,
  `results/s28_nosuppress_e2e_r{1,2,3}_20260812`
