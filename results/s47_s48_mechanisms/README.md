# Simplifying mechanisms and conditions, not calls — 2026-08-13

`s_linker45` cut a quarter of the calls by retiring a tuned constant, which is a cost
result, not a design one. This round goes after the design: one variant removes an
**LLM stage**, the other removes no stage at all and instead merges the **conditions**
the design is spelled out in. Both were sized deterministically off six recorded s25
runs before anything was launched, and both were then run **paired** — s25 and the
variant in the same invocation — six times.

**Both hold.** This is the first mechanism removal in the whole series that costs
nothing, and the first condition merge that is provable rather than argued.

Scripts: `approach/pilot/test_s47_s48_mechanisms.py` (174 checks, no LLM calls),
`approach/pilot/score_runs.py`. Runs: `results/s4748_e2e_r{1..6}_20260813`.

## s47 — one mechanism out: the grounded identity review

s25 judges a partial-name candidate twice. Step 1 asks, with the target component
**withheld**, whether the expression itself denotes a software participant; that
target-blindness is worth 12 false positives and stays. Step 2 then shows the target
plus the sentences naming it and asks whether the two denote the same participant.
Step 2 had never been priced on its own. Off s25's recorded decisions:

> 20.3 candidates per run reach the identity review. It keeps 12.3 (12.2 gold) and
> rejects 8.0 — **of which 5.5 are gold.**

So it trades recall for precision at a bad rate. Removing it makes the partial-name
linker the same shape as the coreference linker — one proposer, one judging call — and
deletes a prompt, the anchor bookkeeping, a four-conjunct evidence gate and the
`alternative` response field.

| | s25 | **s47** | p |
|---|---|---|---|
| judging calls, partial-name | 2 | **1** | |
| calls / run | 89.2 | 87.2 | |
| TP | 181.5 | **187.7** | **0.00** |
| FP | 10.2 | 17.0 | **0.00** |
| macro F1 | 95.66 | **95.84** | 0.53 |
| macro F2 | 95.31 | **96.64** | **0.01** |

**F1 is a wash; F2 is significantly better.** s47's F1 spread is also the tighter of
the two (0.74 against 1.54).

Per project, the review helps exactly one document and hurts the rest:

| project | s25 F1 / F2 | s47 F1 / F2 | |
|---|---|---|---|
| bigbluebutton | 91.34 / 88.41 | **94.15 / 94.75** | TP 53.7 → **59.0** |
| teastore | 99.37 / 99.00 | **100.00 / 100.00** | |
| jabref | 99.10 / 99.63 | 99.55 / 99.82 | |
| mediastore | 98.63 / 97.84 | 98.36 / 97.40 | |
| teammates | **89.84** / 91.69 | 87.15 / 91.21 | FP 8.0 → **12.5** |

It buys precision on the one project that proposes many partial names and costs recall
on the one where partial names carry real links. That is a defensible thing to say
about a step, and a good reason not to keep it: a mechanism that is right on one
document out of five is fitted to that document.

## s48 — no mechanism out: eight condition copies become three predicates

Every stage, prompt, rubric and LLM call is s25's byte for byte. What changes is how
many *conditions* the design has to be described in. All four merges were sized off the
recorded runs first, and each is a provable identity rather than a behavioural claim.

| what was duplicated | where | becomes |
|---|---|---|
"does this sentence state a name of this component?" — the identical expression `any(_find_exact_form(text, n) for n in (name, *aliases))` | the full-name admission filter, the partial-name whole-name exclusion, the coreference antecedent gate | `_states_a_name` |
"is the model's quote really in the sentence?" — two copies of the same normalise-and-substring test | both partial-name judging steps | `_claim_supported` |
"which sentences are near this one?" — two `abs(...) <= C` filters and one `range(max(1, n-C), n+C+1)` walk against the sentence map | denotation, identity review, coreference resolver | `_window` |
three conjuncts that never fire | the identity review's evidence gate | one conjunct |

The mention-label classifier asks the first question too, but *decomposed*, because it
must know which of the two matched — so it keeps its own two calls. That asymmetry is
deliberate and stated.

On the dead conjuncts: the identity review approved only on a listed anchor **and** a
quoted sentence **and** a named alternative, and `evidence_valid` was False **zero**
times in 122 recorded cases over six runs. Two of the three are deleted by exactly the
argument the paper already uses for not adding the claim check to the other judges
("voids 0 verdicts in 25 project-runs"). The substring check stays, because it does
fire — 0.2 denotation verdicts per run. **The prompt is untouched**: the model is still
asked for all three. What goes is code that re-checked two answers and never once
caught one. Demanding a commitment is worth 35.2 true positives in this workflow;
verifying it is worth nothing here, and the two are separable.

| | s25 | **s48** | p |
|---|---|---|---|
| condition copies | 8, in 5 shapes | **3 named predicates** | |
| TP | 181.5 | 182.2 | 0.65 |
| FP | 10.2 | 8.8 | 0.50 |
| macro F1 | 95.66 | 95.96 | 0.50 |
| macro F2 | 95.31 | 95.53 | 0.57 |
| **composition** | | **−0.2** | **0.59** |

The composition statistic is the one that matters for a merge: at **−0.2** the two
arms' link sets differ *less* between arms than within them. When the only remaining
source of difference is the model's own nondeterminism, that is what the number should
look like — and it is what the design predicted, since every prompt renders byte for
byte identically (asserted on real project data for all seven prompt builders).

## What the test pins down

`test_s47_s48_mechanisms.py`, 174 checks, no LLM calls:

* s47: both identity methods gone, no identity prompt anywhere in the file, the
  target-blind step and 13 other methods byte-identical to s25's;
* s48: the merged predicates in place and every copy gone; the two dead conjuncts gone
  from the gate **while the prompt still asks for all three answers**; 34 methods, 10
  rubrics and 7 bounds byte-identical;
* `_states_a_name` against the expression it replaced over every (name, sentence) pair
  on all five benchmarks — 3697 pairs, 0 flips;
* `_window` against **both** old spellings over all 378 sentences — 0 divergences — and
  the coreference prompt's marked context strings compared byte for byte;
* all seven prompt builders rendered on real project data and compared byte for byte.

## Composed: `s_linker49`

Composition is checked, not assumed — this workflow has seven instances of an arm that
held alone and failed in another composition. Here it holds, and it replicates s47's
numbers almost exactly, so the two changes are independent and additive. With the
identity review gone, `_claim_supported` has one call site left and is deliberately not
carried: a one-site helper is not a simplification.

Six paired runs (`results/s49_composed_e2e_r{1..6}_20260813`):

| | s25 | **s49** | p |
|---|---|---|---|
| partial-name judging steps | 2 | **1** | |
| judging steps, whole workflow | 5 | **4** | |
| condition copies | 8, in 5 shapes | **2 named predicates** | |
| calls / run | 89.2 | 87.2 | |
| TP | 182.5 | **187.5** | **0.01** |
| FP | 12.0 | 19.2 | **0.01** |
| macro F1 | 95.66 | 95.43 | 0.50 |
| macro F2 | 95.55 | **96.43** | **0.03** |

**F1 is statistically unchanged; F2 is significantly better.** So s49 is a free
simplification under an F1-led paper and an improvement under an F2-led one — and
either way the design it describes is smaller: *two of the three linkers judge in one
step, and two named conditions replace eight copies.* The full-name judge keeps its two
focused calls, because merging **those** is the one thing this series measured as
significantly worse (s36: F1 −0.7, FP +3.5, both p = 0.01).

## What this round changes about the design story

Nineteen earlier variants established that nothing could be removed without cost, and
the one that held (`s_linker45`) retired a constant rather than a mechanism. This round
finds the exception and explains why it was missed: the identity review was never
ablated alone, only ever measured as half of "the partial-name linker's two-step judge",
and the half that carries the measured value (12 false positives) is the *other* half —
the target-blind denotation step. Independence is what that step provides, and it
provides it whether or not a second step follows.

The condition merges are a different kind of result and worth stating as one: a design
can be described in far fewer conditions than it is written in, and the reduction is
verifiable rather than argued. Eight copies in five shapes were three questions all
along.
