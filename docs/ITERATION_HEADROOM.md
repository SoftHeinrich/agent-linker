# Self-loops, re-reading and feedback: measured before they were built

This branch has no loop anywhere. The head is a strict DAG, and `approach/CLAUDE.md`
records nothing on iteration, feedback or multi-pass reading — the only multi-turn
artifact is `s_linker24_role_orchestrator`, a controller that selects tools once.
So the whole family was open, and the obvious hypotheses were worth pricing at
level 1 of this branch's measurement policy before any of them was built.

All four were priced deterministically, on recorded checkpoints, **at zero API
cost**. Three are refuted and the fourth is inside the noise floor.

## H1 — a loop that accumulates aliases recovers what one draw missed

**Refuted. Zero gain.**

The alias table is genuinely unstable: over three runs it holds 34 distinct terms
of which 27 appear in all three (79% stable), and teammates alone swings between
8, 8 and 10 terms with a Jaccard of 0.50. That instability is real, and it is
what makes an accumulating loop look attractive.

It buys nothing. Gold recall of the deterministic scan:

| alias table | scan gold recall |
| --- | ---: |
| mean single run | 0.897 |
| best single run | 0.897 |
| **union of all three runs** | **0.897** |

The seven unstable terms reach no gold pair that the stable twenty-seven do not
already reach. A loop converging on the union of alias tables converges on the
same link set. `pilot`-style script: `scratchpad/loop_headroom.py`.

## H2 — a second reading pass recovers what the first never proposed

**Largely redundant with a stage that already exists.**

The deterministic scan cannot reach 20 gold links. Of those, pooled over three
runs (60 instances):

| | share |
| --- | ---: |
| already recovered by the **coreference linker** | 68% |
| already recovered by the partial-name linker | 10% |
| **missed by the head entirely** | **22%** (≈4.3 links per run) |

So the re-reading idea is mostly a re-description of the coreference stage. The
genuine headroom is 4.3 links per run, and it is not distributed randomly: over
half of it is three sentences of one document, BigBlueButton S19–S21.

## H3 — a stateful sequential read carries the signal those links need

**The signal is real; the decision rule is not.**

The signal is striking. Of the 20 gold links the scan cannot reach, **19 sit
within five sentences of an established mention of the same component**, and 13
of those exactly one sentence back. The documents describe a component in topical
runs, and the continuation sentences are `It …`, `This means …`, `As such …`.

BigBlueButton S15–S21 is the archetype, and it explains the residual precisely:

```
S15  Scalability of HTML5 server component.        <- a heading; establishes the topic
S19  ... a single nodejs process for bbb-html5 ...  gold: HTML5 Server
S20  This means that bbb-html5 could use ...        gold: HTML5 Server
S21  As of 2.3-alpha-7, bbb-html5 uses 2 ...        gold: HTML5 Server
```

The head assigns all three to HTML5 **Client**. `bbb-html5` is a surface both
siblings can claim, and a judge ruling on one pair at a time never sees that the
section is about the Server.

Implemented as a proposer — carry the last established component onto sentences
that state no name — it does lift recall, from 0.897 to 0.928 at a one-sentence
carry with a referring expression required. But its marginal precision is 0.19
(+6 true, +26 false over five projects), which is well below any proposer already
in the pipeline.

## H4 — discourse state breaks the sibling ties the judge gets wrong

**Refuted: inside the noise floor.**

This was the promising one, because the sibling confusions and the residual false
negatives are the same phenomenon seen from two sides. Over the 50 wrong-component
errors the head makes in three runs, taking the most recently established
component as the tie-breaker:

| | share |
| --- | ---: |
| topic **is** the predicted (wrong) component — makes it worse | 42% |
| topic **is** the gold component, not the predicted one — fixes it | 32% |
| no topic established in the previous 7 sentences | 12% |
| topic includes both / is a third component | 14% |

Net **negative**. Restricting the fact to the cases it can legitimately speak
about — staying silent when the sentence writes a full name of the predicted
component, since such a sentence says what it is about — flips the sign but not
the verdict: of the 38 errors it then speaks about, it fixes 15 and worsens 12,
**net +3 over three runs, or +1 per run.**

This branch's own null arms read TP −4.8 / F1 −0.7 (`s_linker49_null`) and
F1 −1.58 / FP +10.7 (`s_linker75_null`), and the standing run-to-run swing is
±55 links. **+1 per run is not a result.** It is comfortably inside the floor a
delta has to clear.

The reason it fails is instructive: the wrong component is often predicted
*because* it is topical. Topic and error are correlated, so topic cannot
adjudicate them.

## The alternative use of the same budget, measured here

The 2026 literature's sharpest result against loops is that at a matched token
budget, sequential self-refinement loses to **parallel repeated sampling**
("Sample More, Reflect Less", arXiv 2607.28576: Self-Refine and Reflexion trail
best-of-N by 3.6-10.1 points at equal cost). That is directly checkable on this
pipeline, because three paired runs already exist.

Aggregating the head's own three runs by vote:

| decision rule | P | R | F1 |
| --- | ---: | ---: | ---: |
| a link in ≥ 1 of 3 runs | 0.853 | 0.954 | 0.901 |
| **a link in ≥ 2 of 3 runs** | **0.885** | **0.944** | **0.913** |
| a link in all 3 runs | 0.906 | 0.892 | 0.899 |

A majority vote over three independent samples is worth **+1.2 micro F1** over a
single run. Every loop hypothesis above is worth less than that, and the vote
needs no new architecture at all. If the budget for a self-loop exists, the
measured-better place to spend it is parallel resampling.

This also matches the sharpest independence result already on the branch — s25's
two focused calls disagree on 2.7% of candidates while s38's two samples of one
prompt disagree on 0.6%, so *independence comes from asking a different question,
not from resampling the same one*. Both findings survive together: resampling
beats looping, and asking a different question beats both.

## What the literature says about the loop, and where it stops short

The strongest positive evidence for state-gated tools — DocAtlas
(arXiv 2608.07527), which beats a human-expert baseline on MMLongBench-Doc, and
SLEUTH (arXiv 2607.12267), whose ablation shows a re-query trigger helps *only*
when structured state already exists — is real, and both are exactly the
"re-scanning with a grown table returns new candidates" mechanism rather than an
analogy to it.

But all of it is at a scale this task does not have. "Is Progressive Disclosure
All You Need" (arXiv 2607.17598) finds the benefit of agent-decided iterative
reading **grows with corpus size**; here the corpus is one document of a few
hundred sentences and a catalog of 10-30 entities, and H1 showed the discoverable
state space is exhausted by a single pass. The counter-evidence is pointed:
"Do Multimodal Agents Really Benefit from Tool Use?" (arXiv 2606.02357) finds
**93-96% of tool-solved problems are solvable without the tool**, and the full
loop beats neither tool-output-alone nor call-format-alone — much of the apparent
benefit of iterating with tools was extra reasoning wearing a tool's clothes.

If a loop is ever built here, the literature names the one ablation that decides
it: **feed a single non-agentic pass the loop's final state** (the converged
alias table and dossier) and see whether it matches the loop. If it does, the
sequential discovery was never necessary — only the richer inputs were.

## What this closes and what it leaves open

Closed, without spending an E2E batch:

- accumulating aliases across iterations (H1, zero gain);
- re-reading for recall (H2, 68% already done by the coreference stage);
- a topic slot as a decision rule or tie-breaker (H3/H4, net +1 per run).

Still open, and untouched by these measurements:

- **The topic as a computed fact in the evidence bundle**, rather than as a rule.
  This branch's design law is explicit — facts stay in code, weighings go in the
  prompt — and the reference-form field is precisely such a computed fact, worth
  −6.6 TP when withheld. "The most recently established component before this
  sentence is X" is the same kind of fact, and the judge would weigh it rather
  than obey it. Given H4, expect a small effect; it is cheap to price at level 2.
- **Contrastive resolution**, from `docs/CORE_ARCHITECTURE.md`: rule on competing
  sibling components together instead of one at a time. It remains the only
  untested idea that attacks the dominant error directly, and the oracle bound on
  it is macro F1 0.933 → 0.957.

## Why the architecture literature does not rescue the loop either

Two structural facts rule out most of the family before any measurement.

**No gradient, no learned halting.** Every architecture that made adaptive
iteration work — ACT (Graves 2016), PonderNet, Universal Transformers, Huginn
(arXiv 2502.05171), Mixture-of-Recursions — learns its halting policy end to end.
This pipeline is training-free by design, so there is no signal to learn one
from, and what remains is a hand-picked threshold inheriting ACT's own admitted
weakness: "the numerical weight assigned to the time cost has to be hand-chosen"
and behaviour is "quite sensitive" to it. Worse, GATE-06/07 forbid tuning such a
threshold against the gold standard, which removes the only way this literature
normally sets it.

**Saturation is fast where it happens at all.** Levenshtein Transformer
self-terminates at an average of **2.43** iterations with 0.1% of sentences
hitting its cap; Self-Refine's gains concentrate in rounds 1–2 of 4;
Mixture-of-Recursions *degrades* past its trained depth. Nothing here suggests a
long loop; at most it suggests one extra pass.

The one design with a real termination proof is **monotone accretion**: accepted
links form a finite lattice under inclusion, an update that only ever adds is
monotone, and Knaster–Tarski gives a least fixed point reached in at most
|candidate pairs| steps. But that proves the set *stops changing*, not that it
stops at the right set — and a self-referential loop conditioning on its own
prior output is exactly the confirmation-bias setup Arazo et al. (arXiv 1908.02983)
demonstrate. Yarowsky's bootstrapping avoided drift only because it had **two
near-independent constraints that had to agree**; a loop that re-asks one model
about its own answers has no such second signal.

That lands precisely on this branch's own sharpest result: s25's two focused
calls disagree on 2.7% of candidates while s38's two samples of one prompt
disagree on 0.6% — **independence comes from asking a different question, not
from resampling the same one.** A self-loop resamples the same question. It is
the weaker of the two things this branch has already measured.

## The classical reading lineage: memory structure beat rereading

The multi-pass reading literature is where this family began, and its own record
settles the question more cleanly than the LLM-era work does.

Extra passes are real but narrowly useful. Memory Networks went 18.5% → 99.9%
from one hop to two on a task that structurally cannot be solved in one; MemN2N
fell 25.1% → 20.3% → 16.3% error over 1 → 2 → 3 hops; Gated-Attention Readers
went 57% → 65.6% → 68.3% and then **flat at K=4**. Dynamic Memory Networks give
the sharpest picture: bAbI tasks 3/7/8 needed five passes to climb from single
digits to 95%+, while tasks 1/4/6/9/12/20 were already solved in one. Rereading
helps *only* the genuinely compositional items.

And then the result that matters most here: **Recurrent Entity Networks
(Hénaff et al., ICLR 2017) read the document once**, maintaining a gated memory
slot per entity updated as it reads, and solved **20/20 bAbI-10k at 0.5% mean
error** — better than DMN+'s three-pass 2.8%. A slot per entity, updated in a
single pass, beat rereading.

That is exactly the structure H3 found signal for and exactly the conclusion H3
reached from the other direction: the topic-run signal is real, but the answer is
a better memory *structure* inside one pass, not another pass.

Two further constraints from this lineage:

- **Adaptive stopping works only with ground truth to train against.** ReasoNet's
  learned termination gate beat a fixed 2-turn model 98.21% to 76.07% on graph
  reachability, with step count tracking true BFS depth — trained by REINFORCE
  against final correctness. Every mechanism that relied on the model's own
  unverified judgment about whether to continue either failed to help or hurt.
- **SUMIE (arXiv 2406.05079) is the closest LLM-era analogue to this system's
  alias table** — incrementally maintaining an entity-attribute table while
  reading a document — and its recall **degrades** across passes: 84% at turn 2
  down to 74.8% at turn 7, with ~45% of extracted values redundant synonyms and
  ~25% unsupported. That is independent corroboration of H1's zero gain, from a
  benchmark built for exactly this shape of task.

## A result the field appears to lack

The reading-architecture literature compares iteration against a *single* pass —
never against an equal-compute ensemble of independent single passes. Where
papers show gains from looping (MemNN, DMN, ReasoNet, GraphRAG's gleaning), the
compute-matched resampling baseline is absent.

H1 and the vote table above are that missing comparison, run on a real pipeline:
the union of three independent alias-mining passes is worth **zero**, and a
majority vote over three independent full runs is worth **+1.2 micro F1** —
more than every loop hypothesis priced here. If this branch ever writes the
self-loop question up, that is the reportable part.

## The recommendation the evidence supports

Not a loop. **One extra pass that asks a different question.**

That is contrastive resolution (`docs/CORE_ARCHITECTURE.md`): rule on competing
sibling components together rather than one at a time. It is a single bounded
pass, it asks something the existing judges structurally cannot be asked, it is
the independent second signal Yarowsky's design needed, and it targets the error
class these measurements keep converging on — the sibling confusions that are
simultaneously 68% of the false positives and the majority of the residual false
negatives (BigBlueButton S19–S21). Its oracle bound is macro F1 0.933 → 0.957.

## Method note

Every number here came from recorded checkpoints and call logs, with no LLM
calls, following the branch's escalation rule: *do not spend a paired
end-to-end batch to answer a question a checkpoint can answer.* Four hypotheses
in the self-loop family were priced and three refused before any of them reached
code. Scripts: `loop_headroom.py`, `loop_signal.py`, `topic_slot.py`,
`who_gets_them.py`, `slot_breaks_ties.py`, `slot_restricted.py`.
