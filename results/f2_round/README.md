# The F2 round — recall is the binding constraint, and the metric decides the arm

Primary metric **F2**, secondary **F1**, both macro-averaged over the five
projects. Every comparison below puts its arms in the *same* invocation, because
absolute levels drift between invocation sets.

## 0. Price the metric before building anything

At the head's operating point (macro P 87.1, R 97.0):

| change | F2 |
| --- | ---: |
| +1.0 precision | +0.24 |
| +1.0 recall | **+0.76** |

Recall is worth **three times** precision per point. This one line explains every
verdict in this round, and it costs nothing to compute.

## 1. The merged proposer: accepted on F1, refused on F2

Composed, five projects, terra, paired invocation.

| variant | P | R | F1 | F2 | calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| `s_linker90` (head) | 87.1 | **97.0** | 91.5 | **94.6** | 16.4 |
| `s_linker97` (merged reading) | 91.5 | 94.2 | **92.7** | 93.6 | 14.0 |

The same arm is a win on the secondary metric and a loss on the primary one, and
it is 15% cheaper. **Name the primary metric before the arm is built**, or the
cheaper architecture wins the argument on the wrong number.

## 2. Where the head's remaining recall actually goes

The head has **8 false negatives across all five projects**, and **6 are on
sentences that write the component's name** — not coreference at all.

Tracing them to the judge that refused them, on teammates in one recorded run:

| component | coreference judge | named-mention judge |
| --- | ---: | ---: |
| `Logic` | rejected **13** | approved **33** |
| `Storage` | rejected **6** | approved **24** |

with objections of the form *"the Logic component is named directly in the
sentence rather than being a referring expression to an unnamed component"*.

Each judge answered its own question correctly. The pipeline asked "is this link
correct?" and the coreference judge answered "is this a coreference?". When the
extractor had not independently proposed the same pair, a correct link died on a
type technicality — silently, because a rejection leaves no trace.

`s_linker103` acts on this without a prompt change, a merge, or an extra call:
candidates are routed to a judge by the deterministic name relation the code
already computes, applied **after** both proposers have spoken so their mutual
blindness is preserved.

**Measured, and it does not pay under F2:**

| variant | P | R | F1 | F2 | calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| `s_linker90` | 87.4 | 96.0 | 91.0 | 93.8 | 16.6 |
| `s_linker103` | **91.6** | 94.5 | **92.9** | 93.8 | 16.0 |

Precision +4.2, recall **−1.5**, F2 flat — the opposite of the predicted
direction. Re-routing a candidate away from the specialised judge also removes it
from the cases that judge *would* have approved, and the receiving judge brings
its own differently-shaped scepticism. The diagnosis stands and the remedy does
not: it converts one loss into another. Under F1 it would be adopted (+1.9).

What transfers is the instrumentation, not the fix: **inspect verifier rejections
by objection type.** A verifier refusing correct items for being out of scope is
invisible in aggregate metrics, and it accounted for six of the head's eight
residual misses.

## 3. A third blind proposer wins under F2

Composed, five projects, terra, paired invocation.

| variant | P | R | F1 | F2 | calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| `s_linker90` (two proposers + scan) | 88.2 | 94.6 | 91.1 | 93.1 | 16.2 |
| `s_linker101` (+ the reading, blind) | 87.1 | **97.7** | **92.0** | **95.3** | 25.0 |
| delta | −1.1 | **+3.1** | **+0.9** | **+2.15** | +54% |

Recall rises 3.1 points at flat precision, so the arm wins on the primary *and*
the secondary metric.

**Replicated.** A second paired invocation, same five projects:

| variant | P | R | F1 | F2 | calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| `s_linker90` | 87.4 | 96.0 | 91.0 | 93.8 | 16.6 |
| `s_linker101` | 87.5 | **97.2** | **92.1** | **95.1** | 25.2 |

| run | head F2 | s101 F2 | dF2 | dF1 |
| --- | ---: | ---: | ---: | ---: |
| r1 | 93.1 | 95.3 | **+2.15** | +0.9 |
| r2 | 93.8 | 95.1 | **+1.3** | +1.1 |

Positive on both metrics in both runs, always through recall at flat precision.
A third terra run and a luna run are in flight; note that teammates carries most
of the run-to-run variance (head precision 66.7 in r1 against 60.7 in r2), so the
paired, same-invocation comparison is the only valid read.

**This contradicts the current literature**, which puts the marginal value of a
third independent extractor at ≈ +0.01 recall and reports interpretation-diverse
ensembles plateauing at N=3. Two differences are candidates: our third look asks a
genuinely *different question* rather than a paraphrase, and we aggregate by
**union**, where the surveyed systems mostly aggregate by majority vote — which
the same survey identifies as destroying ~90% of realisable ensemble gain.

## 4. Membership framing: refuted, against the literature's prediction

The 2026 result that models judge set membership far better than they author a set
(F1 0.60–0.77 vs 0.26–0.48, a gap stable over a 24× parameter range) predicts that
turning the proposer into a checkbox should raise recall. It does the opposite
here.

| arm | proposals | gold | spurious | precision |
| --- | ---: | ---: | ---: | ---: |
| control | 52.5 | 37.8 | 14.7 | 0.721 |
| `s_linker102` (membership) | 34.5 | **31.3** | 3.2 | **0.908** |

Forcing a decision per component made the model markedly more conservative: the
largest gold loss of any arm measured this session, bought with precision that F2
does not pay for. Recorded as refuted on this pipeline.

## 5. Two measurements that stand alone

**Stage independence.** Jaccard between the two proposal stages' outputs, against
the value expected if they drew independently at their own rates:

| | observed | expected under independence | ratio |
| --- | ---: | ---: | ---: |
| our two stages (different questions) | 0.409 | 0.051 | **12×** |
| AMA (ICLR 2023), paraphrases of one question | 42.2 | 0.25 | **169×** |

Asking two genuinely different questions buys an order of magnitude more
independence than rephrasing one question does.

**Oracle-union ceiling (recall).**

| extractor | resolver | union | union over 3 runs |
| ---: | ---: | ---: | ---: |
| 83.0 | 66.6 | 95.5 | 97.1 |

The head's composed recall is 97.0, above the two LLM stages' union of 95.5 — so
the deterministic scan is already serving as a third blind proposer, for free.
