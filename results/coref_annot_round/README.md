# The shortlist-annotation round — the coreference shortlist, marked with the name linker's verdicts

**Question.** `s_linker122`'s resolver prints a per-case shortlist:

```
NAMED BEFORE THIS CASE: kurento (S68), WebRTC-SFU (S68), FreeSWITCH (S66), FSESL (S60)
```

`_named_before` computes it with `_states_a_name` — a purely **lexical** fact: which
sentences above this one write a component's name. By the time the resolver runs, the
name linker has already judged every one of those mentions and `_run_linker` withholds
its verdicts on purpose ("No linker receives the links the earlier one produced"). The
round asks what handing them over is worth:

```
NAMED BEFORE THIS CASE: kurento (S68, linked), WebRTC-SFU (S68, named only), ...
```

**Where it sits in the ledger.** This is the reading round's Rung I question asked at a
different place. `s_linker100` conditioned the second proposer on the first's *output*
and added **zero pairs in two of three samples**, because conditioning is the opposite
of blindness. Here the conditioning is narrower — not which sentences to look at, but
which antecedents to prefer — so it is not settled by that result, and it is not settled
by `s_linker93` either (that filter removed *cases*; this one re-ranks *antecedents*).

Tooling: `pilot/coref_shortlist_audit.py` (level 1, no calls),
`pilot/coref_annot_pilots.py` + `pilot/run_coref_annot.sh` (level 2).

---

## Level 1 — the reach and the ceiling, off recorded checkpoints, no calls

`pilot/coref_shortlist_audit.py` replays the head's own `_resolve_references` with
`_ask` stubbed, so the batching, the window, the sentence table and `_named_before` are
the variant's and not a re-declaration of them. It reads each run's `linker_name.pkl`
for what the name judge **kept** and what it **saw and refused**.

### R1 — the annotation is a near-constant field, and its informative part is one project

Per run, over three recorded `noanchor_e2e_*_20260914` runs a model:

| model | cases | entries | /case | `linked` | `named only` | of which teammates |
|---|---|---|---|---|---|---|
| terra | 378 | 1017.3 | 2.7 | 846.7 (83.2%) | **170.7** | 159.0 (93%) |
| luna | 378 | 1064.0 | 2.8 | 844.7 (79.4%) | **219.3** | 203.7 (93%) |

**83% of entries would be marked `linked`.** mediastore's shortlist is 140/140 accepted
on both models — the annotation is literally a constant there. The informative rows are
teammates and a handful of teastore/jabref entries, which is the shape the union round
warned about: an evidence field that says the same thing in five rows of six.

### R2/R3 — the population an annotation can change is 0.3 net links a run

Every resolution the recorded runs made, split by whether the **antecedent it cited**
was a mention the name judge kept, refused, or never saw:

| per run | terra | luna |
|---|---|---|
| resolutions **proposed** citing a name-REJECTED antecedent | 12.7 (prec 0.263) | 30.7 (prec 0.207) |
| …for comparison, citing a name-ACCEPTED one | 115.0 (prec 0.696) | 138.3 (prec 0.496) |
| …that the strict coreference judge **keeps** | 0.3 | 3.3 |
| …that are **net new** (not already carried by a name link) | **0.3** | **0.3** |
| **gold** among those net-new | **0.0** | **0.0** |

Read down that column: the resolver does lean on refused mentions — 12.7 and 30.7
resolutions a run, at a third to a half the precision of the accepted ones — and **the
stage behind it already deletes them.** 97% on terra, 89% on luna. What survives is then
deleted a second time by `link`'s merge, because the name linker already carries the pair.

Reproduced on an independent invocation set (`union_e2e_terra_r*_20260911`,
`s_linker120`, different alias tables): net contribution citing a rejected antecedent
**0.0 a run**.

### The ceiling, stated as a number

A *suppressing* annotation — prefer, or require, a `linked` antecedent — can remove at
most **0.3 spurious net links a run on either model, and it has 0.0 gold to lose there.**
Against the branch's recorded harness floor (FP 10.7, TP 4.8) that is **35× below the
noise an E2E could resolve**. In three runs of three, on both models, exactly **one** net
link in the whole benchmark has an antecedent the name judge refused:

```
teammates S131 -> Storage (ant S130) ref='the component'
  "storage.api provides the API of the component to be accessed by the logic component"
```

and it is not gold.

**Level 1 therefore decides the suppressing reading, and the measurement policy says
stop there.** It does not decide the *informing* reading — that the mark changes which
antecedent the resolver picks among the entries it keeps, and so changes what it
proposes. Nothing in a checkpoint answers that, so level 2 was paid for.

---

## Level 2 — the stage pilot, four arms in one invocation

`pilot/coref_annot_pilots.py`. Four arms, all in the same invocation on the same pinned
inputs:

| arm | shortlist |
|---|---|
| `head` | `Facade (S3)` |
| `annot` | `Facade (S3, linked)` / `Storage (S130, named only)` — the mark alone, no rule speaks about it |
| `annot_clause` | the same mark, plus one sentence saying what it means and that a `named only` entry is the weaker antecedent |
| `annot_only` | the suppressing extreme: refused mentions are not listed at all |

`annot` and `annot_clause` are the design law's two halves — the fact in the case, the
weighing in the prompt — so `annot_clause` is the arm the law predicts should work.

**What is pinned.** Both the alias table and the name linker's link set come from the
recorded run's own checkpoints, so every arm is annotated against the *same* verdicts and
nothing about the name stage is resampled into the comparison. Only the resolver and its
own judge are re-run.

**This pilot has no composition risk, and that is checkable rather than argued.** `link`
merges by pair with the earlier linker winning, so the composed link set is exactly
`pinned name links | kept coreference`. Computed that way off the recorded checkpoints it
reproduces the run's own final CSV with a symmetric difference of **0** — TP 183, FP 19,
macro F1 93.55, macro F2 94.77 both ways. The pilot therefore reports the same TP / FP /
macro F1 / macro F2 an E2E batch is read with, carrying the resolver's variance only.

Prompt reach, per five-project run (no calls, `--verify`, against the terra r1 tables):

| arm | resolver bytes | delta |
|---|---|---|
| `head` | 213 006 | — |
| `annot` | 221 794 | +8 788 (+4.1%) |
| `annot_clause` | 231 114 | +18 108 (+8.5%) |
| `annot_only` | 210 365 | −2 641 (−1.2%) |

### terra — three samples, `../results/coref_annot_terra_20260914`

The coreference stage, per five-project run:

| arm | proposed | gold | kept | kept gold | **NET** | **net gold** | net spurious | calls |
|---|---|---|---|---|---|---|---|---|
| `head` | 138.7 | 86.0 | 33.7 | 32.7 | **15.0** | **14.0** | 1.0 | 48.0 |
| `annot` | 156.3 | 89.3 | 34.7 | 33.0 | **16.3** | **14.7** | 1.7 | 49.0 |
| `annot_clause` | 124.3 | 74.3 | 34.7 | 33.0 | **15.3** | **13.7** | 1.7 | 47.7 |
| `annot_only` | 110.3 | 72.3 | 27.7 | 26.3 | **15.3** | **14.0** | 1.3 | 47.0 |

Composed with the pinned name links:

| arm | TP | FP | macro F1 | macro F2 |
|---|---|---|---|---|
| `head` | 183.0 | 19.0 | 93.55 | 94.77 |
| `annot` | 183.7 | 19.7 | 93.58 | 95.07 |
| `annot_clause` | 182.7 | 19.7 | 93.27 | 94.58 |
| `annot_only` | 183.0 | 19.3 | 93.39 | 94.70 |

**The annotation swings the resolver's proposal volume from 110.3 to 156.3 pairs — −20%
to +13% — and moves its net contribution by at most 1.3 pairs and 0.7 gold.** Every
composed statistic is inside the recorded harness floor (TP 4.8, FP 10.7) several times
over.

Set-level, which is the number that settles it. The composed link set is 202 links;
against the head it differs by:

| arm | sample 1 | sample 2 | sample 3 |
|---|---|---|---|
| `annot` | +1 / −0 | +2 / −0 | +1 / −0 |
| `annot_clause` | +1 / −1 | +1 / −0 | +0 / −0 |
| `annot_only` | +1 / −0 | +0 / −0 | +0 / −0 |

**At most two links out of 202, in any arm, in any sample.** The head's own three
samples are identical to each other (symmetric difference 0), so this is not a delta
hiding inside sampling noise — there is almost no sampling noise here to hide in.

### The eight links that moved, and the one the round was aimed at

`pilot/coref_annot_diff.py` differences the composed dumps. Over three samples, five
projects and three arms, the **entire** effect is eight link changes:

| arm | +gold | +FP | −gold | −FP |
|---|---|---|---|---|
| `annot` | 2 | 2 | 0 | 0 |
| `annot_clause` | 0 | 2 | 1 | 0 |
| `annot_only` | 0 | 1 | 0 | 0 |

**Not one arm removes a single false positive**, which is the outcome the annotation
exists to produce.

And the sharpest line of the round: two of those changes are `teammates +S131`, which is
**teammates S131 -> Storage** — the one net link in the whole benchmark, across three
runs of three on both models, whose antecedent the name judge refused (level 1, R4). It
is not gold. The head did not produce it in those samples; `annot` and `annot_clause`
did. **The annotation's entire designed target is one false positive, and marking it
`named only` made the resolver more likely to take it, not less.**

### The mark and the clause point opposite ways, and both wash out

`annot` (the fact, no rule speaking about it) makes the resolver propose **+17.6** pairs
a run; `annot_clause` (the same fact plus "a `named only` entry is the weaker
antecedent") makes it propose **−14.4**. A 32-pair spread between two arms that differ by
one sentence, and the net contributions are 16.3 and 15.3.

This is the union round's "an evidence field restrains when it is stated and misleads
when it is weighted" reproduced at a new field — and an instance where the arm the design
law predicts should work (`annot_clause`, fact in the case, weighing in the prompt) is the
marginally *worse* of the two. The law says where a clause belongs, not that it will buy
anything.

### Why nothing reaches the output: two absorbing stages, measured

The resolver is not indifferent to the mark — it moves 30% of its proposals. What eats
the movement is downstream, and both absorbers are measured here rather than assumed:

1. **The strict coreference judge.** Level 1: it already deletes 97% (terra) / 89% (luna)
   of the resolutions that lean on a mention the name judge refused. Level 2: `kept` is
   33.7–34.7 for three of the four arms across a 32-pair swing in what was proposed.
2. **`link`'s merge.** An earlier linker wins the pair, so a resolution for something the
   name linker already carries changes nothing. `annot_clause` keeps **+1.0** links over
   the head and contributes **+0.3** net.

**A stage that is followed by a reject-by-default judge and a union that an earlier stage
already won cannot be improved by making it a better proposer.** Which is `s_linker92a`'s
false-negative result standing on its head: there the bottleneck moved off the proposer
and onto the gate, and this round is what it feels like to push on the proposer after
that has happened.

### luna — three samples, `../results/coref_annot_luna_20260914`

The laxer model is where this should have worked: its coreference judge keeps 3.3 of the
refused-antecedent resolutions a run against terra's 0.3, so it absorbs least.

| arm | proposed | gold | kept | kept gold | **NET** | **net gold** | net spurious | calls |
|---|---|---|---|---|---|---|---|---|
| `head` | 184.7 | 74.0 | 46.3 | 42.0 | **18.0** | **14.0** | 4.0 | 50.3 |
| `annot` | 178.3 | 70.7 | 42.7 | 38.0 | **19.3** | **14.7** | 4.7 | 49.7 |
| `annot_clause` | 195.0 | 79.0 | 46.7 | 41.7 | **18.7** | **14.0** | 4.7 | 50.0 |
| `annot_only` | 149.0 | 64.0 | 41.7 | 33.0 | **21.0** | **13.0** | **8.0** | 49.0 |

| arm | TP | FP | macro F1 | macro F2 |
|---|---|---|---|---|
| `head` | 176.0 | 40.0 | 89.37 | 91.70 |
| `annot` | 176.7 | 40.7 | 89.31 | 91.73 |
| `annot_clause` | 176.0 | 40.7 | 89.28 | 91.53 |
| `annot_only` | **175.0** | **44.0** | **88.44** | **90.78** |

`annot` and `annot_clause` are neutral again — every statistic inside ±0.7 TP / ±0.7 FP /
±0.2 F2. Link-level over three samples: `annot` +4 gold / +10 FP against −2 gold / −8 FP
(net +0.7 gold, +0.7 FP a run); `annot_clause` +3 / +10 against −3 / −8 (net ±0.0 gold,
+0.7 FP).

**`annot_only` is the one arm on either model that moves anything, and it moves the wrong
way.** TP −1.0, FP +4.0, macro F2 **−0.92**, and the head beats it on F2 in **3 samples of
3** (and on terra is worse-or-equal in 3 of 3) — so **the suppressing arm is never better
than the head in six samples across two models.**

### Suppression does not remove a resolution, it redirects it

This is the round's mechanism result, and it is what a ceiling calculation could not have
predicted. `annot_only` proposes **149.0** pairs a run against the head's 184.7 — it is
offered 2 641 bytes fewer of shortlist and proposes 36 fewer pairs — and yet its **net
spurious doubles, 4.0 to 8.0.** Link-level: **+17 false positives added against 5
removed.**

Taking an entry off the shortlist does not make the resolver abstain. It makes the
resolver reach for the next entry down and attach the same referring expression to the
wrong component. On teammates S131 the suppression does work exactly once — `annot_only`
sample 3 drops it — and it buys that one true deletion at seventeen additions.

`s_linker109` recorded the rule this breaks from the other side: **a discovered fact may
open a case and may not close one.** A name verdict is a discovered fact — another
judge's output, resampled every run — and using it to *withhold* an antecedent ends a case
on evidence that is not stable. The head's shortlist rests only on given input (the
catalog, the document, `_states_a_name`), which is why it has no such failure mode.

---

## Verdict

**REFUTED, on both models, at levels 1 and 2 — do not build it.**

| reading of the proposal | arm | terra | luna |
|---|---|---|---|
| suppress refused antecedents | `annot_only` | F2 −0.07, worse-or-equal 3/3 | **F2 −0.92, FP +4.0, worse 3/3** |
| mark them, no rule | `annot` | F2 +0.30, ±2 links of 202 | F2 +0.03 |
| mark them, one weighing sentence | `annot_clause` | F2 −0.19 | F2 −0.17 |

- **The informative part of the annotation is 17–21% of entries and 93% of it is one
  project.** 83% of the shortlist would carry a constant mark.
- **The resolver is not indifferent to it** — proposal volume moves from 110.3 to 156.3 a
  run on terra and 149.0 to 195.0 on luna, a ±25% swing — **and its net contribution moves
  by at most 1.3 pairs.** Two stages absorb it: a judge that rejects by default, and a
  merge an earlier linker has already won.
- **Nothing recovers a false positive.** Across three arms, two models and six samples,
  no arm shows a net FP reduction, and the one net link the annotation was designed to
  catch (teammates S131 → Storage, not gold) was *added* by two arms on terra.
- **The cost side is real and the benefit side is not**: +4.1% resolver bytes for the
  mark, +8.5% for the mark with its clause.

**What would have to change for this to be worth re-asking.** The absorbing stage is the
coreference judge. If a future head weakens or removes it, the 12.7 / 30.7 refused-
antecedent resolutions a run stop being free and the shortlist mark becomes the cheapest
thing that touches them. Until then the fact is already being used — just downstream, by
a judge that reads the sentence itself rather than a mark about it.


