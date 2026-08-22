# The reading round (s94) — one pass proposes for all three linkers

The head asks the document two questions in two LLM stages: the named-reference
extractor ("which sentences write a component's name?") and the coreference
resolver ("which sentences refer back to one?"). This round asks whether they are
one question at two reference forms.

Every judge, the alias module, the deterministic scan and every batch size are
inherited untouched. Only the proposal side changes.

## Why this is not a merge the ledger already refused

The standing finding — *every consolidation of two LLM decisions into one call
raises recall and lowers precision, twelve variants and no exception* — was
earned on a specific shape. Reading the s26–s35 line row by row, every merge in
it folds **alias discovery** or **judging** into extraction:

| | what was merged |
| --- | --- |
| s26, s60 | alias proposal into the reading |
| s29, s30 | lexical grounding / judging into extraction |
| s31 | review into the proposing call |
| s32–s35 | the judge's rubric into the extraction calls |
| s36, s38 | the two full-name **judging** calls into one |

**No row merges the two proposal stages with each other.** `grep -i coref` over
the ledger returns no merge at all. This round is the untried cell, and it keeps
the property every refused merge broke: a proposer never approves its own list.

## What the recorded runs already say

Three measurements off `../anchors_e2e_terra_r{1,2,3}_20260821`, no LLM calls.

**1. The two proposers already overlap on half their output.**

| | pairs/run | gold | precision |
| --- | ---: | ---: | ---: |
| named-reference extractor | 32.2 | 29.1 | 0.905 |
| coreference resolver | 37.7 | 23.1 | 0.614 |
| proposed by **both** | 17.5 | 16.5 | **0.947** |
| extractor only | 14.7 | 12.6 | 0.855 |
| resolver only | 20.2 | 6.6 | 0.327 |

54% of the extractor's pairs and 46% of the resolver's are proposed by both, and
the pairs both propose are the best either produces. The union then discards the
duplicates, so the resolver spends a large part of its 8 calls per project
re-deriving pairs the extractor already had.

This reproduces the compaction round's largest open finding from a second
direction. That round measured *"half the resolver's output is for sentences that
write the component's name (96.0 judged cases a run on terra, 51.6%)"* and
recorded it as a separate question because fixing it meant **adding** a clause.
Merging the two questions removes the duplication by construction instead.

**2. Anchors are local, so one 50-sentence block is wide enough.**

Over 414 recorded resolutions the antecedent sits a median of **2** sentences
back (mean 2.7, max 14):

| gap | 1 | 2 | 3 | 4 | 5 | 6–10 | >10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| share | 43.5% | 18.1% | 13.3% | 5.8% | 7.5% | 11.1% | 0.7% |

Only **1.0%** of anchors fall outside a fixed 50-sentence block. `EXTRACTION_BATCH`
is already grounded in s27's passage-length effect, so the reading keeps it; the
1% and the block boundaries are covered by a carried per-component note of the
last sentence that named it — the smallest form of the per-entity slot that let
Recurrent Entity Networks solve bAbI in one pass where a three-pass episodic
reader needed three.

**3. The routing shift is 2.1 pairs per project-run.**

In the head every refer-back claim goes to the strict judge. In s94 a claim whose
sentence *states* a name of its component goes to the lenient named judge —
routed by the same relation the head uses, not by the model's choice of field.
Which pairs actually move:

| | pairs/run |
| --- | ---: |
| resolver proposals | 37.7 |
| — sentence states a name | 19.5 (52%) |
| — — also proposed by the extractor (already on the named route) | 17.5 |
| — — **not proposed by the extractor — the shift** | **2.1** |
| — sentence states no name (refer-back route, unchanged) | 18.1 |

The 2.1 shifted pairs carry 0.6 gold (precision 0.29), and the strict judge
already keeps 0.6 of them. Against a recorded null arm that moves FP by 10.7 and
TP by 4.8, a 2.1-pair shift is **inside the floor**.

## What the round buys

| | head (s90) | s94 |
| --- | ---: | ---: |
| LLM calls per project | 16.8 | ~8.8 |
| — extraction + resolution | 9.8 | ~1.8 |
| proposal stages | 2 LLM + 1 scan | 1 LLM + 1 scan |
| judges | 3 | 3, unchanged |
| authored rule text | 3079 B | 3079 B, unchanged |
| batch sizes stated | 3 | 2 (`COREFERENCE_BATCH` unused) |

Roughly **48% fewer LLM calls**, and the narrative collapses from "three linkers
that each find their own candidates" to "read once, judge by the evidence the
sentence gives" — with the judging arrangement the ledger defends left exactly as
measured.

## Status: level 1 done, level 2 owed

Per the measurement policy, escalate and stop at the first level that decides.

- **Level 1, done.** `pilot/test_s94_reading.py`, **47/47 checks**: every judge,
  the alias module and the scan are inherited not redeclared; the three rule
  constants are the head's objects byte for byte and appear verbatim in the
  reading prompt; routing consults the name relation; an invented antecedent
  cannot name a real sentence; the named and refer-back streams are disjoint by
  construction; the pass runs once per document at `EXTRACTION_BATCH`.
  Composition risk priced above at 2.1 pairs/run.
- **Level 2, owed.** Level 1 cannot answer the one question that matters: *does
  one prompt asking both questions propose what two prompts asking them
  separately propose?* That is an LLM behaviour question. Replay both readings on
  the same recorded inputs, N samples a side, and compare gold and spurious at
  the proposal stage before anything composed is bought.
- **Level 4, conditional.** s76 is the warning: cutting the resolver's call count
  by widening its batch read TP −7.0 / F2 −1.8 on the s75 base while the same
  unification was parity on the s25 base. The mechanism here is different — the
  resolution happens inside a call that already reads those sentences, rather
  than in a wider batch of resolution cases — but the risk is the same shape, so
  if the stage pilot is anything but clean this needs paired runs **on both
  models**, per the typed and compaction rounds.

## The ladder — where to go if s94 loses

A merge that loses should not end the round. The ledger predicts *how* it would
lose, and each failure mode has a stronger structure behind it. All four rungs
are built and registered; the fallbacks are not weaker versions of s94, they are
different answers to different failure modes.

| rung | what it is | LLM calls/project | the failure mode it answers |
| --- | --- | ---: | --- |
| **A — `s_linker94`** | one call, both reference forms, flat | ~8.8 | the baseline merge |
| **B — `s_linker95`** | one call, **ordered**: named section, then refer-backs resolved against the list that call just produced | ~8.8 | merging cost precision, or the model skimped one of the two questions |
| **C — union over k readings** | sample the reading k times, union the proposals | ~8.8 × k | the merged reading is unstable run to run |
| **D — `s_linker93`** | both calls kept; the resolver asked **only** about sentences that write no name | ~13.3 | the merge loses on either model, and the cost win is wanted anyway |

**If the merge loses precision** — the direction the standing finding predicts,
since splitting buys precision and merging buys recall — the mechanism is that an
undifferentiated question lets the model report a refer-back for a sentence whose
name it never established, and those land on the lenient judge. **Arm B** answers
it by putting the head's own ordering *inside* the call: the named section is
committed first and the refer-back section resolves against that list rather than
against the whole catalog. This is the structure-first pattern that is the main
positive result of the 2026 document-agent literature (DocSage: schema discovery,
then extraction, then reasoning, +27% over long-context and RAG baselines) and
IRCoT's interleaving of retrieval with the reasoning that consumes it.

**If the merged reading skimps one of the two questions** — attention going to the
name task and the refer-back task getting a thinner read — Arm B's explicit
two-step is again the answer, and Arm D is the guaranteed floor.

**If the merged reading is unstable** — Arm C, and note it is the one rung with a
measured effect already on this pipeline: a majority vote over three independent
runs of the current head scores micro F1 0.913 against 0.901 for a single run.
Union rather than vote is the right operator at the *proposal* stage, because a
proposer's job is reach and the judges decide. The 2026 result that sequential
self-refinement loses to parallel best-of-N at matched cost ("Sample More, Reflect
Less", 3.6–10.1 points at 7B) says to spend any extra budget here rather than on
a gleaning loop.

**If the merge degrades on the long document** — teammates at 198 sentences is
where s27 measured accuracy tracking passage length (84.1 against jabref's 100.0)
— narrow the reading batch before abandoning the merge, then fall to Arm D.

**If the second model refuses what the first accepts** — the typed and compaction
rounds each refused arms on the second model that the first accepted — **Arm D is
the model-robust rung**, because its prompts are the head's. That is proved, not
argued: with the target set unrestricted, s93 renders the head's resolver prompts
**byte-identically** (`pilot/test_s9593_ladder.py`, 21/21).

### Arm D, priced before it was built

The resolver is asked about every sentence in the document. 52% of sentences
write no name of any component, so:

| | today | narrowed |
| --- | ---: | ---: |
| resolver calls per project-run | 8.0 | **4.5** (−44%) |

The exposure is small and computable: of the 4.0 coreference links the pipeline
keeps per project-run, 1.2 are on a sentence that does state a name, and **0.7 of
those are gold** — against a recorded null arm that moves TP by 4.8. This is also
the branch's design law applied rather than argued: which sentences write a name
is a *fact about the case*, `_states_a_name` already computes it, and the
compaction round's open finding only needed a clause because the fact was not
being used.

## Artifacts

- `approach/src/llm_sad_sam/linkers/experimental/s_linker94.py` — the variant, a
  subclass of `SLinker90` overriding exactly `_extract_named_mentions`,
  `_resolve_references`, `_read_document`, `_prompt_reading` and `link`.
- `approach/src/llm_sad_sam/linkers/experimental/s_linker95.py` — rung B, a
  subclass of `SLinker94` overriding exactly `_prompt_reading`.
- `approach/src/llm_sad_sam/linkers/experimental/s_linker93.py` — rung D.

**Numbering note.** The arms were first written at `s_linker91`/`s_linker92`,
which the static round already occupies (its paraphrase heads, tracked and
committed). The collision left `run_ablation.py` with duplicate registry keys —
Python keeps the last, so the reading entries were dead and the names resolved to
the static variants. The arms now live at the free numbers 94 and 95, the
duplicate keys are gone, and the static round's `s_linker90`/`91`/`92` are
byte-unchanged against HEAD.
- `approach/pilot/test_s94_reading.py` — 47 invariants.
- Registered as `s_linker94` (aliases `read`, `reading`) in `run_ablation.py`;
  the registry edit is purely additive, per GATE-01.

---

# Result: the two proposal stages are not mergeable, and the reason is mutual blindness

The stage pilot ran on terra, five documents, three samples a side, every arm
against control in its own invocation. **Six structurally different ways to merge
the two proposal questions were built and measured. All six lose the same links on
the same document.**

## What every arm did

| rung | what it changed | bbb gold | five-project gold | spurious |
| --- | --- | ---: | ---: | ---: |
| control (head) | two stages | **49.0** | 36.1 | 15.1 |
| A `s_linker94` | merged the question, flat, batch 50 | 40.3 | 34.0 | 6.3 |
| E `s_linker96` | + the resolver's batch size (10) | 41.7 | 34.3 | 8.1 |
| F `s_linker97` | + the resolver's per-case obligation | 44.3 | 34.9 | 9.5 |
| G `s_linker98` | + the resolver's context table | 43.0 | 34.8 | 11.1 |
| H `s_linker99` | + "a sentence may name several" | 44.0 | 34.7 | 9.3 |
| I `s_linker100` | read, then glean conditioned on the read | 40.3 | 34.2 | 8.8 |
| C (union over k) | resample the merged read 3× | 43.0 (union) | — | — |

Every merged arm holds gold on mediastore, teastore and jabref and cuts spurious
hard — teastore goes 26.7 → 27.0 gold with spurious 7.7 → 0. The entire aggregate
loss is one document.

## The mechanism, measured rather than argued

bigbluebutton is the only one of the five whose gold is coreference-heavy: 28 of
62 links are refer-backs, 23 with a naming sentence earlier to resolve against,
and **26 links sit on 12 sentences that reference more than one component**. That
last population is the whole story:

| arm | sentences given >1 component | gold on multi-component sentences |
| --- | ---: | ---: |
| control | 10.0 | **19.6 / 26** |
| grain (E) | 4.7 | 14.0 / 26 |
| cases (F) | 4.3 | 14.0 / 26 |
| window (G) | 4.0 | 14.0 / 26 |
| multi (H) | 4.7 | 14.3 / 26 |

The metric does not move under batch size, per-case obligation, context table, or
an explicit instruction to report several. The 5.6-link gap is the deficit.

**What two proposal stages buy is not two questions. It is two looks at the same
sentence that cannot see each other.** The extractor reports the component whose
name is written; the resolver, which never sees that answer, independently asks
what the sentence refers back to and names a *second* component. The union carries
both. One look reports about one component per sentence and moves on.

Rung I is the proof, because it fails *downward*. It was built to decorrelate the
second look by conditioning it on the first and asking only for the remainder —
GraphRAG's gleaning pattern. Conditioning is the opposite of blindness, and the
gleaning pass added **zero pairs in two of three samples on bigbluebutton**
(`Read: 43 pairs ... gleaned 38 sentences -> 43 total`). Told what was already
found for a sentence, the model treats the sentence as finished.

Every failure now has one explanation: merging shares context, gleaning shares it
explicitly and does worst, resampling (C) draws correlated looks, and an
instruction (H) does not create a second look at all.

This also explains the measurement that opened the round. The two proposers
overlap on 54%/46% of their output, but the **non-overlap is exactly the
multi-participant links**, and that is where the coreference-heavy document's gold
concentrates. High overlap did not mean low marginal value.

## The mechanism, at one more level of depth

Splitting every arm's proposals by reference kind shows the loss is not spread
across the merge at all. It is one question, and one direction.

| arm | kind | proposed/run | correct/run | precision |
| --- | --- | ---: | ---: | ---: |
| control | names | 34.8 | 31.5 | 94.1% |
| merged (F) | names | 34.9 | 31.7 | 94.3% |
| control | refers back | 31.5 | 12.6 | 43.9% |
| merged (F) | refers back | **21.7** | 10.2 | **53.9%** |

**Merging costs the naming question nothing measurable.** All of the damage is the
refer-back question emitting 31% fewer claims at 10 points higher precision --
the signature of a *raised evidence threshold*, not of lost ability. The merged
reader becomes more careful about the question it should be least careful about.

Three candidate mechanisms were tested against the recorded proposals and two are
refuted:

* **Per-sentence output cap — refuted.** Components emitted per proposed sentence:
  control 1.247, merged 1.223. Nothing is being truncated.
* **Position decay — not supported.** Gold recall by slot within the 10-sentence
  block runs 87, 83, 67, 100, 99, 85, 93, 90, 81, 74: a mild late dip, no monotone
  decline. The published position effect does not reproduce here.
* **Distraction by a competing name — refuted, and it was our own hypothesis.**
  The refer-back loss is *larger* on sentences that write no name at all (−16.7
  points) than on sentences that name some other component (−9.1).

**The threshold is not reachable by instruction.** Two variants tried:
`s_linker99` told the model a sentence may reference several components (deficit
metric 14.0 → 14.3 of 26); `s_linker104` added the head's own active search
instruction verbatim -- *"identify any pronoun or noun phrase in THAT sentence
that refers back to a component listed above"* -- plus an explicit statement that
the two kinds are judged on their own standards. It went **further the wrong way**:
refer-back proposals 37.6 → 20.0 (−17.6) at precision +17.7.

So the sharpened statement of the finding: **asking two questions in one call
leaves the surface-anchored question untouched and raises the evidence bar on the
one with no surface anchor, and that bar cannot be lowered by telling the model to
lower it.** Eight attempts, two of them aimed squarely at the threshold.

## What this settles for the branch

* The untried cell of the s26–s35 line is now tried. Merging two *proposers* —
  as opposed to a proposer into a judge — **lowers recall and raises precision**,
  the exact inverse of the standing finding, and the standing finding's wording
  should be read as scoped to proposer-into-judge merges.
* The head's two proposal stages are **load-bearing, not habit**. They were the
  round's target and they survive it with a measured reason they did not have
  before.
* The duplication the round set out to remove was never in the authored text —
  `ENTITY_EXTRACTION_RULES` and `COREF_RULES` are already shared constants. It was
  in the call count, and the call count is buying mutual blindness.

## Rungs closed on their own evidence

* **C (union over k readings)** — union over three merged samples reaches 43 gold
  on bigbluebutton against control's 51. A deterministic blind spot; the same
  links are skipped every run, so sampling cannot reach them.
* **D (`s_linker93`, narrow resolver)** — carries a correctness defect found
  before adoption. 12 gold links across the five documents (6.2%) sit on sentences
  that name *only some other* component, which a per-sentence `_nameless` filter
  makes structurally unreachable; bigbluebutton holds 5 of them and the arm lost
  exactly 5.0 there. The obvious repair — filter per component instead of per
  sentence — collapses the filter, since no sentence names all 12 components. Its
  44% call saving and its correctness are in direct conflict.

## Confirmed on the second model

The typed and compaction rounds each refused arms on the second model that the
first accepted, so the decisive arm (F, `s_linker97`) was replayed on luna over
bigbluebutton and teastore, three samples a side, against control in the same
invocation and on luna's own recorded alias table.

| model | bbb gold, control | bbb gold, F | gold on multi-component sentences |
| --- | ---: | ---: | ---: |
| terra | 49.0 | 44.3 | 19.6 → 14.0 of 26 |
| luna | 51.7 | 48.0 | 19.3 → 14.7 of 26 |

The deficit reproduces at the same magnitude on both models, and reproduces
*through the same metric*. On teastore, where only 4 sentences carry more than one
component, luna's merged arm beats control on that population (8.0/8 against
7.3/8) and on gold overall (26.7 against 26.3) — the loss appears only where the
multi-participant population exists, which is what the mechanism predicts.

Worth recording for any future precision-led budget: on luna the merged arm cuts
spurious proposals by 14.8 a project-run at a cost of 1.7 gold, taking proposal
precision from 0.629 to 0.821. This round is run under an F2-led budget, so that
trade is refused here, but it is a real and repeatable effect rather than noise.

---

# Composed under F2: the merged arm is refused, and the metric is why

The stage pilot compared arms on proposals. This is the composed pipeline — every
judge, the alias module and the scan in place — five projects, macro over
projects, terra, one paired run with both variants in the same invocation.

| variant | P | R | F1 | F2 | LLM calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| `s_linker90` (head, two blind proposers) | 87.1 | **97.0** | 91.5 | **94.6** | 16.4 |
| `s_linker97` (one merged proposer) | 91.5 | 94.2 | **92.7** | 93.6 | 14.0 |
| delta | +4.4 | **−2.8** | **+1.29** | **−1.02** | −15% |

Per project, F2: mediastore 97.4 → 96.8, teastore 100.0 → 92.6, teammates
85.0 → 90.4, bigbluebutton 93.8 → 90.2, jabref 96.8 → 97.8.

**Under F1 the merged arm wins and costs 15% less. Under F2 it loses.** Recall
fell 97.0 → 94.2 and the judge tier could not recover it, because a judge only
removes — which is the composed form of the proposal-stage result and of the
branch's own error analysis (~95% of residual false negatives are pairs that never
reached a judge).

This is why the primary metric has to be named before the arm is built. On the
secondary metric this round would have shipped a variant that costs 1.0 F2.

## What the round hands forward

Two recall-led candidates follow directly from the mechanism, both under test:

1. **`s_linker101` — the reading kept as a third blind proposer** rather than a
   replacement. Paired stage-pilot samples put proposal recall at 94.9 → 96.1
   (terra) and 90.4 → 94.4 (luna). If blindness is what buys recall, adding a
   blind look is the move an F2 budget wants; ~21 calls a project against 16.8.
2. **`s_linker102` — the proposer stops authoring and judges membership.** The
   2026 literature prices this failure well outside our own measurement: models
   judge set membership at F1 0.60–0.77 but author the same set at 0.26–0.48, a
   gap that does not close over a 24× parameter range, and planted omissions are
   detected 6–7× less often than planted over-inclusions (arXiv 2608.01000).
   Our rung H already showed phrasing is not the lever (14.0 → 14.3 of 26); this
   changes the reply *shape* instead, to one decision per candidate, so an
   omission has no well-formed short form.
