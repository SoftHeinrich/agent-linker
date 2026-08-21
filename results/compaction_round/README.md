# The compaction round — the prompt is mostly not rules

The typed round removed two restatements and asked what else could go. This one asks
the question the goal states directly: **can every long prompt be made smaller while
performance stays inside the noise on both models?** It follows the branch's
measurement policy — deterministically off recorded runs first, then one stage at a
time on both models, and only then end to end — and it starts by measuring what a
prompt is actually made of, because the answer decides which clauses are worth an arm.

Everything below is read off `s_linker87`'s own end-to-end batch,
`results/dedup_e2e_{terra,luna}_r{1,2,3}_20260821`, three runs a side per model.

## What a prompt is made of (`pilot/judge_prompt_bytes.py`, no LLM calls)

The two families that carry the bytes, rebuilt from the checkpoints and the documents:

**The full-name judging call** — 8.7 calls per five-project run, 20.2 cases a call:

| part of the prompt | bytes/run | share | bytes/call |
|---|---|---|---|
| anchor sentences **repeated inside the same call** | 43 442 | **27.9%** | 5 013 |
| evidence line (`source=`, `span=`, mention) | 25 744 | 16.6% | 2 970 |
| case header | 21 907 | 14.1% | 2 528 |
| anchor sentences, first appearance in the call | 21 346 | 13.7% | 2 463 |
| the case's own sentence | 17 968 | 11.6% | 2 073 |
| the preceding sentence | 15 205 | 9.8% | 1 754 |
| **rule constants** (`LAYERED_ENTITY_RULES` + `QUALIFIED_CLAUSE` + `STRICTER_CLAUSE`) | 8 181 | **5.3%** | 944 |
| the claim-first instruction | 1 699 | 1.1% | 196 |

**The resolver call** — 40 calls per five-project run, 10 cases a call:

| part of the prompt | bytes/run | share | bytes/call |
|---|---|---|---|
| `SENTENCES` rows for sentences the same call prints as a TARGET | 45 124 | **25.4%** | 1 128 |
| `SENTENCES` rows that are context only | 41 264 | 23.2% | 1 032 |
| the case's own TARGET text | 38 698 | 21.8% | 967 |
| preamble (question + input contract + conservatism) | 13 240 | 7.4% | 331 |
| header + `COMPONENTS` + JSON schema | 13 200 | 7.4% | 330 |
| per-case `CONTEXT: sentences Sx-Sy above.` | 12 961 | 7.3% | 324 |
| `COREF_RULES` | 7 600 | 4.3% | 190 |
| case header | 5 670 | 3.2% | 142 |

**The four smaller families, rebuilt the same way** (terra, per five-project run):

| family | bytes/run | of which authored instruction |
|---|---|---|
| extraction (9.0 calls) | 46 312 | 6 156 (13.3%) — `ENTITY_EXTRACTION_RULES` + `QUALIFIED_CLAUSE` |
| alias proposal (5.0) | 38 067 | 1 865 (4.9%) |
| denotation (5.0) | 31 357 | 830 (2.6%) — `QUALIFIED_CLAUSE` |
| alias judge (5.0) | 4 542 | 1 235 (27.2%) |

**So the compaction the goal asks for is not in the prose.** Authored rule text is
3079 B and every earlier round spent itself there; in the two largest families the
rules are 5.3% and 4.3% of what is sent, while **literal repetition is 27.9% and
25.4%**. The arms below are aimed accordingly: two at repetition, three at clauses the
recorded verdicts say decide nothing.

## Which clauses decide anything (`pilot/clause_audit.py`, no LLM calls)

Per five-project run, three runs a side on each model.

| clause / question | terra | luna | reading |
|---|---|---|---|
| strict judge's leniency guard — objection stated yet **approved** | 4/442 (0.9%), 0.3 gold/run | 28/663 (4.2%), **8.3 gold/run** | **not inert on luna** — refused without an arm |
| its enumerated `acts on or produces` ground, cited in the objection | 1.0/run, **0 gold** | 1.7/run, **0 gold** | arm `noartifact` |
| per-case `CONTEXT` range vs. where antecedents actually sit | 16.3/run outside the declared range | 42.7/run outside | the line does not bind — arm `nocasectx` |
| antecedents citing a sentence that is a TARGET of the same call | 159.0/run of 186.0 | 226.3/run of 269.0 | the population `notargetrows` reframes |
| `QUALIFIED_CLAUSE` in the denotation prompt — candidates whose span sits in a dotted path | 2.0/run, **0 gold**, 5 of 6 already `associated` | 2.0/run, **0 gold** | arm `nodenotqual` |
| full-name claim-first vs. the lenient rubric's dead sentence (`claim = none`) | 39 cases, 92.3% rejected, 1.3 gold/run rejected | 30 cases, **100%** rejected | the dead sentence stays (typed round: neutral alone, negative with `nofocus`) |
| alias enumeration item 3, "words of multi-word names" | 1.3 aliases/run (4.1%) | 0.7/run (2.1%) | thin, and the alias table admits — left alone |

Two of these the audit settles on its own, and neither costs a call:

- **the strict judge's guard is load-bearing on the laxer model.** "An objection you
  could raise against most sentences is not a ground for rejecting this one" reads like
  a candidate for deletion — 310 B in every strict judging call — and on terra it
  changes 4 verdicts in 442. On luna it changes 28, **25 of them gold approvals**. This
  is the typed round's asymmetry a fourth time: the strict gate's leniency counterweight
  is what luna leans on, and removing it is predicted to cost ~8 gold links a run there.
  Priced at 0 API cost, refused.
- **the resolver spends half its output on sentences that name the component.** 96.0
  judged cases per run on terra (51.6% of them) and 90.0 on luna (33.5%) are targets
  whose own sentence writes the component's name, which `LAYERED_COREF_RULES` opens by
  saying is not what a coreference link is. The strict judge rejects 230 of terra's 288
  with one recurring objection ("the component is explicitly named in the sentence").
  That is a real inefficiency and it is **not** a compaction: fixing it means *adding* a
  clause to the resolver, and 53 of terra's 58 approvals in that population are gold. It
  is recorded here as the round's largest open finding, not acted on.

## The stage arms

Each replays one stage of `s_linker87` against the checkpoints above, three runs a
side, on both models. Groups are separate invocation sets: compare inside a group,
never across. p floor at n=3 is 0.10.

### The full-name judge, terra (`fullname5`)

Every arm judges the SAME extraction pass of the same run, so it differs from the
control in the judging prompt and in nothing else.

| arm | stage gold | stage spurious | stage bytes/run | composed F1 | F2 | TP | FP | verdict |
|---|---|---|---|---|---|---|---|---|
| ctl | 150.3 | 10.0 | 148 199 | 93.46 | 94.92 | 185.0 | 26.0 | — |
| `anchorblock` | 151.0 | 9.3 | **99 201 (−33%)** | 93.61 | 95.05 | 185.3 | 25.3 | **neutral** (gold p=1.00, F1 p=0.80) |
| **`anchorref`** | 152.0 | 9.7 | **104 158 (−30%)** | **93.81** | **95.20** | **185.7** | 25.3 | **neutral** (gold +1.7 p=0.80, F1 +0.4 p=0.70) |
| `nosource` | 150.0 | 11.3 | 145 091 (−2%) | 92.55 | 94.41 | 184.3 | 27.3 | leans negative (F1 −0.9, p=0.30) |

**The repetition is free to remove and the field is not.** Showing each anchor
sentence once per call — either hoisted into one indexed block or left in the first
case that needs it — takes a third of the judging prompt out with every point estimate
in the arm's favour. Deleting `source=` from the evidence line, 396 bytes a call
against `anchorref`'s 5 000, is the only arm of the four whose point estimates all
point the wrong way, which is the s25 `ablate_all` result again: *repeating the
evidence next to the rubric is not redundant for this model, and neither is naming it.*

### The strict coreference judge, terra (`coref5`)

| arm | stage gold | stage spurious | stage bytes/run | composed F1 | F2 | TP | FP | verdict |
|---|---|---|---|---|---|---|---|---|
| ctl | 36.3 | 4.0 | 81 701 | 93.05 | 94.90 | 185.7 | 26.3 | — |
| `noartifact` | 35.7 | 4.3 | 79 469 (−2.7%) | 92.95 | 94.71 | 185.0 | 26.7 | neutral on terra, 216 B a call |

### The denotation judge, terra (`denot2`) — refused

| arm | stage gold | stage spurious | composed F1 | F2 | TP | FP | verdict |
|---|---|---|---|---|---|---|---|
| ctl | 20.7 | 10.0 | 93.74 | 95.45 | 187.0 | 24.0 | — |
| `nodenotqual` | 21.7 | **25.0** | 91.86 | 94.60 | 187.0 | **39.0** | **refused** (stage spurious +15.0, composed FP +15.0, F1 −1.9) |

**The deterministic screen sized this clause's population at 2.0 candidates a run on
both models and 0 of them gold; deleting it costs 15 spurious partial-name links a
run.** `QUALIFIED_CLAUSE` in the denotation prompt is not doing the job it names. It
says an expression occurring only inside a longer dotted identifier is naming a piece
of that identifier — a population of two candidates a run — and what it is *worth* is
that the judge asked to classify what an expression denotes is reminded, in the same
breath, that an expression can denote a piece of a name rather than a participant. The
count screens the clause; it does not price it. **Fifth instance on this branch of a
surface attribution not being a causal one** (s53, the general round's alias syntax,
the typed round's morphology clause, the bind round's mention label), and the first
where the clause survives at 7.5x its own population. Refused after one cheap group,
and luna was not paid for.

### The correction the first group needed

`anchorblock` and `anchorref` showed every later case the anchor list computed for the
**first** case of that component in the batch. That is not the same list: the bundle
builder drops the case's own sentence and stops at `ANCHOR_LIMIT`, so two cases for one
component see overlapping but unequal lists — over all five projects, of 121 cases that
would be shown someone else's list, **only 19 have an identical one**, and the rest
differ by about one sentence in five each way. The invariants test caught it, not the
stage arm: both arms were neutral-to-positive on terra while silently withholding an
anchor from four cases in five.

The fix is the union: per component per batch, the union of what every case for it
would have shown, written once. Over all five projects that is **lossless for 169 of
169 cases** — no case is shown less than `s_linker87` shows it — and the extra
sentences a case gains are its own sentence (120 of 141), which the case already
prints, or one further naming sentence displaced by the anchor cap (21 of 141). The
union is at most 6 sentences, mean 3.8. `fullname6` measures that form, which is
`s_linker88`'s own code run as an arm.

**Methodological note, and the reason the test is written before the adoption and not
after: a stage arm cannot tell you that your compaction was lossy.** It reports gold
and spurious pairs, and a judge shown four of its five anchors still answers.

### The shipped form, terra (`fullname6`)

| arm | stage gold | stage spurious | stage bytes/run | composed F1 | F2 | TP | FP | verdict |
|---|---|---|---|---|---|---|---|---|
| ctl | 149.0 | 8.3 | 146 337 | 93.64 | 94.79 | 184.0 | 24.3 | — |
| **`anchorunion`** | 149.7 | 7.0 | **106 708 (−27%)** | **94.08** | **95.11** | **184.7** | **23.0** | **QUALITY-NEUTRAL** (TP +0.7 p=0.80, FP −1.3 p=0.70, F1 +0.4 p=0.60, F2 +0.3 p=0.50) |

Every point estimate is in the arm's favour and none is near the n=3 floor of 0.10 —
which is what removing a repetition should look like. Composition p = 0.80.

Composition risk off the recorded checkpoints (`pilot/composition_from_kept.py`, the
branch's step-3 gate): the arm adds 0.7 pairs per run, **0.0** of which a later stage
also proposes and 0.0 of which are already in the recorded final link set, and removes
1.3, of which **1.3 are in the recorded final link set**. Non-zero, so the end-to-end
confirmation is paid for; small, so at three runs a side and not six.

### The shipped form on the second model (`fullname6`, luna)

| arm | stage gold | stage spurious | stage bytes/run | composed F1 | F2 | TP | FP | verdict |
|---|---|---|---|---|---|---|---|---|
| ctl | 153.0 | 23.0 | 161 699 | 89.08 | 92.40 | 183.0 | 55.7 | — |
| **`anchorunion`** | 153.3 | 23.0 | **116 122 (−28%)** | **89.42** | **92.75** | **183.3** | **54.0** | **QUALITY-NEUTRAL** (TP +0.3 p=1.00, FP −1.7 p=0.80, F1 +0.3 p=0.60, F2 +0.4 p=0.50) |

**The one cut that holds on both models is the one that deletes no English.** Every
point estimate favours the arm on terra and on luna, none is near the floor, and the
judging prompt is 27–28% smaller in both.

### The resolver, terra (`resolve3`)

Both arms replay the resolver AND the strict judge behind it — what a resolver
proposes is only a link if that gate keeps it.

| arm | stage gold | stage spurious | stage bytes/run | composed F1 | F2 | TP | FP | verdict |
|---|---|---|---|---|---|---|---|---|
| ctl | 32.3 | 4.0 | 272 345 | 92.93 | 94.71 | 185.0 | 26.7 | — |
| `nocasectx` | 36.3 | 2.3 | 248 822 (−8.6%) | 93.03 | 94.74 | 185.0 | 25.0 | **neutral** (gold +4.0 p=0.60, F1 +0.1 p=0.70) |
| `notargetrows` | **41.0** | **1.7** | **207 518 (−23.8%)** | **93.36** | 94.75 | 184.3 | **24.3** | **composed-neutral** (F1 +0.4 p=0.30, F2 ±0.0 p=1.00), stage **better** (gold +8.7, spurious −2.3, both p=0.20) |

The deterministic screen said the per-case `CONTEXT` range does not bind (16.3
antecedents a run already sit outside it) and that 159.0 of 186.0 antecedents cite a
sentence the same call prints as a TARGET. Removing the duplicate table row for those
sentences does not cost the resolver its antecedents: it **finds more of them** (gold
32.3 → 41.0) and proposes fewer spurious ones, at a quarter less prompt. Luna decides
whether either is adopted.

### The lossy forms on the second model (`fullname5`, luna) — refused

| arm | stage gold | stage spurious | stage bytes/run | composed F1 | F2 | TP | FP | verdict |
|---|---|---|---|---|---|---|---|---|
| ctl | 152.0 | 16.3 | 157 290 | 89.99 | 93.03 | 183.0 | 49.0 | — |
| `anchorblock` | 151.0 | 20.3 | 104 992 (−33%) | 89.54 | 92.76 | 182.7 | 52.3 | leans negative (stage spurious +4.0, F1 −0.4) |
| `anchorref` | 152.0 | **23.0** | 110 151 (−30%) | 89.45 | 92.77 | 183.0 | 53.7 | **refused** — stage spurious **+6.7 (p = 0.10, at the floor)** |
| `nosource` | 151.0 | 17.3 | 154 014 (−2%) | 89.64 | 92.68 | 182.0 | 50.3 | leans negative on both models |

**This is the round's sharpest result, and it is about losslessness, not about
compaction.** The two arms that show a later case *someone else's* anchor list are
neutral on terra and cost luna precision — `anchorref` at stage spurious +6.7, the only
p at the n=3 floor in the whole group. The arm that shows every case at least its own
list (`anchorunion`, measured in its own invocation set above) is **FP −1.7 on the same
model**. Same 27–33% of the prompt removed, opposite sign on the laxer model, and the
difference is one anchor sentence in five.

The typed round's rule was *a prompt cut that holds on the stricter model says nothing
about the laxer one*. This round adds the mechanism for one class of cut: **withholding
evidence reads as neutral where the model is already strict and as a licence where it
is not.** The invariants test is what separated the two arms; the stage arm on terra
could not, and the stage arm on luna would have been read as noise without it.

### The strict judge's enumerated ground on the second model (`coref5`, luna) — refused

| arm | stage gold | stage spurious | composed F1 | F2 | TP | FP | verdict |
|---|---|---|---|---|---|---|---|
| ctl | 50.3 | 8.7 | 89.46 | 92.91 | 184.0 | 53.3 | — |
| `noartifact` | **43.7** | 13.0 | 88.59 | 92.26 | 183.0 | 57.7 | **refused** — stage gold **−6.7 (p = 0.10, at the floor)**, composed F1 −0.9 |

The recorded objections cite the enumerated ground 1.0 times a run on terra and 1.7 on
luna, **0 gold in both**, and terra reads its deletion neutral (gold −0.6, F1 −0.1).
Luna loses 6.7 gold resolutions a run without it. **Fourth instance of the asymmetry
the typed round named, and the second instance in this round of a clause whose
measurable effect is not the effect it states**: what the sentence about data,
artifacts, requests and results buys is not the handful of cases that cite it, it is
the strict judge's willingness to distinguish a thing from the component that acts on
it at all.

### What is adopted, and what is still running

**`s_linker88` = `s_linker87` with each component's anchor sentences written once per
judging call, and no English changed at all.** It is the only arm of the round that is
neutral-or-better on both models, and it is the largest cut: −27% of the judging prompt
on terra, −28% on luna, verified lossless case by case (`pilot/test_s88_anchors.py`,
35 checks).

Refused, each with its number:

| arm | why |
|---|---|
| `anchorref` / `anchorblock` | terra-neutral, **luna stage spurious +6.7 (p = 0.10)** — lossy de-duplication |
| `nosource` | every point estimate negative on both models |
| `noartifact` | terra-neutral, **luna stage gold −6.7 (p = 0.10)** |
| `nodenotqual` | terra stage spurious **+15.0**, composed F1 −1.9 — refused before luna was paid for |
| the strict judge's leniency guard | luna approves 8.3 gold a run over a stated objection; **refused from the checkpoints, 0 API cost** |
| the alias enumeration's third item | 1.3 / 0.7 aliases a run; the alias table trades recall between two linkers (s46, s60), so a thin yield is not worth the arm |

### The resolver on luna (`resolve3`) — one adopted, one refused

| arm | stage gold | stage spurious | stage bytes/run | composed F1 | F2 | TP | FP | verdict |
|---|---|---|---|---|---|---|---|---|
| ctl | 47.7 | 7.0 | 314 417 | 89.39 | 92.84 | 183.7 | 51.0 | — |
| **`nocasectx`** | **52.3** | 7.0 | 296 723 (−5.6%) | **89.48** | 92.52 | 182.7 | 51.3 | **QUALITY-NEUTRAL** (gold +4.7 p=0.50, F1 +0.1 p=0.90) |
| `notargetrows` | 44.0 | 7.3 | 254 358 (−19%) | 88.64 | 91.53 | 181.3 | 53.0 | **refused** — macro F1 **−0.8 (p = 0.10, at the floor)**, QUALITY-CHANGING |

**`nocasectx` holds on both models and both models resolve MORE gold without it**
(terra 32.3 → 36.3, luna 47.7 → 52.3). The line named the ±5 window of one target
while the `SENTENCES` table is the union of ten, and the audit had already found
16.3 antecedents a run on terra (42.7 on luna) citing a sentence outside the range
their own case declares. **`notargetrows` is the round's fourth terra-neutral,
luna-negative arm** — terra reads stage gold +8.7 and composed F1 +0.4 for −23.8% of
the resolver prompt, luna reads gold −3.7 and F1 −0.8 at the floor. The 25.4% of the
resolver call that is a duplicated table row stays.

### The composed head

**`s_linker89` = `s_linker87` + the anchor union + the resolver's range line removed.**
Two changes, **neither of which deletes a rule**; authored rule text is unchanged at
3079 B. `pilot/test_s89_compact.py` asserts the second change in 15 checks (every rule
constant and every method body but `_prompt_coref` byte-identical to s88's, every
resolver prompt of all five projects s88's minus exactly the `CONTEXT` lines, the
input-format contract still in the rendering, 324 B a call and 12 961 B a run).
**Owed: an end-to-end confirmation of the pair.**

### `s_linker88` end to end, three paired runs per model

`TYPED_E2E_CONTROL=s_linker87 TYPED_E2E_TAG=anchors bash pilot/run_typed_e2e.sh
s_linker88 terra luna`, both arms in every invocation.
`results/anchors_e2e_{terra,luna}_r{1,2,3}_20260821`. (terra r3 died on a Flex 429 and
was re-run alone, same two arms, same invocation form.)

| model | arm | TP | FP | macro F1 | macro F2 | calls | F1 range |
|---|---|---|---|---|---|---|---|
| terra | `s_linker87` | 182.3 | 29.0 | 92.23 | 93.87 | 84 | 3.55 |
| terra | **`s_linker88`** | 181.3 | **24.7** | **93.34** | **94.29** | 83 | **1.92** |
| terra | delta | −1.0 (p = 0.80) | −4.3 (0.50) | **+1.1 (0.40)** | +0.4 (0.80) | −1 | |
| luna | `s_linker87` | 179.3 | **37.0** | **90.71** | 92.55 | 86 | 1.59 |
| luna | **`s_linker88`** | **181.7** | 47.3 | 89.90 | **92.58** | 86 | **0.51** |
| luna | delta | +2.3 (p = 0.50) | +10.3 (0.20) | −0.8 (0.20) | +0.0 (1.00) | 0 | |

**Terra is QUALITY-NEUTRAL on all four statistics and macro F1 favours the arm by 1.1.
On luna nothing reaches the n=3 floor of 0.10 either, and the shape is the one this
branch has seen a dozen times: recall up (TP +2.3), precision down (FP +10.3), F2 flat
(+0.0).** The arm's run spread is the tighter of the two on both models (1.92 against
3.55, 0.51 against 1.59). The honest reading is **27–28% of the largest prompt family
removed for no measurable change on terra and no significant change on luna, with
luna's false positives the one number pointing the wrong way** — and it is the number
to watch if this head is taken further, because the stage arm read FP −1.7 there.

Composition is +3.3 (p = 0.10, at the floor) on terra and +3.2 (p = 0.30) on luna: the
two arms do produce somewhat different link sets, which is what changing what a judge
is shown should do.

### `s_linker89` end to end, three paired runs per model

`TYPED_E2E_CONTROL=s_linker87 TYPED_E2E_TAG=compact bash pilot/run_typed_e2e.sh
s_linker89 terra luna`, both arms in every invocation, all six clean.
`results/compact_e2e_{terra,luna}_r{1,2,3}_20260821`.

| model | arm | TP | FP | macro F1 | macro F2 | calls | F1 range |
|---|---|---|---|---|---|---|---|
| terra | `s_linker87` | 186.7 | 25.7 | 93.83 | 95.54 | 83 | **1.04** |
| terra | **`s_linker89`** | 186.3 | **24.0** | **94.19** | **95.68** | 83 | 1.96 |
| terra | delta | −0.3 (p = 1.00) | −1.7 (1.00) | +0.4 (0.60) | +0.1 (1.00) | 0 | |
| luna | `s_linker87` | 176.3 | **42.0** | 89.05 | 91.07 | 86 | **2.36** |
| luna | **`s_linker89`** | **178.7** | 44.0 | **89.38** | **91.88** | 85 | 3.20 |
| luna | delta | +2.3 (p = 0.80) | +2.0 (0.70) | +0.3 (0.90) | +0.8 (0.70) | −1 | |

**QUALITY-NEUTRAL on both models, on all four statistics, with no p anywhere near the
n=3 floor of 0.10** (the smallest is 0.60). Composition is −3.2 (p = 0.90) on terra and
+0.3 (p = 0.70) on luna — the composed head's link sets are not distinguishable from
the control's at this n, which for a change that removes only repetition is the
expected reading, not a null result to explain away.

**The luna false-positive number that the `s88` batch flagged did not persist.** There
it was +10.3 (p = 0.20) against a stage read of −1.7; here, with the same anchor change
plus the resolver cut, it is **+2.0 (p = 0.70)**. Two invocation sets, so these are not
comparable as a trend — the correct statement is that the sign that worried the `s88`
batch is not reproduced in the set that decides the head, and neither set puts it near
significance. Watch it again if the head advances.

**Round result: `s_linker89` is the head.** Two prompt families compacted — the judging
prompt's anchor block written once per call instead of once per case (27–28% of the
largest family, losslessly, proven case-by-case by `pilot/test_s88_anchors.py`) and the
resolver's per-case range line deleted (324 B a call, 12 961 B a run) — for no
measurable quality change on either model, and **no authored rule text removed at all**
(3079 B, unchanged from `s_linker87`).

## Reproducing

```bash
cd approach
# the deterministic work, no LLM calls
../.venv/bin/python pilot/clause_audit.py --variant s_linker87 "dedup_e2e_terra_r*_20260821"
../.venv/bin/python pilot/judge_prompt_bytes.py --variant s_linker87 "dedup_e2e_terra_r*_20260821"

# one stage group, one model (arms are paired inside the invocation)
OPENAI_API_KEY="$OAI_KEY" LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.6-terra \
OPENAI_SERVICE_TIER=flex OPENAI_REASONING_EFFORT=none \
LLM_LOG_DIR=../results/compaction_round/llm_logs_terra_fullname5 \
AB_OUT=../results/compaction_round \
  ../.venv/bin/python pilot/compaction_pilots.py --group fullname5 --model terra --runs 3

# the paired permutation test of every arm against its control
../.venv/bin/python pilot/compaction_round_stats.py --group fullname5 --model terra

# the invariants of the two heads
../.venv/bin/python pilot/test_s88_anchors.py
../.venv/bin/python pilot/test_s89_compact.py

# the end-to-end batches (each arm needs its own --arm block, with its runs)
../.venv/bin/python pilot/score_runs.py \
  --arm s_linker87 ../results/anchors_e2e_terra_r{1,2,3}_20260821 \
  --arm s_linker88 ../results/anchors_e2e_terra_r{1,2,3}_20260821

# the composed head's batch (tag `compact`), and its score
TYPED_E2E_CONTROL=s_linker87 TYPED_E2E_TAG=compact bash pilot/run_typed_e2e.sh \
  s_linker89 terra luna
../.venv/bin/python pilot/score_runs.py \
  --arm s_linker87 ../results/compact_e2e_terra_r{1,2,3}_* \
  --arm s_linker89 ../results/compact_e2e_terra_r{1,2,3}_*   # and luna
```
