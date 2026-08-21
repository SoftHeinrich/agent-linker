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

*(The remaining tables land here as the arms complete.)*

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
```
