# The typed round — one contradiction, one clause, and a closed set of verdicts

Three questions, asked in the order the branch's measurement policy asks them:
deterministically off recorded runs first, then one stage at a time, and only then
end to end.

1. **The full-name judging prompt contradicts itself.** `LAYERED_ENTITY_RULES` says
   *"A mention that says nothing further about the component still counts as a valid
   link"*, and the builder then asks the judge to quote the architectural claim and
   *"decide approve true/false based on that claim"*. One of the two decides every
   case where the sentence makes no claim. Which?
2. **One clause of `ENTITY_EXTRACTION_RULES`** — *"count a name written with different
   spacing, hyphenation or compound joining as that name"* — is the only instruction
   in the module that admits a candidate whose sentence writes no name at all. What
   does it admit, and does it pay on both models?
3. **Can a rubric be a closed set of verdicts instead of prose?** The module already
   has one typed judge (the denotation step answers `participant` / `associated`).
   Typing the other two would replace three prose blocks with four named types and
   let a reject reason be recorded instead of inferred.

## Q1 and Q2, answered without an LLM call

`pilot/entity_prompt_audit.py`, over the three recorded runs a side of
`results/s85_e2e_terra_r*_20260820` (arms `s_linker85` and `s_linker82`) and
`results/audit_e2e_s82luna_r*_20260820`:

**The claim-first instruction wins, unanimously, on both models.**

| model / arm | cases with `claim = none` | rejected | of those, gold |
|---|---|---|---|
| terra `s_linker85` | 37 (12.3/run) | **37 (100%)** | 6 (2.0/run) |
| terra `s_linker82` | 45 (15.0/run) | **45 (100%)** | 5 (1.7/run) |
| luna `s_linker82` | 23 (7.7/run) | **23 (100%)** | 2 (0.7/run) |

So the sentence that says a claimless mention still counts is **inert**: in 105
recorded cases it never once carried a verdict. It is 86 bytes of instruction in
front of 18.5 judging calls per five-project run that decide nothing. Two ways to
resolve the contradiction, and both are arms below: delete the inert sentence
(`nodead`), or honour it and approve `NO_CLAIM` (`typedlenient`) — which the same
audit prices at +2.0 gold and +10.3 spurious per run on terra.

**The morphology clause admits few candidates, and the two models disagree about
their quality.** Candidates whose sentence writes no name of the component at
`ANY_CASE` — the only ones that clause (or a misread) can license:

| model / arm | such candidates per run | gold | spurious |
|---|---|---|---|
| terra `s_linker85` | 3.3 | 2.3 | 1.0 |
| terra `s_linker82` | 3.3 | 2.3 | 1.0 |
| luna `s_linker82` | 12.0 | 2.3 | **9.7** |

On terra the clause is a recall instrument (`bbb-web` for `BBB web`, `WebRTC` for
`WebRTC-SFU`); on luna it is also the licence under which the extractor proposes
`GAE server → GAE Datastore` nine times a run. Same clause, same gold yield,
four times the spurious yield. That is exactly the kind of clause the goal of this
round names, and it is why `nomorph` is measured on both models.

## The prompt surface this round works on

Per five-project run of one arm, from the recorded call log of
`results/s85_e2e_terra_r1_20260820`:

| prompt family | calls/run | prompt bytes/run |
|---|---|---|
| coreference resolution | 40.0 | 190 378 |
| validation (both judges) | 18.5 | 217 294 |
| extraction | 9.0 | 46 808 |
| alias proposal | 5.0 | 38 067 |
| denotation | 5.0 | 30 206 |
| alias judge | 5.0 | 4 772 |

Authored rule constants total **3485 B**. The judging rubrics this round rewrites
are 944 B on the lenient side (`LAYERED_ENTITY_RULES` 394 + `STRICTER_CLAUSE` 384 +
`QUALIFIED_CLAUSE` 166) and 666 B on the strict side, plus ~310 B of builder text
for the strict side's objection paragraph and response field.

*(Stage results and the end-to-end confirmation follow below once the arms land.)*

## The stage arms on terra

Every arm replays one stage of `s_linker85` against the checkpoints of
`results/s85_e2e_terra_r*_20260820`, three runs a side. Within a group the arms share
the same recorded aliases and — for the `fullname*` groups — the same extraction pass,
so an arm differs from its control in the judging prompt and in nothing else.
**Groups are separate invocation sets: compare inside a group, never across.**
Composed numbers union the arm's stage with the same run's recorded other two stages.
p floor at n=3 is 0.10.

| group | arm | stage gold | stage spurious | composed F1 | F2 | TP | FP | verdict |
|---|---|---|---|---|---|---|---|---|
| fullname | ctl | 151.3 | 6.3 | 94.19 | 94.76 | 182.7 | 21.3 | — |
| | `nodead` | 149.3 | 6.3 | 93.63 | 94.07 | 180.3 | 21.3 | **neutral** (gold p=0.30, spur p=1.00) |
| | `typed` | 134.7 | 3.0 | 91.66 | 90.54 | 167.3 | 18.0 | **rejected** (gold −16.7, p=0.10) |
| | `typedlenient` | 146.3 | 3.0 | 93.43 | 93.10 | 177.3 | 18.0 | rejected (gold −5.0, F2 −1.7) |
| fullname2 | ctl | 151.0 | 6.0 | 94.11 | 94.73 | 182.7 | 21.0 | — |
| | `typeddefault` | 142.7 | 7.0 | 92.39 | 92.20 | 174.0 | 22.0 | **rejected** (gold −8.3, p=0.10) |
| | `typedlenient` | 147.7 | 4.0 | 93.70 | 93.63 | 179.3 | 19.0 | rejected (replicates set A) |
| fullname3 | ctl | 150.7 | 6.0 | 94.16 | 94.62 | 182.0 | 21.0 | — |
| | **`nofocus`** | 151.7 | 8.7 | 93.56 | 94.58 | 183.0 | 23.7 | **neutral** (TP +1.0 p=0.50, F2 −0.0 p=0.80) |
| | `compact` (`nofocus`+`nodead`) | 148.0 | 9.0 | 92.84 | 93.59 | 179.3 | 24.0 | worse than either alone |
| extract | ctl | 148.3 | 5.3 | 93.86 | 94.24 | 180.7 | 20.3 | — |
| | `nomorph` | 145.0 | 4.7 | 93.14 | 93.24 | 177.3 | 19.7 | **neutral** (gold −3.3 p=0.50) |
| coref | ctl | 33.3 | 3.7 | 93.52 | 94.08 | 181.0 | 21.7 | — |
| | `typedcoref` | 62.7 | 12.0 | 92.28 | 94.04 | 183.0 | 29.3 | **rejected** (F1 −1.2, p=0.10) |
| coref2 | ctl | 33.3 | 5.7 | 93.01 | 94.03 | 181.3 | 23.7 | — |
| | **`typedcorefstrict`** | 58.7 | 9.3 | 92.97 | 94.46 | 183.7 | 27.0 | **neutral** (F1 −0.0 p=0.90, F2 +0.4 p=0.60) |
| alias | ctl | 147.0 | 5.3 | 93.51 | 93.76 | 179.7 | 20.3 | — |
| | `typedalias` | 146.0 | 6.7 | 92.11 | 91.86 | 177.0 | 21.7 | rejected (table 27.0 → 31.3 terms, prompt *bigger*) |

### What the typed verdicts actually did — the round's mechanism

**Typing a rubric deletes its default, and the default is what each judge's asymmetry
was carrying.** The two judges moved in opposite directions from the same edit:

- the **lenient** full-name gate lost recall — 151.3 → 134.7 gold per run. Naming three
  reject types with no "approve by default" invites the judge to reach for one, and
  `NO_CLAIM` is the one it reaches for: the same 12.3 claimless cases a run that the
  audit found, now with a name;
- the **strict** coreference gate lost strictness — 33.3 → 62.7 gold and 3.7 → 12.0
  spurious. Dropping "when uncertain, reject" for a list of three reject types made a
  merely-plausible resolution reachable.

Restating each default inside the typed rubric fixes the coreference judge
(`typedcorefstrict`: composed F1 −0.0, F2 +0.4) and does **not** fix the full-name judge
(`typeddefault`: gold −8.3 at p = 0.10, worse than `typedlenient`). Three resolutions of
the contradiction were measured — reject `NO_CLAIM`, approve `NO_CLAIM`, restate the
default — and **all three cost recall**, in two independent invocation sets.

**So the answer to "force the validator to give a type" is: yes at the strict judge, no
at the lenient one, and neither is compaction.** The typed coreference rubric is 66
chars *larger* per call than the prose it replaces, and the typed alias rubric 304 larger.
What it buys at the strict judge is that the reject ground becomes a recorded verdict
instead of free text, at F2 +0.4 and F1 −0.0.

**The compaction that does hold on terra is the restatement, not the restructure**:
`VALIDATION_FOCUS` (243 B, carried in all 18.5 judging calls per run) asks for
architectural participation and referential specificity, which `LAYERED_ENTITY_RULES`
states as its approve-condition and `STRICTER_CLAUSE` states as its whole subject.
Dropping it is TP +1.0 and F2 ±0.0. This is the same shape the finetune round measured
once before (dropping the full-name focus tail with nothing added: TP +2.3, FP ±0.0).

## The same arms on luna

Replayed against `results/s85_e2e_luna_r*_20260820`, three runs a side, same design.
Luna is the laxer of the two models — its own end-to-end FP is roughly twice terra's —
so an arm that trades recall for precision has more room here, and the round's two
adoption candidates had to hold in both places.

| group | arm | stage gold | stage spurious | composed F1 | F2 | TP | FP | verdict |
|---|---|---|---|---|---|---|---|---|
| fullname | ctl | 149.0 | 8.3 | 89.49 | 91.16 | 174.7 | 40.3 | — |
| | **`nodead`** | 149.0 | 9.0 | 89.50 | 91.29 | 174.7 | 41.0 | **neutral** (TP identical) |
| | `typed` | 140.3 | 9.0 | 87.87 | 89.11 | 167.0 | 40.7 | **rejected** (gold −8.7, F1 −1.6) |
| | `typedlenient` | 148.0 | 8.7 | 89.57 | 91.32 | 174.7 | 40.7 | neutral here, −5.0 gold on terra |
| fullname3 | ctl | 148.0 | 12.7 | 88.78 | 90.89 | 174.7 | 44.7 | — |
| | **`nofocus`** | 149.7 | 13.7 | 88.88 | 91.19 | 175.7 | 45.7 | **neutral** (F1 +0.1 p=0.90, F2 +0.3 p=0.60) |
| | `compact` (+`nodead`) | 150.0 | 19.7 | 88.33 | 90.93 | 176.0 | 50.7 | worse than either alone, as on terra |
| extract | ctl | 148.7 | 10.0 | 89.46 | 91.31 | 175.3 | 42.0 | — |
| | `nomorph` | 143.7 | 12.0 | 87.82 | 89.41 | 169.3 | 44.3 | **rejected** (gold −5.0, F2 −1.9) |
| coref2 | ctl | 51.3 | 7.3 | 89.18 | 91.17 | 174.7 | 42.7 | — |
| | `typedcorefstrict` | 85.0 | **45.3** | 85.41 | 91.23 | 183.7 | 76.7 | **rejected** (FP +34.0, F1 −3.8) |
| fullname2 | ctl | 148.7 | 10.0 | 89.13 | 91.13 | 174.7 | 42.0 | — |
| | `typeddefault` | 141.7 | 10.7 | 87.96 | 89.59 | 169.0 | 42.7 | rejected (gold −7.0, F1 −1.2) |
| | `typedlenient` | 147.0 | 10.3 | 88.82 | 90.72 | 174.0 | 42.7 | rejected (terra −5.0 gold) |
| coref | ctl | 56.3 | 4.3 | 89.34 | 91.18 | 174.3 | 40.0 | — |
| | `typedcoref` | 86.3 | **48.7** | 85.30 | 91.16 | 183.7 | 79.0 | rejected (F1 −4.0) |
| alias | ctl | 146.7 | 12.0 | 87.63 | 89.54 | 171.3 | 44.0 | — |
| | `typedalias` | 144.0 | 16.7 | 87.32 | 89.34 | 170.0 | 48.7 | rejected (prompt 272 B *larger*) |

### The three results the second model produced

1. **The typed coreference rubric was terra-neutral and is luna-fatal.** Same prompt,
   same default restated, same three runs a side: terra composed F1 −0.0, luna F1 −3.8
   at FP +34.0 (stage spurious 7.3 → 45.3). "When uncertain, reject" is not a sentence
   a laxer model can be given as one type among four. Had this round stopped at terra it
   would have adopted it.
2. **The morphology clause is load-bearing on both models, and the audit's attribution
   of its cost was wrong.** The deterministic screen found 9.7 spurious candidates a run
   on luna whose sentence writes no name at `ANY_CASE` — the population only that clause
   can license. Removing the clause did not remove them: luna's stage spurious went
   *up* (10.0 → 12.0) while gold fell 5.0. The extractor proposes `GAE server → GAE
   Datastore` with or without a licence to; the clause is what buys `bbb-web` for
   `BBB web`. **A surface attribution is not a causal one** — the branch's own s53
   lesson, arriving from a new direction.
3. **`nodead` and `nofocus` are each neutral on both models and negative together.**
   terra `compact` F1 −1.3 against `nofocus` −0.6 and `nodead` −0.6; luna `compact`
   F1 −0.45 at FP +6.0 against `nofocus` +0.1. Once the focus line is gone, the inert
   sentence stops being inert: the focus was carrying the participation requirement the
   claim-first instruction leans on. **A clause is not independently priceable** (s78's
   result, in the other direction). The round therefore removes one clause, not two.

## What is adopted

`s_linker86` = `s_linker85` minus `VALIDATION_FOCUS`, and nothing else.
`pilot/test_s86_nofocus.py` asserts the single change in 75 checks: the constant gone,
every other rule constant byte-identical, every method body but two byte-identical, both
judging prompts rendering byte-identically to s85's once the focus is substituted back,
the deletion sized at exactly 244 B per judging call, the bounds and the deterministic
layer untouched, and GATE-06 re-checked against all five component catalogs.

Composition risk, read off the recorded checkpoints (the branch's step-3 gate): the arm
adds 4.0 pairs per five-project run, of which **0.7** are pairs a later stage also
proposes, and removes 0.3, of which **0.0** are in the recorded final link set. Non-zero,
so the end-to-end confirmation below was paid for; small, so it is three runs a side and
not six.

## Reproducing

```bash
cd approach
# the deterministic audit, no LLM calls
../.venv/bin/python pilot/entity_prompt_audit.py "s85_e2e_terra_r*_20260820"

# one stage group, one model (arms are paired inside the invocation)
OPENAI_API_KEY="$OAI_KEY" LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.6-terra \
OPENAI_SERVICE_TIER=flex OPENAI_REASONING_EFFORT=none \
LLM_LOG_DIR=../results/typed_round/llm_logs_terra_fullname3 \
AB_OUT=../results/typed_round \
  ../.venv/bin/python pilot/typed_prompt_pilots.py \
    --group fullname3 --model terra --runs 3

# the paired permutation test of every arm against its control
../.venv/bin/python pilot/typed_round_stats.py --group fullname3 --model terra

# the invariants, and the end-to-end confirmation
../.venv/bin/python pilot/test_s86_nofocus.py
OAI_KEY=... bash pilot/run_typed_e2e.sh s_linker86 terra luna
```

## Invariants

- **Arms are compared only inside the group that produced them.** Each group is one
  invocation set; the seven controls in the tables above differ by up to 1.4 macro F1
  from each other on the same model and the same recorded inputs, which is the size of
  this pipeline's run-to-run band, not a difference between the controls.
- **No in-set null arm**, per the measurement policy: the harness floor is recorded and
  an arm is not spent re-measuring a constant.
- **The p floor at n=3 is 0.10** and every p in this file is reported against it.

## The end-to-end confirmation

`pilot/run_typed_e2e.sh s_linker86 terra luna`, three paired runs per model, both arms
in every invocation, `gpt-5.6-{terra,luna}` at `OPENAI_REASONING_EFFORT=none` on Flex.
`results/typed_e2e_{terra,luna}_r{1,2,3}_20260821`.

| model | arm | TP | FP | macro F1 | macro F2 | calls | F1 range |
|---|---|---|---|---|---|---|---|
| terra | `s_linker85` | 180.7 | 18.7 | 94.11 | 94.53 | 82 | 1.27 |
| terra | **`s_linker86`** | **184.3** | **18.3** | **94.65** | **95.19** | 81 | 1.23 |
| terra | delta | +3.7 (p = 0.20) | −0.3 (p = 1.00) | +0.5 (p = 0.40) | +0.7 (p = 0.50) | −1 | |
| luna | `s_linker85` | 177.7 | 47.7 | 88.75 | 91.22 | 84 | 0.94 |
| luna | **`s_linker86`** | **179.0** | 49.0 | **89.02** | **91.65** | 84 | 1.99 |
| luna | delta | +1.3 (p = 0.80) | +1.3 (p = 0.60) | +0.3 (p = 0.80) | +0.4 (p = 0.80) | 0 | |

**QUALITY-NEUTRAL on both models on all four statistics**, composition +0.1 (p = 0.50)
on terra and −4.6 (p = 1.00) on luna — the two arms' link sets differ no more between
arms than within them, which is what removing a restatement should look like. Every
point estimate is in `s_linker86`'s favour on both models and none of them is
significant at the n=3 floor of 0.10; the honest reading is **243 B of instruction
removed for no measurable change**, which is what the round set out to buy.

## One more restatement: the resolver's own question (`s_linker87`)

The full-name judge's focus line was a restatement; so is the opening sentence of
`COREF_RULES`. It tells the resolver to "decide whether a pronoun or noun phrase that
refers back in the target sentence refers back to a component named or aliased earlier
in the context" — which the prompt's own preamble, four lines above, already asks, and
asks together with the input contract the question needs (which block is the TARGET,
what to return for a target with no referring expression). s56 measured deleting that
preamble at **TP −16.2**, precisely because it is the contract; this arm deletes the
restatement and keeps the contract, which is the untried other half.

It is also where the bytes are: the resolver is **40 of the ~82 calls a five-project
run makes**, so 163 B off its prompt is ~6.5 kB of instruction per run, against 244 B ×
~8.7 calls for `s_linker86`'s cut.

Arms replay the resolver **and the strict judge behind it** — what a resolver proposes
is only a link if that gate keeps it — three runs a side, composed with the recorded
full-name and partial-name stages of the same run:

| model | arm | stage gold | stage spurious | composed F1 | F2 | TP | FP | verdict |
|---|---|---|---|---|---|---|---|---|
| terra | ctl | 34.3 | 2.3 | 93.59 | 93.90 | 180.3 | 20.3 | — |
| terra | `dedup` | 32.7 | 5.3 | 93.43 | 94.09 | 182.0 | 23.3 | **neutral** (F1 −0.2 p=0.80, F2 +0.2 p=0.80) |
| luna | ctl | 42.7 | 5.7 | 89.20 | 91.02 | 174.7 | 42.0 | — |
| luna | `dedup` | 42.7 | 6.0 | 89.39 | 91.29 | 174.7 | 42.3 | **neutral** (TP ±0.0 p=1.00, F2 +0.3 p=0.80) |

Adopted as `s_linker87`, the round's head: `s_linker85` minus two restatements, nothing
else. Authored rule text **3485 → 3079 B (−11.7%)**; `pilot/test_s87_dedup.py` asserts
the single change in 80 checks, including that the resolver prompt is s86's minus
exactly those 163 B, that the input-format contract is still in the rendering, and that
exactly two constants differ from `s_linker85`. The change is at the last linker, so the
branch's composition precondition is structurally vacuous and the stage arm is the
pipeline answer; the end-to-end batch below was run because a head is quoted end to end.

### The frontier: the strict judge's focus line, priced and refused

The same argument that removed the lenient judge's focus applies verbatim to the strict
one — `COREF_VALIDATION_FOCUS` ("does the referring expression in this sentence actually
refer to the named component as an architectural participant?") is what
`LAYERED_COREF_RULES`'s approve-condition already says, in more words. It does not
survive the second model:

| model | arm | stage gold | stage spurious | composed F1 | F2 | TP | FP | verdict |
|---|---|---|---|---|---|---|---|---|
| terra | ctl | 33.3 | 4.7 | 93.18 | 94.02 | 181.0 | 22.7 | — |
| terra | `nocoreffocus` | 33.3 | 5.0 | 92.91 | 93.82 | 181.0 | 23.3 | neutral (TP ±0.0, F1 −0.3) |
| luna | ctl | 53.7 | 4.0 | 89.39 | 91.12 | 174.3 | 39.3 | — |
| luna | `nocoreffocus` | 54.3 | 5.0 | 88.94 | 91.26 | 175.3 | 45.7 | **refused** (FP +6.3, p = 0.10) |

That is the round's third instance of the same asymmetry: **at the lenient gate a
restatement is redundant, and at the strict gate it is reinforcement.** The typed coref
rubric (luna FP +34.0), the typed rubric with the default restated (luna FP +34.0), and
now the focus deletion (luna FP +6.3) all weaken the same framing and all cost luna
precision, while terra reads each of them neutral. A prompt cut that holds on the
stricter model says nothing about the laxer one.

**The head is `s_linker87`: two deletions, both at places where a sentence was
restated, and nothing else.**
