# Two `s_linker121` ablations — the anchors block, and the scan's one refusal

Two questions, asked together because they are the two pieces of the name stage that
nothing had yet priced on this variant: the `anchors` evidence the judge is shown, and
`_only_inside_another_name`, the single place the deterministic layer ends a case rather
than opening one.

**Anchors stay; the refusal is removed.** A 73-byte clause DOES replace the anchor block
at the stage (§4), and the E2E then splits: `s_linker122` is quality-changing in its
favour on terra (FP -8.0, macro F2 +0.9) and against it on luna (TP -10.3, F2 -2.9), both
at the n=3 floor, so the cut is refused as the head and recorded as priced. The
refusal is worth much less than it looks — the judge rejects 140 of 144 case-samples of
exactly what it blocks, and it can never gain a gold link — so it was taken out for the
simplification it buys: **no predicate in the deterministic layer now ends a case.**

Tooling: `approach/pilot/s121_ablations.py` (the arms, and a `--verify` mode that answers
the deterministic half with no calls), `approach/pilot/anchor_diff.py` (which cases two
arms disagree on, and what evidence each carried — no calls), scored by
`approach/pilot/union_stats.py` — the union round's own paired sign-flip test, over the
dumps here. Reproduce from `approach/`:

    ../.venv/bin/python pilot/s121_ablations.py --verify

    LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.6-terra OPENAI_REASONING_EFFORT=none \
    OPENAI_SERVICE_TIER=default \
      ../.venv/bin/python pilot/s121_ablations.py --samples 3 --dump dump_terra.json
    ../.venv/bin/python pilot/union_stats.py dump_terra.json --arms head noanchor

Every arm of a claim ran in the same invocation as its head, per the measurement policy.

## 1. The anchors block — REFUTED (first set: both models; see §3 for luna)

Arm `noanchor` removes both halves at once: the `anchors` block is not computed and not
printed, and the rule's `anchors` line goes with it. Dropping the block but keeping the
line would leave the judge a line about a field no case carries, which is a different
change. The line is **sliced off `_FIELD_LINES`** rather than retyped, so a drift in the
constant is a drift in the arm.

Per five-project run, three samples, all arms in one invocation per model:

| model | arm | candidates | kept | gold | spurious | precision | calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| terra | head | 296 | 189.0 | 169.7 | 19.3 | 0.898 | 14.0 |
| terra | `noanchor` | 296 | 200.0 | 169.0 | **31.0** | 0.845 | 14.0 |
| luna | head | 296 | 198.7 | 168.3 | 30.3 | 0.847 | 14.0 |
| luna | `noanchor` | 296 | 211.3 | 169.3 | **42.0** | 0.801 | 14.0 |

Paired sign-flip over 15 (sample, project) units, `noanchor` minus head:

| model | gold | p | spurious | p | net (3·gold − sp) | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| terra | −0.7 | 1.000 | **+11.7** | **0.004** | **−13.7** | **0.012** |
| luna | +1.0 | 0.875 | **+11.7** | 0.062 | −8.7 | 0.436 |

**Anchors buy precision and nothing else.** Gold is flat on both models — the sign is not
even consistent — and spurious is +11.7 a run on each, at identical candidates and
identical call count. (§3 re-runs `noanchor` against the post-removal head in a fresh
invocation: terra reproduces at spurious +7.0 / net −11.0, **luna does not**, reading net
+0.7. Both are reported; only within-invocation comparisons count.) The bytes are not free (the union round measured the anchor block
at 27.9% of a judging call, which is why `s_linker88` compacted it), but this says they
are paid for.

**The row it lands on is model-dependent, and that is the round's one transferable note.**

| model | whole name | alias | word only |
| --- | ---: | ---: | ---: |
| terra spurious/unit | +0.33 (p 0.250) | **+1.87 (p 0.062)** | +0.13 (p 0.500) |
| luna spurious/unit | −0.53 (p 0.625) | +0.87 (p 0.062) | **+2.00 (p 0.031)** |

Same cut, same direction in the total, different stream underneath — terra leaks on the
alias row, luna on the word-only row. Both are the rows where the sentence does *not*
write the catalog name as such, which is exactly the population anchors are evidence
about: other sentences of this document that do write it. **The field restrains the rows
that need a document-level fact and is inert on the row that does not** (terra's
whole-name row moves +0.33 spurious, p = 0.250).

## 2. The scan's one refusal — REMOVED, and not for the reason it was expected to be

`_only_inside_another_name` drops a one-word candidate when *every* writing of that word
sits inside some other component's fully written name.

**Level 1, no calls** (`--verify`): over all five projects the refusal fires on **exactly
one**, bigbluebutton, where it drops **12 pairs, of which 0 are gold** — `HTML5` inside
"HTML5 client" claimed for `HTML5 Server`, `Redis` inside "Redis DB." claimed for
`Redis PubSub`, and ten more of that shape. On the other four projects the arm is the
head by construction: same candidates, **byte-identical prompts**, which the same mode
checks. Calls were spent for it only where it differs.

So the ceiling is known before any call is made: **removing the refusal cannot gain a
gold link on this benchmark.** It can only add false positives, bounded at +12 a run.

**Level 2 — and the plain arm changes two things at once.** With the refusal off,
bigbluebutton goes 90 → 102 cases and 4 → 5 judging calls, so the batch boundaries move.
The union round measured the batch boundary as an effect in its own right ("no sentence
of any rule moved that row as far as the batch boundary did"), so a second arm,
`norefusal_split`, holds it fixed: it sends **the head's 4 calls verbatim** and judges the
12 refused pairs in one extra call of their own. Whatever it differs by is the added
cases and nothing else.

Bigbluebutton only, three samples, each arm beside its head in one invocation:

| model | arm | gold | spurious | calls | of the 12 refused pairs |
| --- | --- | ---: | ---: | ---: | --- |
| terra | head | 53.7 | 11.7 | 4.0 | — |
| terra | `norefusal` | 51.7 | 11.0 | 5.0 | **0 approved / 36 rejected** |
| terra | head (split set) | 53.7 | 10.3 | 4.0 | — |
| terra | `norefusal_split` | **53.7** | 11.3 | 5.0 | **0 approved / 36 rejected** |
| luna | head | 50.3 | 12.0 | 4.0 | — |
| luna | `norefusal` | 46.7 | 11.3 | 5.0 | **0 approved / 36 rejected** |
| luna | head (split set) | 47.7 | 11.3 | 4.0 | — |
| luna | `norefusal_split` | 51.0 | 13.7 | 5.0 | **4 approved / 32 rejected** |

**The refusal is very nearly redundant as a filter.** Of 36 case-samples on each model in
each arm, terra's judge rejected all 36 twice over — 72 of 72 — and luna's rejected 68 of
72. `QUALIFIED_CLAUSE` and `STRICTER_CLAUSE` already do the work in the prompt. This is
the `s_linker92b` result again at a different predicate: **the gate rejects what the code
was refusing, so the code's refusal buys only what the gate leaks.**

**What it buys is 0.0 spurious a run on terra and 1.3 on luna, plus a call.** Every
approval is non-gold by construction, since the 12 pairs contain no gold.

**The gold column in this table is noise, and the split arm is what proves it.** The
split arm differs from its head by 12 non-gold pairs only, so it *cannot* change gold —
yet luna reads +3.3. That is a direct measurement of the within-invocation sampling band
on this project: **±3.3 gold over byte-identical prompts.** The plain `norefusal` arm's
−2.0 (terra) and −3.7 (luna) sit inside it, so the batch-reshuffle reading those numbers
invite is **not supported**; terra's split arm reads gold ±0.0 (p = 1.000) against a
plain arm at −2.0, on the same 12 pairs.

**Verdict: REMOVED.** The round first recommended keeping it on cost — 12 lines of code
against +1 judging call and 1.3 spurious a run on the laxer model — and it was removed
anyway, deliberately, for the simplification: **no predicate anywhere in `s_linker121`'s
deterministic layer now ends a case.** That sentence was already in the module as the
justification for `SKIP_QUALIFIED = False` and this is what makes it true.

What was bought and what was paid, stated plainly: the module loses a predicate, a helper
and their two docstrings; it gains one judging call on bigbluebutton (4 → 5, ~1 of the 15
a five-project run now makes) and 1.3 spurious links a run on luna, 0.0 on terra. Both
invariant suites move with it rather than loosening — the one-word reference becomes the
*unrefused* scan and the 12 pairs are pinned as cases that must now be present
(`test_s121_standalone.py` 135/135, `test_s121_union.py` 2783/2783).

The arms live on in the pilot as `refusal` / `refusal_split`, stated in the direction
that now changes something. **A pilot that prices a removed predicate has to own it**, or
the round stops being reproducible the moment the head moves.

## 3. Can a clause do the anchors' job? — the first, verbose attempt (superseded by §4)

The obvious follow-up to §1: **is the anchor block carrying a fact, or is it patching a
rule that does not say enough?** Those have different repairs, and only one of them is
a prompt. Two substitutes, both against the head in one invocation per model, three
samples, five projects, on the post-removal head (308 candidates, 15 calls):

* **`noanchor_clause`** — anchors gone, plus one clause saying what they were evidence
  for: that where the sentence does not write the name in full, the surface is what the
  case *reports*, not something the document has certified, and a short form established
  elsewhere is not established in this sentence. A **weighing**, which is what the design
  law allows a prompt to carry.
* **`anchor_count`** — the block replaced by its cardinality (`Anchors: 2`), and the
  rule's anchors line reduced to match. This separates two readings of §1: is the judge
  restrained by seeing **how** the document writes this name, or merely by learning
  **that** it writes it elsewhere?

Per five-project run, and the paired sign-flip against head over 15 units:

| model | arm | gold | spurious | precision | gold Δ | p | sp Δ | p | net Δ | p |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| terra | head | 169.0 | 21.7 | 0.886 | | | | | | |
| terra | `noanchor` | 167.7 | 28.7 | 0.854 | −1.3 | 0.312 | +7.0 | 0.055 | **−11.0** | **0.016** |
| terra | `noanchor_clause` | 162.7 | 13.3 | **0.924** | **−6.3** | **0.008** | **−8.3** | **0.020** | **−10.7** | **0.027** |
| terra | `anchor_count` | 166.7 | 20.3 | 0.891 | −2.3 | 0.062 | −1.3 | 0.578 | −5.7 | 0.070 |
| luna | head | 162.0 | 31.7 | 0.836 | | | | | | |
| luna | `noanchor` | 163.7 | 36.0 | 0.820 | +1.7 | 0.391 | +4.3 | 0.406 | +0.7 | 0.969 |
| luna | `noanchor_clause` | 161.3 | 22.3 | **0.878** | −0.7 | 0.891 | −9.3 | 0.102 | +7.3 | 0.245 |
| luna | `anchor_count` | 170.7 | 34.3 | 0.833 | **+8.7** | **0.008** | +2.7 | 0.652 | +23.3 | 0.066 |

**The clause works, and that is not the same as being a substitute.** It targets exactly
what §1's error analysis said was leaking: terra's alias row goes from 16.0 spurious
under `noanchor` to **3.0**, against the head's 7.7 — the clause is *better* than the
anchors at the thing the anchors were doing. It pays for it in recall on a row it was
never aimed at: terra's word-only gold **16.7 → 12.0**. A clause cannot know which
surfaces are genuine, so the only thing it can move is the judge's global threshold; the
anchors move what the judge knows about **one name**. **A weighing cannot be aimed, and a
fact is aimed by construction** — which is why the design law's fact/weighing split shows
up here as a precision/recall split rather than as a right and a wrong answer.

**The count is the round's real frontier and it splits by model.** On luna, replacing the
anchor sentences with their number is **+8.7 gold a run (p = 0.008)** at +2.7 spurious —
the word-only row goes 11.0 → 17.7 gold — while on terra the same arm is −2.3 gold
(p = 0.062). Sign-flipped on the measure that matters, so it is **refused** by the
branch's own rule; recorded because it is 26% off the judging call (16.0k chars → 11.9k)
and because the disagreement is informative: on the stricter model the anchors' *content*
is doing work, and on the laxer one the same content is a distraction that costs it gold
it recovers when the block becomes a number.

**§1's luna result did not reproduce in this set** (net +0.7, p = 0.97, against a first
set that read spurious +11.7). Two things changed at once — the invocation set, and the
head, which lost the nesting refusal between the rounds — so this is not a trend, only a
reminder that **the anchors' value is confirmed twice on terra and is unstable on luna.**
Absolute levels drift; only within-invocation comparisons count, and both sets are
reported rather than the better one.

## 4. Removing the anchors outright — `s_linker122`, end to end

§3 said a clause cannot substitute for the anchors. That was measured against a 456-byte
clause that enumerated readings and restated `STRICTER_CLAUSE`. **Rewritten to 73 bytes
it can**, at the stage, and the cut is then worth an E2E of its own: the anchor block is
27.9% of a judging call, the largest single cut available to it.

    That a surface can name this component is not evidence that it does here.

Not a restatement of `STRICTER_CLAUSE`, which is about an ordinary English word
coinciding with a name: the row that leaks is the one the **alias stage** supplies, where
the surface is not an ordinary English word at all, and no rule in the module spoke about
it. `s_linker122` is the head with the block gone, the rule's anchors line gone with it,
and this sentence in their place. Judging call **16 046 → 10 754 chars (−33.0%)**, a
five-project run **86 090 → 63 022 (−26.8%)**, same call count.

Stage, three samples a model, every arm in one invocation with its head:

| arm | terra gold Δ | p | terra sp Δ | p | luna gold Δ | p | luna sp Δ | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `noanchor`, no clause | −0.3 | 1.000 | **+8.0** | **0.031** | −2.7 | 0.312 | +8.0 | 0.266 |
| **73-byte clause** | **±0.0** | 1.000 | −1.0 | 0.750 | −5.0 | 0.438 | **−11.0** | **0.008** |
| 456-byte clause | **−5.3** | **0.004** | −8.0 | 0.009 | −2.7 | 0.543 | −11.7 | 0.016 |

Terra's alias row is the mechanism in one line: head **8.0** spurious, no clause **16.0**,
73-byte clause **7.0** — closed, at a word-only row the long clause wrecked (16.3 → 12.3
gold) and the short one leaves alone (16.3 → 15.7). `noanchor` without a clause has now
reproduced its penalty in **three consecutive invocation sets**.

**End to end, three paired runs a model, both arms in every invocation, arm order
alternating by run** (`pilot/run_noanchor_e2e.sh`,
`../results/noanchor_e2e_{terra,luna}_r{1,2,3}_20260914`, scored by
`pilot/score_runs.py`):

| model | arm | TP | FP | macro F1 | macro F2 | calls | F1 range |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| terra | `s_linker121` | 182.3 | 28.0 | 92.51 | 94.18 | 73.3 | 2.21 |
| terra | **`s_linker122`** | **184.0** | **20.0** | **93.94** | **95.13** | 73.0 | **0.60** |
| luna | `s_linker121` | 185.0 | 42.3 | 90.57 | 93.84 | 75.0 | 2.35 |
| luna | `s_linker122` | **174.7** | 42.7 | 88.79 | 90.94 | 74.7 | 1.61 |

**terra is QUALITY-CHANGING in the arm's favour** — TP +1.7 (p = 0.40), **FP −8.0**
(0.20), macro F1 **+1.4** (0.20), macro F2 **+0.9** (0.10, the n=3 floor) — and **every
s122 run's F2 is above every s121 run's** (94.77/95.01/95.60 against 94.10/94.05/94.40),
at a run spread a quarter of the control's. **luna is QUALITY-CHANGING against it**:
**TP −10.3** (p = 0.10), FP +0.3 (1.00), macro F1 −1.8 (0.20), macro F2 **−2.9** (0.10).
Luna's loss is **pure recall at unchanged precision**, which is the signature of evidence
removed rather than a rule mis-set.

**REFUSED as the head**, on the branch's own sign-flip rule: the same cut is
quality-changing in opposite directions on the two models, both at the n=3 floor. The
saving is real and so is terra's gain; neither buys a regression of 10.3 true links on
the other model.

**Both models' whole effect is one project, and it is the project the error analysis
named.** Per project, mean of three runs, s122 minus s121:

| project | gold | terra ΔTP | terra ΔFP | luna ΔTP | luna ΔFP |
| --- | ---: | ---: | ---: | ---: | ---: |
| teammates | 57 | **+2.0** | **−3.7** | **−8.0** | +2.3 |
| bigbluebutton | 62 | −0.3 | **−4.3** | −2.0 | −1.3 |
| mediastore | 31 | ±0.0 | ±0.0 | +0.7 | ±0.0 |
| teastore | 27 | ±0.0 | ±0.0 | −1.0 | ±0.0 |
| jabref | 18 | ±0.0 | ±0.0 | ±0.0 | −0.7 |

Teammates is 8.0 of luna's 10.3 lost links and both of terra's gains. It is the project
whose alias table binds `GAE` and whose sentences are dotted package paths — the exact
population `pilot/anchor_why.py` found the anchor arms moving (31% of changed cases
against 5.6–9.2% of agreed ones). **One clause, one population, opposite signs.**

**The E2E disagreed with the stage on both models, in opposite directions, and that is
the round's methodological result.** The stage read terra neutral (gold ±0.0, spurious
−1.0) and the composed run reads FP −8.0 and F2 +0.9; the stage read luna −5.0 gold and
the composed run reads TP −10.3, **roughly double**. The coreference linker behind the
name stage neither recovered luna's dropped pairs nor stayed out of the way — luna's
composition statistic is **+24.1 (p = 0.10)**, the largest in this round, against terra's
+1.1 (p = 0.50). The standing caveat on this branch is that a stage arm flatters a change
by hiding composition; here it **understated** the change in both directions at once.

## 5. The clause was over-cut — the scope, and what a fact cannot do for a weighing

§4 refused `s_linker122` as the head on a sign flip: terra quality-changing in its
favour, luna quality-changing against it at **TP −10.3**, pure recall at unchanged
precision. That signature says evidence was removed rather than a rule mis-set, and it
was worth naming before the arm was either dropped or adopted.

**Luna's loss is one sentence.** `pilot/noanchor_fn.py` (no calls) reads the two arms'
per-project link CSVs out of the six E2E runs and counts a lost gold pair by how many
runs lost it, so a pair lost in three runs of three is separated from one lost in one.
Of the 8.0 TP luna loses on teammates, **seven are teammates S1**, each lost in **3 runs
of 3**:

    S1  "Architecture contains UI Component, Logic Component, Storage Component,
         Common Component, Test Driver Component, E2E Component, Client Component."
         -> UI, Logic, Storage, Common, Test Driver, E2E, Client   (7 gold links)

Every one is `writes=whole name`. Not the alias row the clause was aimed at, not the
dotted-path row the anchors were moving: the row that was never in question.

**The cause is a scope the rewrite dropped.** The 456-byte clause of §3 opened with
*"Where this sentence does not write the component's name in full…"*. The 73-byte
rewrite kept the weighing and lost the scope, so it reads on every case. Unscoped it
asks for evidence that the surface is used **for** this component **here**, and S1 is a
bare enumeration that says nothing further about anything — which is exactly the case
the rule's own `MENTION_COUNTS` protects: *"A mention that says nothing further about
the component still counts as a valid link."* The clause and the rule contradict each
other on whole-name bare mentions. Terra resolves it toward the rule, luna toward the
clause, and that is the whole of the sign flip.

**Two repairs, and they are different kinds of thing.**

  * `noanchor_scoped` — the clause with its scope restored, **125 B**. A *weighing*,
    narrowed to the rows it was ever about.
  * `noanchor_plain` — **no clause at all**. The leak is an assertion in the EVIDENCE:
    `WRITES["alias"]` renders the row as *"a short form the document established for
    it"*, and `established for it` is an authority claim the case makes about itself.
    With the block present the judge could check it against the document's own
    sentences; with the block gone nothing in the call can contradict it. So state what
    the match computed and no more — *"a short form listed for it elsewhere in the
    document"*. Provenance stays, authority goes. A *fact*, changed in code, costing
    zero prompt bytes, and it cannot reach a whole-name case through any case line: a
    whole-name case never renders the alias row.

Each arm ran against the head **in its own invocation, with the head and the 73-byte
clause alongside it**, so every column below is a within-set comparison. The two sets
are NOT comparable to each other — the branch's standing finding is that absolute
levels drift between invocation sets, and the 73-byte row drifting between them (terra
+3.3 against +0.3 gold) is that finding reproducing inside this table.

| set | arm | terra gold | terra sp | p | luna gold | luna sp | luna whole-name gold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 73 B, unscoped | +0.3 | −2.3 | 0.328 | −8.7 | −1.0 | **−2.07** |
| A | **125 B, scoped** | ±0.0 | **−3.7** | 0.150 | −2.3 | −0.7 | **−0.13** |
| B | 73 B, unscoped | +3.3 | −0.7 | 0.906 | −5.0 | −3.7 | **−2.07** |
| B | no clause, fact | +2.0 | **+7.3** | **0.047** | +0.3 | +7.0 | −0.47 |

(gold and spurious are per five-project run; the last column is per unit, the row the
defect lives on.)

**The scope is confirmed, at the row the diagnosis named.** In set A the whole-name row
goes **−2.07 → −0.13** on luna, and luna's run-level gold loss goes −8.7 → −2.3, at no
cost on terra — terra's spurious improves, −2.3 → −3.7.

**The clause-free repair is REFUSED, and it is the more interesting result.** It does
what it was built to do: luna's whole-name row goes −2.07 → −0.47 and run-level gold
goes −5.0 → +0.3, S1 recovered without a clause. But it **reopens the row the anchors
were holding** — terra spurious **+7.3 a run (p = 0.047)**, of which the alias row is
+1.47 a unit, and luna +7.0. Changing what the evidence *says* about one field moved
what the judge knows; it did not move the threshold the anchor block was setting. **A
fact cannot do a weighing's job.** That is the branch's design law — facts in code,
weighings in the prompt — read in the direction it is usually not: not only may a
weighing not be smuggled into the evidence, a fact may not be asked to stand in for one.

**What ships is the scoped clause**, and `s_linker122` carries it. The E2E of §4 priced
the unscoped version, so it does not describe the file that ships and the arm is
re-measured end to end against `s_linker121` on both models
(`STAMP=20260914scoped pilot/run_noanchor_e2e.sh`).

## What the ablations say together

The two pieces sit on opposite sides of the branch's design law and the measurements
follow it. **Anchors are a fact the judge cannot derive** — other sentences of the
document are not in the case it is holding — and removing them costs precision.
**The refusal is a fact the judge largely *can* derive**, because the covering name is
written in the very sentence the case prints, and removing it costs almost nothing. The
one thing the judge cannot do is decline to spend the call.

§3 sharpens that into the round's transferable result. The question "is this fact
patching an under-specified rule?" has a **third** answer besides yes and no: the clause
that states what the fact was evidence for is *better than the fact* at the failure the
fact was covering (alias-row spurious 7.7 → 3.0 on terra) and *worse overall*, because a
clause can only move the judge's threshold everywhere while a fact moves what it knows
in one place. **Ask of a candidate clause not whether it repairs the failure, but whether
it can be aimed at it.**

It also produces a concrete error mechanism worth carrying: **the `writes=alias` line is
an assertion the case makes and nothing else in the call can contradict.** The alias
stage bound `GAE`, so six teammates sentences about the platform read to the judge as
sentences about the component, and only the anchors — which show the document writing
`GAE Datastore` — say otherwise. This is the alias table's **third** job, after admitting
full-name candidates and (formerly) suppressing partial ones: it can also mislead one
judge, and the anchors are what the head has against that.

## Caveats

* Stage-level, not end to end. The coreference linker runs behind the name stage and
  re-proposes some of what it declines; no composition check or E2E was bought, because
  both arms read negative and the head does not move.
* The refusal claims rest on 3 samples of one project (the only one where the predicate
  fires), so the sign-flip floor there is p = 0.25. The decisive numbers for it are the
  deterministic ones (12 pairs, 0 gold, 4 of 5 projects untouched) and the 140-of-144
  reject count, neither of which is a paired-mean claim.
* Alias tables are pinned from `consolidation_e2e_terra_r1_20260825`, as every stage
  pilot on this branch does, so the knowledge stage's ~2.8-term run-to-run variation is
  held out of both arms.
