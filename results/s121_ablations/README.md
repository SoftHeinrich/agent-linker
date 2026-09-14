# Two `s_linker121` ablations — the anchors block, and the scan's one refusal

Two questions, asked together because they are the two pieces of the name stage that
nothing had yet priced on this variant: the `anchors` evidence the judge is shown, and
`_only_inside_another_name`, the single place the deterministic layer ends a case rather
than opening one.

**Anchors stay; the refusal is removed.** Anchors are worth ~7-12 spurious links a run
on terra at no recall, and neither of the two substitutes tried in §3 replaces them. The
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

## 3. Can a clause do the anchors' job? — asked, and refused on a sign flip

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
