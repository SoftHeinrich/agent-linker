# The uniform-schema round (s116–s119) — can the three judges write one reply?

`s_linker114` made the three judging passes one loop over three `JudgeSkill`
declarations, which put the remaining differences in a table for the first time. Two of
them are on the wire: the lenient and strict gates reply
`{"validations":[{case, claim[, objection], approve}]}` — one builder, one parser — and
the sortal gate replies `{"judgments":[{case, denotation, claim}]}`: another key, the
verdict before the quote, an enum where the others carry a boolean.

This round asks whether that divergence is load-bearing. Four arms, each one declared
difference at one gate, three samples, both models, **every arm of a gate in the same
invocation**, on candidate sets that are byte-identical across arms because both
proposers in front of these gates are deterministic scans.

Tooling: `pilot/objection_audit.py` (level 1, no calls),
`pilot/test_uniform_schema_arms.py` (104 prompts, no calls),
`pilot/nextgen_pilots.py --gate {lenient,sortal}` driven by
`pilot/run_uniform_round.sh`, scored by `pilot/judge_round_stats.py`.

**`skills` (s114) is carried as the in-set null.** It is the head's judging expressed as
three skills and byte-identical to it by test (284/284 batches over six recorded runs,
`pilot/test_s114_skills.py`), so it is a real arm whose true delta is zero. Every number
below is read against it, and it earns its place immediately: on **luna's sortal gate the
null itself reads net −9.3** (gold −2.0, spurious +3.3) against the control it copies.

## The answer

**No, and the sortal gate is where it fails.** The one arm that actually unifies the
three schemas — `s_linker119`, the sortal gate replying in the other two's key, order
and boolean — is the worst arm of the round on **both** models:

| gate | arm | terra gold / spurious | terra net | luna gold / spurious | luna net |
| --- | --- | ---: | ---: | ---: | ---: |
| lenient | `skills` (null) | 148.7 / 13.7 | 432.3 | 153.3 / 34.3 | 425.7 |
| lenient | `ground` s116 | 146.0 / **8.0** | 430.0 | 146.0 / **8.0** | **430.0** |
| lenient | `verdictfirst` s117 | 147.7 / 15.7 | 427.3 | 150.3 / 31.7 | 419.3 |
| sortal | `skills` (null) | 20.3 / 6.7 | 54.3 | 19.0 / 10.3 | 46.7 |
| sortal | `ground` s118 | 19.7 / 4.7 | 54.3 | 19.0 / 10.7 | 46.3 |
| sortal | `unify` s119 | **15.7** / 1.7 | **45.3** | **11.3** / 3.3 | **30.7** |

`3*gold − spurious`, the F2 derivative at the head's operating point. Per five-project
run, three samples.

- **`s119` is refused outright: net −9.0 (terra) and −16.0 (luna) against the null**,
  gold −4.7 and −7.7, both at the n = 3 floor (p = 0.10). It keeps 17.3 and 14.7 links
  where the null keeps 27.0 and 29.3, at precision 0.904 and 0.773. It did not become a
  worse judge; it became a *stricter* one.
- **The mechanism is the one the typed round named, running backwards.** That round found
  that typing a rubric deletes its default; this finds that **untyping one imports a
  different default.** The enum makes this gate answer a classification question and keep
  only a positive `participant`; `approve`/`reject` is the lenient gate's vocabulary, and
  the lenient gate's default is the opposite of this gate's. The vocabulary carries the
  polarity, so the schema cannot be shared without moving it.
- **The polarity clause of the judge round's law predicted exactly this.** The sortal
  gate's stream is 0.31 / 0.19 gold — the dirtiest of the three — and its default has to
  be reject-by-default. Writing it in the vocabulary of the 0.70 / 0.74 gate is writing
  the wrong base rate into it.

## The field: free at one gate, a sign flip at the other

`objection` is the only part of a uniform schema that costs tokens, and it was priced
before any arm ran (`pilot/objection_audit.py`, six recorded runs, no calls): the strict
gate's ground averages 78 chars on terra and 85 on luna — **5 and 22 on the rows it
approves, 112 and 104 on the rows it rejects** — and the two gates that would gain it
judge 300.3 / 305.7 cases a run. A uniform schema is **+5.9k / +6.5k completion tokens a
run against the 28.6k judging already spends: +20% / +23%. Uniformity is not a saving.**

**At the sortal gate the field is free and buys nothing**: net ±0.0 (terra) and −0.3
(luna), every p = 1.00. **At the lenient gate it is the strongest precision instrument
this branch has measured, and it changes sign between models under F2:**

| model | arm | gold | spurious | net | within-arm spread |
| --- | --- | ---: | ---: | ---: | ---: |
| terra | null | 148.7 | 13.7 | 432.3 | [1, 4, 3] |
| terra | `ground` | 146.0 | **8.0** | 430.0 | [6, 2, 4] |
| luna | null | 153.3 | 34.3 | 425.7 | [22, 18, 4] |
| luna | `ground` | 146.0 | **8.0** | **430.0** | [3, 6, 5] |

- terra: −2.7 gold to save 5.7 spurious = **0.47 false positives per gold link**, against
  the three this budget demands. A loss.
- luna: −7.3 gold to save **26.3** spurious = **3.6 per gold link**. A win, net +4.3
  (p = 0.10), and precision 0.817 → 0.948.
- **Under the standing rule — an arm the second model refuses is refused — this is out.**
  It is recorded as the round's frontier because the *direction* is identical on both
  models and only the exchange rate differs, and because it is the same trade `s_linker111`
  measured (−8.0 / −5.3 gold, −4.3 / −10.0 spurious) with a much better rate on luna.
- **It also stabilises the gate on the model that needs it**, which `s111` did not:
  luna's lenient gate moves 22, 18 and 4 links between identical samples under the null
  and 3, 6 and 5 under this arm; terra's moves 1, 4, 3 and 6, 2, 4, i.e. unchanged.
  `s111` was 2–5× *less* stable than its control on both models. And the arm the round
  refuses on order is the unstable one here: `verdictfirst` reads 2, 14, 14 on terra
  against the null's 1, 4, 3. **Asking for the ground is not
  the same kind of change as asking for the readings**: one adds a field the rubric
  already licenses, the other adds a step the model resamples.
- Where it lands is where the law says: **luna's whole gain is teammates**, the one
  project whose lenient stream is half spurious (27.0 → 2.0 spurious at 48.3 → 44.7
  gold, net +14.0). On the four projects whose stream is 84–90% gold it is flat or
  slightly negative.

## The order: refused in both directions

| arm | gate | who adopts whose order | terra net | luna net | verdict |
| --- | --- | --- | ---: | ---: | --- |
| `s_linker112` | sortal | sortal takes the lenient order (quote first) | +4.7 | −6.7 | refused (judge round) |
| `s_linker117` | lenient | lenient takes the sortal order (verdict first) | **−5.0** | **−6.3** | **refused, both models** |

`s117` loses gold on both models (−1.0, −3.0) and net on both, at the gate with 150 gold
a run and five contributing projects — so this is the cleaner of the two measurements,
and it says the two lenient/strict judges already have the right order. Together the two
arms close the order question: **neither gate may adopt the other's field order**, and
`s_linker48`'s separation — demanding a committed quote is what pays — now has a
measurement at the gate that can carry it.

## What is unifiable, and it is not on the wire

The three verdict *parsers* are one parser. `s_linker114` now declares each skill's
verdict as `verdict_field` plus `verdict_values` (`None` = the head's boolean `approve`
contract) and derives the unanswered-case default from it, so the enum branch and the
boolean branch are one function and each judge's polarity is one expression. Proven, not
argued: `pilot/test_s114_skills.py` runs the head's methods and the variant's side by
side over six recorded runs under **two** stubbed replies — one that answers nothing and
one that answers every case, alternating the verdict — and reads **284/284 judging
batches identical, 1444 kept decision rows identical**.

That second reply is the round's methodological note. The first version of the test
stubbed `_ask` to answer nothing, so no case was ever kept: it proved the prompts and
each judge's *default*, and the kept-set assertion compared two empty sets. It passed
142/142 while the variant's `_classify_denotations` returned `approved: True` on kept
rows where the head returns `False` and lets `_judge_partial_names` write the keep — an
intermediate difference that ends at the same place and that no assertion could see.
**A refactor's equivalence test has to exercise the polarity it is preserving, not only
the default.**

## The round's result

| | decided at | verdict |
| --- | --- | --- |
| one reply schema at all three judges (`s119`) | level 2, both models | **refused** — net −9.0 / −16.0; the boolean imports the wrong default |
| `objection` at the sortal gate (`s118`) | level 2, both models | **refused as pointless** — net ±0.0 / −0.3, and +20% tokens |
| `objection` at the lenient gate (`s116`) | level 2, both models | **refused under the sign-flip rule**; the round's frontier — luna net +4.3 at 3.6 FP per gold, and 4× more stable than the null |
| the lenient gate adopting the sortal order (`s117`) | level 2, both models | **refused** — net −5.0 / −6.3, gold down on both |
| one verdict parser in code (`s114`) | level 1, no calls | **adopted** — 284/284 batches and 1444 kept rows identical |

**The head does not move. `s_linker110` stands.** No arm is composed and none is owed an
E2E.

### What the round establishes

> **A reply schema is not a container for a verdict; it carries the verdict's default.**
> The field set is nearly free to unify, the field *order* is refused at both gates that
> could adopt each other's, and the verdict's *type* cannot be unified at all — because
> `participant`/`associated` and `approve`/`reject` are not two spellings of one question,
> they are two base rates.

Which is the judge round's third clause, arrived at from the other side: that round
observed the three polarities sitting in the base-rate order of the streams that feed
them, and this one shows that the schema is where the polarity lives, so unifying the
schema moves the polarity whether or not the arm means to.
