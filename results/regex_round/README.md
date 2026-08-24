# The regex round (s92a–s92f) — can the entity extraction pass be a scan?

**The question.** `s_linker92`'s full-name linker opens with an LLM pass over the
document that reports which sentences reference which components. Read its own
instruction:

> Report a reference only when the sentence itself writes the component's name,
> spelled as the COMPONENTS list spells it or as one of the KNOWN ALIASES; count a
> name written with different spacing, hyphenation or compound joining as that name.
> […] Among the sentences that do name it, report every one, however incidental the
> mention: whether the mention carries an architectural claim is decided later.

There is no judgement left in that. It is a surface test with the weighing explicitly
deferred to the gate one stage later. **A contract with no judgement in it is a
regex** — and it is one this branch already states: the whole-name row of the
surface-realization relation.

**The verdict.** `s_linker92a` — the extraction call deleted, the contract run as a
scan — **is quality-neutral end to end on terra and recall-led on luna**, and it
closes the branch's "never reached a judge" false-negative bucket almost completely
(terra 4.7 → 0.0 a run, luna 7.7 → 0.3). It removes 9 of the ~84 LLM
calls a five-project run makes, adds no deterministic machinery whatsoever, and reads
macro F2 **+2.0 (terra) / +1.8 (luna)** at macro F1 +0.4 / −0.9 (neither significant).
The four variants built to repair its expected failure modes are all refused: three
because the judge does the job they were built for, one because it made things worse.

Tooling: `pilot/regex_extract_audit.py` (level 1, no calls), `pilot/regex_proposer_pilots.py`
(level 2, the stage arm), `pilot/run_regex_e2e.sh` (level 4, the E2E batch),
`pilot/regex_fn_analysis.py` (the false-negative decomposition, no calls),
`pilot/regex_round_stats.py` and `pilot/score_runs.py` (permutation tests),
`pilot/test_s92abcd_regex.py` (2316 invariant checks, no calls).

---

## Level 1 — the proposer question, settled off the recorded runs

30 recorded run directories of the s89–s92 extraction pass (15 terra, 15 luna) × 5
projects. The scan is replayed over the same documents, the same catalogs and **the
same discovered aliases**, and the candidate sets are compared pair by pair. Per
five-project run, against 195 gold pairs:

| proposer | pairs | gold | prec | +pairs | −pairs | +gold | −gold | newgold | atrisk |
|---|---|---|---|---|---|---|---|---|---|
| LLM extraction | 175.3 | 150.1 | 0.856 | – | – | – | – | – | – |
| …after its judge | 163.2 | 149.0 | 0.913 | | | | | | |
| the run's final link set | 217.1 | 180.6 | 0.832 | | | | | | |
| `as_spelled` | 169.0 | 142.4 | 0.842 | 15.5 | 21.8 | +4.3 | −12.0 | 4.6 | 12.0 |
| **`any_case`** | 221.9 | **158.3** | 0.713 | 53.3 | 6.8 | **+10.6** | −2.4 | **10.2** | 2.4 |
| `any_case`+skip | 196.9 | 158.3 | 0.804 | 32.9 | 11.3 | +10.6 | −2.4 | 10.2 | 2.4 |
| `any_spelling` | 223.1 | 159.1 | 0.713 | 53.5 | 5.7 | +10.8 | −1.8 | 10.2 | 1.8 |
| both, unioned | 223.5 | 159.5 | 0.714 | 53.5 | 5.3 | +10.8 | −1.4 | 10.2 | 1.4 |
| `any_case`, **no aliases** | 172.0 | 133.0 | 0.773 | 38.8 | 42.2 | +8.5 | −25.6 | 8.4 | 25.6 |
| `any_spelling`, no aliases | 176.0 | 137.0 | 0.778 | 39.0 | 38.4 | +8.7 | −21.8 | 8.4 | 21.8 |

`newgold` is gold the scan proposes that the run's whole three-linker pipeline never
linked; `atrisk` is gold the pipeline holds only because the extractor proposed it.

- **The scan's ceiling is 7.8 net gold pairs a run above the extractor's** (+10.2
  reached, −2.4 at risk). This is the branch's own error shape read back: the
  proposer, not the judge, is where the headroom is.
- **The audit reproduces the branch's own relation table exactly.** The two no-alias
  rows are 172/133 and 176/137, which is what `approach/CLAUDE.md`'s name-relation
  table says `ANY_CASE` and `ANY_SPELLING` yield. The scan is not a new rule; it is
  the relation the module already implements, read at the whole-name row.
- **The alias table is what makes the scan work at all.** Without it the scan loses
  25.6 of the extractor's gold. The knowledge stage is untouched by this round and is
  an *input* to it — the round replaces the extraction pass, not the reading of the
  document.
- **`as_spelled` is too tight** (−12.0 gold) and the fidelity axis above `any_case`
  is only 1.2 gold wide. **Which whole-name fidelity the scan uses is worth ~1 gold
  pair a run; whether it is a scan at all is worth ~8.**

### What the audit could not settle

The scan hands the gate 53.3 more pairs a run. Replaying the gate over the recorded
verdicts brackets the arm — every pair it was never shown counted as rejected, then
as approved:

| arm | policy | TP | FP | macro F1 | macro F2 |
|---|---|---|---|---|---|
| control | – | 180.6 | 36.4 | 91.04 | 92.93 |
| `any_case` | reject | 178.5 | 33.6 | 91.00 | 92.50 |
| `any_case` | approve | 187.8 | 75.7 | 87.16 | 92.70 |
| `any_case`+skip | reject | 178.5 | 32.3 | 91.19 | 92.58 |
| `any_case`+skip | approve | 187.8 | 54.4 | 89.29 | 93.84 |
| both+skip | reject | 179.5 | 32.7 | 91.32 | 92.84 |
| both+skip | approve | 188.8 | 54.8 | 89.43 | 94.11 |

A bracket ~2 pp of F1 wide is not an answer, so level 2 was paid for.

---

## Level 2 — the stage arm, four arms in one invocation, both models

Every arm judges in the same invocation, on the same recorded aliases, composed with
the same run's untouched partial-name and coreference stages. Three runs a side.

**terra**

| arm | proposed | TP | FP | macro F1 | macro F2 |
|---|---|---|---|---|---|
| `ctl` (LLM extraction) | 177.3 | 180.3 | 27.3 | 91.98 | 93.09 |
| **`scan` (s92a)** | 222.0 | **186.7** | 32.7 | 92.43 | **95.12** |
| `scan+e` (s92e) | 222.0 | 182.7 | 32.7 | 91.65 | 93.82 |
| `scan+f` (s92f) | 222.0 | 181.0 | **26.3** | **93.07** | 94.20 |

**luna**

| arm | proposed | TP | FP | macro F1 | macro F2 |
|---|---|---|---|---|---|
| `ctl` | 174.7 | 180.0 | 39.7 | 90.46 | 92.57 |
| **`scan` (s92a)** | 222.0 | **190.7** | 59.0 | 89.61 | **94.35** |
| `scan+e` (s92e) | 222.0 | 189.3 | 70.7 | 87.84 | 92.99 |
| `scan+f` (s92f) | 222.0 | 184.7 | 51.3 | 89.73 | 93.07 |

Permutation test against `ctl`, n = 3 (p floor 0.10):

| arm | model | TP | FP | macro F1 | macro F2 |
|---|---|---|---|---|---|
| `scan` | terra | +6.3 (0.10) | +5.3 (0.30) | **+0.4 (0.10)** | **+2.0 (0.10)** |
| `scan` | luna | +10.7 (0.10) | +19.3 (0.30) | −0.9 (0.60) | +1.8 (0.20) |
| `scan+f` | terra | +0.7 (1.00) | −1.0 (0.70) | **+1.1 (0.10)** | +1.1 (0.10) |
| `scan+f` | luna | +4.7 (0.10) | +11.7 (0.20) | −0.7 (0.70) | +0.5 (0.50) |
| `scan+e` | terra | +2.3 (0.50) | +5.3 (0.30) | −0.3 (0.40) | +0.7 (0.40) |
| `scan+e` | luna | +9.3 (0.10) | +31.0 (0.10) | −2.6 (0.20) | +0.4 (0.70) |

**Calls.** 9.0 extraction calls a run go; the wider candidate set costs 2.0 more
judging calls. Net **−7.0 of ~84 calls a five-project run (−8.4%)**, and one whole
prompt constant (`ENTITY_EXTRACTION_RULES`) plus two methods leave the module.

---

## Level 4 — end to end, three runs per model

`pilot/run_regex_e2e.sh`, `../results/regex_e2e_{terra,luna}_r{1,2,3}_20260822`.
**One arm.** `s_linker92` is byte-unchanged by this round and has three recorded
five-project runs per model from the same week
(`../results/solo_e2e_{terra,luna}_r{1,2,3}_20260821`), so it was not re-run.
**That makes the control cross-set, which this branch normally forbids** — the in-set
claim is the stage arm's; this batch exists to answer the one thing a stage arm
structurally cannot, which is composition.

| | terra ctl | terra scan | | luna ctl | luna scan | |
|---|---|---|---|---|---|---|
| TP | 178.3 | 181.0 | +2.7 (p 0.40) | 177.3 | **188.7** | +11.3 (p 0.10) |
| FP | 27.3 | 32.3 | +5.0 (p 0.40) | 45.0 | 71.7 | +26.7 (p 0.20) |
| macro F1 | 92.14 | 91.36 | −0.8 (p 0.30) | 89.03 | 87.93 | −1.1 (p 0.40) |
| macro F2 | 93.22 | 93.10 | −0.1 (p 0.80) | 91.45 | **93.30** | **+1.9 (p 0.10)** |
| calls | 83.2 | **75.3** | −9.5% | 85.2 | **79.0** | −7.3% |
| F1 run range | 1.64 | 1.29 | | 0.18 | 3.93 | |

**terra is QUALITY-NEUTRAL on all four statistics** (smallest p 0.30). **luna
reproduces the stage arm's reading**: macro F2 +1.9 at the n=3 floor, TP +11.3, macro
F1 −1.1 and not significant.

### Every run, and macro F2 per project

| model | run | ctl TP/FP | ctl F2 | scan TP/FP | scan F2 |
|---|---|---|---|---|---|
| terra | r1 | 176 / 28 | 92.40 | 182 / 38 | 92.99 |
| terra | r2 | 177 / 26 | 93.20 | 180 / 26 | 93.25 |
| terra | r3 | 182 / 28 | 94.04 | 181 / 33 | 93.07 |
| terra | **mean** | 178.3 / 27.3 | **93.22** (range 1.64) | 181.0 / 32.3 | **93.10** (range **0.26**) |
| luna | r1 | 177 / 51 | 91.66 | 191 / 87 | 92.98 |
| luna | r2 | 175 / 36 | 90.64 | 185 / 78 | 92.13 |
| luna | r3 | 180 / 48 | 92.03 | 190 / 50 | 94.78 |
| luna | **mean** | 177.3 / 45.0 | **91.45** (range 1.40) | 188.7 / 71.7 | **93.30** (range 2.66) |

macro F2 per project, mean of three runs:

| project | gold | terra ctl | terra scan | Δ | luna ctl | luna scan | Δ |
|---|---|---|---|---|---|---|---|
| mediastore | 31 | 97.40 | 90.32 | **−7.08** | 96.53 | 97.25 | +0.72 |
| teastore | 27 | 99.00 | 100.00 | +1.00 | 95.12 | 97.80 | +2.67 |
| teammates | 57 | 88.62 | 86.65 | −1.97 | 85.71 | 87.36 | +1.65 |
| bigbluebutton | 62 | 82.86 | 89.64 | **+6.78** | 81.69 | 85.89 | +4.20 |
| jabref | 18 | 98.19 | 98.90 | +0.71 | 98.18 | 98.18 | ±0.00 |

**On luna the scan is F2-positive on four projects of five and negative on none.** On
terra it is positive on three and its whole macro deficit is one project:
**mediastore, −7.08, and the cause is three false positives with one name.** On a
31-gold project the scan links `DB` on three sentences that use the common noun
"database", which the alias stage bound to that component — every run, both models
(luna pays the same 3 FP there and still gains, because its control was not at 0 FP).
That is one alias, one project, and it is `STRICTER_CLAUSE`'s population again, which
is what `s_linker92f` was built for.

**Terra's run spread collapses from 1.64 to 0.26 F2.** A deterministic proposer removes
one of the pipeline's two sampled stages, and it shows: the arm is six times tighter
than the control across runs on terra, though not on luna (2.66 against 1.40), where
the gate's higher approve rate on the added pairs re-introduces the variance.

### The E2E is weaker than the stage arm on terra, and the decomposition says why

The stage arm read terra at macro F2 +2.0; the E2E reads −0.1. Per source, from the
per-variant link CSVs (the phase states of the 0821 runs are not usable for this —
two variants wrote that namespace):

| source | terra ctl TP/FP | terra scan TP/FP | luna ctl TP/FP | luna scan TP/FP |
|---|---|---|---|---|
| `full_name` | 143.3 / 10.3 | **147.7 / 11.3** | 147.7 / 18.7 | **160.3 / 35.7** |
| `partial_name` | 20.3 / 14.0 | 16.3 / 18.3 | 14.0 / 21.0 | 13.0 / 28.3 |
| `coreference` | 14.7 / 3.0 | 17.0 / 2.7 | 15.7 / 5.3 | 15.3 / 7.7 |
| total | 178.3 / 27.3 | 181.0 / 32.3 | 177.3 / 45.0 | 188.7 / 71.7 |

**The change lands where it can reach and is clean there**: on terra the full-name
stage is **TP +4.4 at FP +1.0**. What eats the gain is downstream — `partial_name`
gives back 4.0 TP (mostly relabelling: a pair both linkers propose is now tagged by
the earlier one) and adds 4.3 FP of its own, at a stage this change does not touch and
whose judge runs at ~0.6 precision. **That is the composition effect the stage arm
cannot see, and it is why the batch was owed.** On luna the effect is at the full-name
gate itself (TP +12.6 at FP +17.0), which is the stage arm's 0.736 approve rate on the
added pairs showing up end to end.

### Why it regresses where it does — the scan against the extractor, pair by pair

`pilot/regex_regression_analysis.py` (no calls). Every pair in the symmetric
difference of the two arms' final link sets, pooled over the three paired runs, split
by **the stage it sits at**. Only `full_name` is a stage this change touches:

| | terra TP+ | TP− | FP+ | FP− | net TP | net FP | | luna net TP | net FP |
|---|---|---|---|---|---|---|---|---|---|
| **`full_name`** | 12.0 | 5.7 | 7.7 | 6.7 | **+6.3** | **+1.0** | | **+11.3** | **+18.7** |
| `partial_name` | 0.0 | 4.7 | 10.3 | 6.0 | −4.7 | +4.3 | | −0.3 | +5.3 |
| `coreference` | 1.3 | 0.3 | 1.7 | 2.0 | +1.0 | −0.3 | | +0.3 | +2.7 |
| total | | | | | +2.7 | +5.0 | | +11.3 | +26.7 |

**On terra the change's own stage is TP +6.3 at FP +1.0 and the whole net regression
is at stages it does not touch.** `partial_name` gives back 4.7 TP and adds 4.3 FP on
its own, from a judge that runs at ~0.6 precision in a different invocation set. That
is not the scan; it is what a cross-set control buys you.

What *is* the scan is three mechanisms, and they are separable.

**1. Lowercase surfaces — the use/mention judgement the extractor was doing for free.**
Of the full-name false positives the scan adds, **6.7 of 7.7 on terra and 24.6 of 27.3
on luna are lowercase**, recomputed from the proposer itself:

| surface written | matched to | via | terra a run | luna a run |
|---|---|---|---|---|
| "database" | `DB` | alias | 3.0 | 3.0 |
| "common" | `Common` | name | – | 3.7 |
| "e2e" | `E2E` | name | – | 3.7 |
| "logic" | `Logic` | name | 1.0 | 3.3 |
| "client" | `Client` | name | 0.3 | 2.7 |
| "storage" | `Storage` | name | – | 2.0 |
| "test driver" | `Test Driver` | name | 1.0 | 1.0 |
| "front-end" | `UI` | alias | – | 1.3 |
| "back end" | `Logic` | alias | 0.7 | 1.0 |

**This is the answer to "what does the LLM extractor do that a regex cannot".** Its
instruction says *"report a reference only when the sentence itself writes the
component's name"*, and the model reads *the name* as the proper noun rather than the
homograph — it is applying use/mention judgement at **proposal** time, silently, and
that was never written down anywhere. The scan cannot: it delegates the distinction to
`STRICTER_CLAUSE` at the gate, and that gate is lenient by construction ("approve by
default"). **The precision cost of the swap is exactly the amount of use/mention
judgement that was implicitly happening inside the extraction call.** It is also why
the two models differ by a factor of three here: terra's gate applies the clause,
luna's largely does not.

**2. A hard dependency on the alias table, where the extractor had a soft one.**
mediastore loses 3.0 TP a run on terra, all `FileStorage`, all "not proposed": S33,
S35 and S36 write **"DataStorage"**, and this batch's knowledge stage discovered
`{AudioAccess, ReEncoder, Database}` — **no `DataStorage`**, where the control run's
table had it. The extractor could still report those sentences; the scan can only
match `N(c)`. **The swap converts a soft match into a hard dependency on a stage that
varies by ~2.8 terms a run.** The same three-term table is also what fires "database"
→ `DB` three times on the same project, so **one alias table costs mediastore 3 TP and
3 FP at once** — which is the whole of that project's −7.08 F2.

**3. Fidelity, and this one refutes the round's own earlier refusal.**
bigbluebutton loses `BBB web` on S30 and S78, whose sentences write **"bbb-web"**. The
alias is `BigBlueButton web application → BBB web`; `ANY_CASE` does not match a
hyphen-joined writing of a space-separated name and **`ANY_SPELLING` does**. Re-priced
on this batch's *own* alias tables rather than the recorded control's:

| | pairs | gold | union pairs | union gold | extra gold |
|---|---|---|---|---|---|
| terra | 202.7 | 151.7 | 205.0 | 153.7 | **+2.0** (for +2.3 pairs) |
| luna | 220.0 | 161.0 | 220.7 | 161.0 | ±0.0 (for +0.7 pairs) |

The level-1 audit priced the fidelity axis at **+0.8 gold a run** and this report
refused `s_linker92c`/`s_linker92d` on that basis. On the alias tables the arm actually
ran with it is **+2.0 gold on terra at +0.3 non-gold pairs** — a strictly better trade
than the one that was refused. **The refusal was measured on the control's alias
tables, and which spellings a table contains is exactly what varies between runs.**
`s_linker92d` is therefore **re-opened, not refused**: it is the one arm that addresses
a real terra loss without touching the judge.

**4. Name nesting between siblings, unaddressed by any clause.** bigbluebutton S60
writes "FreeSWITCH Event Socket Layer (fsels)"; the scan matches the catalog name
`FreeSWITCH` inside the longer name of a *different* component (`FSESL`), capitalised,
so neither `QUALIFIED_CLAUSE` (dotted identifiers) nor `STRICTER_CLAUSE` (ordinary
vocabulary) speaks about it. 1.0 FP a run on terra. The extractor reads the whole
phrase and does not propose it. This is the sibling-confusion mechanism again, on the
proposer side, and nothing in the round addresses it.

### Two candidate repairs, priced at level 1 and not built

- **Union the alias table across runs — REFUTED, for free.** If the scan's hard
  dependency on `N(c)` is a sampling problem, unioning the knowledge stage's output
  over k runs should fix it. Over this batch's own tables: terra gold **151.7 (k=1) →
  151.3 (k=2) → 151.0 (k=3)** at 202.7 → 218.0 pairs; luna 161.0 → 162.3 → 163.0 at
  220.0 → 233.0. **It buys pairs and no gold on terra.** The reason is that the term
  it would need — `DataStorage` — is in **none** of the three draws, while the
  control's 0821 draw had it: this is not sampling within the batch, it is a
  knowledge-stage draw that went against the arm. Worth stating precisely, because it
  narrows mechanism 2: the scan's hard dependency is real, but mediastore's 3 TP would
  not have been lost with the control's table, and a cheap k-run union does not
  recover it.
- **A longest-name-wins gate for sibling nesting — priced, not built.** Pairs whose
  matched span is strictly inside a longer catalog or alias name of a *different*
  component: **1.0 a run on terra, 0.0 of them gold** (the `FreeSWITCH` inside
  `FreeSWITCH Event Socket Layer` case). A free deterministic gate, and one false
  positive a run is below anything this branch has adopted a rule for. Recorded so the
  next round does not re-derive it.

---

## The three variants the judge made unnecessary

`s_linker92b`, `s_linker92c` and `s_linker92d` were built before the stage arm ran,
each to repair a failure mode the audit predicted. All three are **refused, and the
reason is the same in every case: the thing they add is already being done.**

- **`s_linker92b`** does not propose a name written only inside a longer dotted
  identifier — 21.0 pairs a run, 0 of them gold. The audit's approve-end bracket said
  that was worth 21.3 false positives. **The gate rejects them itself: 21/21 on terra
  and 12/19 on luna, and the 7 luna approvals contain no gold at all.** `QUALIFIED_CLAUSE` is
  in the prompt and it works. This is the design law confirmed from a new direction —
  a *weighing* stays folded even when its population grows tenfold.
- **`s_linker92c`** transcribes the deleted prompt's morphology clause as a second
  fidelity. It buys +0.8 gold and +1.3 pairs a run for ~25 lines and a second relation
  point in the code.
- **`s_linker92d`** unions both fidelities, as the branch's name-relation table
  prescribes. +1.2 gold over `92b`, best bracket of the four, most code. Every gold
  pair it adds is already in the pipeline's final link set by another route.

All three remain in the tree, registered and priced, because a refusal that cannot be
reproduced is not a result. **`s_linker92d`'s refusal did not survive the E2E** — see
mechanism 3 above: on the alias tables the arm actually ran with, the union is +2.0
gold a run on terra rather than +0.8, and it is the only arm that recovers a real terra
loss without touching the judge. `s_linker92b` and `s_linker92c` stand refused.

---

## The judge templates — one refuted, one real and optional

The stage arm named the residue exactly. Of the pairs the scan adds outside a
qualified path, the gate approves 44% (terra) / 74% (luna) at roughly one gold in
three, and the recurring approvals are all one shape:

| surface in the sentence | matched to | why it is not a link |
|---|---|---|
| "database" | `DB`, via a discovered alias | the common noun, not the component |
| "logic" | `Logic` | the ordinary word, lowercased |
| "client" | `Client` | the ordinary word, lowercased |
| "testing" | `Test Driver`, via an alias | the activity, not the component |
| "front-end" | `UI`, via an alias | a generic term the alias stage bound |

That is `STRICTER_CLAUSE`'s population, and the clause is **already in the prompt**.
Restating it is the repair this round refuses — s86 measured a restatement at this
gate as redundant, and the typed round found that at the lenient gate a restatement
buys nothing. So neither template adds a rule, a clause or a code gate. Both change
only **the order the reply is written in**, which is `s_linker106`'s mechanism moved
to a different question, and both render `s_linker92`'s strict branch byte for byte,
so the coreference judge is untouched.

- **`s_linker92e` — quote the surface first, then the claim, then decide. ✗ refuted.**
  It *loses* gold on terra (stage gold 152.0 → 147.7) and adds false positives on luna
  (59.0 → 70.7, macro F1 −2.6). Echoing the surface is not deliberating about it;
  it moved the answer without making the model weigh anything.
- **`s_linker92f` — list the readings that surface could have here, name the one it
  has, then the claim, then decide. ✓ real, on terra.** Best macro F1 of any arm in
  the round (93.07, +1.1 over control, p = 0.10) at **FP 26.3, below the control's
  27.3** — it takes the scan's whole added-FP cost back out. On luna it cuts the
  scan's added FP (59.0 → 51.3) but pays 6.0 TP for it, so F2 falls from 94.35 to
  93.07.

**Nothing enumerates the readings for the model** — the prompt does not say "the name
or the ordinary word". The model supplies its own candidates, which is what makes it a
thinking template and not a rule. That distinction is the round's transferable result:
`92e` and `92f` differ only in whether the model is asked to *write down what it sees*
or to *weigh what it could be*, and only the second moves anything.

**Which to take depends on the measure the paper leads with.** On F2 the bare scan is
the best arm on both models. On F1 `s_linker92f` is the only arm that beats the control
on terra while holding luna. `s_linker92a` is the round's head because it is better on
F2 on both models and is the smaller change: `92f` adds 313 characters of template to
every judging call to recover an F1 point that is neutral on the second model.

---

## Caveats

- **n = 3 per model.** Every p in the tables is at or near the 0.10 floor. Read the
  deltas against the recorded harness floor (FP +10.7, TP +4.8) rather than zero.
- **The E2E control is cross-set**, by decision: `s_linker92` is byte-unchanged by
  this round, so its 0821 runs were reused rather than re-bought. The branch's rule is
  that absolute levels drift between invocation sets (s49's FP mean read 10.7 to 16.8
  across five sets in one day), so the E2E numbers carry that exposure and the in-set
  claim rests on the stage arm. The two agree on luna and disagree on terra's F2 by
  2.1 pp; the per-source decomposition attributes that to `partial_name`, a stage
  neither arm changes.
- **The luna FP level.** The scan takes luna from 45.0 to 71.7 false positives a run
  end to end. F2 rises anyway (+1.9) and F1 falls 1.1 (p = 0.40), but a reviewer will
  ask. `s_linker92f` is the priced answer rather than a hidden one, and luna's F1 run
  range of 3.93 (against the control's 0.18) says the arm is also less stable there.
- **The whole round is measured on gpt-5.6 terra and luna only.** The scan itself is
  model-independent; what is model-dependent is the gate behind it, and the two models
  differ by a factor of two on exactly the population the scan adds (approve rate 0.444
  vs 0.736).
- The recorded phase states the arms replay were written by `s_linker89`-named
  directories that both `s_linker89` and `s_linker92` wrote to. The two share the
  extraction pass byte for byte, so the recorded candidate sets are one extractor's;
  the coreference stage they compose with may be either variant's, identically for
  every arm.

## The false-negative analysis

`pilot/regex_fn_analysis.py` (no calls). Every gold pair a run misses is labelled by
**the furthest it got** across all three linkers, read from that run's recorded phase
states: `fn/rejected` means some stage proposed it and its judge said no — a judging
failure; `fn/unproposed` means no stage proposed it at all — a proposing failure. Each
is then asked what could ever have reached it: the sentence writes a **whole name** of
the component (catalog name or discovered alias, either whole-name fidelity), **one
word** of it (the partial-name linker's row), or **no surface** at all (coreference
only). Per five-project run, gold 195:

| | terra ctl | terra scan | | luna ctl | luna scan | |
|---|---|---|---|---|---|---|
| linked | 180.3 | **186.7** | +6.3 | 180.0 | **190.7** | +10.7 |
| `fn/rejected` | 10.0 | 8.3 | −1.7 | 7.3 | 4.0 | −3.3 |
| **`fn/unproposed`** | **4.7** | **0.0** | **−4.7** | **7.7** | **0.3** | **−7.3** |
| …@ whole-name | 4.7 | 0.0 | −4.7 | 7.7 | 0.3 | −7.3 |
| …@ one-word | 0.0 | 0.0 | – | 0.0 | 0.0 | – |
| …@ no surface | 0.0 | 0.0 | – | 0.0 | 0.0 | – |
| `fn/rejected` @ whole-name | 5.0 | 3.3 | −1.7 | 3.7 | 0.0 | −3.7 |
| `fn/rejected` @ one-word | 4.0 | 4.0 | ±0.0 | 3.0 | 3.3 | +0.3 |
| `fn/rejected` @ no surface | 1.0 | 1.0 | ±0.0 | 0.7 | 0.7 | ±0.0 |

**The scan closes the unproposed bucket.** 4.7 → 0.0 on terra and 7.7 → 0.3 on luna:
after the swap, essentially **every remaining false negative reached a judge**. This
inverts the branch's standing error-shape result — "95% of false negatives never
reached a judge, the proposer is the bottleneck, not the gate" — for this pipeline.
It is now the gate.

**And every one of those unproposed pairs was reachable by the tightest thing the
round measures.** The `@ one-word` and `@ no surface` rows of `fn/unproposed` are
**0.0 in every column**: the extractor never lost a link that needed morphology,
inflection or context. What it lost, 4.7–7.7 times a run, were sentences that
**literally write the component's name** and were simply not reported. That is the
whole case for the swap in one row.

**What is left is judging, and it concentrates.** The residual false negatives are
dominated by the partial-name denotation judge on sibling components — `HTML5 Server`
in bigbluebutton is 3 of luna's 4 residual FNs a run and 3 of terra's 8, on sentences
that write one word of a name two components share. That is the same sibling-confusion
mechanism the error-shape analysis found on the *precision* side, appearing on the
recall side, and it is not a proposer problem: those pairs are proposed every run and
declined every run. The remaining `@ no surface` pairs (0.7–1.0 a run, e.g. teastore
S26 against `Persistence`) are coreference-only by construction and are the floor no
lexical layer can move.

Consequence for what to do next: **the round moves the branch's headroom from the
proposer to the judge**, which makes the error-shape analysis's oracle discriminator
(macro F1 0.933 → 0.957 over the candidates already produced) the live prize rather
than a better proposer.

## What the scan does not reach

Asked directly: does the scan cover the false negatives the LLM extractor never
proposed? Per five-project run, off the same 30 recorded runs:

| | pairs |
|---|---|
| gold the extractor never proposed | 44.9 |
| …the scan proposes | 10.6 |
| …neither proposes | 34.3 — of which **30.1 are already linked** by the partial-name and coreference linkers |
| the pipeline's actual false negatives | **14.4** |
| …the scan reaches | **10.2 (71%)** |
| …residue | 4.2 |

Of the residue, **3.2 a run are already proposed by the partial-name scan** — so they
are a judging question, not a proposing one — and **0.5 a run is out of reach of any
lexical scan at any fidelity or extent** (one recurring case: teastore S26 against
`Persistence`). **Replacing the extractor with a scan removes 71% of the pipeline's
remaining false negatives and moves what is left off the proposer.**
