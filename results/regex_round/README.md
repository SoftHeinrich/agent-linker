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
scan — **is adopted at the stage level on both models.** It removes 9 of the ~84 LLM
calls a five-project run makes, adds no deterministic machinery whatsoever, and reads
macro F2 **+2.0 (terra) / +1.8 (luna)** at macro F1 +0.4 / −0.9 (neither significant).
The four variants built to repair its expected failure modes are all refused: three
because the judge does the job they were built for, one because it made things worse.

Tooling: `pilot/regex_extract_audit.py` (level 1, no calls), `pilot/regex_proposer_pilots.py`
(level 2, the stage arm), `pilot/regex_round_stats.py` (permutation test),
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
reproduced is not a result.

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

- **n = 3 per model, one invocation set per model.** Every p in the tables is at or
  near the 0.10 floor. Read the deltas against the recorded harness floor (FP +10.7,
  TP +4.8) rather than against zero.
- **The stage arm is not an E2E.** It replays one stage against recorded checkpoints
  and composes with the same run's other two stages. Composition risk is real here —
  the scan adds pairs the coreference linker also proposes — so an end-to-end batch is
  owed before `s_linker92a` becomes the head, and this report does not claim it.
- **The luna FP level.** The bare scan takes luna from 39.7 to 59.0 false positives a
  run. F2 rises anyway, and F1 falls 0.9 (p = 0.60), but a reviewer will ask, and
  `s_linker92f` is the priced answer to that question rather than a hidden one.
- The recorded phase states the arms replay were written by `s_linker89`-named
  directories that both `s_linker89` and `s_linker92` wrote to. The two share the
  extraction pass byte for byte, so the recorded candidate sets are one extractor's;
  the coreference stage they compose with may be either variant's, identically for
  every arm.

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
