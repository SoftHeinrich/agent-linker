# `s_linker123` — the union judge's evidence fields merged into one

The head hands its judge three named facts, `writes` / `alternatives` / `mention`, and
a reader who takes three names for three facts is being misled by the format. This
round replaces them with two, `written` and `competitors`, and prices the whole change
at **level 1 against the recorded s122 runs — 11,586 checks, no LLM calls.**

Arm: `s_linker123` (alias `written`), a subclass staged directly on `SLinker122`.
Audit: `approach/pilot/written_field_audit.py` (`--verify`, `--cells`, `--bytes`).
Staging assertions: `approach/pilot/s123_inherits.py`.
Reproduce from `approach/`:

    ../.venv/bin/python pilot/s123_inherits.py
    ../.venv/bin/python pilot/written_field_audit.py

## 1. Why, measured before it was designed

Replaying every recorded judging call of the s122 round — 184 calls, 3753 cases, both
models, both arms — says the field set has two defects that are not stylistic.

* **One fact is printed twice.** `mention=via known alias` fires on exactly the 598
  cases where `writes` already says "a short form the document established for it":
  **598 of 598, in both directions.** The judge is told the same thing in two
  vocabularies.
* **`mention` has two reachable values of the five `MentionType` declares**, and the
  surviving one can only ever co-occur with the whole name. It is returned when
  `_find_exact_form` matches, which is the same `ANY_CASE` predicate `naming` is decided
  with, so the two fields were never independent.

## 2. The design, and why the merge is entailed rather than fitted

One field, `written`, taking the four values the recorded runs actually produce, plus
`competitors` for what was `alternatives` — a name that answers "alternatives to what?".
The evidence line is then two fields: one always present, one when the match found it.

    Case 1: "HTML5 client" -> HTML5 Client
      Evidence: written=whole name, competitors=HTML5 Server
    Case 7: "bbb-web" -> BBB web
      Evidence: written=whole name (qualified)

`qualified` is folded into a *value* rather than kept as a field because the containment
is **entailed**: it implies the whole name is written, since both go through
`_find_exact_form`. That has one precondition — `SKIP_QUALIFIED = False`, which makes
`_writes_name` and `_find_exact_form` the same relation — and `_written_as` asserts it
rather than trusting it. A value chosen by unification is defensible; one chosen because
it happened to hold on five documents is not.

Parenthesised and not comma-separated: the evidence line separates fields with commas,
so `written=whole name, qualified` would be indistinguishable from two fields, to a
reader and to every tool that parses these lines — including this round's own audit.

The real gain is upstream of the wire: **one `_written_as` replaces two predicates that
had to agree.** `MentionType`, `RETAINED_MENTION_TYPES` and `_mention_label` lose their
reason to exist, and the two facts that were consistent only because they happened to
call the same helper become one that cannot disagree. `naming` survives in the decision
record, because six pilots and every recorded phase state read it — but it is *derived
from* `written` there, not computed beside it.

## 3. What the four values are, and what the judge does with them

Off the recorded `judge_decisions` and the benchmark gold standard, 3753 judged cases,
0 unmapped:

| `written=` | cases | share | gold | gold rate | keep rate | over-keep | precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| `whole name` | 1728 | 46.0% | 1572 | 0.910 | 0.925 | +0.016 | 0.959 |
| `whole name (qualified)` | 336 | 9.0% | 24 | **0.071** | **0.125** | +0.054 | 0.286 |
| `short form` | 598 | 15.9% | 318 | 0.532 | **0.741** | **+0.209** | 0.707 |
| `one word` | 1091 | 29.1% | 244 | 0.224 | 0.249 | +0.026 | 0.533 |
| total | 3753 | 100.0% | 2158 | 0.575 | 0.628 | +0.053 | 0.850 |

**The one field spans a 12.8× gold density**, and the judge's keep rate tracks it on
three of four rows. So `written` alone carries what the bundle discriminates on, which
is the strongest argument for merging: there was never a second axis.

**The value the design was most tempted to drop is the sharpest one.** `qualified` takes
the whole-name row from 0.910 gold to 0.071 and the judge follows it, 0.925 to 0.125.
Reducing `mention` to this one flag keeps the only value that was doing anything, and
retro-justifies `QUALIFIED_CLAUSE` on data the compaction round never had.

A value's gold rate is a property of the data; what the judge did with it is a property
of the model, and the two rows that matter split:

| model | `written=` | cases | gold rate | keep rate | over-keep | precision |
|---|---|---:|---:|---:|---:|---:|
| terra | `whole name (qualified)` | 168 | 0.071 | 0.048 | −0.024 | **1.000** |
| luna | `whole name (qualified)` | 168 | 0.071 | 0.202 | +0.131 | **0.118** |
| terra | `short form` | 284 | 0.493 | 0.690 | +0.197 | 0.699 |
| luna | `short form` | 314 | 0.567 | 0.787 | +0.220 | 0.713 |

Terra honours the identifier fact almost perfectly and luna barely at all, so
shortening that signal touches the one fact the laxer model already mishandles. Both
models over-approve the `short form` row by ~0.20, which is the row whose duplicate
line this variant removes.

## 4. Level 1 — 11,586 checks, and the rebuild is faithful

`--verify` rebuilds every judging call of every recorded run from the benchmark and that
run's own recorded alias table, with the head and with the arm:

* **The head's rebuilt prompts are byte-identical to the ones actually sent** on
  **30 of 30 project-runs** (see §5 for the one substitution that makes this exact).
  Without that, everything above would be a simulation rather than a measurement.
* **The candidate sets are equal on 30 of 30 project-runs** and the call count is
  unchanged everywhere: the scan does not move, so no composition question arises from
  the proposer side.
* **Outside the evidence lines and the rule's field block, the prompts are identical** —
  preamble, sentence window, clause, demand, cases and reply contract all byte-equal,
  checked line by line over 1870 evidence lines.
* **`written` reproduces the head's `naming` on all 1870 bundles**, and `competitors`
  equals `alternatives` on all of them.

`pilot/s123_inherits.py` pins the staging: MRO is `SLinker123 -> SLinker122 -> object`,
**19 of 19 shared methods byte-identical** (both scans, the merged stream, the whole
judging loop, and the entire coreference linker including `_named_before`), the arm
overrides exactly six names and all six are the evidence vocabulary, the coreference
shortlist is identical on **378 of 378 sentences**, and `SURFACE_NOT_EVIDENCE` is
present in every judging call on all five projects.

## 5. The finding the audit was not looking for: the head no longer renders its own runs

The rebuild differed from the sent prompts by exactly one sentence, on every project-run.
`s_linker122`'s clause was **re-scoped after the recorded E2E was bought**
(`s122 promoted: standalone, and the clause gets its scope back`):

    recorded: That a surface can name this component is not evidence that it does here.
    head    : Where the sentence does not write the name in full, that a surface can
              name this component is not evidence that it does here.

With the recorded clause substituted the rebuild is byte-identical on 30 of 30
project-runs, so this is the *only* difference — but it matters twice over.

**The recorded E2E describes an arm the head is not.** `results/s121_ablations/README.md`
§4 refuses s122 on a luna regression of TP −10.3, and the registry's own note says the
unscoped clause costs luna **−2.07 gold a unit on the whole-name row and seven gold
links end to end** on a bare enumeration of component names. Those are the same rows.
So the refusal verdict rests on the clause version that was afterwards shown to be the
faulty one, and **no run exists for the head as it now stands.** Before s122 is reported
or s123 is compared against it, that E2E has to be re-bought — and the comparison this
round's arm needs is against the *scoped* head, in the same invocation.

**And the cell tables above inherit it.** Gold rates are properties of the data and
stand. Keep rates were produced under the unscoped clause and are the baseline of the
arm that ran, not of the head.

## 6. Bytes

Whole judging calls rebuilt, 92 calls over six runs:

| | |
|---|---|
| evidence lines | 94 703 → 62 786 B, **−33.7%** |
| whole judging call | mean **−311 B**, best −695, worst **+33** |
| per five-project run | **−4768 B** |

The rule's field block goes from three lines to two but absorbs one sentence defining
"qualified", a fixed +37 B a call — which is why a one-case call can come out 33 B
*larger*. The readability is the point; the bytes are a side effect, and half the size
an earlier estimate suggested because that one counted both arms' calls per run.

## 7. What is owed

A stage arm, and nothing more until it reads. The candidate sets are equal and the call
count is unchanged, so there is no composition question from the proposer; what changes
is what the judge reads, which is a prompt change and has to be sampled.

Three arms in one invocation per model, fixed recorded candidates, alias table pinned,
three samples a side:

* `head` — s122 as it now stands, **with the scoped clause** (which its own E2E did not
  have)
* `written` — the new vocabulary, **duplicate retained**
* `written_nodup` — plus the 598 duplicate lines dropped

Read gold and spurious **per value of `written`**, not on the total, and read the
`short form` row first. Splitting the duplicate out is deliberate: this branch has twice
been caught pricing a clause on its own when its population was changed by the other
half of the same edit, so both arms belong in the same invocation.

## Caveats

* Every number here is off recorded runs of the *unscoped* arm (§5). Gold densities are
  data; keep rates are not.
* The cell table pools the s121 and s122 arms, which differ in whether the anchor block
  was printed. That pooling affects keep rates — the s122 round found the anchors' effect
  landing on the alias row on terra — and not gold rates.
* `naming` is retained in the decision record for compatibility with six pilots and the
  recorded phase states. It is derived, not computed, but it is still a second word for
  one fact and should go when those readers are updated.
