# The union round (s120) — one judge, one rule, evidence computed from the match

`s_linker110` judges its two name streams with two prompts whose rubrics state opposite
defaults, and routes each case by which scan proposed it. This round asks whether one
judge can do it: **one rule that says what a trace link is and how to read each piece of
evidence**, every candidate in one case format, with the difference between candidates
carried by the evidence the match computed rather than by a second rubric.

**It can, and it is better than the arrangement it replaces.** Stage pilot on fixed
recorded candidates, both arms in the same invocation, five projects, alias table pinned:

| model | samples | gold | p | spurious | p | net (3·gold − sp) | p | precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| terra | 5 | **+0.2** | 1.000 | **−12.6** | **0.000** | **+13.2** | **0.025** | 0.878 → **0.939** |
| luna | 3 | **+0.3** | 1.000 | **−17.7** | **0.008** | **+18.7** | **0.011** | 0.789 → **0.859** |

Gold-neutral on both models, spurious down on both, **at the same 14 judging calls and
one prompt instead of two**. Deltas are the union minus the control that ran beside it,
per five-project run; p is a two-sided sign-flip permutation test over the paired
(sample, project) units (25 on terra, 15 on luna).

Tooling: `approach/pilot/union_pilots.py` (the arms), `union_stats.py` (paired tests),
`union_diff.py` (error analysis), `union_composition.py` (level 3),
`union_defensibility.py` (GATE-06/07, 25 checks), `test_s120_union.py` (2593 invariants,
no calls). The arm: `s_linker120` (`unijudge`); every iteration of its rule:
`union_iterations.py`. Reproduce from `approach/`:

    OPENAI_MODEL_NAME=gpt-5.6-terra LLM_BACKEND=openai \
      ../.venv/bin/python pilot/union_pilots.py --samples 5 --dump dump.json
    ../.venv/bin/python pilot/union_stats.py dump.json

## The design that worked

One rule, in four parts, and **every part that states a criterion is a verbatim slice of
one of the head's own constants**:

1. **What a link is.** *A trace link holds between a sentence and a component when the
   sentence makes an architectural claim about that component — when it says something
   about that component as a participant in the system this document describes.* Plus
   `LAYERED_ENTITY_RULES`' own second sentence: *a mention that says nothing further
   about the component still counts as a valid link.*
2. **What a case is.** The expression, the sentence, the evidence, and — *where the
   sentence writes a name of it* — the component that name reaches. A case whose
   sentence writes no name of any component carries none.
3. **How to read each field.** `writes` (whole name / established short form / one word
   of the name), `alternatives` (other components whose names carry the same word),
   `mention`, `anchors` — one line each, each saying what the field is evidence *of*,
   and that none of it is a verdict.
4. **When to reject.** `LAYERED_ENTITY_RULES`' positive-ground sentence, `STRICTER_CLAUSE`
   (scoped to cases that carry a component, because that is what it is about),
   `QUALIFIED_CLAUSE`, and `LAYERED_COREF_RULES`' acts-on clause.

And three things the **evidence** decides, not the rubric: what a case prints, which
batch it joins, and what its call can be asked (a call with no component in any case
gets no catalog and answers the head's denotation contract — there is nothing to approve
against).

## The trail: thirteen iterations, one change each

Every row is the union minus the control that ran beside it, three samples unless noted,
per five-project run. `net` = 3·gold − spurious.

| it | change | terra gold | terra sp | net | luna gold | luna sp | net |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | two standards, paraphrased; alternatives = a ground to reject | −11.3 | −0.7 | −33.2 | | | |
| v2 | both standards quoted verbatim; alternatives made inert | −4.0 | −12.7 | +0.7 | | | |
| v3 | + word-only case blinded (s25's refusal, in the bundle) | −4.0 | −13.4 | +1.4 | | | |
| v4 | + quote demand made row-aware | −0.7 | +3.3 | −5.4 | | | |
| v5 | + each row answers its own verdict field | −3.0 | +3.7 | −12.7 | | | |
| v6 | − `alternatives` and recency removed from the case | +0.7 | **+26.4** | −24.3 | | | |
| v7 | **one rule, no rows**, one case format | −4.0 | −7.4 | −4.6 | | | |
| v8 | + `writes` carries the denotation question, not a strictness gradient | −0.7 | −1.0 | −1.1 | −10.3 | −7.0 | −23.9 |
| v9 | + the component slot filled by the evidence | | | | −11.0 | −17.7 | −15.3 |
| v10 | + a case with no component answers `denotation` | | | | −21.0 | −22.0 | −41.0 |
| v11 | + `STRICTER_CLAUSE` scoped to cases carrying a component | | | | −12.0 | −22.3 | −13.7 |
| v12 | + the evidence groups the batch | | | | −7.3 | −6.3 | −15.6 |
| **v13** | **+ the call carries what its batch's evidence has** | **+0.2** | **−12.6** | **+13.2** | **+0.3** | **−17.7** | **+18.7** |

(terra v13 at five samples; every other row at three.)

## What the round learned, beyond the arm

**1. An evidence field restrains when it is stated and misleads when it is weighted.**
Iteration 1 stated the alternative set as a ground for rejecting and lost **7.6 gold** on
a bucket that is 0.765 gold. Iteration 6 removed the same field from the case entirely
and gained **26.4 spurious** at +0.7 gold — the whole-name row kept 158 of 172 cases.
Same field, opposite errors, and between them the design law with a number on each side:
facts belong in the case, weighings in the rule, and a fact that has been turned into a
weighing is worse than either.

**2. The company a case keeps is part of its evidence.** Three successive rewrites of
the rule (v9, v10, v11) left luna's word-only row at 9.7–12.3 gold against a control's
21.3–22.0. Grouping those cases by what the match computed — a code fact, no prompt
change — moved it to 15.3 in one step (v12), and letting the call carry only what its
batch's evidence has finished it at 22.0 (v13). **No sentence of any rule moved that row
as far as the batch boundary did.**

**3. `s_linker25`'s refusal is about the question, not the target.** Showing the
component to a case whose sentence writes one word of a name cost luna 9.4 gold; blinding
it back recovered 1.4 of 6.4 on terra and nothing on luna. What that stream loses to is
being asked an *identity* question, in whichever of the several ways a merged prompt can
ask one — the target in the header, the catalog above it, the claim demand, or the
neighbouring cases. Remove them all and the row is fine; remove any three and it is not.

**4. A row-free prompt cannot carry a per-row reply contract.** Iteration 10 asked cases
with no component to answer `denotation`; luna kept answering `approve`, the parser
rejected what the contract had not answered, and the row kept **1.0 of 81**. Iteration 5
could carry the same contract because it still had rows to hang it on. A contract needs
a boundary the model can see — which is why v13 puts it on the *call*, where the boundary
is real.

## Defensibility — 25 checks, zero corpus bytes

The union is the easiest place on the branch to smuggle in a fitted clause: it is the one
prompt that can see every stream's failure mode at once. `union_defensibility.py` forbids
it mechanically:

- **Quotation is checked, not claimed.** `MENTION_COUNTS`, `POSITIVE_GROUND` and
  `ACTS_ON` are *slices* of `LAYERED_ENTITY_RULES` and `LAYERED_COREF_RULES`, verified to
  appear in the constant they came from; `STRICTER_CLAUSE` and `QUALIFIED_CLAUSE` appear
  verbatim. `LAYERED_ENTITY_RULES`' first sentence — "Approve the link by default" — is
  deliberately **not** carried, and its absence is asserted: a default belongs to a
  stream, and this rule has none.
- **Every authored sentence declares its ground** (general / SE-practice / prior-work),
  and `corpus` is inadmissible. The residue is the definition of a trace link, the input
  contract, and one line per evidence field.
- **GATE-06**: 0 of 63 catalog words and project names appear. **GATE-07**: no dotted or
  joined identifier, no document-shape word, and every quoted token in the residue is an
  evidence field name.
- **What the judge punches on** is printed and checked to be two things only: whether the
  sentence makes an architectural claim about the component, and what the expression
  denotes where no name is written. Neither mentions a document, a layout, a spelling or
  a syntax.

## Composition (level 3)

| model | pairs added (gold) | coref proposes anyway | pairs removed (gold) | coref re-proposes | **gold at risk** |
| --- | ---: | ---: | ---: | ---: | ---: |
| terra | 32 (15) | 10 | 44 (10) | 19 | **0** |
| luna | 36 (7) | 4 | 88 (6) | 64 | **2 distinct** |

On terra the stage arm **is** the pipeline answer for this change. On luna two distinct
gold pairs (teammates S185 `Logic`, bigbluebutton S73 `HTML5 Server`) are removed and not
re-proposed downstream — ~1.3 a run, **below the recorded TP floor of 4.8**, so an E2E
batch would measure drift rather than the change. Reported rather than resolved.

## Status

`s_linker120` runs iteration **v13** (`union_iterations.ACTIVE`), and any earlier
iteration can be run as its own arm beside it. The head does not move on this evidence
alone: what is measured is one stage, and adoption into the pipeline is an E2E decision
the branch takes for a whole variant. What this round establishes is that **the two name
judges were not two questions** — one rule, one format, and evidence computed from the
match reproduce them at better precision and lower cost.
