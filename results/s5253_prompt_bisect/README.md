# Bisecting a failed prompt generalization — 2026-08-14

`s_linker51` restated nine of the workflow's ten rule constants as the principles they
enumerate and lost **2.4 macro F1 (p = 0.00)**, entirely on precision (FP +10.3). The
round-1 traces pointed at the alias table: three quarters of the extra false positives
arrived on sentences that state no component name and contain a term only s51's table
carried, and the terms were layer and tier names — `core`, `outer shell`,
`intermediate layer`, `back end`, `storage layer`. The clause s51 had dropped from the
alias judge says exactly that a phrase naming a grouping is not an alias for one
element of it.

Two arms put that reading to six paired runs:

- **`s_linker52`** — s51 with all three knowledge-side rules back at s49's wording.
  Six of nine generalizations survive.
- **`s_linker53`** — s51 with **one subordinate clause** restored inside the alias
  judge: *"…or identifies anything other than that one component, including a
  grouping that encompasses several elements."* 51 bytes.

Runs: `results/s5253_e2e_r{1..6}_20260813`, all four arms in the same invocations.
Scripts: `approach/pilot/test_s52_s53_prompts.py`, `approach/pilot/stage_diff.py`,
`approach/pilot/score_runs.py`.

## Result — the reading was wrong, and that is the finding

| | s49 | s51 | **s52** | **s53** |
|---|---|---|---|---|
| generalized | — | all nine | six (knowledge reverted) | all nine + 1 clause |
| rule text | 4 022 B | 2 461 B | 2 804 B | 2 517 B |
| TP | 188.5 | 189.2 | 186.8 | 188.8 |
| FP | 10.7 | 21.0 | **17.5** | 21.2 |
| macro F1 | 96.76 | 94.41 | 94.63 | 94.26 |
| macro F2 | 97.19 | 96.33 | 95.90 | 96.02 |
| F1 vs s49 (p) | — | **-2.4 (0.00)** | **-2.1 (0.00)** | **-2.5 (0.00)** |
| FP vs s49 (p) | — | **+10.3 (0.00)** | **+6.8 (0.00)** | **+10.5 (0.00)** |

Reverting the **whole** knowledge side recovers 3.5 of 10.3 false positives and
**0.3 F1 of 2.4**. Restoring the single grouping clause recovers **nothing**. Neither
arm comes near s49.

So the alias table is a *third* of the damage at most, and the specific clause the
round-1 traces indicted is not the mechanism. Two things went wrong in that reading,
and both are general:

1. **A surface attribution is not a causal one.** "This link arrived through a term
   only s51's alias table has" says which *surface* carried the link, not which
   prompt change produced it. Every alias is fed into the extraction prompt, so a
   candidate admitted by a looser *extraction* rule and a candidate admitted by a
   looser *alias judge* look identical at the link level.
2. **The alias table is not stable enough to attribute from.** s49 and s50 have
   byte-identical knowledge prompts and still build tables differing by 2.8 terms per
   run; `back end` and `front-end` appear in some s49 tables too. A term that is
   "only in this arm" over six runs can be sampling.

Held to the one comparison that controls the proposer — the terms **both** arms
propose, where any difference in the table is the judge's — the alias judge shows
**zero verdict flips** between s52 and s49, and only three between s51 and s49
(`core`, `outer shell`, one bigbluebutton term). That is the honest size of the
clause: three terms, worth ~0 F1.

## Where the damage actually is

With the knowledge side reverted, s52 still loses 2.1 F1, and its remaining
difference from s49 is the three rules the **full-name linker** reads. Held to shared
candidates, so the proposer is constant:

    s52 vs s49: 174.2 shared candidates/run, 168.2 same verdict
      approved only by s52   4.8/run   gold 1.2   spurious 3.5
      approved only by s49   1.2/run   gold 1.2   spurious 0.0

The full-name judge, asked as a principle rather than as four numbered
reject-conditions, approves **3.5 more false positives per run and no more gold**.
The rest of s52's FP gap comes from the extraction rule proposing more: teammates
full-name proposals 68 → 79 in the paired run, accepted 50 → 63.

The two candidates for that are `LAYERED_ENTITY_RULES` (four numbered conditions
become "for example …") and `ENTITY_EXTRACTION_RULES` (the aside *"even if the
compound identifier is semantically related to the component"* is dropped, on the one
document where 62 of 198 sentences carry a dotted identifier). Round 3 —
`s_linker54` (full-name family reverted) and `s_linker55` (only the coreference side
generalized) — separates them from the rest.

## Standing rules this round adds

- **Eighth instance of a trace-derived reading not surviving its arm**, and the first
  where the reading was *directionally* right (the alias table does contribute 3.5
  FP) and still wrong about the mechanism. Size a clause off traces to choose what to
  run; never to conclude.
- **A prompt clause and the prompt stage it sits in are different objects.** The
  knowledge family is worth 3.5 FP as a family and its most-indicted single clause is
  worth ~0. Family-level and clause-level ablation answer different questions, and the
  clause-level one is the one the paper wants.
