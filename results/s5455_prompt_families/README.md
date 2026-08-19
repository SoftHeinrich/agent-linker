# Which prompt families are guidelines, and which are rulebooks — 2026-08-14

Three rounds partition the workflow's ten rule constants into three families and
price each one. The question throughout: how much of this hand-written English is
*accreted* — enumerations of cases seen on five documents — and how much is the
design?

| family | constants | bytes | calls/run |
|---|---|---|---|
| knowledge | `DOC_KNOWLEDGE_{EXTRACTION,JUDGE}_RULES`, `ALIAS_EXCLUSION_RULES` | 767 | 15 |
| full-name | `ENTITY_EXTRACTION_RULES`, `P1_FOCUS`, `LAYERED_ENTITY_RULES` | 1 313 | 36 |
| coreference | `COREF_RULES`, `COREF_VALIDATION_FOCUS`, `LAYERED_COREF_RULES` | 1 773 | 54 |

(`P2_FOCUS`, 169 B, is already a general question and is carried verbatim by every
arm.)

Five arms, six paired runs each, arms always inside the same invocation. Runs:
`s5051_e2e_r{1..6}`, `s5253_e2e_r{1..6}`, `s5455_e2e_r{1..6}`, all 2026-08-13.

## Result

| arm | generalized | rule text | instr. B/run | macro F1 vs s49 | p | macro F2 | p |
|---|---|---|---|---|---|---|---|
| **s50** | coreference resolution rule only | -10% | **-27%** | **-0.2** | 0.71 | -0.7 | 0.05 |
| **s55** | **whole coreference family** | **-19%** | **-31%** | **-0.0** | **0.90** | -0.3 | 0.39 |
| s54 | coreference + knowledge | -28% | -34% | -1.1 | **0.00** | -0.6 | 0.05 |
| s52 | coreference + full-name | -30% | -41% | -2.1 | **0.00** | -1.3 | 0.00 |
| s51 | all nine | -39% | -44% | -2.4 | **0.00** | -0.9 | 0.00 |
| s53 | all nine, one clause back | -37% | -44% | -2.5 | **0.00** | -1.2 | 0.00 |

Reading the arms against each other, the families are close to additive and each
carries a different verdict:

- **coreference — free.** `s55` is neutral on every measure (TP -1.5 p = 0.27,
  FP -1.5 p = 0.58, F1 -0.0 p = 0.90, F2 -0.3 p = 0.39, composition +0.9 p = 0.19),
  and it was in position 3 of its invocation, where the harness's own null arm costs
  0.7 F1 (see `results/null_calibration/`). Three rules, 1 773 B, 54 calls per run —
  the biggest instruction item in the workflow — restate as guidelines for nothing.
- **knowledge — ~1.1 F1.** `s54` reverts the full-name family and still loses 1.1
  (p = 0.00).
- **full-name — ~2.1 F1.** `s52` reverts the knowledge family and still loses 2.1
  (p = 0.00). This is the largest single family effect measured.

## What the coreference family loses without cost

`COREF_RULES` (760 B, 40 calls/run — half the instruction budget) gives up two
lettered clauses, a five-phrase list and an alias-shape enumeration for one sentence:

> Resolve when the surrounding sentences make one component the clear antecedent,
> under any form the document uses for it. Avoid resolving when two or more equally
> plausible antecedents exist.

The deterministic audit had said clause (b) — resolve to the section topic *without*
a name repetition — licensed **0.0 of 578 recorded resolutions**, because
`_antecedent_states_name` discards exactly what it permits. The arm confirms it, and
adds something the audit could not see: the generalized rule makes the resolver
*propose* far more (mediastore 6.8 → 11.7 per run, jabref 1.2 → 9.7) and the strict
coreference judge rejects the surplus. **The enumeration was doing work the
downstream judge already does** — which is the same shape as the s38 result that two
samples of one prompt are not independent, and of the s25 result that a judging step
must be separate from what it judges.

`LAYERED_COREF_RULES` gives up its gloss of "architectural claim" and its three named
fragment shapes; `COREF_VALIDATION_FOCUS` gives up three listed pronoun forms. What
stays is the asymmetry that is the design: this gate rejects when uncertain, the
full-name gate approves.

## What cannot be generalized, and why it is not arbitrary

Both load-bearing families are on the **admitting** side of the cascade.

The full-name family (2.1 F1) is the extraction rule and the two judging rules for
the linker that produces 88% of all links. Held to the candidates both arms judged —
so the proposer is constant — the generalized judge approves **3.5 more false
positives per run and no more gold**. Its four numbered reject-conditions become "for
example" illustrations, and the model stops applying them as a gate. The extraction
rule's dropped aside (*"even if the compound identifier is semantically related to
the component"*) costs the rest: teammates full-name proposals 68 → 79, accepted
50 → 63, on the one document where 62 of 198 sentences carry a dotted identifier.

The knowledge family (1.1 F1) is the alias table, which is the workflow's only
structure that *admits* rather than rejects — a looser rubric there manufactures
candidates for every later stage. But the single clause the round-1 traces indicted
(a grouping of several elements is not an alias for one of them) is worth **~0** on
its own: `s53` restores it and recovers nothing. The family matters; that clause
does not.

**So the shape of the answer is: the prompts of the *rejecting* stage are guidelines
and can say so; the prompts of the *admitting* stages are not, and the reason is
structural rather than lexical.** A rejecting stage that over-rejects is caught by
recall it never had; an admitting stage that over-admits has no downstream that can
tell the difference.

## Method note

Every delta here should be read against the null arm's, not against zero: six paired
runs of two byte-identical linkers in this harness report TP -4.8 (p = 0.00) and
macro F1 -0.7 (p = 0.03). `results/null_calibration/` has that measurement and what
it invalidates.
