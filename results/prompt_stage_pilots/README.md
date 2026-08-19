# Ablating prompts one stage at a time — 2026-08-14

The three E2E rounds priced prompt **families** at six paired five-project runs each,
about 90 minutes apiece, against a harness whose null arm reports 0.7 macro F1 and
4.8 TP between byte-identical linkers (`results/null_calibration/`). That is a heavy
and blunt instrument for a question about one sentence of English.

This round replaces it for screening: **replay one stage with the two wordings against
the same recorded inputs**, five samples a side, permutation-test the stage's own
output. Everything upstream is a fixed checkpoint rather than a fresh LLM sample, so
the arm measures the prompt and not the pipeline's spread. Eleven arms cost minutes,
not hours.

Harness: `approach/pilot/prompt_stage_pilots.py` (it asserts first that its
re-declared prompt builders render byte-identically to `s_linker49`'s, so an arm is
measuring the swapped constant and not the re-declaration). Reports: the JSON files
beside this one.

## Results

Stage output, five samples per arm, exact permutation test:

| stage | what varies | TP Δ | p | FP Δ | p |
|---|---|---|---|---|---|
| alias judge | `DOC_KNOWLEDGE_JUDGE_RULES` | +0.0 | 1.00 | −0.4 | 1.00 |
| full-name judge | `P1_FOCUS` | +0.0 | 1.00 | +0.2 | 1.00 |
| alias proposer | extraction + exclusion rules | +0.6 | 0.63 | +1.8 | 0.33 |
| coreference judge | focus + `LAYERED_COREF_RULES` | +1.8 | 0.30 | **−1.2** | 0.05 |
| full-name judge | `LAYERED_ENTITY_RULES` | +0.0 | 1.00 | **+2.4** | **0.01** |
| full-name judge | both of the above | −1.0 | 0.37 | **+2.2** | **0.01** |
| extraction | `ENTITY_EXTRACTION_RULES` | +6.2 | 0.03 | **+20.2** | **0.01** |
| coreference resolution | drop the whole preamble (s56) | **−16.2** | **0.01** | **+14.0** | **0.01** |

and, composing the three that cleared, through the knowledge stage and the whole
full-name linker so each arm builds its own alias table and judges its own candidates:

| | TP | FP | composition |
|---|---|---|---|
| s49 | 152.4 | 7.8 | |
| s59 (three cleared clauses) | 151.4 | 7.6 | +0.0 |
| | −1.0 (p = 0.23) | −0.2 (p = 1.00) | p = 0.45 |

## Two hypotheses refuted before an E2E run was paid for

**`s_linker56` — the "duplicated" coreference preamble.** `_prompt_coref` opens with a
paragraph and then appends `COREF_RULES`, which says the same three things. Deleting
the paragraph costs **16.2 true positives per run at the resolution stage** and adds
14.0 false positives. Decomposing it says why:

| coreference resolution stage | TP | FP |
|---|---|---|
| full paragraph | 73.4 | 23.1 |
| drop only "Be conservative … CERTAIN" | 96.8 (**+23.2**, p = 0.01) | 35.0 (+11.4) |
| drop only the protocol sentences | 80.4 (+9.2, p = 0.02) | 34.8 (+11.8) |
| drop both (s56) | 57.0 (**−16.2**) | 36.6 (+14.0) |

The two are strongly non-additive, and the protocol sentences are not restated
instruction at all: they bind the per-case structure — which block is the TARGET, and
that a target with no referring expression yields nothing — which `COREF_RULES` never
says. **What read as duplication is an input-format contract.** The strictness
sentence is a separate object again: a recall brake worth 23 TP for 11 FP at this
stage, which the strict judge and the `_unlinked` subtraction downstream turn into the
parity s55 measured end to end.

**`s_linker58` — the full-name extraction rule.** Generalizing it adds **20.2 false
positives per run** at the stage that feeds everything else, for 6.2 true positives.
The clause that buys them is the aside *"even if the compound identifier is
semantically related to the component"*, on the one document of five where 62 of 198
sentences carry a dotted identifier.

## What the clause level shows that the family level could not

The E2E rounds concluded the full-name family costs ~2.1 macro F1 and is load-bearing
as a family. It is not uniform: **`P1_FOCUS` generalizes for exactly nothing (TP ±0.0,
FP +0.2, both p = 1.00) while `LAYERED_ENTITY_RULES` costs 2.4 false positives on the
very same candidates (p = 0.01).** Same stage, same call, same inputs — 289 bytes of
it are decoration and 692 bytes are the gate. Likewise the knowledge family: its judge
rubric is free (p = 1.00 on both measures), its proposer rules are not clean.

That refines the round-3 rule rather than overturning it. **A prompt clause is
removable when something downstream rejects by default.** The coreference rules sit in
front of a gate that rejects when uncertain, and they go. The alias judge's own rubric
is itself the rejecting step, and the *rest* of that rubric still rejects, so a
generalized wording holds. `P1_FOCUS` sits in front of `P2_FOCUS` and
`LAYERED_ENTITY_RULES`, which are the gate — so the focus line goes and the rubric
does not. `ENTITY_EXTRACTION_RULES` sits in front of a judge that approves by default,
and it stays.

## The variant this leaves

`s_linker59` takes the coreference family, `P1_FOCUS` and the alias judge rubric —
every clause that cleared, and no others. Rule text 4022 → 2960 B (−26%); instruction
bytes per five-project run 60 892 → 40 081 (−34%). `pilot/test_s56_s57_s58_prompts.py`
asserts that every constant in it is either s49's or the generalized wording, in the
intended place, with no method body or class attribute changed.

## Method note

A stage arm is a screen, not a verdict — this branch has eight instances of a stage
arm pointing opposite to the composed pipeline, always on precision, because the three
linkers subtract from one another. What changes with this round is the *cost* of being
wrong: a refutation now costs minutes, and only a surviving candidate needs the six
paired runs. The two refutations here would each have cost ~90 minutes of E2E to reach
the same answer, and the `corefpre_task` / `corefpre_strict` decomposition — the part
that explains *why* — is not reachable end to end at all.
