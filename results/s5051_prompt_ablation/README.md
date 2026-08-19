# Ablating the prompts, not the pipeline — 2026-08-13

Twenty variants have ablated this workflow's *structure*: stages, calls, batch
constants, code predicates. No mechanism has come out without a measured cost. That
leaves the surface nothing in the series has touched — the hand-written English —
and the question a reviewer will ask about it directly:

> Are these prompts general guidelines, or a rulebook grown against five benchmark
> documents?

This round answers it. `approach/pilot/prompt_audit.py` prices every clause off
`s_linker49`'s own six recorded runs before any call is paid for; two variants then
put the two readings of "minimize" to six paired runs each.

Scripts: `approach/pilot/prompt_audit.py` (deterministic, no LLM calls),
`approach/pilot/test_s50_s51_prompts.py` (68 checks), `approach/pilot/stage_diff.py`,
`approach/pilot/source_stats.py`, `approach/pilot/score_runs.py`.
Runs: `results/s5051_e2e_r{1..6}_20260813`.

## What the instructions actually are

Ten constants, 4022 bytes, carried into 88 calls per five-project run:

| clause | bytes | calls/run | bytes/run |
|---|---|---|---|
| COREF_RULES | 760 | 40 | **30 400** |
| LAYERED_ENTITY_RULES | 692 | 18 | 12 456 |
| LAYERED_COREF_RULES | 723 | 7 | 5 061 |
| ENTITY_EXTRACTION_RULES | 332 | 9 | 2 988 |
| P1_FOCUS | 289 | 9 | 2 601 |
| COREF_VALIDATION_FOCUS | 290 | 7 | 2 030 |
| DOC_KNOWLEDGE_JUDGE_RULES | 405 | 5 | 2 025 |
| P2_FOCUS | 169 | 9 | 1 521 |
| DOC_KNOWLEDGE_EXTRACTION_RULES | 240 | 5 | 1 200 |
| ALIAS_EXCLUSION_RULES | 122 | 5 | 610 |
| **total** | **4 022** | | **60 892** |

60.9 kB of instruction against 948.6 kB of prompt — **6.4% of what this workflow
sends is rule text**. So a trim buys no meaningful token cost. What it buys is one
fewer hand-written stipulation to defend, which is the only reason to want it.

## What each enumeration can reach (deterministic, six s49 runs)

- **The qualified-path rule is written five times** (alias exclusion, entity
  extraction, `P1_FOCUS`, both judging rubrics) and **one of five documents has
  dotted identifiers at volume**: teammates 62 of 198 sentences, against 1–6 on the
  others and those are `e.g` / `i.e` / `React.js`. At the full-name judge, 4.5
  candidates per run have the component name *only* inside a path; **0.0 of them are
  gold** and 2.2 are rejected.
- **`COREF_RULES` clause (b)** — resolve a role-referential phrase to the section
  topic *even without a direct name repetition* — licensed **0.0 of 578 recorded
  resolutions**. Every resolution in six runs on five projects had the component's
  name or a discovered alias inside the ±5 context sentences shown, which is
  clause (a). The code gate `_antecedent_states_name` independently discards what
  clause (b) permits.
- **Its five listed role phrases** cover 17.3 resolutions per run, of which 15.3 are
  the single word `it`; the other four listed forms total 2.0, against twenty-odd
  document-specific noun phrases the model reports unprompted (`the logic component`,
  `the webui`, `the image provider`).
- **`LAYERED_ENTITY_RULES`'s four numbered reject-conditions**: of 14.7 rejections
  per run, 2.2 match condition (1) lexically and 1.8 match (2). **73% rest on (3)
  and (4)** — the two conditions that name no surface form.
- **The terminal-word alias sentence** covers 1.7 antecedents per run.

A prohibition's effect shows as *absence*, so these numbers bound the population at
risk, not the clause's value. The exception is clause (b), which is a *permission*:
its licensed population is empty, and that is a real screen.

## The two arms

| | s49 | **s50** | **s51** |
|---|---|---|---|
| change | — | `COREF_RULES` stated as one guideline | nine of ten constants generalized |
| rule text | 4 022 B | 3 615 B | **2 461 B (-39%)** |
| instruction B/run | 60 892 | 44 612 (-27%) | **34 143 (-44%)** |
| calls/run | 87.3 | 87.7 | 88.0 |
| TP | 188.0 | 185.0 | 188.5 |
| FP | 12.5 | 9.7 | **20.2** |
| macro F1 | 96.57 | 96.41 | **94.43** |
| macro F2 | 97.01 | 96.36 | 96.06 |
| F1 vs s49 (p) | — | **-0.2 (0.71)** | **-2.1 (0.00)** |
| F2 vs s49 (p) | — | -0.7 (0.05) | -1.0 (0.02) |
| TP vs s49 (p) | — | -3.0 (0.01) | +0.5 (0.68) |
| FP vs s49 (p) | — | -2.8 (0.20) | **+7.7 (0.00)** |

Six paired runs each, all three arms in the same invocation.

## s51 — generalizing everything fails, and the cause is one structure

`stage_diff.py` attributes the +7.7 false positives by the linker that produced them
and by whether the sentence states the component's name at all:

| | gold | spurious |
|---|---|---|
| GAINED, full-name, name stated | 1.3 | 4.8 |
| GAINED, full-name, **via an alias only s51's table has** | 0.2 | **5.8** |
| GAINED, coreference | 0.2 | 0.7 |
| LOST, all sources | 1.9 | 4.1 |

So **three quarters of the damage arrives through the alias table**, and the terms
are all of one kind:

    teammates   'back end'   -> Logic     2.2/run   spurious
    teammates   'front-end'  -> UI        2.0/run   spurious
    jabref      'core'       -> model     0.8/run   spurious
    jabref      'center'     -> model     0.8/run   spurious
    bigbluebutton 'conversion process' -> Presentation Conversion  1.0/run  spurious

and the tables grow exactly where they do: jabref 1.0 → 4.7 terms, teammates
7.2 → 10.2, teastore 6.7 → 10.0. jabref's four new terms are `core`, `center`,
`intermediate layer`, `outer shell`.

Those are **layer and tier names — groupings**, and the clause s51 dropped from the
alias judge is the one that says a phrase naming a grouping of several elements is
not an alias for any one of them. That was the hypothesis this round ended on.

> **The bisect overturned it** (`results/s5253_prompt_bisect/`). Reverting the whole
> knowledge side recovers 3.5 of the 10.3 false positives and 0.3 F1 of 2.4;
> restoring the grouping clause alone recovers nothing. The alias table is a third
> of the damage, and the indicted clause is worth ~0.
>
> Two general reasons the reading failed. **A surface attribution is not a causal
> one**: every alias is fed into the extraction prompt, so a candidate admitted by a
> looser extraction rule and one admitted by a looser alias judge look identical at
> the link level. And **the table is not stable enough to attribute from**: s49 and
> s50 have byte-identical knowledge prompts and still build tables differing by 2.8
> terms per run — `back end` and `front-end` appear in some s49 tables too. Held to
> the terms *both* arms propose, where the proposer is constant, the alias judge
> flips three verdicts.

The correct attribution, from the bisect: most of the loss is the three rules the
**full-name linker** reads. Held to shared candidates, the generalized full-name
judge approves 3.5 more false positives per run and no more gold.

Consequence for the architecture, already stated in the s26 and s46 rounds: **the
alias table is this workflow's only admitting structure**, so a looser rubric in the
knowledge stage manufactures candidates for every later stage rather than merely
failing to reject one. That remains true — it is worth 3.5 FP — it is just not the
main effect here.

## s50 — and a measurement problem the whole series shares

Read at face value, s50 is F1-neutral (-0.2, p = 0.71) and loses 3.0 true positives
at p = 0.01. But `COREF_RULES` is read by the **last** of three linkers, and the
link-level diff puts 3.2 of those lost true positives on the **full-name** linker,
which runs first and whose every prompt byte is identical between the arms. *A change
cannot lose links in a stage it does not reach.*

`source_stats.py` re-runs the same exact permutation test per source:

| source | reachable by the change? | TP delta (p) | FP delta (p) | composition (p) |
|---|---|---|---|---|
| coreference | **yes** | +0.2 (1.00) | +1.2 (0.21) | +0.2 (0.18) |
| full_name | no | -2.0 (0.15) | -2.3 (0.36) | +1.9 (**0.03**) |
| partial_name | no | -1.2 (0.29) | -1.7 (**0.01**) | +1.4 (**0.03**) |
| all | — | -3.0 (**0.01**) | -2.8 (0.20) | +3.1 (**0.01**) |

On the one stage the change can reach, it is neutral on every measure. The p = 0.01
verdict is assembled from stages it cannot reach.

The mechanism is in the harness: **each arm re-runs the whole pipeline, including the
stages it does not modify, and those stages are LLM calls with their own spread.**
Pairing arms inside one invocation controls the model, the day and the ordering; it
does not control the *upstream sampling*. Concretely, s49 and s50 have byte-identical
knowledge prompts and still build alias tables that differ by 2.8 terms per run
across the five projects — and a smaller table means fewer full-name candidates.

This is not a caveat about one arm. Twenty variants in this branch have been judged
on whole-pipeline p values computed this way. A null arm (`s_linker49_null`,
byte-identical to s49 apart from its checkpoint namespace) is running to calibrate
it; results in `results/null_calibration/`.

What s50 *does* change is real but absorbed: the generalized rule makes the resolver
propose far more resolutions (mediastore 6.8 → 11.7 per run, jabref 1.2 → 9.7) and
the strict coreference judge rejects the extra ones. The enumeration was doing work
the downstream judge already does.

## Where this leaves the claim

- **The instructions are not free-floating prose that can be shortened at will**: the
  one arm that rewrote all of them lost 2.1 F1, decisively, on precision.
- **But the failure is localized, not diffuse.** Three quarters of it is one clause
  of one rubric, in the one stage that admits rather than rejects.
- **But the failure is not where the traces said it was.** Round 2
  (`results/s5253_prompt_bisect/`) reverted the knowledge side (`s_linker52`) and
  restored the single indicted clause (`s_linker53`): 3.5 of 10.3 false positives and
  nothing, respectively. Round 3 (`s_linker54`, `s_linker55`) tests the full-name
  family, which is where the shared-candidate verdict flips actually are.
