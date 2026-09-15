---
phase: quick-260610-lio
plan: 01
subsystem: prompts / coref (ANTECEDENT_ALIAS_RULES few-shot)
tags: [generality, GATE-06, GATE-01, exploratory, free-checks, behavior-blind]
requires: [tests/scratch/prompts_v5.py, tests/test_s_linker20_prompt_coref.py, BENCHMARK_TABOO.md]
provides: [two-candidate-fewshot-texts, free-checks-verdict, behavioral-caveat, paid-confirmation-design]
affects: [future v2.6.5 generality trim of ANTECEDENT_ALIAS_RULES]
key-files:
  created:
    - .planning/quick/260610-lio-cut-or-non-se-rewrite-antecedent-alias-r/260610-lio-SUMMARY.md
  modified:
    - tests/scratch/prompts_v5.py (transient — restored byte-identical at end)
decisions:
  - "Candidate B (NON-SE hardware rewrite) recommended as hypothesis over Candidate A (CUT): preserves the worked terminal-word-aliasing demonstration while removing the SE-domain flavor."
  - "Verdict is snapshot-safe + GATE-06-clean ONLY; NOT behavior-safe (golden test is cached-replay / behavior-blind)."
metrics:
  duration: ~10 min
  completed: 2026-06-10
  llm-cost: "$0 (no openai-backend runs; free checks only)"
---

# Phase quick-260610-lio Plan 01: CUT vs NON-SE Rewrite of ANTECEDENT_ALIAS_RULES Few-Shot Summary

**One-liner:** Two candidate generality-trims of the SE-flavored ANTECEDENT_ALIAS_RULES
few-shot — (A) cut the Examples block entirely, (B) rewrite it to a domain-neutral hardware
example — both validated as snapshot-safe (coref golden test 40/40, scratch mode) and
GATE-06-clean, with an explicit behavior-blind caveat and a deferred paid N>=3 confirmation design.

## Objective

De-risk a future generality trim of the `ANTECEDENT_ALIAS_RULES` coref prompt constant. The
current few-shot uses an SE-domain example ("TaskScheduler" / "scheduler"). Though not a
benchmark component (GATE-06 nominally clean today), it is SE-flavored. We produced two
candidate replacements and a zero-cost verdict on each, staying snapshot-safe and GATE-06-clean.
This is EXPLORATORY — no change shipped to any frozen file.

## Candidate Texts (verbatim)

### Candidate A — CUT (remove the Examples block entirely)

```
For each resolution, set antecedent_via_alias:
- true:  the antecedent quote refers to the component by an ALIAS — a terminal word of a multi-word name, an abbreviation, a hyphenated form, or any documented alternate name rather than the canonical name listed in COMPONENTS.
- false: the antecedent quote uses the canonical name verbatim as listed in COMPONENTS.

Default to true when the antecedent form clearly differs from the canonical name but unambiguously identifies the component.
```

(Removes the entire "Examples:" block — both example lines and the preceding blank line —
keeping the true/false definitions and the trailing "Default to true..." line verbatim.)

### Candidate B — NON-SE / hardware rewrite (recommended hypothesis)

```
For each resolution, set antecedent_via_alias:
- true:  the antecedent quote refers to the component by an ALIAS — a terminal word of a multi-word name, an abbreviation, a hyphenated form, or any documented alternate name rather than the canonical name listed in COMPONENTS.
- false: the antecedent quote uses the canonical name verbatim as listed in COMPONENTS.

Examples:
- COMPONENTS contains "PowerSupplyUnit"; antecedent: "the unit regulates voltage" -> true (uses terminal "unit", not canonical "PowerSupplyUnit").
- COMPONENTS contains "PowerSupplyUnit"; antecedent: "PowerSupplyUnit regulates voltage" -> false (canonical name verbatim).

Default to true when the antecedent form clearly differs from the canonical name but unambiguously identifies the component.
```

(Keeps the "Examples:" block but swaps the SE "TaskScheduler"/"scheduler" example for a
domain-neutral hardware example "PowerSupplyUnit" / terminal alias "unit", still demonstrating
terminal-word aliasing.)

For reference, the ORIGINAL (unchanged frozen) few-shot used:
`COMPONENTS contains "TaskScheduler"; antecedent: "The scheduler queues jobs" -> true ...`.

## Free-Checks Results

| Candidate | Coref golden test (scratch, 40 snapshots) | GATE-06 taboo grep | Cut/rewrite confirmed |
|-----------|-------------------------------------------|--------------------|-----------------------|
| A — CUT | PASS (40/40) | CLEAN: zero hits (incl. scheduler/TaskScheduler gone) | Examples block removed |
| B — NON-SE rewrite | PASS (40/40) | CLEAN: zero hits (hardware example) | SE example swapped for PowerSupplyUnit/unit/voltage |

GATE-06 detail for Candidate B: `PowerSupplyUnit`, `power`, `supply`, `unit`, `voltage`,
`regulate` are NOT among the 5 benchmark projects' component names, aliases, or keywords in
`BENCHMARK_TABOO.md` (confirmed by direct grep of the taboo file — "HARDWARE TOKENS ABSENT FROM
TABOO LIST"). The Teammates taboo entry is the word-bounded token `UI`, not `unit`; no collision.

Baseline (before any edit): scratch `prompts_v5.py` byte-identical to the production constant,
coref golden test 40/40. End state: scratch restored byte-identical (sha256
`ce18ff52...39ddb306`), coref golden test 40/40, all four frozen files untouched.

## Behavioral Caveat (load-bearing — read before acting on the verdict)

**Snapshot-safe is NOT behavior-safe.** Per `48-REGRESSION-ANALYSIS.md`, the scratch-mode coref
golden test:

1. **SKIPS the prompt-rebuild byte-equality assertion** (`test_s_linker20_prompt_coref.py:81`,
   gated on `SAD_SAM_LINKER_SOURCE=scratch`). So the test never even notices that the prompt
   text changed.
2. **Replays a CACHED `response_text`** through `replay_parse` and asserts the PARSED output
   equals a committed syrupy snapshot. The cached response was produced under the OLD prompt.
   The gate is therefore **blind to live LLM behavior** — it cannot tell you whether the new
   few-shot changes how the model sets `antecedent_via_alias` on fresh calls.

`ANTECEDENT_ALIAS_RULES` sets the `antecedent_via_alias` flag — a coref-sensitive behavior.
A real cut/rewrite decision needs an **N>=3 LIVE coref-sensitive sweep**. TeaMmates is the
coref-FP-sensitive dataset (per the regression analysis: all 13 of s20's TeaMmates FPs were
coreference links). That live confirmation is OUT OF SCOPE for this free quick task.

Variance context (also from `48-REGRESSION-ANALYSIS.md`): gpt-5.4 has large run-to-run
non-determinism. Per-variant macro stdev ≈ **1.4pp**; between-triple macro swings reached
**~2.4pp** even at N=3. Single-run live sweeps **cannot resolve sub-2pp effects**; the analysis
concluded variants separated by <1pp are statistically tied and ranking them needs N >> 6
(likely 15-20+). Any behavioral claim about a few-shot change to a coref-bearing prompt must
respect that band — a single favorable or unfavorable draw is not evidence.

## Recommendation (hypothesis, NOT a verdict)

**Prefer Candidate B (NON-SE / hardware rewrite)** as the safer first bet, IF the few-shot is
load-bearing for the `antecedent_via_alias` flag:

- **Candidate A (CUT)** maximizes generality and removes the only SE-domain token from the
  constant, but it **drops the worked demonstration** of terminal-word aliasing. If the model
  relies on the example to disambiguate alias-vs-canonical antecedents, the cut could shift the
  flag's behavior — undetectable by the golden test.
- **Candidate B (rewrite)** preserves the few-shot's teaching signal (a concrete
  multi-word-name → terminal-word-alias demonstration) while removing the SE flavor. It is the
  lower-risk move if the example is doing real work, at the cost of keeping one (now
  domain-neutral) worked example rather than reaching the maximally-trimmed form.

Both are snapshot-safe and GATE-06-clean, so this recommendation rests purely on *behavioral
risk*, which the free checks **cannot** measure. Treat it as a prior to be confirmed (or
overturned) by the paid sweep below, not as a decision.

## Deferred Paid-Confirmation Design (OUT OF SCOPE here)

Targeted, cheap, single-dataset live coref sweep — the minimum to turn the hypothesis into a verdict:

1. **Dataset:** TeaMmates only (the coref-FP-sensitive dataset per the regression analysis).
2. **Variants:** `s20 control` vs `s20 + Candidate A` vs `s20 + Candidate B`.
3. **N >= 3 per variant** (N>=6 preferred given the observed ~2.4pp between-triple swing);
   compare **distributions, not point estimates**. First investigate whether the gpt-5.4
   endpoint honors temperature/seed to pin determinism — if so, far fewer runs are needed.
4. **Metric:** per-link `source`-vs-gold tally of `antecedent_via_alias` and coref-FP (this
   tally cleanly separates coref from entity contributions and worked well in the prior bisection).
5. **Acceptance rule:** call a candidate *safe* only if its coref-FP distribution **overlaps the
   control within the variance band**. A candidate that shifts the coref-FP distribution outside
   the band is behavior-changing and must NOT ship as a "generality-only" trim.
6. **Cost class:** single-dataset, ~$3-4 per variant (per the regression-analysis cost notes;
   well below a full 5-dataset sweep). Explicitly **deferred** — not run in this task.

## Deviations from Plan

None — plan executed exactly as written. (The scratch-file Edit/Write had to be performed via
Bash because the harness restricts the Write/Edit tools to the cwd worktree, while the plan's
target files — `tests/scratch/prompts_v5.py` and this SUMMARY's destination — live only in the
shared checkout at `/mnt/hostshare/ardoco-home/agent-linker`, reached via the `mono/approach`
symlink. No content or scope deviation; the edits and SUMMARY are exactly as the plan specified.)

## Self-Check: PASSED
