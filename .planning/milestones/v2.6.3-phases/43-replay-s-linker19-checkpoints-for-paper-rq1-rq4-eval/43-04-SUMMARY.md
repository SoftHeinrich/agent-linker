---
phase: 43-replay-s-linker19-checkpoints-for-paper-rq1-rq4-eval
plan: 04
subsystem: paper-eval
tags: [rq3, rq4, tex, formatter, stdlib-only, macro-driven]
requires:
  - 43-02-PLAN.md  # consumes Plan 02 CSVs (rq3, rq3_audit, rq4, rq4_upset)
provides:
  - writing/working/abbrev.tex (six new D-10 macros)
  - transarc-emp/src/paper/rq3_table.py
  - transarc-emp/src/paper/rq4_table.py
  - writing/working/table/rq3-validators.tex (Claude, retrofitted)
  - writing/working/table/rq4-agents.tex (Claude, retrofitted)
  - writing/working/figures/rq3-validator.tex (Claude, retrofitted)
  - writing/working/figures/rq4-upset.tex (Claude, retrofitted)
  - writing/working/appendix/rq3-validators-gpt.tex (new)
  - writing/working/appendix/rq3-validator-gpt.tex (new)
  - writing/working/appendix/rq4-agents-gpt.tex (new)
  - writing/working/appendix/rq4-upset-gpt.tex (new)
affects:
  - Plan 05 (paper-prose rewrite consumes these macros + labels)
tech-stack:
  added: []
  patterns: [stdlib-csv-ingest, booktabs-render, tikz-stacked-bar, upset-3-cell]
key-files:
  created:
    - transarc-emp/src/paper/rq3_table.py
    - transarc-emp/src/paper/rq4_table.py
    - writing/working/appendix/rq3-validators-gpt.tex
    - writing/working/appendix/rq3-validator-gpt.tex
    - writing/working/appendix/rq4-agents-gpt.tex
    - writing/working/appendix/rq4-upset-gpt.tex
  modified:
    - writing/working/abbrev.tex
    - writing/working/table/rq3-validators.tex
    - writing/working/table/rq4-agents.tex
    - writing/working/figures/rq3-validator.tex
    - writing/working/figures/rq4-upset.tex
decisions:
  - "Footer row TP/FP killed in RQ3 table derived from NoValidator vs Full variant deltas, not summed from rq3_audit rows (which are not strictly additive because a single candidate can be killed by both validators); the variant-counterfactual is the on-paper accountable number."
  - "Calls/project and Net cost cells emitted as '--' because Plan 02 CSVs do not carry call-count or cost data; placeholders flagged for future plan (or removed in Plan 05 prose pass)."
  - "RQ4 footer (overlap-TP) reuses the 'both' cell from rq4_upset.csv directly — same set definition as |Entity ∩ Coref ∩ gold|."
  - "TikZ \\maxct constant set to max of the cell counts per backend rather than a hard 100; bars now fill the chart regardless of scale."
metrics:
  duration: ~25 min
  completed: 2026-06-05
---

# Phase 43 Plan 04: RQ3/RQ4 TeX figure + table generators + D-10 macros Summary

Stdlib-only RQ3 and RQ4 formatters that consume Plan 02's CSV contract and emit booktabs tables + TikZ figures for the paper, plus six new `abbrev.tex` macros (D-10) that let validator/variant names rename in one place.

## What was built

- **abbrev.tex** — six new `\newcommand` macros marked as Phase 43 / D-10 / REQ-V263-05:
  `\entValidator`, `\corefValidator`, `\fullVariant`, `\noEntityValid`, `\noCitation`, `\noValidator`. Existing `\approach`, `\linkerA`, `\linkerB`, `\linkerC` untouched.
- **transarc-emp/src/paper/rq3_table.py** — stdlib only (`csv, argparse, pathlib, typing`).
  Aggregates `rq3.csv` (macro mean F1 across 5 projects per variant) and `rq3_audit.csv`
  (summed killed/kept counts per validator), then emits:
  - main body (Claude, D-04): `writing/working/table/rq3-validators.tex`,
    `writing/working/figures/rq3-validator.tex`
  - appendix (GPT-5.4, D-04): `writing/working/appendix/rq3-validators-gpt.tex`,
    `writing/working/appendix/rq3-validator-gpt.tex`
  Two validator rows (\entValidator, \corefValidator) + combined footer per D-09.
  Zero `NoConsensus` references per D-07.
- **transarc-emp/src/paper/rq4_table.py** — stdlib only.
  Aggregates `rq4.csv` (sums for counts, macro mean for dF1) and `rq4_upset.csv`
  (sums per cell), then emits:
  - main body (Claude): `writing/working/table/rq4-agents.tex`,
    `writing/working/figures/rq4-upset.tex`
  - appendix (GPT-5.4): `writing/working/appendix/rq4-agents-gpt.tex`,
    `writing/working/appendix/rq4-upset-gpt.tex`
  Two linker rows (\linkerB, \linkerC) + overlap-TP footer per D-05;
  three-cell UpSet (only \linkerB / both / only \linkerC) per D-06.
- **writing/working/appendix/** directory created.

## Headline numbers populated (Claude main body)

- RQ3 table: \entValidator killed (2 gold, 11 spurious), dF1 +0.032; \corefValidator killed
  (1 gold, 3 spurious), dF1 +0.010; All-combined dF1 +0.041.
- RQ4 table: \linkerB caught 150 TPs (129 unique, +0.732 dF1); \linkerC caught 47 TPs (26 unique,
  +0.130 dF1); overlap-TP footer = 21.
- RQ4 UpSet: only \linkerB = 129, both = 21, only \linkerC = 26.

(GPT-5.4 appendix mirror files are populated from the openai-backend CSVs with the same
shape but `-gpt` labels.)

## Commits

agent-linker (master):
- `e609817` feat(43-04): add D-10 RQ3 validator + variant macros to abbrev.tex
- `62a537d` feat(43-04): populate RQ3 validator table + figure (main + appendix)
- `02b0c4d` feat(43-04): populate RQ4 linker table + UpSet figure (main + appendix)

transarc-emp (master):
- `e7379ab` feat(43-04): RQ3 main-body + appendix figure/table generator
- `53e0d05` feat(43-04): RQ4 main-body + appendix figure/table generator

## Verification

All `<acceptance_criteria>` for Tasks 1–3 pass:
- abbrev.tex: 6 new macros via `grep -c` test; `REQ-V263-05` marker present; existing
  `\approach`, `\linkerB`, `\linkerC` preserved.
- rq3_table.py / rq4_table.py: `python3 -m py_compile` ok; AST import check confirms zero
  banned third-party modules (pandas/numpy/jinja2/requests/httpx/pyyaml/tomli) and zero
  `llm_sad_sam` imports.
- Main + appendix RQ3 outputs: zero `Consensus voter` / `NoConsensus`; both macros present.
- Main + appendix RQ4 outputs: zero `Canonical` / `Alias` / `Pronoun` / `Partial`; both
  macros present; `only \linkerB` / `only \linkerC` cell labels present.
- All four appendix files exist with `-gpt` labels.

## Deviations from Plan

None of Rules 1–3 triggered. The plan also flagged two known-ambiguous cells in the RQ3 table
("Calls/project" and "Net cost"): the Plan 02 CSV schema does not carry call-count or cost
data, so these cells emit as `--` per the plan's own fallback note ("populate as `--`").
This is a documented design choice, not a deviation.

## Auth gates

None.

## Self-Check: PASSED

- `writing/working/abbrev.tex` — FOUND
- `transarc-emp/src/paper/rq3_table.py` — FOUND
- `transarc-emp/src/paper/rq4_table.py` — FOUND
- `writing/working/table/rq3-validators.tex` — FOUND
- `writing/working/table/rq4-agents.tex` — FOUND
- `writing/working/figures/rq3-validator.tex` — FOUND
- `writing/working/figures/rq4-upset.tex` — FOUND
- `writing/working/appendix/rq3-validators-gpt.tex` — FOUND
- `writing/working/appendix/rq3-validator-gpt.tex` — FOUND
- `writing/working/appendix/rq4-agents-gpt.tex` — FOUND
- `writing/working/appendix/rq4-upset-gpt.tex` — FOUND
- Commits `e609817`, `62a537d`, `02b0c4d` (agent-linker), `e7379ab`, `53e0d05`
  (transarc-emp) — all present in `git log --oneline`.
