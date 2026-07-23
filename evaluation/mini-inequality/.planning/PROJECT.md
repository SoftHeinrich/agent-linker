# Data-Inequality Mini-Study

> **Isolated sub-project.** This `.planning/` directory is self-contained and scoped
> to the `mini-inequality/` study only. It deliberately does NOT share the repo-root
> `.planning/` (which tracks the active **v1.2 Component-Centric Metric Suite**,
> Phases 7-9). Run GSD phase commands against this subdir; never let this milestone
> write to root `.planning/`. Branch: `gsd/mini-data-inequality`.

## What This Is

A self-contained, stdlib-only mini study under `mini-inequality/` that quantifies the
**concentration inequality** of trace links in the ARDoCo benchmark (how a few large
components / files own most of the gold link mass) and uses that empirical evidence to
**verify the paper's distributional-inequality claims** and **motivate the proposed
four-metric evaluation suite**. Audience: the authors of the `alinker-paper`
(architecture-driven trace-link evaluation paper).

## Core Value

The inequality numbers this study produces must be (1) **faithful** — reproducible from
the bundled benchmark with zero cross-module imports, sanity-checked against the
canonical `src/bias/component_suite.py` `gold_gini` and the `eval.tex` tables — and
(2) **claim-grounding** — every distributional-inequality statement the paper makes
(and every `XX` placeholder) is backed by a number this study computed.

## Current Milestone: v0.1 Data-Inequality Mini-Study

**Goal:** Measure trace-link concentration inequality (Gini, Lorenz, top-k share,
enrollment expansion factor) across all 5 projects for both `sad-code` and `sad-sam`;
verify the paper's inequality claims against the computed numbers; fill the paper's
open `XX` placeholders; and empirically demonstrate why file/link micro-F1 needs the
four-metric suite.

**Target features:**
- A self-contained inequality engine (`mini-inequality/inequality.py`) → CSV
- A paper-claim verification report (MATCH / MISMATCH / STALE per claim) + resolved placeholders
- A baseline-exploitation argument (Top-3 / random) motivating the metric suite
- Paper-ready Gini/Lorenz table (+ optional Lorenz-curve data/figure)

## Requirements

### Validated (v0.1 — 2026-06-21)

- [x] **INEQ-01** Per-component gold inequality (Gini/Lorenz/top-k/min-median-max), both tasks, 5 projects → CSV — *Phase 1, `inequality.py`*
- [x] **INEQ-02** Per-file concentration + per-sentence links/sentence distribution (sad-code; Gini 0.331→0.645) — *Phase 1*
- [x] **INEQ-03** Enrollment expansion (525→18,660, 217.6×) + **gold structural** component→file amplification (max fan-out 972). NOTE: re-scoped — the TransArc actual-error cascade (36→3,457) was dropped as system-specific per user directive — *Phase 1*
- [x] **CLAIM-01** Paper data-inequality claims extracted into a checklist with source locations — *Phase 2, `CLAIM_CHECK.md`*
- [x] **CLAIM-02** Each claim labelled MATCH/MISMATCH/PARTIAL/SYSTEM-SPECIFIC with computed value (6 MATCH, 1 PARTIAL, 1 system-specific) — *Phase 2*
- [x] **CLAIM-03** `intro.tex` `XX` placeholders resolved (5 projects, 4 metrics, 70% JabRef, trivial-baseline F1 0.353); pipeline/approach deferred — *Phase 2-3*
- [x] **MOTIV-01** Top-3 + random baselines show micro-F1 inflation (0.353/0.381 > random) vs the 4-metric suite — *Phase 3, `MOTIVATION.md`*
- [x] **OUT-01** Self-contained stdlib engine + reports, no cross-module imports, sanity-checked vs canonical numbers — *Phase 1*
- [x] **OUT-02** Paper-ready Gini/Lorenz table (`.tex`+`.csv`) + Lorenz pgfplots figure source — *Phase 3*

### Active

(None — v0.1 complete. See `.planning/milestones/v0.1-phases/` for archived phase artifacts.)

### Out of Scope

- Touching repo-root `.planning/` or the v1.2 milestone — strict isolation; this study lives only under `mini-inequality/`
- New trace-link *recovery* (this study measures the dataset + scores existing results; it does not produce a linker)
- Re-deriving the full 13-column metric suite — `mini-src/` already does the metric panel; this study is about the *distribution*, not per-system scores
- Third-party deps (pandas/numpy/matplotlib) — stdlib only; figures emitted as pgfplots/TeX or plain data files

## Context

- **Parent workspace**: TransArc-EMP research workspace (two pillars: TransArc empirical study + benchmark bias analysis). The repo-root `.planning/` is mid-milestone (v1.2); this sub-study is isolated to avoid colliding with it.
- **Live paper**: `/mnt/hostshare/ardoco-home/alinker-paper` (`main.tex` + `sections/`). Inequality claims live in `sections/metric.tex` (1.0×→217.6× expansion; long-tail both tasks), `sections/eval.tex` (long tail dominates the average; Top-3/random baselines exploit inequality), `sections/intro.tex` (unfilled `XX` placeholders: % gold-mass concentration on three sentences, trivial-baseline file F1). Local mirror: `writing/eval.tex` Ch1 "Distributional Inequality" (Gini 0.331→0.645; 96.0× / 36→3,457 cascade).
- **Benchmark data**: `/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark/` (5 projects: mediastore, teastore, teammates, bigbluebutton, jabref). Gold standards: `goldstandard_sad_*-sam_*.csv`, `goldstandard_sam_*-code_*.csv`, `goldstandard_sad_*-code_*.csv`; code models `*.acm`.
- **Existing reference code** (read for definitions, do NOT import): `src/bias/component_suite.py` (`_gini`, `gold_gini` = Gini of gold #sentences-per-component), `mini-src/metrics.py` (enrollment, gold loaders, normalize_path — the stdlib pattern to mirror).
- **Key prior findings**: enrollment inflation up to ~217× (JabRef); gold link mass heavily right-skewed; LLM linkers nail popular components and abandon the long tail; file-level micro-F1 summarizes a handful of large components, not the recovery task.

## Constraints

- **Tech stack**: Python 3, stdlib only (`csv`, `json`, `collections`, `math`, `pathlib`, `argparse`). No requirements.txt.
- **Self-contained**: no imports from `src/` or `mini-src/`; definitions copied verbatim and sanity-checked for agreement.
- **Reproducibility**: benchmark/result roots derived from file location, overridable via `$TRANSARC_BENCHMARK` / `$TRANSARC_RESULTS_DIR`.
- **No benchmark leakage**: no benchmark-derived word lists (workspace CLAUDE.md rule). Distributional stats only.
- **Isolation (hard)**: never write to repo-root `.planning/`; commit only `mini-inequality/**` on branch `gsd/mini-data-inequality`.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Isolated branch + nested `mini-inequality/.planning/` | User requirement: must not collide with active v1.2 root `.planning/` | ✓ Held — all commits touched only `mini-inequality/**` |
| Stdlib-only, self-contained (mirror `mini-src/`) | Faithful, dependency-free, sanity-checkable reduction | ✓ Done — AST-verified stdlib-only |
| Measure GOLD distribution as primary; results/baselines secondary | "Data inequality of the dataset" is an intrinsic property of the gold standard | ✓ Done — gold-only engine + gold-only baselines |
| Verify against the live `alinker-paper`, mirror to local `eval.tex` Ch1 | alinker-paper is the live target; eval.tex Ch1 is the local source of the same claims | ✓ Done — CLAIM_CHECK.md audits both |
| **Drop the TransArc cascade (36→3,457); re-pivot INEQ-03 to gold structural amplification** | User directive: "no TransArc-specific; benchmark distribution." The cascade is an actual-error attribution (other pillar), not a gold property | ✓ Done — Phase 1 re-pivot; cascade labelled SYSTEM-SPECIFIC in CLAIM_CHECK.md |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition:**
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After the milestone:**
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?

---
*Last updated: 2026-06-21 — v0.1 Data-Inequality Mini-Study COMPLETE (all 9 requirements validated; isolated sub-project).*
