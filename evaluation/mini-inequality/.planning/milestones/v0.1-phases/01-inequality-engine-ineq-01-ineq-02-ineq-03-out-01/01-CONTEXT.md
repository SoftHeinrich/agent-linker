# Phase 1: Inequality Engine — INEQ-01, INEQ-02, INEQ-03, OUT-01 - Context

**Gathered:** 2026-06-21
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — grey-area tables accepted/overridden by user

<domain>
## Phase Boundary

Deliver a self-contained, stdlib-only inequality engine at the study root —
`inequality.py` (i.e. `mini-inequality/inequality.py` from the repo root) — that
measures the **gold** trace-link concentration inequality of the ARDoCo benchmark
and reproduces the paper's distributional-inequality numbers:

- **INEQ-01** — per-component gold link-count inequality (Gini, Lorenz points,
  top-k share, min/median/max) for `sad-code` (component level) and `sad-sam`,
  across all 5 projects → CSV.
- **INEQ-02** — per-file (post-enrollment) concentration + per-sentence gold
  links-per-sentence distribution for `sad-code`.
- **INEQ-03** — per-project enrollment expansion factor (component decision →
  file-level pairs) and the aggregate component-FP → file-FP cascade.
- **OUT-01** — self-contained artifacts (engine + `reports/` MD/CSV), stdlib only,
  no cross-module imports, sanity-checked against canonical numbers.

IN scope: measuring the dataset distribution + reproducing the paper's Ch1
inequality tables. OUT of scope (later phases): paper-claim MATCH/MISMATCH audit
& `XX` placeholder resolution (Phase 2); Top-3/random baseline exploitation,
paper-ready Gini/Lorenz **table/figure** for ingestion, Lorenz **TeX figure**
(Phase 3 / OUT-02). New trace-link recovery is never in scope.

**Isolation (HARD):** this study lives only under `mini-inequality/`. All GSD
commands and the engine run with project root = `mini-inequality/`. Never write
to repo-root `.planning/` (active **v1.2** milestone). Commit only
`mini-inequality/**` on branch `gsd/mini-data-inequality`.
</domain>

<decisions>
## Implementation Decisions

### Area 1 — Output & report shape
- **Per-task CSVs**: `reports/inequality_sad_code.csv` and
  `reports/inequality_sad_sam.csv` (column sets differ by task), plus a combined
  long-format `reports/inequality.csv` is acceptable if cheap. Mirror the
  `COMPONENT_SUITE_{level}.csv` / `metrics.py` panel style.
- **Per-project rows + an `AVG`/`Total` row** (matches `metrics.py average_row`
  and the eval.tex tables which report per-project then aggregate).
- **Markdown report = tables-first, each section prefixed by a one-line headline**
  stating the claim (e.g. "Gini ranges 0.331→0.645"), mirroring eval.tex Ch1 prose
  so numbers are paste-ready. Report file: `reports/INEQUALITY.md`.
- **Outputs land in `mini-inequality/reports/`** (i.e. `reports/` relative to the
  study root). Engine at the study root (`inequality.py`).

### Area 2 — Distribution, Lorenz & top-k  (user: "use the most inequality-ed ones")
- **The engine computes SEVERAL distinct Ginis — do NOT conflate them.** Each
  has its own sanity target:
  - **Per-sentence enrolled sad-code links** (`tab:sent_gini`) — the headline
    **Gini 0.331 (MediaStore) → 0.645 (Teammates)**, Top-3 %, Min/Median/Max.
    This is INEQ-02 and the number the paper cites. Frozen eval.tex literal.
  - **# distinct sentences per component** (sad-code component level & sad-sam) —
    matches `component_suite.gold_gini` (INEQ-01). NOT in eval.tex; agreement is
    by-construction (copied `_gini`, same universe).
  - **SAM-CODE files per architectural element** (`tab:samcode_skew`) — Gini
    **0.400→0.694**, the `|files(m)|` fan-out that drives the cascade. Frozen
    eval.tex literal. Computed for INEQ-01/INEQ-03.
- **ALSO report the more-skewed supplementary units** (per user "most
  inequality-ed"): **# links (pairs) per component** and **# files per component**.
  Supplementary views, not sanity targets.
- **Lorenz: emit the FULL cumulative curve** (per-component points: cumulative
  population %, cumulative mass %) to CSV, **plus an 11-point decile summary** in
  the markdown.
- **Emit the raw Lorenz CSV now** (`reports/lorenz_*.csv`, pgfplots-friendly:
  `cum_pop_pct,cum_mass_pct`). The formatted TeX figure is deferred to Phase 3.
- **Concentration set = top-1, top-3, top-10% shares + the Palma ratio**
  (top-10% mass ÷ bottom-40% mass). top-1/top-3 are required by INEQ-01; top-10%
  and Palma are the "most inequality-ed" extras.

### Area 3 — Sanity-check gate
- **Source of truth = frozen eval.tex literals**, embedded as cited expected
  constants (Gini 0.331…0.645 from `tab:sent_gini`; enrolled total 18,660 and
  per-project factors from `tab:enrollment`; cascade 36→3,457 / 96.0× from
  `tab:amplification`). The engine recomputes each value with its OWN copied
  `_gini`/enrollment and asserts agreement — imports from `src/`/`mini-src/` are
  forbidden, so component_suite is NOT called live; agreement of the copied
  definitions with the published numbers IS the faithfulness proof.
- **Tolerance: |Δ| ≤ 0.005 on Gini** (eval.tex reports 3 decimals); **exact match
  on integer counts** (18,660; 36; 3,457; per-project enrolled counts).
- **Fail loud on mismatch — non-zero exit + a diff table** (project, metric,
  expected, computed, Δ). Matches the workspace "fail loud" convention (WR-03).
- **The sanity check runs by default at the end of every run** (prints a `CHECK`
  section); a `--check-only` flag runs just the gate.

### Area 4 — Expansion & structural amplification  (RE-PIVOTED per user: "do not use anything TransArc-specific; re-pivot towards benchmark distribution")
- **Enrollment expansion = pure-gold structural**: reproduce eval.tex
  `tab:enrollment` (Raw, Dir. Entries, Enrolled, Factor) per project; totals 525
  raw → 18,660 enrolled (35.5× avg), per-project 1.0× (MediaStore) → 217.6×
  (JabRef). From raw sad-code gold + `.acm` code model via copied
  `enroll`/`normalize_path`. sad-code only (Q4 override).
- **Structural component→file amplification (REPLACES the TransArc cascade)**:
  the dataset-intrinsic amplification driver = the gold SAM-CODE files-per-
  architectural-element fan-out `|files(m)|` (gold only, no system results).
  Report its distribution (AEs, min/median/max, Gini, top-3 conc) — this IS
  eval.tex `tab:samcode_skew` (Gini 0.400→0.694) — plus the structural statement:
  one component-level decision expands to up to `max|files(m)|` file-level pairs
  (worst case 972 = JabRef `logic`, 348 = Teammates `ui`); aggregate amplification
  = enrollment factor (total enrolled ÷ total component decisions). NO result/
  system files are read anywhere in the engine.
- **DROPPED — eval.tex `tab:amplification` (36→3,457 / 96.0×)**: that is a
  TransArc actual-error attribution (real sad-code FPs decomposed by transitive
  cause over a system's output), NOT a gold/benchmark property and NOT reproducible
  from gold alone. Per the user directive it is OUT of this dataset-inequality
  study; it belongs to the TransArc empirical pillar, not here.
- **Headline numbers featured** (all gold/benchmark): enrollment 525→18,660
  (35.5× avg, 217.6× JabRef); files-per-component Gini 0.400→0.694 with max fan-out
  972; per-sentence Gini 0.331→0.645.
- **Q4 — the enrollment-expansion table is sad-code-only** (sad-sam has no
  enrollment; sad-sam is still fully measured for the distribution in Area 1/2).

### Claude's Discretion
- Exact CSV column names/order, markdown layout details, argparse flag names
  (beyond `--check-only`), and internal function decomposition — mirror
  `mini-src/metrics.py` style.
- Whether to also emit the combined long-format `inequality.csv` (nice-to-have).
- Choice of Palma denominator edge-case handling (empty bottom-40%).
</decisions>

<code_context>
## Existing Code Insights

### Reusable definitions (COPY verbatim, do NOT import — isolation rule)
- `src/bias/component_suite.py`:
  - `_gini(values)` — the Gini formula to copy (sorted, cumulative form).
  - `gold_gini` = `_gini([len(sentences) for each component])` — the canonical
    "#sentences-per-component" inequality the sanity check must reproduce.
  - `_code_inputs` — the reconciled sad-code universe: collapse (sentence, file)
    → (sentence, component) via the enrolled SAM-CODE map, **dropping files with
    no SAM-CODE component** (apply the same drop rule consistently).
- `mini-src/metrics.py` (the stdlib pattern to mirror):
  - `enroll(gold, code_files)` — expand directory-trailing-`/` gold entries to
    individual files.
  - `normalize_path(path)` — strip leading `Implementation/`.
  - `load_code_model_files` (parse `.acm` JSON), `load_gs_sad_sam`,
    `load_gs_sad_code_raw`, `load_file_to_comps` (file→{component} via SAM-CODE).
  - Benchmark/result roots derived from file location (`_ARDOCO_HOME = parents[2]`)
    with `$TRANSARC_BENCHMARK` / `$TRANSARC_RESULTS_DIR` overrides — replicate so
    `inequality.py` at the study root resolves the same `ardoco-home` root.
  - CLI/table/CSV pattern (`argparse`, `print_table`, `write_csv`, `average_row`).

### Data inputs (GOLD / benchmark only — NO system results)
- Benchmark gold: `goldstandard_sad_*-sam_*.csv`, `goldstandard_sam_*-code_*.csv`,
  `goldstandard_sad_*-code_*.csv`; code models `*.acm` — 5 projects.
- The engine reads NO `results/` files (re-pivot: no TransArc-specific inputs).

### Path-relativity note (PREVENTS the double-nesting trap)
The engine and all GSD commands run with **project root = `mini-inequality/`**.
- Engine path = `inequality.py` (study root) = `mini-inequality/inequality.py`
  from repo root — do NOT create `mini-inequality/mini-inequality/…`.
- Outputs = `reports/…` (study root) = `mini-inequality/reports/…` from repo root.
- Reference modules/docs are one level up: `../src/bias/component_suite.py`,
  `../mini-src/metrics.py`, `../writing/eval.tex`.
</code_context>

<canonical_refs>
## Canonical References (downstream agents MUST read these)

Paths are repo-root-relative (repo root = `/mnt/hostshare/ardoco-home/transarc-emp`).
From the study root (`mini-inequality/`), prefix with `../`. The `alinker-paper`
refs live one level above the repo root (`/mnt/hostshare/ardoco-home/alinker-paper`).

- `writing/eval.tex` — **local Ch1 source of the sanity targets**. Key tables:
  `tab:enrollment` (lines ~57-74: 525→18,660, 35.5×, 1.0×→217.6×),
  `tab:amplification` (lines ~156-173: 36→3,457, 96.0×; per-project sad-sam FPs &
  induced file FPs), `tab:sent_gini` (Gini 0.331→0.645, Top-3 %). MUST read before
  implementing the sanity gate and the expansion/cascade tables.
- `src/bias/component_suite.py` — `_gini`, `gold_gini`, `_code_inputs` (copy the
  definitions; sanity-check agreement). READ, do not import.
- `mini-src/metrics.py` — `enroll`, `normalize_path`, `.acm`/gold loaders, CLI
  pattern (mirror). READ, do not import.
- `mini-src/check.py` — example of the sanity-check pattern in this workspace.
- `/mnt/hostshare/ardoco-home/alinker-paper/sections/metric.tex` — expansion
  1.0×→217.6×, long-tail-both-tasks (claim source for Phase 2; context here).
- `/mnt/hostshare/ardoco-home/alinker-paper/sections/eval.tex` — long tail
  dominates the average; Top-3/random exploit inequality (Phase 3 context).
- `/mnt/hostshare/ardoco-home/alinker-paper/sections/intro.tex` — `XX`
  placeholders (Phase 2/3 targets; not resolved in Phase 1).
- Benchmark root: `/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark/`
  (5 projects; gold standards + `.acm`). Overridable via `$TRANSARC_BENCHMARK`.
- (NOT used) `results/` — the engine reads no system results after the re-pivot.
</canonical_refs>

<specifics>
## Specific Ideas

- The engine must reproduce these EXACT published numbers (faithfulness gate):
  - `tab:sent_gini` per-sentence enrolled sad-code Gini: 0.331 / 0.448 / 0.645 /
    0.472 / 0.527 (MediaStore/TeaStore/Teammates/BBB/JabRef); Top-3 % 27.1 / 35.2
    / 20.3 / 21.3 / 70.0.
  - `tab:samcode_skew` files-per-AE Gini: 0.400 / 0.694 / 0.452 / 0.513 / 0.612;
    AEs 19/19/14/22/6; enrolled 60/164/1,616/730/1,956; Top-3 Conc 43.3/68.9/52.4/38.4/98.6%.
  - `tab:enrollment`: Raw total 525 → Enrolled 18,660 (35.5× avg); per-project
    factor 1.0× / 10.1× / 35.5× / 11.6× / 217.6×.
  - (DROPPED) `tab:amplification` 36→3,457 — TransArc-specific, out of scope per
    user re-pivot; the structural amplification driver is the `tab:samcode_skew`
    fan-out above (max 972 / 348).
- All three frozen gold tables (`tab:sent_gini`, `tab:samcode_skew`,
  `tab:enrollment`) were pre-verified to reproduce EXACTLY from the copied
  `mini-src/metrics.py` definitions + `_gini` (probe run 2026-06-21).
- "Most inequality-ed" (user, Area 2): prefer the richest concentration
  characterization — supplementary #links/#files-per-component units, full Lorenz
  curve, top-10% + Palma — alongside the sanity-anchored #sentences unit.
- sad-sam is omitted from the **enrollment-expansion table only** (Area 4 Q4),
  not from the inequality distribution.
</specifics>

<deferred>
## Deferred Ideas

- Paper-ready Gini/Lorenz **table/figure** for ingestion + Lorenz **TeX figure**
  → Phase 3 (OUT-02).
- Top-3 / random **baseline exploitation** argument → Phase 3 (MOTIV-01).
- Paper-claim **MATCH/MISMATCH/STALE audit** + `XX` placeholder resolution
  → Phase 2 (CLAIM-01/02/03).
</deferred>
