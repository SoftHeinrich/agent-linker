# mini-rq34 — validator contribution (RQ3) & per-module ablation (RQ4)

A self-contained, stdlib-only mini study that computes the **paper's RQ3 and RQ4
metrics** directly from the agent-linker *running results*. It is the RQ3/RQ4
counterpart to `mini-inequality/` (RQ2 motivation) and `mini-src/` (RQ1/RQ2
link- and component-level metrics).

- **RQ3 — validator contribution.** A candidate link is a TP if it is in the
  gold standard, an FP otherwise. For each of the two validators (the
  `\entValidator` two-pass evidence gate on the entity linker, and the
  `\corefValidator` quoted-antecedent gate on the coreference linker): how many
  TPs it rejects (cost), how many of those rejected TPs are *unique* to it (TPs the
  other validator does not also reject), and how many FPs it rejects (benefit). The
  `Full / No*Valid / NoValidator` variant macro-F1 is still reported as raw F1,
  but no per-validator ΔF1.
- **RQ4 — per-module ablation.** For each linker (`Entity` = `\linkerB`,
  `Coref` = `\linkerC`): true positives caught, *unique* true positives no
  other linker caught, false positives, and the leave-one-out macro-F1 delta —
  plus the `only_E / both / only_C` overlap decomposition of the gold links.

Both at the **doc-to-model (SAD-SAM)** grain, across all five ARDoCo projects.

## Why a dedicated folder (and why it reads pickles, not `paper-result/`)

RQ3 needs each validator's *per-link accept/reject decisions* and RQ4 needs each
linker's *independent validated output*. The normalized `paper-result/*.csv`
files are per-project P/R/F1 summaries — they carry neither the candidate-vs-
validated split nor the per-linker provenance. That information survives only in
the run's `phase_cache` pickles, so this study reads those directly. This is the
one mini-study that depends on run internals rather than scored result CSVs.

## Inputs

The canonical N=3 `s_linker21` sweep (v2.6.6) in the agent-linker repo:

| Backend  | Paper role      | Results slot (default)                      |
|----------|-----------------|---------------------------------------------|
| `openai` | main body       | `../agent-linker/results/v2.6.6_s21_gpt`    |
| `claude` | appendix mirror | `../agent-linker/results/v2.6.6_s21_sonnet` |

Per `run{1,2,3}/<project>/` it reads
`phase_cache/s_linker21/<backend>/<project>/{layer3,layer4,final}.pkl`:

- `layer3` → entity linker `candidates` + validator-approved `validated`
  (→ entity kept/rejected sets).
- `layer4` → `coref_raw` + `coref_validated` (→ coref kept/rejected sets).
- `final` → the emitted link set (= entity-kept ∪ coref-kept, deduped).

Gold standard: `goldstandard_sad_*-sam_*.csv` under `$TRANSARC_BENCHMARK`.

Roots derive from this file's location; override via `$TRANSARC_BENCHMARK`,
`$RQ34_VARIANT`, `$RQ34_CLAUDE_SLOT`, `$RQ34_OPENAI_SLOT` (e.g. point these at
the prior `s_linker20_union` / `v2.6.5_s20union*` slots for a side-by-side).

## Method notes (faithful to `working/sections/results.tex`)

- **RQ3 is measured from logged decisions, not by re-running.** The headline
  per-validator signal is its rejected/kept TP/FP counts plus the *unique rejected
  TP* (TPs it rejects that the other validator does not). The
  `Full / NoEntityValid / NoCitation / NoValidator` "validator removed" link
  sets (final set with that validator's rejected links added back) are still
  derived per project and macro-averaged, but reported as raw macro-F1 only —
  no per-validator ΔF1.
- **RQ4 uses set overlap as the headline**, because leave-one-out is
  contaminated (removing one linker lets the other recover some of its hits).
  The leave-one-out `delta_f1_if_removed` is still emitted, but the
  `only_E / both / only_C` cells are the figure.
- **N=3 → run-aware aggregates.** Top-level CSVs include `run1`, `run2`, `run3`,
  and an `average` row. The per-project drill-down CSVs are still emitted from
  one coherent run (so counts are real integers): the **median-macro-F1** run.
  Force a drill-down/aggregate run with `--run runN`.
- **RQ2-lens companion.** `rq34_rq2.py` composes the RQ3/RQ4 SAD-SAM link sets
  through recovered SAM-CODE links and scores them with the RQ2 doc-to-code
  metric suite, exposing whether validator/linker effects remain visible in
  file F1/F2, sentence coverage, noise, worst-component F1/F2, and
  harmonic-component F1/F2 (the panel is copied from `mini-src/metrics.py`, which
  pairs every F1 with its recall-weighted F2).

## Vendored types (copied, not imported)

The pickles are instances of agent-linker dataclasses. Per the mini-* "copy,
don't import" rule, `_alinker_types.py` is a verbatim copy of the approach
repo's `llm_sad_sam/core/data_types_v2.py` (`SadSamLink`, `CandidateLink`, …);
`rq34.py` registers it under that module path before unpickling. Nothing is
imported from the agent-linker package, so this runs on a bare interpreter even
if the approach repo is not installed.

## Usage

```bash
python3 rq34.py                 # both backends → reports/
python3 rq34.py --backends claude
python3 rq34.py --run run1      # force a specific run
python3 rq34.py --no-validate   # skip the ablation-JSON cross-check
python3 rq34_rq2.py             # RQ3/RQ4 variants scored with RQ2 metrics
```

## Outputs (`reports/`)

CSV only — no TeX. Dataset-wide aggregates are summed/averaged over the 5
projects per run; top-level reports include all three runs plus a run-average.

| File | Content |
|------|---------|
| `rq3_validators.csv` | Per run/backend/validator (+combined): rejected/kept × TP/FP and `unique_rejected_tp` (TPs rejected by this validator but not the other; blank on the `all_combined` row) **summed over the 5 projects**; `average` rows are means over runs. |
| `rq3_variants.csv` | Per run/backend: macro-F1 of each variant (`Full/NoEntityValid/NoCitation/NoValidator`). Raw F1 only — no ΔF1. |
| `rq4_linkers.csv` | Per run/backend/linker (+overlap row): TPs caught, unique TPs, FPs **summed over the 5 projects**, ΔF1 if removed; `average` rows are means over runs. |
| `rq4_variants.csv` | Per run/backend: macro-F1 of entity-only / coref-only / full link sets. |
| `rq34_rq2_variants.csv` | RQ3 variants after SAD-SAM→SAD-CODE composition, scored with the RQ2 doc-to-code metric panel. |
| `rq34_rq2_linkers.csv` | RQ4 linker sets after SAD-SAM→SAD-CODE composition, scored with the RQ2 doc-to-code metric panel. |
| `RQ34_RQ2_INVESTIGATION.md` | Short interpretation of the RQ2-lens variant/linker deltas. |
| `<backend>/<project>/rq3.csv` | Per project: 4 variant rows: tp/fp/fn/f1. |
| `<backend>/<project>/rq3_audit.csv` | Per project: 2 validator rows: rejected/kept TP/FP + unique_rejected_tp. |
| `<backend>/<project>/rq4.csv` | Per project: 2 linker rows: tps_caught/unique_tps/fps/delta_f1_if_removed. |
| `<backend>/<project>/rq4_upset.csv` | Per project: 3 cells: only_E/both/only_C. |
| `<backend>/runs_summary.csv` | All 3 runs' per-project + macro F1; canonical marked. |

**Aggregation:** run rows sum counts over the **5 projects of one coherent
run** (RQ4 also averages its leave-one-out ΔF1). `average` rows are means of the
three run rows, so count-like columns may be fractional there. Per-project, un-summed numbers are in the
`<backend>/<project>/` CSVs for the canonical or forced run.

## Verification

`rq34.py` cross-checks every Full-variant `tp/fp/fn` against the run's
`ablation_*.json` and prints `validate=OK` per backend (mismatches are listed).
The canonical-run macro-F1 (`claude` run1 0.9318, `openai` run3 0.9338; the
median-macro run per backend) is reproduced from the s21 sweep's phase cache.

## Conventions (inherited)

- Python 3, **stdlib only** — no `requirements.txt`, no third-party deps.
- The benchmark layout, the gold loader and the F-measures are imported from
  `mini-src/metrics.py` (the tree's shared core), so RQ3/RQ4 score with the same
  arithmetic as RQ1/RQ2; only the agent-linker types are vendored (above).
- No benchmark-derived word lists (workspace leakage rule) — IDs and counts only.
