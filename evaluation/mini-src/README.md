# mini-src — the code that reproduces the paper's tables

Self-contained and stdlib-only. Everything needed to rebuild every results table
in the paper lives here, and nothing else does: an exploratory probe or a
one-off audit belongs in `../studies/`.

| Stage | Module | Produces |
|---|---|---|
| shared core | `metrics.py` | the benchmark layout, the confusion matrix and F-measures, the gold loaders, the CSV writer |
| RQ1 / RQ2 input | `build_alinker_extracts.py` → `build_dump.py` | the arm's slots in the normalized link dump |
| RQ1 / RQ2 | `rq12.py` | `reports/RQ12_BIGTABLE.csv`, `reports/RQ12_PERPROJECT.csv` |
| RQ3 / RQ4 | `rq34.py`, `rq34_rq2.py`, `rq4_floor.py` | `reports/rq34/<arm>/` |
| reshape | `rq_tables.py` | one small CSV per paper float, under `reports/tex_src/` |
| render | `csv_to_tex.py` | one booktabs `.tex` per float, under `reports/tex/` |
| bridge | `sync_paper.py` | copies them into the paper; `--check` is the drift guard |
| gates | `check.py`, `gen_csv_to_temp.py` | frozen metric goldens; a no-overwrite regeneration diff |

Run them in that order (`HOWTO-REGENERATE-RQ.md` has the copy-pasteable script).
The reshape and render stages compute nothing — every cell they emit is copied
from an engine's CSV, so a number can only enter the paper through an engine.

## The metrics

`metrics.py` is the project's **sole** metrics implementation and the shared core
of the whole `evaluation/` tree: the benchmark layout, the confusion matrix and
F-measures, the gold loaders and the dict-row CSV writer live here once, and
every engine plus `mini-inequality/` and `../studies/` imports them. Nothing
outside it re-derives an F-measure. The former canonical
stack (`src/lib/metrics_api.py`, `src/bias/component_suite.py`, the RQ2/bias
side-analyses, and the `generate_tables.py` table pipeline) has been retired —
it is preserved on the `master` (legacy) branch (`git show master:<path>`), not
on this `mini` branch. The redundancy analysis that justified the reduction
(legacy `reports/RQ2_METRIC_REDUNDANCY.md`) showed the dropped columns carried
no independent ranking signal (Spearman ρ ≥ 0.85 with a kept metric, ~0
system-pair reversals).

## What it computes

| Task       | Metrics                                                                |
|------------|-----------------------------------------------------------------------|
| `sad-code` | file P/R/F1+F2; per-component F1+F2 (micro); **worst-component F1+F2**; **harmonic-mean component F1+F2** |
| `sad-sam`  | link P/R/F1+F2; **component miss rate** (CMR%) |

**Every F1 ships with its F2.** F2 is recall-weighted (a missed gold link costs
4× a spurious one), which is the asymmetry that matters for recovered links: a
reader can discard a wrong link but cannot see one that was never proposed. The
paper reports the pair in every table, so each F1 key here has an `_f2` twin
computed by the same aggregation — for the tail metrics that means the min and
the harmonic mean *of the per-component F2s*, not the F2 of the worst-F1
component.

The **size-aware suite** (worst-component + harmonic mean) is the paper's
`metric.tex` headline: it weights each architecture component equally rather
than each link pair, so a system that abandons one documented component scores
0 even when file-level F1 stays high. Both flavours zero out on the same
abandoned component. `sad-sam` keeps no per-component view — without enrolment it
collapses onto link F1 (ρ = +1.00).

Dropped as redundant (appendix-only): sentence F1, decision F1, weighted F1,
MCC, MAP, ACF1, NDG, HUS, and the per-component *macro* mean (washes the tail
back out; ρ ≈ 0.85–0.96 with file F1).

### Definitions

- **per-component grouping (D-01):** each `(sentence, file)` link contributes one
  `(sentence, component)` pair per SAM-CODE component that owns the file; files
  mapping to no component are dropped (same rule for gold and result).
- **interface drop (D-12):** `Interface:` model elements are excluded from the
  file→component map. Every interface shares its code extent with a `Component:`
  twin (0 interface-only files) and the doc-to-model gold never links a sentence
  to an interface, so interfaces add no unique code and no documentation signal;
  dropping them makes the component count the distinct architectural units
  (7/10/6/9/6) and removes a per-component distortion on MediaStore/TeaStore. The
  tail metrics are invariant to it (they don't change on any project).
- **per-component F1 (micro):** one P/R/F1 over all `(sentence, component)` pairs;
  `component_f2` is F2 of that same P/R pair.
- **worst-component F1/F2:** minimum per-component F1 (resp. F2) over a project's
  gold components — one abandoned component drives either to 0.
- **harmonic-mean component F1/F2:** harmonic mean of per-component F1 (resp. F2)
  over gold components; also 0 if any component is missed.
- **component miss rate (CMR%):** the doc-model size-aware metric — share of gold
  (sentence, component) assignments whose component recovers no correct link.
  Reported in **percent** (the paper's `\cmr` is the same quantity as a share in
  [0,1]); named *silent-failure mass / SFM* until 2026-08-27.

> The integer **component miss count (CMC)** was reported beside CMR until
> 2026-09-01. It was dropped: no `.tex` table and no downstream engine read the
> column, and CMR prices the same abandonment. CSVs written before that date keep
> the column; readers select by name, so they still load.

> A per-sentence **noise rate** (mean FP/(TP+FP) over predicted sentences) was
> reported here until 2026-08-27. It was dropped: the paper never defined it, and
> it is not 1 − precision, so the two figures could not be reconciled by a reader.

## Usage

```bash
# All five projects, bundled TransArc results:
python3 mini-src/metrics.py --task sad-code
python3 mini-src/metrics.py --task sad-sam

# One project, also dump a CSV:
python3 mini-src/metrics.py --task sad-code --project jabref --csv /tmp/panel.csv

# Score arbitrarily-named result CSVs (column dialect auto-detected):
python3 mini-src/metrics.py --task sad-code \
    --results-dir /path/to/run \
    --result-pattern 's_linker21_{project}_links.csv'
```

Benchmark and result roots default to the bundled tree and can be overridden via
`$TRANSARC_BENCHMARK` / `$TRANSARC_RESULTS_DIR` or the `--results-dir` flag.

## RQ3 and RQ4 — judges, modules, and the floor

Three engines, all writing into `reports/rq34/<arm>/`. They live here rather than
in a study of their own because they feed body floats.

- **`rq34.py` — RQ3 (judge contribution) and RQ4 (per-module ablation).** RQ3 needs
  each judge's per-link accept/reject decisions and RQ4 needs each linker's
  independent output; neither survives into the scored link CSVs, so this is the one
  engine that reads the run's **phase state** directly. Per phase
  (`linker_{full_name,partial_name,coreference}`) it recovers the judged-and-kept
  links plus one `judge_decisions` record per candidate, scores every link against
  the SAD-SAM gold standard, and cross-checks each Full variant's `tp/fp/fn` against
  the run's `ablation_*.json` (`validate=OK`).
  RQ3 is measured from those *logged decisions*, not by re-running with a judge
  removed; RQ4's headline is the `only_X / shared` set overlap, because leave-one-out
  is contaminated — with one linker gone the others recover some of its hits. The
  leave-one-out ΔF1 is emitted anyway, marked as the contaminated comparison.
- **`rq34_rq2.py` — the doc-to-code lens.** Composes each RQ3/RQ4 link set through
  the recovered SAM-CODE links and scores it with the RQ2 size-aware panel, to show
  whether the effects survive the change of grain. `RQ2_LENS_DECISION.md` records
  what that answered: yes for RQ4, no for RQ3.
- **`rq4_floor.py` — RQ4's total floor.** One linking call against the whole
  workflow. That arm records no `linker_*` phases, so `rq34.py` structurally cannot
  read it — there are no stages to attribute — and this engine scores both sides end
  to end off the predicted-link CSVs instead.

**One knob picks the arm.** `rq34.py`'s `ARMS` table maps the reported arm to its
phase-state variant *and* its run sweep, so `$ALINKER_ARM` moves both together;
setting them apart is how a report directory ends up holding another arm's numbers.
A bare run of any of the three engines reproduces the committed
`reports/rq34/<arm>/` byte for byte — `gen_csv_to_temp.py` asserts exactly that.

**The default output belongs to the default run.** Each engine rewrites its whole
report directory rather than merging into it, so scoring a different sweep
(`--runs-from`, `--head-runs`/`--arm-runs`) or a subset (`--backends`, `--run`) would
leave the arm's published numbers partly overwritten by a differently-scoped run.
Those flags therefore make `--csv-root` required, and the engine stops with the reason
instead of writing. `$ALINKER_ARM` is the deliberate exception: it moves the input and
the output together.

**Vendored types, not imported.** The phase-state files are pickles of agent-linker
dataclasses, and unpickling needs those classes importable under their original
module path. `_alinker_types.py` is a verbatim copy of the approach repo's
`llm_sad_sam/core/data_types_v2.py`; `rq34.py` registers it under that path before
unpickling. Nothing is imported from the agent-linker package, so this runs on a bare
interpreter whether or not the approach repo is installed.

## Verification

`check.py` is a self-contained regression: it scores the bundled TransArc
results with `metrics.py` and asserts the panel reproduces a frozen golden table
to 1e-4. The goldens were validated at retirement against the then-canonical
`metrics_api` (primary panel) and the interface-dropped `component_suite`
(`worst_component_f1` == `min_comp`).

```bash
python3 mini-src/check.py     # → PASS
```

Bundled TransArc headline averages: sad-code file F1 .80 (F2 .78) / comp F1 .82
(F2 .77) / worst-comp F1 .54 (F2 .51) / harmonic F1 .67 (F2 .65) / cov .75;
sad-sam link F1 .80 (F2 .78) / cov .79 / CMR 7.1%.
