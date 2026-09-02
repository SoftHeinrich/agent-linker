# Regenerating the RQ1–RQ4 results by hand

This is the manual, step-by-step recipe for rebuilding every number behind the
paper's research questions, from the raw agent-linker run outputs to the scored
CSVs and the `.tex` floats the paper reads. Everything here is **stdlib-only
Python 3** — no `pip install`, no `requirements.txt`.

The canonical arm is **`s_linker110`** (the scan, plus the sibling-name refusal
and the resolver's antecedent shortlist) on two
GPT-5.6 backends: **terra = paper body, luna = mirror**. Every engine below
defaults to it. The retired arms (`s_linker21`, `s_linker20_union`) were dropped
from the roster on 2026-08-26. Their link dumps are still in `sota-links/`, and
`rq34.py` still scores them via `RQ34_ARM=s21`, but `rq12.py` no longer lists
them and nothing regenerates a paper number from them.

---

## Layout and environment

Every command below is run from the replication-package root, the directory that
holds `evaluation/`, `results/` and `sota-links/`. The scripts derive their roots
from their own location, so a bare run works; export these only to point at data
outside the tree:

```bash
cd /mnt/hostshare/ardoco-home/alinker-replication-package
export TRANSARC_BENCHMARK=/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark
export SOTA_LINKS=$PWD/sota-links
export TRANSARC_RESULTS_DIR=$PWD/evaluation/mini-data
```

| what | where |
|---|---|
| doc-model / doc-code link dumps | `sota-links/{model-doc/aalinker,doc-code/aalinker-composed}/{terra,luna}_s110/run{1,2,3}/` |
| per-phase state (RQ3/RQ4) | `results/consolidation_e2e_{terra,luna}_r{1,2,3}_20260825/phase_states/s_linker110/` |
| no-knowledge sweep | `results/consolidation_noknow_e2e_{terra,luna}_r{1,2,3}_20260902/` |
| RQ4 floor sweep | `results/onecall_e2e_{terra,luna}_r{1,2,3}_20260902/` (control: `noevidence_e2e_…`) |
| RQ1/RQ2 output | `evaluation/reports/RQ12_{BIGTABLE,PERPROJECT}.csv` |
| RQ3/RQ4 output | `evaluation/reports/rq34/s110/` (+ `s110_floor`, `s110_noknow`, `s110_noknow_luna`) |
| paper tables | `evaluation/reports/tex_src/*.csv` → `evaluation/reports/tex/*.tex` |

`rq34.py` and `rq4_floor.py` find the run root via `ALINKER_RESULTS` (auto-detected
for both layouts). The reported arm is one knob: `rq34.py`'s `ARMS` table maps
`$ALINKER_ARM` to the phase-state variant *and* the run sweep together, so a bare run
of any engine writes the arm it read. Seven modules declare `DEFAULT_ARM` and `check.py`
fails if any two disagree.

---

## Quick reference — full rebuild from the recorded runs (no LLM calls, ~2 min)

```bash
# (a) sota slots for this arm, if absent: run CSVs -> extracts -> dump
python3 evaluation/mini-src/build_alinker_extracts.py --variant s_linker110 \
    --out results/s110_extracts \
    --model terra results/consolidation_e2e_terra_r{1,2,3}_20260825 \
    --model luna  results/consolidation_e2e_luna_r{1,2,3}_20260825
EXTRACTS_DIR=$PWD/results/s110_extracts SOTA_LINKS=$PWD/sota-links \
  DUMP_CONFIG=terra_s110 DUMP_MANIFEST_TAG=s110_terra \
  python3 evaluation/mini-src/build_dump.py                       # terra_s110
EXTRACTS_DIR=$PWD/results/s110_extracts SOTA_LINKS=$PWD/sota-links DUMP_BE_DIR=luna \
  DUMP_BE_TAG=gpt-5.6-luna DUMP_CONFIG=luna_s110 DUMP_MANIFEST_TAG=s110_luna \
  python3 evaluation/mini-src/build_dump.py                       # luna_s110

# (b) RQ1 + RQ2
python3 evaluation/mini-src/rq12.py

# (c) RQ3 + RQ4  (bare = the reported arm; RQ34_ARM=s21 for the retired phase layout)
python3 evaluation/mini-src/rq34.py
python3 evaluation/mini-src/rq34_rq2.py

# (d) the RQ4 "No knowledge" row (see §4)

# (e) the RQ4 floor: the workflow against one linking call (see §5)
python3 evaluation/mini-src/rq4_floor.py

# (f) paper tables
python3 evaluation/mini-src/rq_tables.py
python3 evaluation/mini-src/csv_to_tex.py
PAPER_DIR=$PWD/paper python3 evaluation/mini-src/sync_paper.py --only rq

# (g) verify
python3 evaluation/mini-src/check.py                      # metric goldens -> PASS
python3 evaluation/mini-src/gen_csv_to_temp.py            # data CSVs reproduce -> exit 0
PAPER_DIR=$PWD/paper python3 evaluation/mini-src/sync_paper.py --check
```

Steps (b), (c) and (e) take no path arguments: each engine's default output is the
directory `rq_tables.py` reads for the same arm, and `gen_csv_to_temp.py` in step (g)
asserts a bare run of all four reproduces the committed CSVs byte for byte.

---

## Quick reference — scoring a candidate arm against the incumbent

The paper reports one arm (`s110`). Every generator resolves its inputs from
`$ALINKER_ARM` (default `s110`), so a candidate is scored by setting one variable
instead of editing paths in seven files. The incumbent keeps the unsuffixed names;
a candidate is written *beside* it, never over it:

| | incumbent (`s110`) | candidate (e.g. `s120`) |
|---|---|---|
| dump slots | `sota-links/**/{terra,luna}_s110` | `…/{terra,luna}_s120` |
| RQ1/RQ2 CSVs | `reports/RQ12_BIGTABLE.csv` | `reports/RQ12_BIGTABLE_s120.csv` |
| RQ3/RQ4 reports | `reports/rq34/s110` | `reports/rq34/s120` |
| reshaped + rendered | `reports/tex_src`, `reports/tex` | `reports/tex_src_s120`, `reports/tex_s120` |

A candidate also needs a row in `rq34.py`'s `ARMS` table (its phase-state variant and
run sweep); without one the RQ3/RQ4 engines fall back to the default arm's runs.

Nothing is synced into the paper until the arm decision is made — `sync_paper.py`
always reads the incumbent directories.

```bash
# (1) extracts from the candidate's recorded E2E runs. s110's are the consolidation
#     round's (they carry an in-set s_linker92a control -- see "the control" note below),
#     so no LLM calls are needed. A round whose runs were never recorded is the one case
#     that does cost calls; rq12.py --arm <arm> then stops with a "no dump slots" error
#     naming the four missing directories rather than scoring a partial set.
python3 evaluation/mini-src/build_alinker_extracts.py --variant s_linker110 \
    --out $PWD/results/s110_extracts \
    --model terra results/consolidation_e2e_terra_r{1,2,3}_20260825 \
    --model luna  results/consolidation_e2e_luna_r{1,2,3}_20260825

# (1b) THE CONTROL. The consolidation runs scored s_linker92a in the SAME invocations,
#      so build that as its own arm and compare against it. Comparing s110 against the
#      paper's s92a instead is cross-set: the two differ by three days of API drift as
#      well as by the arm, and on CMR that difference is larger than the arm's.
python3 evaluation/mini-src/build_alinker_extracts.py --variant s_linker92a \
    --out $PWD/results/s92actl_extracts \
    --model terra results/consolidation_e2e_terra_r{1,2,3}_20260825 \
    --model luna  results/consolidation_e2e_luna_r{1,2,3}_20260825
# build_dump.py defaults its roots in-tree, so only the cell knobs are needed. Name
# both the config slot and the manifest tag, or the candidate overwrites the incumbent.
EXTRACTS_DIR=$PWD/results/s110_extracts \
  DUMP_CONFIG=terra_s110 DUMP_MANIFEST_TAG=s110_terra \
  python3 evaluation/mini-src/build_dump.py
EXTRACTS_DIR=$PWD/results/s110_extracts DUMP_BE_DIR=luna \
  DUMP_BE_TAG=gpt-5.6-luna DUMP_CONFIG=luna_s110 DUMP_MANIFEST_TAG=s110_luna \
  python3 evaluation/mini-src/build_dump.py

# (2) score both arms (no LLM calls)
python3 evaluation/mini-src/rq12.py                      # incumbent, unsuffixed
python3 evaluation/mini-src/rq12.py --arm s110           # candidate,  _s110

# (3) the verdict: per-run deltas + sign agreement, not just the Average row
python3 studies/compare_arms.py s110 --base s92actl \
    --csv evaluation/reports/ARM_COMPARE_s110_vs_inset.csv   # the read to trust
python3 studies/compare_arms.py s110 --base s92a \
    --csv evaluation/reports/ARM_COMPARE_s110_vs_paper.csv   # vs the arm the paper ships
```

`studies/compare_arms.py` lives outside the pipeline because it feeds no float. It
exists because the Average row cannot settle this question: on this
benchmark one run moves the headline metrics by more than a typical arm delta, so a
mean whose runs disagree on the sign is reported as `INSIDE NOISE` however large it is.
Read the verdicts as:

- `BETTER` / `WORSE` — every run agrees on the sign and |mean| ≥ sd.
- `WEAK` — signs agree, but the mean sits inside one sd.
- `INSIDE NOISE` — the runs disagree on the sign; the mean is not evidence.
- `NO CHANGE` — every per-run delta is exactly zero.

The size-aware block (`dm CMR%`, `dc worst F1`, `dc harm F1`) is reported next to the
headline block on purpose: the paper's own argument is that link-level F1 is the wrong
place to read an architecture-traceability result, and that applies to picking an arm too.

### Promoting the winner

If the candidate wins, promote it by moving the default rather than by renaming data —
set `DEFAULT_ARM` in `mini-src/rq12.py`, `mini-src/rq_tables.py`, and
`mini-src/csv_to_tex.py` (the three are asserted to agree by `check.py`), then re-run the
full-rebuild quick reference above and `sync_paper.py`. The losing arm keeps its
suffixed CSVs, so the comparison stays reproducible after the promotion.

---

## 0. What lives where (the two data forms)

The RQs are computed by **two** scoring engines that read **two different forms**
of the same agent-linker runs:

| RQs | Engine | Reads | Why |
|-----|--------|-------|-----|
| RQ1, RQ2 | `rq12.py` | the normalized **`sota-links/` dump** (built from the run *extracts*) | needs only the final link sets, scored against gold |
| RQ3, RQ4 | `rq34.py`, `rq34_rq2.py` | the run **phase state** directly | needs per-judge and per-linker decisions that the final link set throws away |
| RQ4 floor | `rq4_floor.py` | the run's **link CSVs** | the floor arm records no phases, so there are no stages to attribute |

```
agent-linker runs ──► link CSVs ──► extracts ──► sota-links dump ──► rq12 ──────► RQ1, RQ2
       ├────────────► phase_states ──────────────────────────────► rq34 ───────► RQ3, RQ4
       └────────────► link CSVs ───────────────────────────────────► rq4_floor ─► RQ4 floor
```

### sota config slots

`sota-links/` stores each arm as a normalized link dump in a config slot:

| Config slot | Backend | Arm | Built by |
|-------------|---------|-----|----------|
| `terra_s110`, `luna_s110` | GPT-5.6-terra / -luna | **`s_linker110` — canonical** | `build_dump.py` (all defaults) |
| `terra_s92a`, `luna_s92a` | GPT-5.6-terra / -luna | `s_linker92a` — the arm s110 replaced; kept because `../studies/compare_arms.py --base s92a` reads it | `build_dump.py` (env-overridden) |
| `terra_s92actl`, `luna_s92actl` | GPT-5.6-terra / -luna | `s_linker92a` scored **in-set**, off the consolidation runs — the honest base for the promotion | `build_dump.py` (env-overridden) |
| `gpt-5.4_s21`, `sonnet_s21` (+ `_noknow`) | gpt-5.4 / claude | `s_linker21` — retired | `build_dump.py` (env-overridden) |
| `gpt-5.4_full`, `sonnet_full` | gpt-5.4 / claude | `s_linker20_union` — retired | `build_unified.py` |

Each slot holds `model-doc/aalinker/<slot>/run{1,2,3}/<project>.csv` (doc→model)
and `doc-code/aalinker-composed/<slot>/run{1,2,3}/<project>.csv` (doc→code,
composed through the ArCoTL model→code bridge). Gold is read from
`$TRANSARC_BENCHMARK`.

---

## 1. Build the normalized sota dump (needed for RQ1 / RQ2)

Skip this if the slots are already populated — they are committed. The build is
**additive and idempotent**: each pass writes exactly the one slot its env names
and then rebuilds `UNIFIED_MANIFEST.csv` by aggregating every per-task manifest,
so run order does not matter.

```bash
# (a) gold standards + ArCoTL model->code bridge + SOTA baselines (TransArC, Artemis).
#     The dump build below depends on the gold + bridge this produces, so run it first.
python3 sota-links/build_unified.py

# (b) the canonical s110 slots, from the extracts built in the quick reference.
#     terra is every default, so it needs no env at all; luna names its own cell.
python3 evaluation/mini-src/build_dump.py
EXTRACTS_DIR=$PWD/results/s110_extracts DUMP_BE_DIR=luna DUMP_BE_TAG=gpt-5.6-luna \
  DUMP_CONFIG=luna_s110 DUMP_MANIFEST_TAG=s110_luna \
  python3 evaluation/mini-src/build_dump.py
```

`build_dump.py` knobs: `EXTRACTS_DIR`, `DUMP_BE_DIR` (which backend dir of the
extracts tree to read), `DUMP_BE_TAG` (manifest backend column), `DUMP_CONFIG`
(slot name), `DUMP_MANIFEST_TAG` (`_manifest_<tag>.csv`), `DUMP_KNOW`
(`full`/`noknow`, manifest column only). It bails out without writing if the
extracts cell it was pointed at is empty.

Each run prints a `model-doc F1 vs gold` integrity figure. At time of writing:
terra_s110 **0.9385**, luna_s110 **0.8923** (15 cells each). The arm s110 replaced reads
terra_s92a **0.9136**, luna_s92a **0.8793**; the ~2.5pp gap on terra is a useful tell
that a slot was built from the wrong extracts.

---

## 2. RQ1 (link / file P/R/F1) and RQ2 (size-aware panel)

Both come out of `mini-src/`, which scores the sota dump against gold. No new
metric code — `metrics.py` is the sole implementation, pinned by `check.py`.

```bash
python3 evaluation/mini-src/rq12.py --csv $PWD/evaluation/reports/RQ12_BIGTABLE.csv
#   -> reports/RQ12_BIGTABLE.csv    (one row per system x run + average; superset of every RQ1/RQ2 cell)
#   -> reports/RQ12_PERPROJECT.csv  (per system x project, whole suite — feeds the per-project big table)
```

The roster is the two canonical approach arms plus the two baselines (Artemis,
TransArC/SWATTR), and two `Delta (approach - Artemis)` rows.

### Which CSV column feeds which paper table

| Paper table | CSV | Columns |
|-------------|-----|---------|
| body RQ1 (`tab:rq1`) | `RQ12_BIGTABLE.csv` | `doc_to_model_link_{precision,recall,f1,f2}`, `doc_to_code_file_{precision,recall,f1,f2}` |
| body RQ2 (`tab:rq2`) | `RQ12_BIGTABLE.csv` | `doc_to_model_link_{f1,f2}`, `doc_to_model_component_miss_rate`, `doc_to_code_file_{f1,f2}`, `doc_to_code_{worst,harmonic}_component_{f1,f2}` |
| appendix per-project / per-run | `RQ12_PERPROJECT.csv` / `RQ12_BIGTABLE.csv` | the whole suite |

---

## 3. RQ3 (judge contribution) and RQ4 (per-module ablation)

Both come out of `rq34.py`, which reads the run phase state directly (it needs the
candidate-vs-judged split and the per-linker provenance that the final link set
discards). Both write into `reports/rq34/<arm>/`, which is where `rq_tables.py` reads
the same arm — so neither takes a path argument.

```bash
python3 evaluation/mini-src/rq34.py
#   -> reports/rq34/s110/rq3_validators.csv, rq3_variants.csv
#   -> reports/rq34/s110/rq4_linkers.csv,    rq4_variants.csv, rq4_variants_perproject.csv
#   -> reports/rq34/s110/<backend>/<project>/{rq3,rq3_audit,rq4,rq4_upset}.csv
#   -> reports/rq34/s110/<backend>/runs_summary.csv

python3 evaluation/mini-src/rq34_rq2.py
#   -> reports/rq34/s110/rq34_rq2_{variants,linkers}.csv (+ _perproject)
#   -> reports/rq34/s110/RQ34_RQ2_INVESTIGATION.md
```

`rq34.py` cross-checks every Full-variant `tp/fp/fn` against the run's
`ablation_*.json` and prints `validate=OK` per backend.

Useful flags / env knobs:

```bash
python3 evaluation/mini-src/rq34.py --runs-from TMPL     # a different sweep ({model}, {i})
python3 evaluation/mini-src/rq34.py --backends terra     # one backend only
python3 evaluation/mini-src/rq34.py --run run1           # force a drill-down run
python3 evaluation/mini-src/rq34.py --no-validate        # skip the ablation-JSON cross-check
ALINKER_ARM=s92a python3 evaluation/mini-src/rq34.py     # a different arm, one knob
RQ34_ARM=s21 python3 evaluation/mini-src/rq34.py         # the retired two-judge layout
```

Each engine writes its whole report directory rather than merging into it, so a
single-backend or off-sweep run would leave the arm's reported numbers partly
overwritten. All four flags above therefore make `--csv-root` required, and the script
stops with the reason rather than writing. `$ALINKER_ARM` is the exception: it moves the
input *and* the output together, which is the point of the `ARMS` table.

---

## 4. The RQ4 no-knowledge row

Measured on this arm by `approach/pilot/run_consolidation_e2e_noknow.sh <terra|luna>`
(variant `s_linker110_noknow`, three five-project runs per model, live calls), then
scored with the same two engines pointed at that sweep. Naming a non-default sweep (or
a subset of the backends) makes `--csv-root` required, so this run cannot land on top of
the arm's reported numbers:

```bash
RUNS='consolidation_noknow_e2e_{model}_r{i}_20260902'
python3 evaluation/mini-src/rq34.py     --runs-from "$RUNS" --ablation-key s_linker110_noknow \
    --backends terra --csv-root evaluation/reports/rq34/s110_noknow
python3 evaluation/mini-src/rq34_rq2.py --runs-from "$RUNS" \
    --backends terra --csv-root evaluation/reports/rq34/s110_noknow
# luna goes to reports/rq34/s110_noknow_luna (RQ34_NOKNOW in rq_tables.py)
```

---

## 5. The RQ4 floor — the workflow against one linking call

`s_linker110_onecall` receives the document, the component list and the discovered
alias table and returns the final link set: no scan, no window, no evidence bundle, no
antecedent shortlist, no judge, no union. The head's four rubrics render verbatim, so
what the arm removes is the arrangement and not the guidance.

Runs: `approach/pilot/run_onecall_e2e.sh <terra|luna>` (three five-project runs a
model, live calls) -> `results/onecall_e2e_{terra,luna}_r{1,2,3}_20260902/`.

```bash
python3 evaluation/mini-src/rq4_floor.py
#   -> evaluation/reports/rq34/s110_floor/rq4_floor.csv
# rq_tables.py reshapes it into reports/tex_src/rq4_floor.csv -> tab:rq4-floor
```

`--backends terra` rewrites the whole file with that backend only, so pass it just for
a spot check and re-run bare before `rq_tables.py`.

The arm has **no `linker_*` phases**, so `rq34.py` cannot read it -- there are no stages
to attribute -- which is why this engine scores end to end off the predicted-link CSVs
instead. It re-derives no F-measure: the confusion matrix is `metrics.prf_counts`.

Two properties of these numbers must travel with them wherever they are quoted:

- **the control is CROSS-SET by decision.** The head runs come from
  `noevidence_e2e_*_20260902`, a different invocation from the arm's. `s_linker110` on
  terra read macro F1 93.85 in one 2026-09-02 set and 92.90 in another, so ~1 F1 of
  invocation drift sits on every delta.
- **the floor is not asked to quote.** The head demands every judge quote the sentence
  before ruling, worth 35.2 TP on its own (`results/s25_design_pilots/`). The floor
  removes that alongside the arrangement, so the deltas are **upper bounds** on what the
  arrangement is worth, not point estimates.

Measured 2026-09-02, mean of three runs, QUALITY-CHANGING on both backends (every p at
the n=3 floor): terra macro F1 **92.90 -> 84.39**, macro F2 94.45 -> 86.60; luna macro
F1 **89.35 -> 76.23**, macro F2 94.58 -> 83.10; LLM calls 73-75 -> 15. The loss is
**not monotone in document length** -- teastore (43 sentences) is the worst project on
both models while teammates (198) is milder -- so `s_linker27`'s length effect does not
explain it.

`rq_tables.py` picks the row up automatically per backend and prints a NOTE for
any backend whose no-knowledge slot is absent, so a missing row is never mistaken
for a measured zero.

Both backends are measured (2026-09-02, on s110): the alias table is worth **7.6pp
macro F1 / 11.9pp macro F2** on terra and **3.3 / 8.4pp** on luna, and **37.8pp of
worst-component F1** on terra. Flex tier returned `flex_unavailable` (429) partway
in, so the luna half ran with `OPENAI_SERVICE_TIER=default OPENAI_ENFORCE_FLEX=0`;
terra completed on flex. Both sweeps validate cell-for-cell against their own
`ablation_*.json`.

---

## 6. Paper tables: per-RQ CSVs → TeX

The paper's RQ floats are **generated**, not hand-typed. Two stdlib scripts sit on
top of the CSVs above:

```bash
# (a) reshape the wide CSVs into one small "this is the table" CSV per float
python3 evaluation/mini-src/rq_tables.py       # -> reports/tex_src/*.csv (12 files)

# (b) render each tex_src CSV into a booktabs .tex via the SPECS registry
python3 evaluation/mini-src/csv_to_tex.py      # -> reports/tex/*.tex (12 files)
```

`rq_tables.py` does NO metric math — it only selects rows/columns from the CSVs in
§2–§4 (it reads the no-knowledge `rq34_rq2_*` for the RQ4 "No knowledge" row, so
run §4 first). `csv_to_tex.py` is a declarative renderer: edit the `SPECS` list to
change columns, headers, precision, bolding, or captions. Re-running is
byte-identical.

| Paper float (label) | tex_src CSV | rendered .tex | grain |
|---------------------|-------------|---------------|-------|
| body RQ1 `tab:rq1` | `rq1.csv` | `rq1-results.tex` | terra, macro |
| body RQ2 `tab:rq2` | `rq2.csv` | `rq2-results.tex` | terra, macro size-aware |
| body RQ3 `tab:rq3-confusion` | `rq3.csv` | `rq3-confusion.tex` | terra, mean of 3 runs |
| body RQ4 `tab:rq4` | `rq4.csv` | `rq4-results.tex` | terra, macro |
| appendix `tab:rq3-runs` | `rq3_runs.csv` | `rq3-runs.tex` | both backends, per run + avg |
| appendix `tab:detailed-perproject` | `bigtable_rq12_perproject.csv` | `big-table-perproject.tex` | both backends, per project + Average |
| appendix `tab:detailed-perrun` | `bigtable_rq12_perrun.csv` | `big-table-perrun.tex` | both backends, per run + avg |
| appendix `tab:rq4-perproject` | `bigtable_rq4_perproject.csv` | `rq4-bigtable-perproject.tex` | both backends, per project + Average |
| appendix `tab:rq4-run{1,2,3}` / `tab:rq4-runavg` | `rq4_run{1,2,3}.csv`, `rq4_runavg.csv` | `rq4-run{1,2,3}.tex`, `rq4-runavg.tex` | both backends, per run |

**Sync into the paper.** `sync_paper.py` is the single bridge: it copies every
generated `.tex` and its `tex_src` companion `.csv` into the paper (the four body
tables to `table/`, the rest to `appendix/`) and refreshes
`gold_concentration.{tex,csv}` too. The file set is derived from
`csv_to_tex.SPECS`, so it tracks table adds/removes automatically.

```bash
PAPER_DIR=$PWD/paper python3 evaluation/mini-src/sync_paper.py            # copy
PAPER_DIR=$PWD/paper python3 evaluation/mini-src/sync_paper.py --check     # drift guard, exit 1 on drift
PAPER_DIR=$PWD/paper python3 evaluation/mini-src/sync_paper.py --only rq   # skip the gold pair
```

The copied files carry a `% GENERATED ... do not edit by hand` header; edit the
CSV specs and re-render instead. `--only gold` regenerates the OUT-02 inequality
pair and needs `$TRANSARC_BENCHMARK`; `../studies/mini-inequality/check_paper_table.py`
is a back-compat wrapper around that slice.

---

## 7. Verification

```bash
python3 evaluation/mini-src/check.py                      # frozen-golden regression on metrics.py -> PASS
python3 evaluation/mini-src/gen_csv_to_temp.py
PAPER_DIR=$PWD/paper python3 evaluation/mini-src/sync_paper.py --check
```

* `check.py` asserts `metrics.py` reproduces a frozen golden table to 1e-4, so any
  arithmetic drift in the RQ1/RQ2 numbers is caught. It also reads the `DEFAULT_ARM`
  literal out of all seven generators and fails if any two disagree.
* `gen_csv_to_temp.py` re-runs the four data generators (`rq12`, `rq34`, `rq34_rq2`,
  `rq4_floor`) with **bare defaults** into a scratch dir and byte-compares every
  produced file against the committed copy — RQ1/RQ2 against `reports/`, RQ3/RQ4
  against `reports/rq34/<arm>/` (override with `$RQ34_REPORTS`, which `rq_tables.py`
  reads too). Exit 0 = everything reproduces; the working tree is never written.
  Because it runs the engines bare, it is also the guard that a default output path
  still lands where `rq_tables.py` reads.
* RQ3/RQ4 additionally carry their own `validate=OK` cross-check inside `rq34.py`.

Bundled-TransArc headline (a quick smoke reference): sad-code file F1 .80 (F2 .78)
/ worst-comp F1 .54 (F2 .51) / harmonic F1 .67 (F2 .65) / cov .75; sad-sam link
F1 .80 (F2 .78) / cov .79 / CMR 7.1%. `check.py` needs `$TRANSARC_RESULTS_DIR`
pointed at `mini-data/`; without it every cell SKIPs and it now fails rather than
printing a vacuous PASS. Every F1 the suite reports has an F2 beside it, so a
panel showing one without the other is a stale generator.

---

## 8. Reproduce from a clone

Everything the paper reports rebuilds from a clone of this repo plus the public
ARDoCo benchmark tree — no dev-machine-only state. The recorded runs are
committed: the normalized link dumps (`sota-links/`), the neutral extracts
(`results/s110_extracts/`), and the per-phase state RQ3/RQ4 reads
(`results/consolidation_e2e_*/phase_states/`, `results/consolidation_noknow_e2e_*/`).

| Layer | Beyond the clone | Available? |
|-------|------------------|------------|
| Paper tables (`rq_tables` → `csv_to_tex`) | nothing (committed CSVs only) | ✅ |
| RQ1 / RQ2 (`rq12.py`) | committed `sota-links/` dump + benchmark | ✅ |
| RQ3 / RQ4 (`rq34`, `rq34_rq2`) | committed `results/*/phase_states/` + benchmark | ✅ |
| RQ4 no-knowledge row | committed `results/consolidation_noknow_e2e_*/` + benchmark | ✅ |
| RQ4 floor (`rq4_floor.py`) | committed `results/{onecall,noevidence}_e2e_*/` + benchmark | ✅ |
| Metric self-test (`check.py`) | benchmark (uses committed `mini-data/`) | ✅ |

```bash
git clone <this repo> alinker && cd alinker
export TRANSARC_BENCHMARK=/path/to/ardoco/core/tests-base/src/main/resources/benchmark

# every engine, then the three verifiers
python3 evaluation/mini-src/rq12.py
python3 evaluation/mini-src/rq34.py
python3 evaluation/mini-src/rq34_rq2.py
python3 evaluation/mini-src/rq4_floor.py
python3 evaluation/mini-src/rq_tables.py && python3 evaluation/mini-src/csv_to_tex.py
python3 evaluation/mini-src/check.py
python3 evaluation/mini-src/gen_csv_to_temp.py
PAPER_DIR=$PWD/paper python3 evaluation/mini-src/sync_paper.py --check
```

After running, `git status` is **clean** — every regenerated CSV and `.tex`
matches what is committed. `rq34.py` vendors the pickle classes (stdlib-only), so
no agent-linker install is needed to read the phase states.

The raw LLM logs and checkpoints under `results/` are recorded too, but no paper
number depends on them.
