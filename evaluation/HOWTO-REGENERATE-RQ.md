# Regenerating the RQ1–RQ4 results by hand

This is the manual, step-by-step recipe for rebuilding every number behind the
paper's research questions, from the raw agent-linker run outputs to the scored
CSVs and the `.tex` floats the paper reads. Everything here is **stdlib-only
Python 3** — no `pip install`, no `requirements.txt`.

The canonical arm is **`s_linker126`** on two GPT-5.6 backends:
**terra = paper body, luna = mirror**. Every engine below defaults to it. Its E2E sweep
is `results/greedymerge_e2e_{terra,luna}_r{1,2,3}_20260916v2`, which scored the
`s_linker123` in-set control in the same invocations. The normalized control slots are
named `s123gctl`; the arm's own slots are named `s126`.

**Two per-arm SHAPES, not just per-arm paths.** `rq34.py`'s `PHASE_SETS` gives this arm
**two** judges (`linker_name.pkl`, `linker_coreference.pkl`) where the pre-union arms have
three, so RQ3 has two rows; RQ4 prices the same two phases, one row per linker. (Until
2026-09-19 `FORM_SETS` split the name phase on the stage label each link carries, to keep
RQ4 at three proposal forms; the split was retired because this arm ships a single name
linker, so a standalone partial-name row prices a component the pipeline no longer has.) An arm with no one-call
floor sweep of its own -- `s126` has none, `s_linker126_onecall` was never built -- has
no floor table at all: `rq4_floor.py` refuses, `rq_tables.py` drops the CSV,
`csv_to_tex.py` skips the table and `sync_paper.py` deletes the previous arm's copy from
the paper. The floor's control is the arm itself, so it cannot be borrowed. The retired arms (`s_linker21`, `s_linker20_union`) were dropped
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
| doc-model / doc-code link dumps | `sota-links/{model-doc/aalinker,doc-code/aalinker-composed}/{terra,luna}_s126/run{1,2,3}/` |
| per-phase state (RQ3/RQ4) | `results/greedymerge_e2e_{terra,luna}_r{1,2,3}_20260916v2/phase_states/s_linker126/` |
| no-knowledge sweep | `results/greedymerge_noknow_e2e_{terra,luna}_r{1,2,3}_20260916v2/` |
| RQ4 floor sweep | none on this arm (s110's is `results/onecall_e2e_*_20260902/`) |
| RQ1/RQ2 output | `evaluation/reports/RQ12_{BIGTABLE,PERPROJECT}.csv` |
| RQ3/RQ4 output | `evaluation/reports/rq34/s126/` (+ `s126_noknow`, `s126_noknow_luna`) |
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
python3 evaluation/mini-src/build_alinker_extracts.py --variant s_linker126 \
    --out results/s126_extracts \
    --model terra results/greedymerge_e2e_terra_r{1,2,3}_20260916v2 \
    --model luna  results/greedymerge_e2e_luna_r{1,2,3}_20260916v2
EXTRACTS_DIR=$PWD/results/s126_extracts SOTA_LINKS=$PWD/sota-links \
  DUMP_CONFIG=terra_s126 DUMP_MANIFEST_TAG=s126_terra \
  python3 evaluation/mini-src/build_dump.py                       # terra_s126
EXTRACTS_DIR=$PWD/results/s126_extracts SOTA_LINKS=$PWD/sota-links DUMP_BE_DIR=luna \
  DUMP_BE_TAG=gpt-5.6-luna DUMP_CONFIG=luna_s126 DUMP_MANIFEST_TAG=s126_luna \
  python3 evaluation/mini-src/build_dump.py                       # luna_s126

# (b) RQ1 + RQ2
python3 evaluation/mini-src/rq12.py

# (c) RQ3 + RQ4  (bare = the reported arm; RQ34_ARM=s21 for the retired phase layout)
python3 evaluation/mini-src/rq34.py
python3 evaluation/mini-src/rq34_rq2.py

# (d) the RQ4 "No knowledge" row (see §4)

# (e) the RQ4 floor: NOT on this arm -- s_linker126_onecall (like s_linker120_onecall
#     before it) was never built, so rq4_floor.py refuses and rq_tables.py drops the
#     table (see §5)

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

The paper reports one arm (`s126`, promoted 2026-09-16; `s120` is the incumbent it
replaced). Every generator resolves its inputs from `$ALINKER_ARM` (default `s126`),
so a candidate is scored by setting one variable instead of editing paths in seven
files. The incumbent keeps the unsuffixed names; a candidate is written *beside* it,
never over it:

| | incumbent (`s126`) | candidate (e.g. `s130`) |
|---|---|---|
| dump slots | `sota-links/**/{terra,luna}_s126` | `…/{terra,luna}_s130` |
| RQ1/RQ2 CSVs | `reports/RQ12_BIGTABLE.csv` | `reports/RQ12_BIGTABLE_s130.csv` |
| RQ3/RQ4 reports | `reports/rq34/s126` | `reports/rq34/s130` |
| reshaped + rendered | `reports/tex_src`, `reports/tex` | `reports/tex_src_s130`, `reports/tex_s130` |

A candidate also needs a row in `rq34.py`'s `ARMS` table (its phase-state variant and
run sweep); without one the RQ3/RQ4 engines fall back to the default arm's runs.

Nothing is synced into the paper until the arm decision is made — `sync_paper.py`
always reads the incumbent directories.

```bash
# (1) extracts from the candidate's recorded E2E runs. s120's are the union round's
#     (they carry an in-set s_linker110 control -- see "the control" note below), so no
#     LLM calls are needed. A round whose runs were never recorded is the one case that
#     does cost calls; rq12.py --arm <arm> then stops with a "no dump slots" error
#     naming the four missing directories rather than scoring a partial set.
python3 evaluation/mini-src/build_alinker_extracts.py --variant s_linker120 \
    --out $PWD/results/s120_extracts \
    --model terra results/union_e2e_terra_r{1,2,3}_20260911 \
    --model luna  results/union_e2e_luna_r{1,2,3}_20260911

# (1b) THE CONTROL. The union runs scored s_linker110 in the SAME invocations, so build
#      that as its own arm and compare against it. Comparing s120 against the paper's
#      previous s110 numbers instead is cross-set: the two would differ by weeks of API
#      drift as well as by the arm, and on CMR that difference is larger than the arm's.
python3 evaluation/mini-src/build_alinker_extracts.py --variant s_linker110 \
    --out $PWD/results/s110ctl_extracts \
    --model terra results/union_e2e_terra_r{1,2,3}_20260911 \
    --model luna  results/union_e2e_luna_r{1,2,3}_20260911
# build_dump.py defaults its roots in-tree, so only the cell knobs are needed. Name
# both the config slot and the manifest tag, or the candidate overwrites the incumbent.
EXTRACTS_DIR=$PWD/results/s110ctl_extracts \
  DUMP_CONFIG=terra_s110ctl DUMP_MANIFEST_TAG=s110ctl_terra \
  python3 evaluation/mini-src/build_dump.py
EXTRACTS_DIR=$PWD/results/s110ctl_extracts DUMP_BE_DIR=luna \
  DUMP_BE_TAG=gpt-5.6-luna DUMP_CONFIG=luna_s110ctl DUMP_MANIFEST_TAG=s110ctl_luna \
  python3 evaluation/mini-src/build_dump.py

# (2) score both arms (no LLM calls)
python3 evaluation/mini-src/rq12.py                      # incumbent, unsuffixed
python3 evaluation/mini-src/rq12.py --arm s110ctl        # the in-set control, _s110ctl

# (3) the verdict: per-run deltas + sign agreement, not just the Average row
python3 studies/compare_arms.py s120 --base s110ctl \
    --csv evaluation/reports/ARM_COMPARE_s120_vs_inset.csv   # the read to trust
```

The worked example above (steps 1-3) is left as recorded for the s120-vs-s110
promotion, since it is still the concrete precedent for how a promotion's verdict is
computed. **The current promotion is s126-vs-s120**, run the same way but with
`--variant s_linker126 --out results/s126_extracts` (see the rebuild recipe above)
against the in-set `s123gctl` control (the arm the greedy-merge round actually paired
in its own invocations — not s120 itself, which would be cross-set); its numbers and
the override rationale are in `../approach/CLAUDE.md`'s status header, and the
comparison CSV is `evaluation/reports/ARM_COMPARE_s126_vs_s123gctl.csv`.

That is the promotion that happened on 2026-09-11, and its verdict is the one the
paper rested on before s126: on terra every one of the six moving metrics reads BETTER with 3/3 sign
agreement (doc-model \fone +2.25, \ftwo +1.96, doc-code \fone +2.21, \ftwo +1.12,
worst-component +4.46, harmonic +2.77); on luna both doc-code metrics read BETTER and
the rest are INSIDE NOISE with positive means. CMR is 0.0% for both arms on both
backends, so the union changes precision and the tail, not coverage.

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
set `DEFAULT_ARM` in all SEVEN modules that declare it -- `build_dump.py`, `rq12.py`,
`rq34.py`, `rq34_rq2.py`, `rq4_floor.py`, `rq_tables.py`, `csv_to_tex.py` (they are
asserted to agree by `check.py`, which is what stops a half-promotion), then re-run the
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
| `terra_s126`, `luna_s126` | GPT-5.6-terra / -luna | **`s_linker126` — canonical** | `build_dump.py` (all defaults) |
| `terra_s123gctl`, `luna_s123gctl` | GPT-5.6-terra / -luna | `s_linker123` scored **in-set**, off the greedy-merge runs | `build_dump.py` (env-overridden) |
| `terra_s120`, `luna_s120` | GPT-5.6-terra / -luna | `s_linker120` — the arm s126 replaced | `build_dump.py` (env-overridden) |
| `terra_s110`, `luna_s110` | GPT-5.6-terra / -luna | `s_linker110` — the arm s120 replaced, scored off its own consolidation runs | `build_dump.py` (env-overridden) |
| `terra_s110ctl`, `luna_s110ctl` | GPT-5.6-terra / -luna | `s_linker110` scored **in-set**, off the union runs — the honest base for the s120 promotion | `build_dump.py` (env-overridden) |
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

# (b) the canonical s126 slots, from the extracts built in the quick reference.
#     terra is every default, so it needs no env at all; luna names its own cell.
python3 evaluation/mini-src/build_dump.py
EXTRACTS_DIR=$PWD/results/s126_extracts DUMP_BE_DIR=luna DUMP_BE_TAG=gpt-5.6-luna \
  DUMP_CONFIG=luna_s126 DUMP_MANIFEST_TAG=s126_luna \
  python3 evaluation/mini-src/build_dump.py
```

`build_dump.py` knobs: `EXTRACTS_DIR`, `DUMP_BE_DIR` (which backend dir of the
extracts tree to read), `DUMP_BE_TAG` (manifest backend column), `DUMP_CONFIG`
(slot name), `DUMP_MANIFEST_TAG` (`_manifest_<tag>.csv`), `DUMP_KNOW`
(`full`/`noknow`, manifest column only). It bails out without writing if the
extracts cell it was pointed at is empty.

Each run prints a `model-doc F1 vs gold` integrity figure. At time of writing:
terra_s126 **0.9347**, luna_s126 **0.8953** (15 cells each). The in-set control reads
terra_s123gctl **0.9241** and luna_s123gctl **0.8881**; the gaps are a useful tell that a
slot was built from the wrong extracts.

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
| body RQ2 (`tab:rq2`) | `RQ12_PERPROJECT.csv` (+ `RQ12_BIGTABLE.csv` for the Average panel) | `doc_to_model_link_{f1,f2}`, `doc_to_model_component_miss_rate`, `doc_to_code_file_{f1,f2}`, `doc_to_code_{worst,harmonic}_component_{f1,f2}` |
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

Measured on this arm by `approach/pilot/run_s126_e2e_noknow.sh <terra|luna>`
(variant `s_linker126_noknow`, three five-project runs per model, live calls), then
scored with the same two engines pointed at that sweep. Naming a non-default sweep (or
a subset of the backends) makes `--csv-root` required, so this run cannot land on top of
the arm's reported numbers:

```bash
RUNS='greedymerge_noknow_e2e_{model}_r{i}_20260916v2'
python3 evaluation/mini-src/rq34.py     --runs-from "$RUNS" --ablation-key s_linker126_noknow \
    --backends terra --csv-root evaluation/reports/rq34/s126_noknow
python3 evaluation/mini-src/rq34_rq2.py --runs-from "$RUNS" \
    --backends terra --csv-root evaluation/reports/rq34/s126_noknow
# luna goes to reports/rq34/s126_noknow_luna (RQ34_NOKNOW in rq_tables.py)
```

---

## 5. The RQ4 floor — the workflow against one linking call

The reported `s126` arm has no one-call floor sweep, so no floor row is included in
the paper. The retained historical sweep belongs to `s110`: `s_linker110_onecall`
receives the document, the component list and the discovered
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

Both no-knowledge backends are measured on `s126`: the alias table is worth **6.7pp
macro F1 / 11.9pp macro F2** on terra and **5.4 / 11.0pp** on luna, and **33.2pp of
worst-component F1** on terra. Both sweeps validate cell-for-cell against their own
`ablation_*.json`.

---

## 6. Paper tables: per-RQ CSVs → TeX

The paper's RQ floats are **generated**, not hand-typed. Two stdlib scripts sit on
top of the CSVs above:

```bash
# (a) reshape the wide CSVs into one small "this is the table" CSV per float
python3 evaluation/mini-src/rq_tables.py       # -> reports/tex_src/*.csv (14 files)

# (b) render each tex_src CSV into a booktabs .tex via the SPECS registry
python3 evaluation/mini-src/csv_to_tex.py      # -> reports/tex/*.tex (13 files)
```

`rq_tables.py` does NO metric math — it only selects rows/columns from the CSVs in
§2–§4 (it reads the no-knowledge `rq34_rq2_*` for the RQ4 "No knowledge" row **and for
the whole knowledge-off half of the judges x knowledge grid**, so run §4 first; without
it `build_rq5` skips `rq5.csv` and the table is not rendered). `csv_to_tex.py` is a declarative renderer: edit the `SPECS` list to
change columns, headers, precision, bolding, or captions. Re-running is
byte-identical.

**One float is rendered outside the column registry.** The body RQ2 float prints
the same metric block twice across the page -- two projects side by side,
separated by a vertical rule, each project block introduced by a row naming both
of its projects -- so every header band repeats once per panel, which a list of
columns does not express. Its spec says `"render": "panels"` and carries one
panel's bands (`groups`, `subgroups`, `headers`) and `metrics` in place of
`cols`; `render_panels()` renders it and `check_specs()` checks those bands
against one panel's width. The pairing itself is data, not layout: `rq_tables.py`
writes `rq2.csv` with a `left_*`/`right_*` column pair per metric, so the CSV
stays row-for-row what the table prints. The spec still names the same
csv/out/label triple, so `sync_paper.py` and the missing-CSV skip treat it like
every other table.

**Two bolding rules, one per table orientation.** Both are computed at render
time from the CSV — no winner is ever written into a spec or a `.tex` by hand.
A table whose systems are the *rows* bolds down a column: `{"bold": "max"|"min"}`
on a column marks the best row for that metric (`extrema`). A table whose systems
are the *columns* — the body RQ1 float — bolds across a row instead, with
`"row_bold": "by_position"`: `position_groups()` reads the comparison groups off
the spec's own columns (every column prints the same metric tuple in the same
order, so position 0 is precision against precision, position 1 recall against
recall, …), and `row_winners()` takes the argmax per group per row. Add or drop a
system column and the groups follow; nothing restates the system or metric list.
`{"by": "position", "mode": "min"}` flips the direction, and an explicit list of
`{"fields": [...], "kind":, "mode":}` groups still works for columns that are not
positionally aligned.

The argmax runs on the rounded value the reader sees, so two systems differing
only below the last shown decimal are both bolded rather than one carrying an
invisible lead; genuine ties bold every tied cell. A spec that uses `row_bold`
also sets `"summary_bold_values": False`, so the Average row's numbers carry
winner bold only (its row *labels* stay bold). On import `check_specs()` resolves
`row_bold`, rejecting a ragged column set (`by_position`) or a group naming a
field the table does not print.

| Paper float (label) | tex_src CSV | rendered .tex | grain |
|---------------------|-------------|---------------|-------|
| body RQ1 `tab:rq1` | `rq1_transposed.csv` | `rq1-results.tex` | terra, per project + Average |
| body RQ2 `tab:rq2` | `rq2.csv` | `rq2-results.tex` | terra, per project in two panels + Average |
| body RQ3 `tab:rq3-confusion` | `rq3.csv` | `rq3-confusion.tex` | terra, mean of 3 runs |
| body RQ4 `tab:rq4` | `rq4.csv` | `rq4-results.tex` | terra, macro |
| body RQ4 `tab:judges-knowledge` | `rq5.csv` | `rq5-knowledge-judges.tex` | terra, judges x knowledge, mean of 3 runs |
| appendix `tab:rq3-runs` | `rq3_runs.csv` | `rq3-runs.tex` | both backends, per run + avg |
| appendix `tab:detailed-perproject` | `bigtable_rq12_perproject.csv` | `big-table-perproject.tex` | both backends, per project + Average |
| appendix `tab:detailed-perrun` | `bigtable_rq12_perrun.csv` | `big-table-perrun.tex` | both backends, per run + avg |
| appendix `tab:rq4-perproject` | `bigtable_rq4_perproject.csv` | `rq4-bigtable-perproject.tex` | both backends, per project + Average |
| appendix `tab:rq4-run{1,2,3}` / `tab:rq4-runavg` | `rq4_run{1,2,3}.csv`, `rq4_runavg.csv` | `rq4-run{1,2,3}.tex`, `rq4-runavg.tex` | both backends, per run |

**Sync into the paper.** `sync_paper.py` is the single bridge: it copies every
generated `.tex` and its `tex_src` companion `.csv` into the paper (the body tables
listed in `sync_paper.BODY` to `table/`, the rest to `appendix/`) and refreshes
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
