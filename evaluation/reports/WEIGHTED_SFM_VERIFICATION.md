# Weighted-SFM verification

Date: 2026-08-26

SFM is computed as the percentage of gold `(sentence, component)` assignments
whose component is abandoned. A sentence linked to two components therefore
contributes one unit for each component assignment.

## Command and configuration

Run from the repository root:

```bash
export TRANSARC_BENCHMARK=/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark
export SOTA_LINKS="$PWD/sota-links"
export TRANSARC_RESULTS_DIR="$PWD/evaluation/mini-data"
python3 evaluation/mini-src/rq12.py --csv evaluation/reports/RQ12_BIGTABLE.csv
python3 evaluation/mini-src/rq_tables.py
python3 evaluation/mini-src/csv_to_tex.py
PAPER_DIR="$PWD/paper" python3 evaluation/mini-src/sync_paper.py --only rq
python3 evaluation/mini-src/check.py
```

## Result

The regenerated RQ2 macro SFM values are: approach (GPT-5.6-terra) `1.9355%`,
Artemis `4.6218%`, and TransArC `7.0968%`. The final regression-check result was:

```text
PASS: mini-src/metrics.py reproduces the frozen golden panel (sad-code + sad-sam).
```

## Paper build check

The attempted command was `cd paper && latexmk -pdf -interaction=nonstopmode
-halt-on-error main.tex`. It could not run because this environment has no TeX
engine installed (`/bin/bash: latexmk: command not found`; `pdflatex` and
`tectonic` are also unavailable). The generated TeX tables were nevertheless
regenerated and synchronized into `paper/` by the command above.
