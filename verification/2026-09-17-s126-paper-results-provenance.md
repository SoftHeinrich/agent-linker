# s126 paper-results and provenance verification — 2026-09-17

## Scope

- Paper result literals were checked against the generated `s126` RQ tables.
- The evaluation pipeline was checked with `s126` as the common default arm.
- RQ1–RQ4 reports were regenerated into a temporary directory and compared with the repository copies.
- The paper-table bridge was checked for drift.
- A PDF build was attempted.

## Commands and results

### Metric and arm-default gate

```bash
python3 evaluation/mini-src/check.py
```

Result:

```text
OK    arm-default   every generator reports arm 's126' (7/7 found)
PASS: mini-src/metrics.py reproduces the frozen golden panel (10 cells, sad-code + sad-sam).
```

### Full CSV regeneration

```bash
python3 evaluation/mini-src/gen_csv_to_temp.py
```

Result:

```text
[rq12] arm=s126
[rq34] runs-from = greedymerge_e2e_{model}_r{i}_20260916v2  (variant s_linker126)
[rq34-rq2] runs-from = greedymerge_e2e_{model}_r{i}_20260916v2  (variant s_linker126)
RESULT: all generated CSVs reproduce the committed repo copies. Repo untouched.
```

### Paper-table drift guard

```bash
PAPER_DIR=$PWD/paper python3 evaluation/mini-src/sync_paper.py --check
```

Result:

```text
absent for this arm: rq4-floor.tex
absent for this arm: rq4_floor.csv
IN SYNC: all 26 paper file(s) match the generated output. (2 absent for this arm)
```

### Paper prose arithmetic

A read-only Python assertion recomputed the prose replacements from
`paper/table/rq2-results.csv`, `paper/table/rq3-confusion.csv`, and
`paper/table/rq4-results.csv`.

Result:

```text
PASS: paper prose replacement values recompute from s126 RQ2/RQ3/RQ4 CSVs
PASS: headline deltas DM F1/F2=12.5/12.1pp; DC F1/F2=6.9/5.3pp
PASS: tail deltas vs Artemis=30.5/27.6pp; vs TransArC=25.8/23.6pp
PASS: combined judges=145.7 FP, 10.7 TP, -15.0/-5.8pp
PASS: terra knowledge contribution=6.7/11.9pp DM F1/F2
```

### Whitespace validation

```bash
git -C paper diff --check -- main.tex sections/intro.tex sections/results.tex sections/discussion.tex sections/conclusion.tex
git diff --check -- evaluation/HOWTO-REGENERATE-RQ.md evaluation/CLAUDE.md evaluation/mini-src/build_dump.py evaluation/mini-src/csv_to_tex.py evaluation/mini-src/rq12.py evaluation/mini-src/rq34.py evaluation/mini-src/rq34_rq2.py evaluation/mini-src/rq4_floor.py evaluation/mini-src/rq_tables.py
```

Result: both commands exited 0 with no output.

### PDF build

```bash
./scripts/build-paper.sh
```

Result:

```text
latexmk is required to build the paper (install TeX Live with latexmk).
```

The container has no `latexmk`, `pdflatex`, `lualatex`, `xelatex`, or `tectonic`, so
the PDF build could not run. The deterministic metric, regeneration, arithmetic, and
paper-sync checks above completed successfully.

### s126 implementation checks

```bash
cd approach
python3 pilot/test_s126.py
```

Result:

```text
ModuleNotFoundError: No module named 'lxml'
```

The repository virtual environment points to the unavailable interpreter
`/home/yu/miniconda3/bin/python3`, while the available interpreter does not contain
`lxml`. As a dependency-independent fallback, syntax and launcher checks passed:

```bash
python3 -m py_compile src/llm_sad_sam/linkers/experimental/s_linker126.py run_ablation.py pilot/test_s126.py pilot/test_s126_standalone.py
bash -n pilot/run_s126_e2e_noknow.sh
```

## Paper prose corrections

The live prose was corrected to match the regenerated `s126` evidence:

- Terra run 1 has doc-model CMR `1.9355%`, while runs 2 and 3 have `0%`;
  therefore the paper now reports one abandoned component in one of three runs.
- The aggregate precision and recall claims are kept at the aggregate grain.
- Per-project claims now state that doc-model F1 leads on five of five projects,
  doc-code F1 leads on four of five, and both doc-model F-measures improve on
  four of five.

Verification commands:

```bash
python3 evaluation/mini-src/check.py
python3 evaluation/mini-src/gen_csv_to_temp.py
PAPER_DIR="$PWD/paper" python3 evaluation/mini-src/sync_paper.py --check
git -C paper diff --check -- sections/intro.tex sections/results.tex
```

Results:

```text
OK    arm-default   every generator reports arm 's126' (7/7 found)
PASS: mini-src/metrics.py reproduces the frozen golden panel (10 cells, sad-code + sad-sam).
RESULT: all generated CSVs reproduce the committed repo copies. Repo untouched.
IN SYNC: all 26 paper file(s) match the generated output. (2 absent for this arm)
git diff --check: exit 0, no output
```
