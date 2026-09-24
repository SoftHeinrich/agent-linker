# Compact inference-cost table, 2026-09-24

## Configuration

The `s126` cost engine reads three s126/GPT-5.6-terra runs and three
Artemis/GPT-5.6-luna runs per project. It retains the project-oriented
`inference_cost.csv` and writes a table-shaped
`inference_cost_by_system.csv`. The generated table has two rows, ArchLinker
and Artemis, and one input/output pair per project and Total column. Values are thousands
of tokens, rounded to one decimal for display. The run sets are unpaired.

## Commands and text results

Run from the repository root:

```text
$ python3 evaluation/mini-src/inference_cost.py
PASS: 30 project/run usage records; means of three runs written to .../evaluation/reports

$ python3 evaluation/mini-src/csv_to_tex.py
[csv2tex] wrote .../evaluation/reports/tex/inference-cost.tex
[csv2tex] 13 tables written under .../evaluation/reports/tex, 1 skipped (no source CSV for arm s126)

$ PAPER_DIR=$PWD/paper python3 evaluation/mini-src/sync_paper.py --only rq
absent for this arm: rq4-floor.tex
absent for this arm: rq4_floor.csv
synced 26 file(s) into .../paper (2 absent for this arm)

$ python3 verification/verify_inference_cost_compact.py
PASS cost: 2 system rows, 12 input/output tuples, 24 source values match

$ python3 verification/verify_rq1_task_rows.py
PASS RQ1: 12 single-line task rows; 36 system cells match source scores and two-decimal SD; metric headers, short caption, and cost placement verified

$ PAPER_DIR=$PWD/paper python3 evaluation/mini-src/sync_paper.py --only rq --check
absent for this arm: rq4-floor.tex
absent for this arm: rq4_floor.csv
IN SYNC: all 26 paper file(s) match the generated output. (2 absent for this arm)

$ git diff --check -- evaluation/mini-src/csv_to_tex.py evaluation/mini-src/inference_cost.py evaluation/reports/tex/inference-cost.tex evaluation/reports/tex_src/inference_cost_by_system.csv verification/verify_inference_cost_compact.py
(no output; exit 0)

$ git -C paper diff --check -- table/inference-cost.tex table/inference-cost.csv
(no output; exit 0)

$ bash scripts/build-paper.sh
latexmk is required to build the paper (install TeX Live with latexmk).
(exit 1)
```

The cost engine was also rerun with `--out` set to a temporary directory.
`INFERENCE_COST_PERRUN.csv`, `tex_src/inference_cost.csv`, and
`tex_src/inference_cost_by_system.csv` each matched the repository copy byte
for byte. The command and text result were:

```bash
python3 - <<'PY'
import subprocess
import sys
import tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as directory:
    out = Path(directory)
    run = subprocess.run(
        [sys.executable, 'evaluation/mini-src/inference_cost.py', '--out', str(out)],
        check=True, capture_output=True, text=True)
    print(run.stdout.strip())
    for relative in ('INFERENCE_COST_PERRUN.csv', 'tex_src/inference_cost.csv',
                     'tex_src/inference_cost_by_system.csv'):
        assert (out / relative).read_bytes() == (
            Path('evaluation/reports') / relative).read_bytes(), relative
print('PASS: all 3 inference-cost CSVs reproduce byte for byte')
PY
```

```text
PASS: 30 project/run usage records; means of three runs written to <temporary directory>
PASS: all 3 inference-cost CSVs reproduce byte for byte
```

The PDF layout remains unverified because `latexmk` is absent.
