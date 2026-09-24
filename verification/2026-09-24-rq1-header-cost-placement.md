# RQ1 metric header and cost-table placement, 2026-09-24

## Configuration

The `s126` RQ1 table uses `rq1_transposed.csv` on GPT-5.6-terra. The generator
prints `Precision/Recall; F1/F2` under each system. Its caption is one sentence
of eight words. The inference-cost table retains its separate source CSV and
its unpaired terra/luna run scope. The RQ sync places its include and existing
explanation in the RQ1 subsection of `paper/sections/results.tex`.

## Commands and text results

Run from the repository root:

```text
$ python3 evaluation/mini-src/csv_to_tex.py
[csv2tex] wrote .../evaluation/reports/tex/rq1-results.tex
[csv2tex] wrote .../evaluation/reports/tex/inference-cost.tex
[csv2tex] 13 tables written under .../evaluation/reports/tex, 1 skipped (no source CSV for arm s126)

$ PAPER_DIR=$PWD/paper python3 evaluation/mini-src/sync_paper.py --only rq
absent for this arm: rq4-floor.tex
absent for this arm: rq4_floor.csv
placed cost table and explanation in RQ1: .../paper/sections/results.tex
synced 26 file(s) into .../paper (2 absent for this arm)

$ python3 verification/verify_rq1_task_rows.py
PASS RQ1: 12 single-line task rows; 36 system cells match source scores and two-decimal SD; metric headers, short caption, and cost placement verified

$ PAPER_DIR=$PWD/paper python3 evaluation/mini-src/sync_paper.py --only rq --check
absent for this arm: rq4-floor.tex
absent for this arm: rq4_floor.csv
IN SYNC: all 26 paper file(s) match the generated output. (2 absent for this arm)

$ python3 evaluation/mini-src/check.py
PASS: mini-src/metrics.py reproduces the frozen golden panel (10 cells, sad-code + sad-sam).

$ git diff --check -- evaluation/mini-src/csv_to_tex.py evaluation/mini-src/sync_paper.py evaluation/reports/tex/rq1-results.tex verification/verify_rq1_task_rows.py
(no output; exit 0)

$ git -C paper diff --check -- sections/results.tex table/rq1-results.tex
(no output; exit 0)

$ bash scripts/build-paper.sh
latexmk is required to build the paper (install TeX Live with latexmk).
(exit 1)
```

The placement rule was also exercised on the preceding committed Results
section (`paper` commit `89f9570`):

```bash
python3 - <<'PY'
import subprocess
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, 'evaluation/mini-src')
from sync_paper import sync_rq1_cost_placement
old = subprocess.check_output(
    ['git', '-C', 'paper', 'show', '89f9570:sections/results.tex'], text=True)
with tempfile.TemporaryDirectory() as directory:
    paper = Path(directory)
    section = paper / 'sections' / 'results.tex'
    section.parent.mkdir()
    section.write_text(old)
    assert sync_rq1_cost_placement(paper, True) == 1
    assert sync_rq1_cost_placement(paper, False) == 0
    assert sync_rq1_cost_placement(paper, True) == 0
    assert section.read_text() == Path('paper/sections/results.tex').read_text()
print('PASS cost placement: drift detected, block moved, and second check clean')
PY
```

Text result: the first check printed `DRIFT` with a diff moving the include
and three explanation sentences from Summary to RQ1; the sync printed
`placed cost table and explanation in RQ1`; the final line printed
`PASS cost placement: drift detected, block moved, and second check clean`.
PDF layout remains unverified because `latexmk` is absent.
