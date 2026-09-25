# Cost prose update, 2026-09-24

## Scope and data

The RQ1 text uses `paper/table/inference-cost.csv`.
Its figures are means of three runs per project.
The approach uses s126 with GPT-5.6-terra, including replacement MediaStore runs.
Artemis uses three separate GPT-5.6-luna runs with token logging.
The run sets are unpaired and use different models and dates.

## Commands and text results

Run from the repository root:

```text
$ python3 verification/verify_inference_cost_compact.py
PASS cost: 2 system rows, 12 input/output tuples, 24 source values match

$ python3 verification/verify_rq1_task_rows.py
PASS RQ1: 12 single-line task rows; 36 system cells match source scores and two-decimal SD; metric headers, short caption, and cost placement verified

$ PAPER_DIR=$PWD/paper python3 evaluation/mini-src/sync_paper.py --only rq --check
absent for this arm: rq4-floor.tex
absent for this arm: rq4_floor.csv
IN SYNC: all 26 paper file(s) match the generated output. (2 absent for this arm)

$ git -C paper diff --check -- sections/eval.tex sections/results.tex
(no output; exit 0)

$ bash scripts/build-paper.sh
latexmk is required to build the paper (install TeX Live with latexmk).
(exit 1)
```

The PDF build could not complete because `latexmk` is unavailable.
Thus, the final page layout remains unverified.

The prose and numeric checks used this command:

```bash
python3 - <<'PY'
import csv
import re
import subprocess
from pathlib import Path

patch = subprocess.check_output(['git', '-C', 'paper', 'diff', '--unified=0', '--', 'sections/eval.tex', 'sections/results.tex'], text=True)
lines = [line[1:].strip() for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++')]
prose = [line for line in lines if line and not line.startswith(('%', '\\'))]
prose += [line for line in lines if line.startswith('\\autoref')]
for line in prose:
    for sentence in re.split(r'(?<=[.!?])\s+', line):
        count = len(sentence.split())
        assert count < 15, (count, sentence)
with Path('paper/table/inference-cost.csv').open() as source:
    rows = {row['system']: row for row in csv.DictReader(source)}
results = Path('paper/sections/results.tex').read_text()
for system, numbers in {'approach': ('98.6', '22.7'), 'Artemis': ('20.0', '23.4')}.items():
    for key, display in zip(('Total_input_k', 'Total_output_k'), numbers):
        assert f"{float(rows[system][key]):.1f}" == display
        assert display in results
print(f'PASS prose: {len(prose)} changed sentences contain fewer than 15 words')
print('PASS totals: all four RQ1 token figures match the committed cost CSV')
PY
```

Text result:

```text
PASS prose: 13 changed sentences contain fewer than 15 words
PASS totals: all four RQ1 token figures match the committed cost CSV
```
