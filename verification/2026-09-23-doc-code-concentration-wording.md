# Doc-code link-concentration wording audit, 2026-09-23

## Configuration

- Scope: active `paper/main.tex`, `paper/sections/*.tex`, and `paper/appendix/*.tex`; archived paper material, bibliography titles, historical study notes, and benchmark data were not edited.
- Fixed input: the five project gold standards under `benchmark/`, read through `evaluation/mini-inequality/inequality.py`; the paper CSV is `paper/table/gold_concentration.csv`.
- Paper comparison revision: `1d0253443ae9a266654dfd969f3e1f7b62b82ea3`.
- The wording change is intentional: "long-tailed" is replaced by measured link concentration or the named per-component metric. "Highly concentrated" is a descriptive interpretation of the reported shares, not a fitted distribution claim.

## Commands and text results

From the repository root, the fixed-gold check was:

```bash
python3 - <<'PY'
import csv
import sys
from pathlib import Path
sys.path.insert(0, 'evaluation/mini-inequality')
import inequality
rows = list(csv.DictReader(Path('paper/table/gold_concentration.csv').open()))
assert len(rows) == 5
for row in rows:
    measured = inequality.compute_sadcode_link_conc(row['project'].lower())
    assert float(row['doc_code_gini']) == round(measured['link_gini'], 3)
    assert float(row['doc_code_top3_pct']) == round(measured['link_top3_pct'], 1)
    assert int(row['doc_code_links']) == measured['links_total']
    print(f"PASS {row['project']}: {measured['comp_n']} gold-reachable components; top-three share {row['doc_code_top3_pct']}%; Gini {row['doc_code_gini']}")
print('PASS: five doc-code table rows agree with fixed gold inputs')
PY
```

```text
PASS MediaStore: 9 gold-reachable components; top-three share 74.6%; Gini 0.542
PASS TeaStore: 6 gold-reachable components; top-three share 86.8%; Gini 0.474
PASS Teammates: 7 gold-reachable components; top-three share 77.1%; Gini 0.479
PASS BigBlueButton: 10 gold-reachable components; top-three share 69.3%; Gini 0.532
PASS JabRef: 6 gold-reachable components; top-three share 99.3%; Gini 0.591
PASS: five doc-code table rows agree with fixed gold inputs
```

The wording and numerical-preservation check was run from `paper/`:

```bash
python3 - <<'PY'
from collections import Counter
from pathlib import Path
import re
import subprocess
base = '1d0253443ae9a266654dfd969f3e1f7b62b82ea3'
files = ['main.tex', 'appendix/big-table-perrun.tex', 'sections/conclusion.tex',
         'sections/discussion.tex', 'sections/eval.tex', 'sections/intro.tex',
         'sections/metric.tex', 'sections/motivation.tex', 'sections/results.tex']
for name in files:
    before = subprocess.check_output(['git', 'show', f'{base}:{name}'], text=True)
    after = Path(name).read_text()
    def numbers(text):
        prose = '\n'.join(line for line in text.splitlines()
                          if not line.lstrip().startswith('%'))
        return Counter(re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?', prose))
    assert numbers(before) == numbers(after), name
    print(f'PASS {name}: visible numeric tokens unchanged')
assert Path('table/gold_concentration.csv').read_bytes() == subprocess.check_output(
    ['git', 'show', f'{base}:table/gold_concentration.csv'])
print('PASS table/gold_concentration.csv: unchanged')
PY
git diff --check -- main.tex appendix/big-table-perrun.tex sections/conclusion.tex sections/discussion.tex sections/eval.tex sections/intro.tex sections/metric.tex sections/motivation.tex sections/results.tex
if rg -n -i 'long.tail|long tail|\btail\b' main.tex sections appendix --glob '*.tex'; then exit 1; else echo 'PASS: no long-tail terminology in active TeX'; fi
```

```text
PASS main.tex: visible numeric tokens unchanged
PASS appendix/big-table-perrun.tex: visible numeric tokens unchanged
PASS sections/conclusion.tex: visible numeric tokens unchanged
PASS sections/discussion.tex: visible numeric tokens unchanged
PASS sections/eval.tex: visible numeric tokens unchanged
PASS sections/intro.tex: visible numeric tokens unchanged
PASS sections/metric.tex: visible numeric tokens unchanged
PASS sections/motivation.tex: visible numeric tokens unchanged
PASS sections/results.tex: visible numeric tokens unchanged
PASS table/gold_concentration.csv: unchanged
PASS: no long-tail terminology in active TeX
```

`git diff --check` produced no output and exited 0. This was a text and fixed-input audit; no model runs or benchmark scores were changed.
