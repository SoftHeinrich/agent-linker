# Paper caption synchronization

The generator uses the exact captions currently authored in the paper for RQ1,
inference cost, RQ3, and RQ4, including punctuation. Verification renders the
four tables from the existing s126 CSVs into a temporary directory, checks
caption equality with the paper and checks that all other generated content
is unchanged, then updates the generated copies.

Scope: caption synchronization only. The paper cost table's separately edited
price column and footnote removal are not implemented by this change; a full
paper sync would still overwrite those edits. No full paper sync was run.

## Command

Run from the repository root with Python 3 and default `ALINKER_ARM=s126`:

```bash
python3 - <<'CHECK'
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, 'evaluation/mini-src')
import csv_to_tex as c
names = {'rq1-results.tex', 'inference-cost.tex', 'rq3-confusion.tex', 'rq4-results.tex'}
def caption(text):
    return next(line for line in text.splitlines() if line.startswith('\\caption{'))
def without_caption(text):
    return '\n'.join(line for line in text.splitlines() if not line.startswith('\\caption{'))
original_out = c.TEX_OUT
with tempfile.TemporaryDirectory() as directory:
    c.TEX_OUT = Path(directory)
    for spec in c.SPECS:
        if spec['out'] not in names:
            continue
        c.render_panels(spec) if spec.get('render') == 'panels' else c.render(spec)
        rendered = (c.TEX_OUT / spec['out']).read_text()
        paper = (Path('paper/table') / spec['out']).read_text()
        old = (original_out / spec['out']).read_text()
        assert caption(rendered) == caption(paper), spec['out']
        assert without_caption(rendered) == without_caption(old), spec['out']
        (original_out / spec['out']).write_text(rendered)
        print('PASS:', spec['out'], 'matches paper caption; non-caption output unchanged')
print('PASS: four captions verified using existing s126 table CSVs; no metric recomputation')
CHECK
```

## Text results

```text
[csv2tex] wrote /tmp/tmphzjqt_t9/rq1-results.tex
PASS: rq1-results.tex matches paper caption; non-caption output unchanged
[csv2tex] wrote /tmp/tmphzjqt_t9/inference-cost.tex
PASS: inference-cost.tex matches paper caption; non-caption output unchanged
[csv2tex] wrote /tmp/tmphzjqt_t9/rq3-confusion.tex
PASS: rq3-confusion.tex matches paper caption; non-caption output unchanged
[csv2tex] wrote /tmp/tmphzjqt_t9/rq4-results.tex
PASS: rq4-results.tex matches paper caption; non-caption output unchanged
PASS: four captions verified using existing s126 table CSVs; no metric recomputation
Exit status: 0
```
