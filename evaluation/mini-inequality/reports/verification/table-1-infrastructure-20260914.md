# Table 1 infrastructure verification

Date: 2026-09-14

## Commands

```bash
python3 evaluation/mini-inequality/motivation.py
python3 evaluation/mini-src/sync_paper.py --only gold /mnt/hostshare/ardoco-home/agent-linker/paper
python3 evaluation/mini-src/sync_paper.py --check --only gold /mnt/hostshare/ardoco-home/agent-linker/paper
python3 - <<'PY'
from pathlib import Path
tex = Path('paper/table/gold_concentration.tex').read_text()
csv = Path('paper/table/gold_concentration.csv').read_text()
assert '\\textbf{Sentences}' in tex
assert '\\textbf{Components}' in tex
assert '\\textbf{Top-3 (\\%)}' in tex
assert '\\textbf{\\makecell[c]{Lines of code\\\\(thousands)}}' in tex
for forbidden in ('\\textbf{Sent.}', '\\textbf{Comp.}', '\\textbf{$K$}', '\\textbf{Med}', '\\textbf{Max}', '\\textbf{Top-3\\%}'):
    assert forbidden not in tex, forbidden
assert 'MediaStore & 37 & 9 & 4' in tex
assert 'Teammates & 198 & 7 & 145' in tex
assert 'BigBlueButton & 87 & 10 & 159' in tex
assert csv.startswith('project,sentences,components,lines_of_code_thousands,links,median,max,gini,top3_pct\n')
print('PASS: Table 1 has unabbreviated headers, K-based Components, Top-3 (%), and summed published lines-of-code totals.')
PY
git diff --check -- evaluation/mini-inequality/motivation.py evaluation/mini-inequality/reports/out02_concentration.csv evaluation/mini-inequality/reports/out02_concentration.tex paper/table/gold_concentration.csv paper/table/gold_concentration.tex evaluation/mini-inequality/reports/verification/table-1-infrastructure-20260914.md
```

## Result

```text
[motivation] seed=0 reports=/mnt/hostshare/ardoco-home/agent-linker/evaluation/mini-inequality/reports (OUT-02 table only)
synced 2 file(s) into /mnt/hostshare/ardoco-home/agent-linker/paper
IN SYNC: all 2 paper file(s) match the generated output.
PASS: Table 1 has unabbreviated headers, K-based Components, Top-3 (%), and summed published lines-of-code totals.
```

The scoped `git diff --check` completed with no output and exit status 0.

## Lines-of-code provenance

The original benchmark paper used by TransArC and ArTEMiS reports rounded kLOC
for the primary programming languages in its Table 1. This table sums those
published values: MediaStore 4; TeaStore 12; TEAMMATES 145; BigBlueButton 159;
and JabRef 157 thousand lines of code. Source: Fuchß et al., *Establishing a
Benchmark Dataset for Traceability Link Recovery Between Software Architecture
Documentation and Models* (ECSA 2022), Table 1,
https://publikationen.bibliothek.kit.edu/1000160708/151135839.

## PDF build status

```bash
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The PDF build could not run because no TeX engine is installed in this environment:
`latexmk`, `pdflatex`, `xelatex`, `lualatex`, and `tectonic` were all unavailable.
