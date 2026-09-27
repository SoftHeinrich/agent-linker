# Gold link concentration comparison, 2026-09-23

## Configuration

- Input: `benchmark/` in this checkout, resolved by `evaluation/mini-src/metrics.py` as `/mnt/hostshare/ardoco-home/agent-linker/benchmark`.
- Projects: MediaStore, TeaStore, Teammates, BigBlueButton, and JabRef.
- Scope: gold standards only; no system arm, model inference, or repeated runs.
- Grain: doc-model `(model element ID, sentence)` gold pairs from `load_gs_sad_sam`; doc-code enrolled `(sentence, file)` gold pairs grouped by the existing `compute_sadcode_link_conc` mapping. Each task's inequality uses its gold-reachable components. Doc-code links to shared files contribute to every mapped component's distribution; links to unmapped files remain in the distinct link total.
- Shared Components column: all component elements in the base PCM repositories, parsed from the checked-in XML files. The selection of base repositories matches the model inputs in `approach/run_ablation.py`. The counts are `14`, `11`, `8`, `12`, and `6`, matching [ArTEMiS Table 6](https://arxiv.org/pdf/2511.02434#page=19) for the revised benchmark.
- Dataset overview cells: sentence counts come from the checked-in architecture text; lines of code come from the checked-in `benchmark/<project>/README.md` cloc tables. `PRIMARY_LANGUAGES` records the existing per-project language selection; each selected cloc count is rounded to thousands before summing, matching the prior table values.
- Metrics: per-component median and maximum link count, Gini, and top-three share. The table source is `evaluation/mini-inequality/reports/out02_concentration.csv`.

## Commands and text results

Run from the repository root:

```text
$ python3 evaluation/mini-inequality/motivation.py
[motivation] seed=0 reports=/mnt/hostshare/ardoco-home/agent-linker/evaluation/mini-inequality/reports (OUT-02 table only)

$ python3 evaluation/mini-src/sync_paper.py --only gold paper
synced 2 file(s) into paper

$ python3 evaluation/mini-inequality/inequality.py --check-only
SANITY CHECK PASSED (tol: Gini<=0.005, counts exact)

$ python3 evaluation/mini-src/sync_paper.py --check --only gold paper
IN SYNC: all 2 paper file(s) match the generated output.

$ cmp -s evaluation/mini-inequality/reports/out02_concentration.csv paper/table/gold_concentration.csv
(no output; exit 0)

$ cmp -s evaluation/mini-inequality/reports/out02_concentration.tex paper/table/gold_concentration.tex
(no output; exit 0)

$ git diff --check
(no output; exit 0)

$ git -C paper diff --check
(no output; exit 0)
```

The inequality sanity gate checked the five existing sentence Gini values, the five SAM-code Gini values, SAM-code component counts, enrollment counts, maximums, factors, and the overall enrollment totals. Every row passed.

The cloc parser returned `4`, `12`, `145`, `159`, and `157` thousand lines for MediaStore, TeaStore, Teammates, BigBlueButton, and JabRef, respectively. These are the same values as the prior table. No displayed numeric cell is manually entered in the paper table.

An independent Python read of the five doc-model gold files recomputed Gini as the sum of all pairwise absolute differences divided by twice the number of gold-reachable components times the total link count. It also parsed base PCM repositories, compared each generated doc-code metric with the prior committed table, and counted the TeX columns. Reproduce that check from the repository root with:

```bash
python3 - <<'PY'
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from xml.etree import ElementTree

sys.path.insert(0, 'evaluation/mini-src')
import metrics

path = Path('evaluation/mini-inequality/reports/out02_concentration.csv')
rows = list(csv.DictReader(path.open()))
prior = list(csv.DictReader(subprocess.check_output(
    ['git', 'show', 'HEAD:evaluation/mini-inequality/reports/out02_concentration.csv'],
    text=True).splitlines()))
assert len(rows) == 5
repositories = {
    'mediastore': 'mediastore/model_2016/pcm/ms.repository',
    'teastore': 'teastore/model_2020/pcm/teastore.repository',
    'teammates': 'teammates/model_2021/pcm/teammates.repository',
    'bigbluebutton': 'bigbluebutton/model_2021/pcm/bbb.repository',
    'jabref': 'jabref/model_2021/pcm/jabref.repository',
}
published_counts = (14, 11, 8, 12, 6)
for project, old, published in zip(metrics.PROJECTS, prior, published_counts):
    row = next(r for r in rows if r['project'] == old['project'])
    for key in ('sentences', 'lines_of_code_thousands'):
        assert row[key] == old[key], (project, key)
    for key in ('links', 'median', 'max', 'gini', 'top3_pct'):
        assert row['doc_code_' + key] == old[key], (project, key)
    pcm = ElementTree.parse(Path('benchmark') / repositories[project]).getroot()
    components = [el for el in pcm.iter()
                  if el.tag.rsplit('}', 1)[-1] == 'components__Repository']
    assert int(row['components']) == len(components) == published
    counts = list(Counter(c for c, _ in metrics.load_gs_sad_sam(project)).values())
    total = sum(counts)
    gini = sum(abs(a-b) for a in counts for b in counts) / (2 * len(counts) * total)
    assert (int(row['doc_model_links']), float(row['doc_model_median']),
            int(row['doc_model_max']),
            row['doc_model_gini'], row['doc_model_top3_pct']) == (
            total, median(counts), max(counts), f'{gini:.3f}',
            f'{100 * sum(sorted(counts, reverse=True)[:3]) / total:.1f}')
    assert float(row['doc_model_gini']) < float(row['doc_code_gini'])
    print(f"PASS {old['project']}: {published} PCM components; model Gini {row['doc_model_gini']} < code Gini {row['doc_code_gini']}; code metrics unchanged")
tex = Path('paper/table/gold_concentration.tex').read_text().splitlines()
abbr = ('MS', 'TS', 'TM', 'BBB', 'JR')
data = [line for line in tex if line.startswith(tuple(
    f"{r['project']} ({short}) & " for r, short in zip(rows, abbr)))]
assert len(data) == 5 and all(line.count('&') == 13 for line in data)
assert any(line.count('\\textbf{Comp.}') == 1 and
           '\\multicolumn{5}{c}{\\textbf{Doc-model}}' in line and
           '\\multicolumn{5}{c}{\\textbf{Doc-code}}' in line for line in tex)
print('PASS 5 table rows and 14 columns per row; one Comp. column')
PY
```

The text result after the column-panel layout was:

```text
PASS MediaStore: 14 PCM components; model Gini 0.306 < code Gini 0.542; code metrics unchanged
PASS TeaStore: 11 PCM components; model Gini 0.179 < code Gini 0.474; code metrics unchanged
PASS Teammates: 8 PCM components; model Gini 0.261 < code Gini 0.479; code metrics unchanged
PASS BigBlueButton: 12 PCM components; model Gini 0.370 < code Gini 0.532; code metrics unchanged
PASS JabRef: 6 PCM components; model Gini 0.222 < code Gini 0.591; code metrics unchanged
PASS 5 table rows and 14 columns per row; one Comp. column
```

The checked paper table has one row per project and one shared Comp. column, with separate doc-model and doc-code link-statistic panels. The panels have separate rules and a space between them. Project labels include abbreviations for the result tables. The table uses the paper's existing `adjustbox` package to cap its width at `\textwidth` if needed.

No TeX compiler (`latexmk`, `pdflatex`, or `tectonic`) is available in this environment, so PDF rendering was not run. The generated TeX has five data rows with fourteen columns each, and the paper copy matches it byte for byte.
