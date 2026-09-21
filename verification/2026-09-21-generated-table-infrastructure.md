# Generated-table infrastructure verification — 2026-09-21

## Scope and configuration

- Branch/base: `master` at `eee5c72b`
- Python: 3.13.13
- Evaluation arm: `s126` (the repository default)
- Paper directory: `$PWD/paper`
- Scope: generated-file headers and font size for all 13 generated floats;
  project abbreviations in the two per-project wide tables; generated/paper
  copy agreement.

## Regeneration

```bash
python3 evaluation/mini-src/csv_to_tex.py
PAPER_DIR="$PWD/paper" python3 evaluation/mini-src/sync_paper.py --only rq
PAPER_DIR="$PWD/paper" python3 evaluation/mini-src/sync_paper.py --only gold
```

Result:

```text
[csv2tex] 12 tables written under evaluation/reports/tex, 1 skipped (no source CSV for arm s126)
synced 24 file(s) into paper (2 absent for this arm)
synced 2 file(s) into paper
```

The skipped float is `rq4-floor.tex`; its source CSV is absent for `s126`, as
reported by the generator.

## Metric regression gate

```bash
python3 evaluation/mini-src/check.py
```

Result:

```text
OK    arm-default   every generator reports arm 's126' (7/7 found)
PASS: mini-src/metrics.py reproduces the frozen golden panel (10 cells, sad-code + sad-sam).
```

## Rendering invariants

```bash
python3 - <<'PY'
from pathlib import Path

root = Path('.')
gold = root / 'evaluation/mini-inequality/reports/out02_concentration.tex'
wide = [
    root / 'evaluation/reports/tex/big-table-perproject.tex',
    root / 'evaluation/reports/tex/rq4-bigtable-perproject.tex',
]
tex_outputs = sorted((root / 'evaluation/reports/tex').glob('*.tex')) + [gold]
assert len(tex_outputs) == 13, len(tex_outputs)
for path in tex_outputs:
    text = path.read_text()
    assert text.startswith('% GENERATED '), path
    assert '\\centering\\footnotesize' in text, path
    assert '\\centering\\scriptsize' not in text, path
for path in wide:
    text = path.read_text()
    for raw in ('mediastore', 'teastore', 'teammates', 'bigbluebutton', 'jabref'):
        assert f'& {raw} &' not in text, (path, raw)
    for short in ('MS', 'TS', 'TM', 'BBB', 'JR'):
        assert f'& {short} &' in text, (path, short)
print('PASS: 13 generated floats have the standard header and footnotesize')
print('PASS: both per-project wide tables use all five PROJECT_ABBR values')
PY
```

Result:

```text
PASS: 13 generated floats have the standard header and footnotesize
PASS: both per-project wide tables use all five PROJECT_ABBR values
```

## Paper drift and whitespace guards

```bash
PAPER_DIR="$PWD/paper" python3 evaluation/mini-src/sync_paper.py --check
git diff --check -- \
  evaluation/mini-inequality/motivation.py \
  evaluation/mini-inequality/reports/out02_concentration.tex \
  evaluation/mini-src/csv_to_tex.py \
  evaluation/reports/tex/big-table-perproject.tex \
  evaluation/reports/tex/rq4-bigtable-perproject.tex
```

Result:

```text
absent for this arm: rq4-floor.tex
absent for this arm: rq4_floor.csv
IN SYNC: all 26 paper file(s) match the generated output. (2 absent for this arm)
```

The scoped `git diff --check` exited 0 with no output.
