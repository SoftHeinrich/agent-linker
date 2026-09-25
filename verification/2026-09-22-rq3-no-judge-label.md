# RQ3 no-judge row label

Date: 2026-09-22. Configuration: `s126`; body backend `terra`; appendix backends
`terra` and `luna`, each with runs 1–3 and the average. This is a table-label
change. The engine audit row remains `none`, and the metric variant remains
`NoValidator`.

The `rq_tables.py` row key for the all-off configuration is now `no_judge`, and
`csv_to_tex.py` renders it as `No judge`. The body and appendix CSVs and TeX were
regenerated and copied into the paper repository.

## Verification

From the repository root:

```text
$ python3 verification/2026-09-21-rq3-real-metrics-check.py
[OK] rq3.csv: No judge is the last row of each block
[OK] rq3_runs.csv: No judge is the last row of each block
[OK] rq3-confusion.tex: bands [2, 2, 2, 3] = 9 cols + 1 label(s) == 10 header cells
[OK] rq3-runs.tex: bands [2, 2, 2, 3] = 9 cols + 3 label(s) == 12 header cells
[OK] body Full                     145.7 10.7 184.0 24.0 .91/.96\,;\,.93/.95 0.6 .86/.91\,;\,.88/.90 .77/.78 .91/.91
[OK] body \nameValidator{}         100.0 9.7 169.0 22.0 .75/.99\,;\,.84/.91 0.6 .76/.95\,;\,.84/.90 .56/.70 .82/.90
[OK] body \corefValidator{}        53.3 2.3 34.3 2.0 .77/.97\,;\,.85/.92 0.0 .78/.92\,;\,.83/.88 .68/.77 .86/.90
[OK] body No judge                 0.0 0.0 194.7 166.3 .67/1.00\,;\,.78/.89 0.0 .71/.96\,;\,.80/.88 .55/.70 .78/.88
[OK] appendix: 32 rows checked against the engine CSVs
[OK] no-judge row: 0 rejects and kept_tp == the candidate pool, in every backend/run block

RESULT: PASS
```

The checker reads the engine CSVs directly, checks every printed metric cell,
and confirms the all-off row's zero rejections and candidate-pool true positives.
The paper copies were checked byte for byte with:

```text
$ python3 - <<'PY'
from pathlib import Path
pairs = [
    ('evaluation/reports/tex_src/rq3.csv', 'paper/table/rq3-confusion.csv'),
    ('evaluation/reports/tex_src/rq3_runs.csv', 'paper/appendix/rq3-runs.csv'),
    ('evaluation/reports/tex/rq3-confusion.tex', 'paper/table/rq3-confusion.tex'),
    ('evaluation/reports/tex/rq3-runs.tex', 'paper/appendix/rq3-runs.tex'),
]
for source, copy in pairs:
    assert Path(source).read_bytes() == Path(copy).read_bytes(), f'drift: {copy}'
    print(f'OK {copy}')
PY
OK paper/table/rq3-confusion.csv
OK paper/appendix/rq3-runs.csv
OK paper/table/rq3-confusion.tex
OK paper/appendix/rq3-runs.tex
```

`git diff --check` on the changed evaluation files passed with no output.
