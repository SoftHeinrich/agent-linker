# Appendix Table 7 split verification

Date: 2026-09-30. Configuration: default paper arm `s126`, terra and luna,
three runs per stochastic system, five projects. Released Artemis GPT-5.4
and deterministic TransArC retain their single-run reference rows.

The former 70-row combined float is now Table 7 (approach, 30 rows) and
Table 8 (Artemis plus reference results, 40 rows). The reshape layer writes
one CSV per table; the renderer registry feeds both TeX/CSV pairs to the
paper sync. The pipeline removes the retired combined artifacts. No metric
computation or numerical values changed.

Verification from the `agent-linker` root:

```sh
python3 - <<'PYCODE'
import csv, sys, tempfile
from collections import Counter
from pathlib import Path
sys.path.insert(0, 'evaluation/mini-src')
import rq_tables as rq, csv_to_tex as tex, sync_paper as sync
source = list(csv.DictReader(rq.RQ12_PERPROJECT_PERRUN.open()))
keys = {(system, run, project)
        for system, _, runs in rq.PERRUN_SYSTEMS
        for run in runs for project in rq.PROJECTS}
fields = ['system', 'run', 'project'] + rq.SUITE_COLS
expected = [tuple(r[f] for f in fields) for r in source
            if (r['system'], r['run'], r['project']) in keys]
actual = []
for system, count in [('approach', 30), ('artemis', 40)]:
    rows = list(csv.DictReader((rq.TEX_SRC / f'bigtable_rq12_{system}.csv').open()))
    assert len(rows) == count
    actual.extend(tuple(r[f] for f in fields) for r in rows)
assert len(actual) == 70 and Counter(actual) == Counter(expected)
print('PASS: 70 rows and all source metric cells preserved without duplication')
for gen, subdir, name in sync.rq_pairs():
    if name.startswith(('big-table-approach.', 'big-table-artemis.')):
        assert gen.read_bytes() == (Path('paper') / subdir / name).read_bytes()
assert sync.remove_retired_rq_tables(Path('paper'), True) == 0
print('PASS: both TeX/CSV pairs in sync; retired combined artifacts absent')
original_tex = tex.TEX_OUT
with tempfile.TemporaryDirectory() as tmp:
    rq.TEX_SRC = Path(tmp) / 'csv'
    rq.main()
    tex.TEX_SRC = rq.TEX_SRC
    tex.TEX_OUT = Path(tmp) / 'tex'
    tex.main()
    for system in ('approach', 'artemis'):
        name = f'big-table-{system}.tex'
        assert (tex.TEX_OUT / name).read_bytes() == (original_tex / name).read_bytes()
print('PASS: full reshape/render reproduces both tables byte-for-byte')
PYCODE

mkdir -p /tmp/appendix-split-build
/tmp/rq2-tectonic.grZsvV/tectonic -k --keep-logs \
  -o /tmp/appendix-split-build paper/main.tex \
  > /tmp/appendix-split-build/build.txt 2>&1
```

Results: all assertions PASS; Tectonic exit 0. Auxiliary labels place Table 7
on page 23 and Table 8 on page 24. Both pages were rendered with PyMuPDF and
visually inspected: all rows, captions, and the reference footnote fit within
the page. No overflow warning names either new table. Existing warnings in
other paper sections and RQ4 tables remain outside this change.

The paper's existing edits were preserved. Only the two new TeX/CSV pairs were
copied through `sync_paper.rq_pairs()` during this change, avoiding replacement
of independently edited body tables. Routine regeneration uses `rq_tables.py`,
`csv_to_tex.py`, then `sync_paper.py --only rq paper` as documented.
