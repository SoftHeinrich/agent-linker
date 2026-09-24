# Results prose at displayed table precision — 2026-09-24

## Scope and configuration

The active Results section reports GPT-5.6-terra scores. The two LLM systems
use means of three runs. The generated RQ1–RQ4 tables display F-scores to two
decimal places, so percentage-point differences in the Results prose were
recomputed from those displayed values. CMR and judge counts retain their
displayed one-decimal precision. This is a presentation check, not a new model
run or a comparison between different invocation sets.

The paper subagent independently audited `paper/sections/results.tex` against
`paper/table/rq1-results.tex` through `rq4-results.tex`, the source CSVs, and
the committed causal-claims report. It confirmed the corrected project-count
claims and found three further inconsistencies: MediaStore `DB` and
`Reencoding` had exchanged gold-link shares; weighted CMR was described as a
component count; and the no-knowledge link-level loss was called negligible.
It also found that the named route's unique true-link count is a three-run mean
of 149.67, so the prose now says "about 150" per run. The RQ4 table does not
isolate either route without knowledge, so the corresponding causal assertion
was removed.

## Displayed-value arithmetic

Command (no raw CSV precision enters the differences):

```bash
python3 - <<'PY'
from decimal import Decimal
from pathlib import Path
import re

root = Path('paper/table')
def vals(file, prefix):
    row = next(line for line in (root / file).read_text().splitlines() if line.startswith(prefix))
    return [Decimal(x) for x in re.findall(r'(?<![A-Za-z0-9])(?:\d+\.\d+|\.\d+)', row)]
def gap(a, b):
    return int((a - b) * 100)
r1 = vals('rq1-results.tex', r'\textbf{Avg}')
r2a = vals('rq2-results.tex', 'AL & TM')
r2b = vals('rq2-results.tex', 'AT &  & .76')
r2c = vals('rq2-results.tex', 'S/T &  & .71')
r3f = vals('rq3-confusion.tex', 'Full &')
r3n = vals('rq3-confusion.tex', 'No judge &')
r4f = vals('rq4-results.tex', 'Full &')
r4n = vals('rq4-results.tex', 'No knowledge &')
r4name = vals('rq4-results.tex', r'\linkerN{} only &')
checks = {
 'RQ1 doc-model F1 vs Artemis': gap(r1[2], r1[6]),
 'RQ1 doc-model F2 vs Artemis': gap(r1[3], r1[7]),
 'RQ1 doc-code F1 vs Artemis': gap(r1[14], r1[18]),
 'RQ1 doc-code F2 vs Artemis': gap(r1[15], r1[19]),
 'RQ2 doc-code worst F1 vs Artemis': gap(r2a[14], r2b[14]),
 'RQ2 doc-code harmonic F1 vs Artemis': gap(r2a[16], r2b[16]),
 'RQ2 doc-code worst F1 vs TransArc': gap(r2a[14], r2c[14]),
 'RQ3 F1 judging gain': gap(r3f[6], r3n[6]),
 'RQ3 F2 judging gain': gap(r3f[7], r3n[7]),
 'RQ4 doc-model F1 knowledge gain': gap(r4f[2], r4n[2]),
 'RQ4 doc-model F2 knowledge gain': gap(r4f[3], r4n[3]),
 'RQ4 doc-code F1 knowledge gain': gap(r4f[7], r4n[7]),
 'RQ4 doc-code worst F1 knowledge gain': gap(r4f[9], r4n[9]),
 'RQ4 doc-model F1 implicit-route gain': gap(r4f[2], r4name[2]),
 'RQ4 doc-model F2 implicit-route gain': gap(r4f[3], r4name[3]),
}
for name, pp in checks.items():
    print(f'{name}: {pp:+d} pp')
assert list(checks.values()) == [12, 12, 7, 6, 30, 28, 26, 15, 6, 6, 12, 11, 33, 4, 7]
print('PASS: 15 displayed-table deltas match results prose')
PY
```

```text
RQ1 doc-model F1 vs Artemis: +12 pp
RQ1 doc-model F2 vs Artemis: +12 pp
RQ1 doc-code F1 vs Artemis: +7 pp
RQ1 doc-code F2 vs Artemis: +6 pp
RQ2 doc-code worst F1 vs Artemis: +30 pp
RQ2 doc-code harmonic F1 vs Artemis: +28 pp
RQ2 doc-code worst F1 vs TransArc: +26 pp
RQ3 F1 judging gain: +15 pp
RQ3 F2 judging gain: +6 pp
RQ4 doc-model F1 knowledge gain: +6 pp
RQ4 doc-model F2 knowledge gain: +12 pp
RQ4 doc-code F1 knowledge gain: +11 pp
RQ4 doc-code worst F1 knowledge gain: +33 pp
RQ4 doc-model F1 implicit-route gain: +4 pp
RQ4 doc-model F2 implicit-route gain: +7 pp
PASS: 15 displayed-table deltas match results prose
```

The other changed differences were checked directly against their displayed
rows: RQ1 MediaStore `0.95 - 0.93 = 2` pp and equal displayed F2, JabRef
`0.97 - 0.89 = 8` pp, TeaStore `1.00 - 0.72 = 28` pp; RQ2 TransArc
harmonic `0.91 - 0.68 = 23` pp; RQ3 one-judge removals `9/8` pp F1 and
`4/3` pp F2; RQ4 no-knowledge harmonic `0.91 - 0.52 = 39` pp and
named-route-only doc-code F1 `0.88 - 0.84 = 4` pp. MediaStore's gold doc-code
link shares are `DB` 47.5% and `Reencoding` 1.7%; TeaStore's
`ImageProvider` share is 45.3% (committed causal-claims report, lines 194–198).

## Checks and limits

```bash
python3 evaluation/mini-src/check.py
git -C paper diff --check -- sections/results.tex
git diff --check
./scripts/build-paper.sh
python3 scripts/check-paper-numeric-claims.py --self-test
```

Results:

```text
PASS: mini-src/metrics.py reproduces the frozen golden panel (10 cells, sad-code + sad-sam).
git -C paper diff --check: exit 0, no output
git diff --check: exit 0, no output
latexmk is required to build the paper (install TeX Live with latexmk).
PASS self-test: TeX comments, numeric tokens, citation stripping, and change detection
FAIL paper numeric-claim audit: 51 error(s); 211 active statements
```

The PDF build is blocked by missing `latexmk`. The manuscript-wide numeric
ledger's digest and line scopes already conflict with active Introduction,
Motivation, Discussion, and other prose. Results summary lines are also outside
its current scope. Its failure means those lines need a separate review before
the ledger can be refreshed; it does not contradict the table arithmetic above.
