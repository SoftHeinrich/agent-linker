# Architecture-driven suite terminology

Replaced the former suite name and its metric/view references throughout active paper TeX sources, including the RQ2 caption. Updated the table generator and generated RQ2 report to preserve the naming on regeneration. Historical archives, notes, and prior verification evidence retain their original wording.

Configuration: repository root, Python 3, default `ALINKER_ARM=s126`, existing table CSV inputs. This is a wording-only check; no model runs or metric recomputation.

## Initial check (superseded by the refined terminology check below)

```bash
python3 - <<'CHECK'
from pathlib import Path
import re
import sys
import tempfile
sys.path.insert(0, 'evaluation/mini-src')
import csv_to_tex as c
paths = [p for p in Path('paper').rglob('*.tex')
         if not {'archive', 'notes', 'verification'} & set(p.relative_to('paper').parts)]
remaining = [str(p) for p in paths if re.search(r'size[-\s]+aware', p.read_text(), re.I)]
assert not remaining, remaining
print(f'PASS: no former suite terminology in {len(paths)} active paper TeX files')
spec = next(s for s in c.SPECS if s['out'] == 'rq2-results.tex')
with tempfile.TemporaryDirectory() as directory:
    c.TEX_OUT = Path(directory)
    c.render_panels(spec) if spec.get('render') == 'panels' else c.render(spec)
    rendered = (c.TEX_OUT / spec['out']).read_text()
    for target in [Path('paper/table') / spec['out'], Path('evaluation/reports/tex') / spec['out']]:
        assert rendered == target.read_text(), target
        print(f'PASS: regenerated RQ2 table exactly matches {target}')
print('PASS: existing s126 CSV inputs; no metric recomputation')
CHECK
git diff --check -- evaluation/mini-src/csv_to_tex.py evaluation/reports/tex/rq2-results.tex
git -C paper diff --check
```

## Text results

```text
PASS: no former suite terminology in 29 active paper TeX files
[csv2tex] wrote /tmp/tmp3wymktmk/rq2-results.tex
PASS: regenerated RQ2 table exactly matches paper/table/rq2-results.tex
PASS: regenerated RQ2 table exactly matches evaluation/reports/tex/rq2-results.tex
PASS: existing s126 CSV inputs; no metric recomputation
Exit status: 0
```

The full paper `git diff --check` exited 2 due to existing whitespace in the independently modified `figures/approach-overview.pdf` and `sections/results.tex:213` (`The component-level scores move with it. `). These lines were not changed by this terminology edit. This does not block the terminology or regeneration checks. Scoped checks follow:

```text
$ git diff --check -- evaluation/mini-src/csv_to_tex.py evaluation/reports/tex/rq2-results.tex
Exit status: 0
$ git -C paper diff --check -- main.tex sections/intro.tex sections/discussion.tex sections/eval.tex sections/motivation.tex sections/conclusion.tex table/rq2-results.tex
Exit status: 0
```

## Final semantic audit and refined scope

The user's clarification distinguishes the suite name from descriptive properties: use architecture-driven for the suite, retain size-aware for gains and the contribution bullet's description of its metrics, and explain size-aware evaluation in the metrics section. The initial zero-occurrence check above records an intermediate state and is superseded by this check.

Reviewed every active occurrence in the inventory below against the suite definitions in `paper/sections/metric.tex`, the RQ2 and RQ4 table headers, and component aggregation in `evaluation/mini-src/metrics.py`.

- Suite names, headings, captions, and references to the suite's measures consistently use architecture-driven.
- The contribution bullet retains three size-aware metrics as a property, and the empirical-study bullet retains size-aware gains.
- The metrics section explains architecture-driven evaluation through architecture-model components and size-aware evaluation through differences in component link counts.
- RQ3 identifies the doc-model measure; RQ4 refers to task-specific measures in the plural.
- Doc-code aggregation is described as minimum and harmonic mean over per-component scores. This does not imply that doc-model CMR discards its sentence weights.
- The diagnostic passage distinguishes aggregate warnings from identifying components through underlying results.
- Abstract, conclusion, research question, motivation, remaining result/discussion references, and appendix heading refer to the defined suite or its measures. No numerical results or formulas were changed by these terminology edits. This audit does not revalidate all numerical claims elsewhere in the paper.

Configuration: Python 3, repository root, default `ALINKER_ARM=s126`, existing table CSVs; no model runs or metric recomputation. Historical archives and review evidence remain unchanged.

### Reproducible command

```bash
python3 - <<'CHECK'
from pathlib import Path
import re
import sys
import tempfile
sys.path.insert(0, 'evaluation/mini-src')
import csv_to_tex as c
paths = sorted(p for p in Path('paper').rglob('*.tex')
               if not {'archive', 'notes', 'verification'} & set(p.relative_to('paper').parts))
expected = {
    'paper/sections/intro.tex': ['of three size-aware metrics', 'with larger size-aware gains.'],
    'paper/sections/metric.tex': ['The suite also provides size-aware evaluation']}
for p in paths:
    text = p.read_text()
    assert not re.search(r'size[-\s]+aware\s+(?:evaluation\s+|metric\s+)?suite', text, re.I), p
    assert not re.search(r'architecture-driven\s+gains', text, re.I), p
    kept = [line for line in text.splitlines() if re.search(r'size[-\s]+aware', line, re.I)]
    wanted = expected.get(str(p), [])
    assert len(kept) == len(wanted), (p, kept)
    for phrase in wanted:
        assert any(phrase in line for line in kept), (p, phrase)
print(f'PASS: suite references audited across {len(paths)} active TeX files')
print('PASS: size-aware retained for gains and metric properties, including the new explanation')
print('Terminology inventory for semantic review:')
for p in paths:
    for number, line in enumerate(p.read_text().splitlines(), 1):
        if re.search(r'architecture-driven|size-aware', line, re.I):
            print(f'{p}:{number}: {line}')
spec = next(s for s in c.SPECS if s['out'] == 'rq2-results.tex')
with tempfile.TemporaryDirectory() as directory:
    c.TEX_OUT = Path(directory)
    c.render_panels(spec) if spec.get('render') == 'panels' else c.render(spec)
    rendered = (c.TEX_OUT / spec['out']).read_text()
    for target in [Path('paper/table') / spec['out'], Path('evaluation/reports/tex') / spec['out']]:
        assert rendered == target.read_text(), target
        print(f'PASS: regenerated RQ2 table exactly matches {target}')
CHECK
```

### Text results

```text
PASS: suite references audited across 29 active TeX files
PASS: size-aware retained for gains and metric properties, including the new explanation
Terminology inventory for semantic review:
paper/appendix/detailed-results.tex:12: \subsection{RQ1 and RQ2: Comparison and Architecture-Driven Metrics}
paper/main.tex:113: \newcommand{\avgfone}{average \fone}% headline aggregate (macro-averaged across projects); NOT for the standard-vs-architecture-driven contrast, which stays "link-level \fone".
paper/main.tex:152: Under an architecture-driven suite that adds per-component measures, the gap widens to 31\,pp on worst-component \fone and 29\,pp on harmonic per-component \fone.
paper/sections/conclusion.tex:8: Alongside the approach we introduce an architecture-driven evaluation suite that adds per-component measures to the standard link-level metrics on doc-code, and the \cmrname{} (\cmr) on doc-model.
paper/sections/conclusion.tex:10: Under the architecture-driven suite this lead widens to $+31$\,pp worst-component \fone{} and $+29$\,pp harmonic-mean per-component \fone.
paper/sections/discussion.tex:17: The architecture-driven suite reuses the benchmark annotations and can flag component-level weaknesses.
paper/sections/discussion.tex:26: The architecture-driven suite exposes this headroom where the link-level average suggests near-saturation.
paper/sections/discussion.tex:69: our architecture-driven metrics express this through worst-component and harmonic-mean per-component \fone{} on doc-code,
paper/sections/eval.tex:27:     \item \label{rq:metrics} \emph{What component-level behavior do architecture-driven metrics reveal beyond link-level ones?}
paper/sections/eval.tex:33: We therefore examine whether the proposed architecture-driven metrics reveal component-level misses, which are difficult to see in the standard metrics.
paper/sections/eval.tex:108: We evaluate the same recovered links with the architecture-driven suite defined in \autoref{sec:metric:suite}.
paper/sections/intro.tex:123: % ; link-level \fone\ understates this gain, as the architecture-driven suite below shows.
paper/sections/intro.tex:136: We therefore introduce an architecture-driven evaluation suite that measures how recovery covers the full architecture model, 
paper/sections/intro.tex:153:     \item an \textbf{architecture-driven evaluation suite} of three size-aware metrics for architecture trace-link recovery (\autoref{sec:metric}).
paper/sections/intro.tex:154:     \item an \textbf{empirical study} showing that \approach{} improves over the strongest baseline by $13$\,pp \fone\ ($12$\,pp \ftwo) on doc-model and $9$\,pp \fone\ ($6$\,pp \ftwo) on doc-code, with larger size-aware gains.
paper/sections/metric.tex:1: \section{An Architecture-Driven Metric Suite}
paper/sections/metric.tex:32: The suite is architecture-driven because it evaluates recovery at the level of components in the architecture model.
paper/sections/metric.tex:33: The suite also provides size-aware evaluation by exposing component-level recovery that pooled link-level scores can obscure when component link counts differ.
paper/sections/motivation.tex:65: \subsection{Architecture-Driven Evaluation}
paper/sections/motivation.tex:115: %SPEC [ev-close] role: close with the paper move; introduce architecture-driven metrics that expose dropped components.
paper/sections/motivation.tex:116: We therefore quantify doc-code link concentration across all five projects and introduce architecture-driven metrics
paper/sections/results.tex:91: % RQ2 body float = tab:rq2 (architecture-driven suite, GPT-5.6-terra). GENERATED by csv_to_tex.py
paper/sections/results.tex:95: \autoref{tab:rq2} reports the architecture-driven suite for all compared systems on both tasks.
paper/sections/results.tex:102: The doc-code measures of the architecture-driven suite aggregate per-component scores; the gaps are larger on these measures.
paper/sections/results.tex:116: The architecture-driven metrics do not favor \approach{} everywhere.
paper/sections/results.tex:123: The architecture-driven metrics also reveal room for improvement that the standard average hides.
paper/sections/results.tex:125: All three systems share this pattern: high link-level averages coexist with weak component-level scores, which the architecture-driven metrics expose.
paper/sections/results.tex:134: The architecture-driven metrics reveal three behaviors that link-level scores hide.
paper/sections/results.tex:139: Third, the link-level and architecture-driven measures can rank systems differently.
paper/sections/results.tex:174: On the doc-model measure of the architecture-driven suite, judging adds no component miss:
paper/sections/results.tex:206: \autoref{tab:rq4} reports the four variants on both tasks, on the reference metrics and the task-specific measures of the architecture-driven suite.
paper/sections/results.tex:262: The architecture-driven suite exposes component-level differences that the standard metric hides, including \approach{}'s own remaining weaknesses.
paper/table/rq2-results.tex:4: \caption{RQ2 architecture-driven metrics by project on GPT-5.6-terra.}
[csv2tex] wrote /tmp/tmpws8dsxem/rq2-results.tex
PASS: regenerated RQ2 table exactly matches paper/table/rq2-results.tex
PASS: regenerated RQ2 table exactly matches evaluation/reports/tex/rq2-results.tex
Exit status: 0
```
