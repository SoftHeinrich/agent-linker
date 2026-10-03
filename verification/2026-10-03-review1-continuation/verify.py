#!/usr/bin/env python3
"""Audit review1 edits against displayed cells; run from any directory."""
import csv
from decimal import Decimal
import hashlib
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'evaluation/mini-src'))
import csv_to_tex as render

SRC = ROOT / 'evaluation/reports/tex_src'
PAPER = ROOT / 'paper'
BASE_PAPER = 'b1278c8f214761339bd310ff1890f93efe5a7b6b'

def rows(name):
    return list(csv.DictReader((SRC / name).open()))

def shown(row, field):
    return Decimal(render.fmt(row[field], 'f2'))

def delta(a, af, b, bf=None):
    return (shown(a, af) - shown(b, bf or af)) * 100

def prose(name):
    return '\n'.join(re.split(r'(?<!\\)%', line)[0] for line in (PAPER / name).read_text().splitlines())

checks = 0

def claim(name, fragment):
    global checks
    assert fragment in prose(name), (name, fragment)
    checks += 1
    print(f'PASS prose {name}: {fragment}')

rq1 = {(r['project'], r['task']): r for r in rows('rq1_transposed.csv')}
rq2 = {r['system']: r for r in rows('rq2.csv') if r['right_project'] == 'Average'}
rq3 = {r['judge']: r for r in rows('rq3.csv')}
rq4 = {r['variant']: r for r in rows('rq4.csv')}
dm, dc = rq1['Average', 'DM'], rq1['Average', 'DC']
a, b, t = (rq2[k] for k in ('approach', 'Artemis', 'TransArC'))
f, n, c, off = (rq3[k] for k in ('full_on', 'name', 'coref', 'no_judge'))
full, named, nk = (rq4[k] for k in ('Full', 'Name', 'No knowledge'))

print('Configuration: s126; five ARDoCo projects; GPT-5.6-terra; mean of three runs for stochastic systems.')
print('RQ3 rescoring uses the recorded full runs; RQ4 No knowledge uses separate recorded runs.')
print('Convention: format each source score with the table renderer, then subtract displayed scores, times 100.')
f2 = delta(dm, 'approach_f2', dm, 'Artemis_f2')
harm = delta(a, 'right_dc_harm_f1', b)
claim('main.tex', f'by {f2:.0f}\\,pp in the recall-weighted')
claim('main.tex', f'and {harm:.0f}\\,pp on harmonic')
claim('main.tex', 'yet they are hard to recover:')
claim('sections/intro.tex', f'$0.95$ \\avgftwo, ${f2:.0f}$\\,pp above it')
claim('sections/intro.tex', f'(${f2:.0f}$\\,pp \\ftwo)')
claim('sections/intro.tex', 'scores displayed to two decimal places in the tables.')
claim('sections/results.tex', 'scores displayed to two decimal places in the tables.')
claim('sections/results.tex', f"from \\Artemis{{}}'s $0.83$ to $0.95$ ($+{f2:.0f}$pp)")
claim('sections/results.tex', f'on doc-code it leads by $+{delta(dc,"approach_p",dc,"Artemis_p"):.0f}$pp precision')
claim('sections/results.tex', f'by ${delta(dm,"approach_f1",dm,"pipeline_f1"):.0f}$pp and ${delta(dc,"approach_f1",dc,"pipeline_f1"):.0f}$pp.')
claim('sections/results.tex', f'($+{harm:.0f}$pp).')
claim('sections/results.tex', f'$+{harm:.0f}$pp harmonic-mean')
claim('sections/conclusion.tex', f'$+{harm:.0f}$\\,pp harmonic-mean')
claim('sections/motivation.tex', f'by ${delta(b,"right_dc_file_f1",t):.0f}$\\,pp of link-level')
claim('sections/motivation.tex', f'by ${delta(t,"right_dc_worst_f1",b):.0f}$\\,pp on worst-component')
claim('sections/motivation.tex', f'and ${delta(t,"right_dc_harm_f1",b):.0f}$\\,pp on the harmonic mean')
claim('sections/results.tex', f'a loss of ${delta(f,"dm_f1",off):.0f}$pp,')
claim('sections/results.tex', f'Removing both costs ${delta(f,"dm_f1",off):.0f}$pp')
claim('sections/results.tex', f'costs ${delta(f,"dm_f1",n):.0f}$ and ${delta(f,"dm_f1",c):.0f}$pp')
gain2 = delta(full, 'doc_to_model_macro_f2', named)
gain1 = delta(full, 'doc_to_model_macro_f1', named)
claim('sections/results.tex', f'Its ${gain2:.0f}$pp \\ftwo{{}} gain is ${gain2/gain1:.1f}$ times its ${gain1:.0f}$pp \\fone{{}} gain.')
claim('sections/results.tex', f'configuration is ${delta(full,"doc_to_model_macro_f1",nk):.0f}$\\,pp lower')
claim('sections/results.tex', f'and ${delta(full,"doc_to_model_macro_f2",nk):.0f}$\\,pp \\avgftwo\\ ($0.95')
claim('sections/discussion.tex', 'ranging from $4$ to $159$ thousand lines of code')
assert 'two orders of magnitude' not in prose('sections/discussion.tex')
assert PAPER.joinpath('main.tex').read_text().splitlines()[0] == r'\documentclass[acmsmall,screen,review,anonymous]{acmart}'
print('PASS FSE 2027 class matches official track instructions (URL in README.md).')

# Compare every numeric row, including bold appendix averages, with the pre-edit paper commit.
# Labels and notes may change; counts, scores, order and bolding must not.
with tempfile.TemporaryDirectory() as tmp:
    render.TEX_OUT = Path(tmp)
    for filename, directory in [('rq3-confusion.tex', 'table'), ('rq3-runs.tex', 'appendix')]:
        spec = next(s for s in render.SPECS if s['out'] == filename)
        render.render(spec)
        regenerated = (Path(tmp) / filename).read_text()
        paper_file = PAPER / directory / filename
        assert regenerated == paper_file.read_text()
        assert regenerated == (ROOT / 'evaluation/reports/tex' / filename).read_text()
        for label in (r'\noNameValid{}', r'\noCitation{}', r'\noValidator{}'):
            assert label in regenerated
        assert 'Configuration &' in regenerated
        assert 'including deterministic exclusions' in regenerated
        old = subprocess.check_output(['git', '-C', str(PAPER), 'show', f'{BASE_PAPER}:{directory}/{filename}'], text=True)
        # The appendix prefixes Backend and Run; remove all label columns.
        nlabels = len(spec['labels'])
        def cells(text):
            return [line.split(' & ')[nlabels:] for line in text.splitlines()
                    if len(line.split(' & ')) == nlabels + len(spec['cols'])
                    and re.match(r'(?:\\textbf\{)?\d', line.split(' & ')[nlabels])]
        assert cells(old) and cells(old) == cells(regenerated)
        print(f'PASS {filename}: reproducible, paper mirror exact, labels explicit, numeric cells unchanged ({len(cells(old))} rows).')

print(f'PASS {checks} prose assertions.')
for name in ('rq1_transposed.csv', 'rq2.csv', 'rq3.csv', 'rq3_runs.csv', 'rq4.csv'):
    print(f'SHA256 {name}: {hashlib.sha256((SRC/name).read_bytes()).hexdigest()}')
