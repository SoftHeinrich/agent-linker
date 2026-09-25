#!/usr/bin/env python3
"""Independent checks of RQ1 SD, replacement provenance, and token-only output."""
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
def read(name):
    return list(csv.DictReader((ROOT/name).open()))

raw = read('evaluation/reports/RQ12_PERPROJECT_PERRUN.csv')
big = read('evaluation/reports/RQ12_BIGTABLE.csv')
for row in read('evaluation/reports/RQ12_SD.csv'):
    source = big if row['project'] == 'Average' else raw
    selected = [r for r in source if r['system'] == row['system']
                and r['run'] in ('run1', 'run2', 'run3')
                and (row['project'] == 'Average' or r['project'] == row['project'])]
    assert len(selected) == 3
    expected = statistics.stdev(float(r[row['metric']]) for r in selected)
    assert abs(expected-float(row['sd'])) < 0.000062
print('PASS: all sample SD cells independently match exported runs (CSV rounding tolerance 0.000062)')
for run in (1, 2, 3):
    old = ROOT/f'results/greedymerge_e2e_terra_r{run}_20260916v2'
    new = ROOT/f'results/ms_replacement_run{run if run > 1 else ""}_20260924'
    assert (old/'s_linker126_mediastore_links.csv').read_bytes() == (new/'s_linker126_mediastore_links.csv').read_bytes()
    assert (old/'s_linker126_mediastore_links.csv.bak_20260924').exists()
    for phase in ('final', 'knowledge', 'linker_name', 'linker_coreference'):
        rel = f'phase_states/s_linker126/openai/mediastore/{phase}.pkl'
        assert (old/rel).read_bytes() == (new/rel).read_bytes()
print('PASS: all three replacement link sets and phase states match their source runs; original links retained')
usage = read('evaluation/reports/INFERENCE_COST_PERRUN.csv')
summary = read('evaluation/reports/tex_src/inference_cost.csv')
assert len(usage) == 30
assert len(summary) == 6
assert set(summary[0]) == {'project', 'approach_input_k', 'approach_output_k', 'Artemis_input_k', 'Artemis_output_k'}
for row in summary:
    for system in ('approach','Artemis'):
        subset = [r for r in usage if r['system']==system and (row['project']=='Total' or r['project']==row['project'])]
        for metric in ('input_k','output_k'):
            assert abs(sum(float(r[metric]) for r in subset)/3-float(row[f'{system}_{metric}'])) < 1e-6
for name in ('rq1-results', 'inference-cost'):
    assert (ROOT/f'evaluation/reports/tex/{name}.tex').read_bytes() == (ROOT/f'paper/table/{name}.tex').read_bytes()
tex=(ROOT/'paper/table/rq1-results.tex').read_text()
assert tex.count('$\\pm$') == 48
assert len(read('paper/table/rq1-results.csv')) == 12
cost=(ROOT/'paper/table/inference-cost.tex').read_text()
assert 'Time (s)' not in cost and 'Calls' not in cost
print('PASS: token-only table means/totals, 12 RQ1 rows, 48 inline SD entries, and paper sync')
print('Token totals (thousands):', summary[-1])

quality = {r['system']: r for r in big if r['run'] == 'average'}
a = quality['approach (GPT-5.6-terra)']
b = quality['Artemis (GPT-5.6-terra)']
for metric in ('doc_to_model_link_f1', 'doc_to_model_link_f2', 'doc_to_code_file_f1',
               'doc_to_code_file_f2', 'doc_to_code_worst_component_f1',
               'doc_to_code_harmonic_component_f1'):
    print(f"Quality: {metric}: approach={a[metric]}, Artemis={b[metric]}, gap={(float(a[metric])-float(b[metric]))*100:.1f}pp")
abstract = (ROOT/'paper/main.tex').read_text().split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
for number in ('12.7', '6.6', '29.6', '27.6'):
    assert number in abstract
for name in ('intro', 'conclusion'):
    prose = (ROOT/f'paper/sections/{name}.tex').read_text()
    for number in ('0.936', '0.955', '0.879', '12.7', '6.6'):
        assert number in prose
print('PASS: updated headline scores and gaps present in abstract, introduction, and conclusion')
