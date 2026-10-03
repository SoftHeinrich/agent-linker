#!/usr/bin/env python3
"""Recorded call usage for the s126 terra and luna sweeps of 2026-10-02 and the
September 24 Artemis/luna runs. The scored Artemis terra runs logged no token usage,
so Artemis_terra comes from a separate token re-run on 2026-10-03 (replication/artemis/run.sh).

Counts successful responses with reported usage, including logged repair calls.
This is a token-usage report, not a monetary estimate.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

import metrics as m

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = tuple(m.PROJECTS)
METRICS = ('input_k', 'output_k')


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def artemis_rows(content, system, model, run, log):
    records = {}
    project = None
    for line in content.splitlines():
        start = re.search(r'ArDoCo - Starting (\w+)$', line)
        if start and start[1] in PROJECTS:
            project = start[1]
            assert project not in records
            records[project] = dict(calls=0, input_k=0., output_k=0.)
        tokens = re.search(r'(?:prompt[12]|repair) tokens: input=(\d+), output=(\d+), total=(\d+)', line)
        if tokens:
            assert project is not None and int(tokens[1])+int(tokens[2]) == int(tokens[3])
            rec = records[project]
            rec['calls'] += 1
            rec['input_k'] += int(tokens[1])/1000
            rec['output_k'] += int(tokens[2])/1000
    assert set(records) == set(PROJECTS)
    rows = []
    for project, rec in records.items():
        assert rec.pop('calls') >= 2
        rows.append(dict(system=system, model=model, run=run, project=project,
            **rec, source=str(log.relative_to(ROOT)), sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
    return rows


def collect():
    rows = []
    for run in (1, 2, 3):
        for backend, system in (('terra', 'approach'), ('luna', 'approach_luna')):
            for project in PROJECTS:
                directory = ROOT / 'results' / f'greedymerge_e2e_{backend}_r{run}_20261002'
                files = list(directory.glob(f'llm_logs/s_linker126_openai_{project}_*_calls.json'))
                assert len(files) == 1, files
                calls = json.loads(files[0].read_text())
                assert calls and all(c['success'] and c['model'] == f'gpt-5.6-{backend}' for c in calls)
                assert all(c.get('token_usage') and c['token_usage']['total_tokens'] ==
                           c['token_usage']['prompt_tokens'] + c['token_usage']['completion_tokens'] for c in calls)
                reports = list(directory.glob('ablation_*.json'))
                assert len(reports) == 1, reports
                result = json.loads(reports[0].read_text())[project]['s_linker126']
                assert result['llm_calls'] == len(calls)
                rows.append(dict(system=system, model=f'gpt-5.6-{backend}', run=run, project=project,
                    input_k=sum(c['token_usage']['prompt_tokens'] for c in calls)/1000,
                    output_k=sum(c['token_usage']['completion_tokens'] for c in calls)/1000,
                    source=str(files[0].relative_to(ROOT)),
                    sha256=hashlib.sha256(files[0].read_bytes()).hexdigest()))
        log = ROOT / f'sota-links/_build-logs/run-artemis-gpt-5.6-luna-run{run}.log'
        content = log.read_text()
        assert 'BUILD SUCCESS' in content
        rows += artemis_rows(content, 'Artemis', 'gpt-5.6-luna', run, log)
    # Separate token re-run of Artemis on terra (2026-10-03); one log holds all three runs,
    # each ending with the run's link-count summary written by replication/artemis/run.sh.
    log = ROOT / 'replication/artemis/logs/tokens-artemis-gpt-5.6-terra-20261003.log'
    content = log.read_text()
    ends = [(int(mk[1]), mk.start()) for mk in re.finditer(r'^run(\d) doc-model mediastore: \d+ links$', content, re.M)]
    assert [r for r, _ in ends] == [1, 2, 3], ends
    start = 0
    for run, end in ends:
        segment = content[start:end]
        assert 'BUILD SUCCESS' in segment
        rows += artemis_rows(segment, 'Artemis_terra', 'gpt-5.6-terra', run, log)
        start = end
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'evaluation/reports')
    args = parser.parse_args()
    rows = collect()
    write(args.out/'INFERENCE_COST_PERRUN.csv', rows)
    summary = []
    for project in (*PROJECTS, 'Total'):
        row = dict(project=project)
        for system in ('approach', 'approach_luna', 'Artemis', 'Artemis_terra'):
            selected = [r for r in rows if r['system'] == system and
                        (project == 'Total' or r['project'] == project)]
            for metric in METRICS:
                per_run = [sum(r[metric] for r in selected if r['run'] == run) for run in (1, 2, 3)]
                row[f'{system}_{metric}'] = f'{statistics.mean(per_run):.6f}'
        summary.append(row)
    write(args.out/'tex_src/inference_cost.csv', summary)
    compact = []
    for system in ('approach', 'approach_luna', 'Artemis', 'Artemis_terra'):
        row = {'system': system}
        for project_row in summary:
            project = project_row['project']
            for metric in METRICS:
                row[f'{project}_{metric}'] = project_row[f'{system}_{metric}']
        compact.append(row)
    write(args.out/'tex_src/inference_cost_by_system.csv', compact)
    print(f'PASS: {len(rows)} project/run usage records; means of three runs written to {args.out}')


if __name__ == '__main__':
    main()
