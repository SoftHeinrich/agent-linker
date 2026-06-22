"""Run SLinker19EvJudgeMin on the 3 remaining projects × 2 backends.

Compares against existing baseline phase_cache (no baseline rerun — those exist
from prior canonical runs).
"""
from __future__ import annotations
import csv, json, os, pickle, time, sys
from pathlib import Path

_ENV = Path('/mnt/hostshare/ardoco-home/agent-linker/.env')
if _ENV.exists():
    for line in _ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        k, v = line.split('=', 1)
        os.environ[k.strip()] = v.strip()

sys.path.insert(0, str(Path(__file__).parent / "src"))
from ablation_evjudge import SLinker19EvJudgeMin, score, load_gold, PROJECTS as _MS_JR_PROJECTS
from llm_sad_sam.linkers.experimental.s_linker19 import SLinker19
from llm_sad_sam.llm_client import LLMBackend

BENCH = '/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark'
PROJECTS = {
    'teastore': {
        'text':  f'{BENCH}/teastore/text_2020/teastore.txt',
        'model': f'{BENCH}/teastore/model_2020/pcm/teastore.repository',
        'gold':  f'{BENCH}/teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv',
    },
    'teammates': {
        'text':  f'{BENCH}/teammates/text_2021/teammates.txt',
        'model': f'{BENCH}/teammates/model_2021/pcm/teammates.repository',
        'gold':  f'{BENCH}/teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv',
    },
    'bigbluebutton': {
        'text':  f'{BENCH}/bigbluebutton/text_2021/bigbluebutton.txt',
        'model': f'{BENCH}/bigbluebutton/model_2021/pcm/bbb.repository',
        'gold':  f'{BENCH}/bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv',
    },
}

def load_gold_local(proj):
    g = set()
    with open(PROJECTS[proj]['gold']) as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            if len(row) < 2: continue
            try: g.add((int(row[1].strip()), row[0].strip()))
            except: continue
    return g

def score_local(links, proj):
    gold = load_gold_local(proj)
    pred = set((lk.sentence_number, lk.component_id) for lk in links)
    tp, fp, fn = pred & gold, pred - gold, gold - pred
    P = len(tp) / max(1, len(tp) + len(fp))
    R = len(tp) / max(1, len(gold))
    F1 = 2*P*R / max(1e-9, P+R)
    return dict(tp=len(tp), fp=len(fp), fn=len(fn),
                precision=round(P, 4), recall=round(R, 4), f1=round(F1, 4),
                gold=len(gold), pred=len(pred))

def baseline_score(proj, backend):
    p = f'results/phase_cache/s_linker19/{backend}/{proj}/final.pkl'
    if not os.path.exists(p): return None
    d = pickle.load(open(p, 'rb'))
    return score_local(d['final'], proj)

def run(proj, backend_name, model):
    paths = PROJECTS[proj]
    linker = SLinker19EvJudgeMin(backend=LLMBackend(backend_name), model=model)
    print(f'\n##### evjudge_min / {proj} / {backend_name} ({model}) #####', flush=True)
    t0 = time.time()
    links = linker.link(paths['text'], paths['model'])
    elapsed = round(time.time() - t0, 1)
    s = score_local(links, proj)
    s['elapsed_s'] = elapsed; s['project'] = proj; s['backend'] = backend_name
    print(f'>> evjudge_min/{proj}/{backend_name}: TP={s["tp"]} FP={s["fp"]} FN={s["fn"]} F1={s["f1"]} ({elapsed}s)', flush=True)
    return s

def main():
    cases = [
        ('bigbluebutton', 'claude', 'sonnet'),
        ('bigbluebutton', 'openai', 'gpt-5.4'),
    ]
    rows = []
    for proj, backend, model in cases:
        try:
            ev = run(proj, backend, model)
            bl = baseline_score(proj, backend)
            rows.append({'project': proj, 'backend': backend, 'baseline': bl, 'evjudge_min': ev})
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append({'project': proj, 'backend': backend, 'error': str(e)})

    print('\n\n══════════════════ COMPARISON (3 remaining projects) ══════════════════')
    print(f'{"project":<14} {"backend":<7} {"pass":<14} {"TP":>3} {"FP":>3} {"FN":>3}  {"P":>6} {"R":>6} {"F1":>6}')
    print('─' * 78)
    for r in rows:
        if 'error' in r:
            print(f'ERROR {r["project"]}/{r["backend"]}: {r["error"]}'); continue
        b, e = r['baseline'], r['evjudge_min']
        if b: print(f'{r["project"]:<14} {r["backend"]:<7} {"baseline":<14} {b["tp"]:>3} {b["fp"]:>3} {b["fn"]:>3}  {b["precision"]:>6.4f} {b["recall"]:>6.4f} {b["f1"]:>6.4f}')
        print(f'{r["project"]:<14} {r["backend"]:<7} {"evjudge_min":<14} {e["tp"]:>3} {e["fp"]:>3} {e["fn"]:>3}  {e["precision"]:>6.4f} {e["recall"]:>6.4f} {e["f1"]:>6.4f}')
        if b:
            print(f'{"":<14} {"":<7} {"Δ":<14} {e["tp"]-b["tp"]:>+3} {e["fp"]-b["fp"]:>+3} {e["fn"]-b["fn"]:>+3}  {e["precision"]-b["precision"]:>+6.4f} {e["recall"]-b["recall"]:>+6.4f} {e["f1"]-b["f1"]:>+6.4f}')
        print()

    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = f'results/ablation_evjudge_rest_{ts}.json'
    json.dump(rows, open(out_path, 'w'), indent=2)
    print(f'Saved {out_path}')

if __name__ == '__main__':
    main()
