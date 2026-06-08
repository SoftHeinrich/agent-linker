"""Phase 1 judge ablation — minimal-prompt variant.

End-to-end: (mediastore, jabref) × (claude, openai).

Plan:
  1. Load OPENAI_API_KEY from agent-linker/.env.
  2. Re-run canonical SLinker19 to restore baseline phase_cache (overwritten by prior runs).
  3. Run SLinker19EvJudgeMinimal which adds Evidence: lines per alias and KEEPS
     the original DOC_KNOWLEDGE_JUDGE_RULES (no permissive override). Saves to a
     separate phase_cache via `_VARIANT_NAME = "s_linker19_evjudge_min"`.
  4. Compare.

Hypothesis being tested: providing evidence alone — without a permissive override
rule — recovers the mediastore alias gap (DataStorage, AudioAccess) while letting
the existing tier/metaphor rejection rule continue to reject jabref's metaphors
(core, outer shell, intermediate layer).
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

# ─── Load .env from agent-linker ─────────────────────────────────────────────
_ENV = Path('/mnt/hostshare/ardoco-home/agent-linker/.env')
if _ENV.exists():
    for line in _ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        os.environ[k.strip()] = v.strip()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_sad_sam.linkers.experimental.s_linker19 import SLinker19
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.linkers.experimental.prompts_v5 import (
    DOC_KNOWLEDGE_JUDGE_EXAMPLES,
    DOC_KNOWLEDGE_JUDGE_RULES,
)

BENCH = '/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark'
PROJECTS = {
    'mediastore': {
        'text': f'{BENCH}/mediastore/text_2016/mediastore.txt',
        'model': f'{BENCH}/mediastore/model_2016/pcm/ms.repository',
        'gold':  f'{BENCH}/mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv',
    },
    'jabref': {
        'text':  f'{BENCH}/jabref/text_2021/jabref.txt',
        'model': f'{BENCH}/jabref/model_2021/pcm/jabref.repository',
        'gold':  f'{BENCH}/jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv',
    },
}


def load_gold(proj):
    g = set()
    with open(PROJECTS[proj]['gold']) as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            if len(row) < 2: continue
            try: g.add((int(row[1].strip()), row[0].strip()))
            except: continue
    return g


# ─── Minimal evjudge subclass ────────────────────────────────────────────────

class SLinker19EvJudgeMin(SLinker19):
    """SLinker19 + minimal-prompt evidence judge: same rules, evidence added.

    Separate phase_cache via `_VARIANT_NAME` override — does NOT overwrite the
    canonical s_linker19 cache.
    """
    _VARIANT_NAME = "s_linker19_evjudge_min"

    def _learn_document_knowledge(self, sentences, components):
        self._evidence_sentences = sentences
        return super()._learn_document_knowledge(sentences, components)

    def _prompt_doc_knowledge_judge(self, comp_names, mapping_list):
        sentences = getattr(self, '_evidence_sentences', [])
        rebuilt = []
        for line in mapping_list:
            m = re.match(r"'(.+?)' -> (.+)", line.strip())
            if not m:
                rebuilt.append(line); continue
            term = m.group(1)
            pat = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
            ev = []
            for s in sentences:
                if pat.search(s.text):
                    ev.append(f'    S{s.number}: "{s.text[:200]}"')
                    if len(ev) >= 4: break
            ev_block = '\n'.join(ev) if ev else "    (no occurrences in document)"
            rebuilt.append(f"{line}\n  Evidence:\n{ev_block}")
        # Original rules unchanged — no permissive override.
        return f"""JUDGE: Review these component name mappings for correctness. Each mapping is paired with Evidence sentences from the document where the proposed term appears.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(rebuilt)}

{DOC_KNOWLEDGE_JUDGE_EXAMPLES}

{DOC_KNOWLEDGE_JUDGE_RULES}

Return JSON:
{{"approved": ["term1", "term2"]}}
JSON only:"""


# ─── Runner ─────────────────────────────────────────────────────────────────

def score(links, proj):
    gold = load_gold(proj)
    pred = set((lk.sentence_number, lk.component_id) for lk in links)
    tp, fp, fn = pred & gold, pred - gold, gold - pred
    P = len(tp) / max(1, len(tp) + len(fp))
    R = len(tp) / max(1, len(gold))
    F1 = 2 * P * R / max(1e-9, P + R)
    return dict(tp=len(tp), fp=len(fp), fn=len(fn),
                precision=round(P, 4), recall=round(R, 4), f1=round(F1, 4),
                gold=len(gold), pred=len(pred),
                fp_keys=sorted(f'S{s}' for s, _ in fp),
                fn_keys=sorted(f'S{s}' for s, _ in fn))


def run(linker_cls, proj, backend_name, model):
    paths = PROJECTS[proj]
    backend = LLMBackend(backend_name)
    linker = linker_cls(backend=backend, model=model)
    print(f'\n##### {linker_cls.__name__} / {proj} / {backend_name} ({model}) #####', flush=True)
    t0 = time.time()
    links = linker.link(paths['text'], paths['model'])
    elapsed = round(time.time() - t0, 1)
    s = score(links, proj)
    s['elapsed_s'] = elapsed
    s['variant'] = linker_cls.__name__
    s['backend'] = backend_name
    s['project'] = proj
    print(f'>> {linker_cls.__name__}/{proj}/{backend_name}: TP={s["tp"]} FP={s["fp"]} FN={s["fn"]} '
          f'P={s["precision"]:.4f} R={s["recall"]:.4f} F1={s["f1"]:.4f} ({elapsed}s)', flush=True)
    return s


def main():
    cases = [
        ('mediastore', 'claude', 'sonnet'),
        ('jabref',     'claude', 'sonnet'),
        ('mediastore', 'openai', 'gpt-5.4'),
        ('jabref',     'openai', 'gpt-5.4'),
    ]

    all_results = []
    # Pass A: re-establish baseline (overwrites phase_cache/s_linker19/)
    print('\n══════ Pass A: rebuild baseline SLinker19 ══════')
    for proj, backend, model in cases:
        try:
            all_results.append({'pass': 'baseline', **run(SLinker19, proj, backend, model)})
        except Exception as e:
            import traceback; traceback.print_exc()
            all_results.append({'pass': 'baseline', 'project': proj, 'backend': backend, 'error': str(e)})

    # Pass B: minimal evjudge (saves to phase_cache/s_linker19_evjudge_min/)
    print('\n══════ Pass B: evjudge minimal ══════')
    for proj, backend, model in cases:
        try:
            all_results.append({'pass': 'evjudge_min', **run(SLinker19EvJudgeMin, proj, backend, model)})
        except Exception as e:
            import traceback; traceback.print_exc()
            all_results.append({'pass': 'evjudge_min', 'project': proj, 'backend': backend, 'error': str(e)})

    # Comparison
    print('\n\n══════════════════ COMPARISON ══════════════════')
    print(f'{"project":<14} {"backend":<7} {"pass":<14} {"TP":>3} {"FP":>3} {"FN":>3}  {"P":>6} {"R":>6} {"F1":>6}')
    print('─' * 78)
    by_key = {}
    for r in all_results:
        if 'error' in r:
            print(f'  ERROR {r["project"]:<10} {r["backend"]}: {r["error"]}')
            continue
        key = (r['project'], r['backend'])
        by_key.setdefault(key, {})[r['pass']] = r
    for key, runs in by_key.items():
        proj, backend = key
        b = runs.get('baseline'); e = runs.get('evjudge_min')
        if b:
            print(f'{proj:<14} {backend:<7} {"baseline":<14} {b["tp"]:>3} {b["fp"]:>3} {b["fn"]:>3}  {b["precision"]:>6.4f} {b["recall"]:>6.4f} {b["f1"]:>6.4f}')
        if e:
            print(f'{proj:<14} {backend:<7} {"evjudge_min":<14} {e["tp"]:>3} {e["fp"]:>3} {e["fn"]:>3}  {e["precision"]:>6.4f} {e["recall"]:>6.4f} {e["f1"]:>6.4f}')
        if b and e:
            print(f'{"":<14} {"":<7} {"Δ":<14} {e["tp"]-b["tp"]:>+3} {e["fp"]-b["fp"]:>+3} {e["fn"]-b["fn"]:>+3}  {e["precision"]-b["precision"]:>+6.4f} {e["recall"]-b["recall"]:>+6.4f} {e["f1"]-b["f1"]:>+6.4f}')
        print()

    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = f'results/ablation_evjudge_min_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
