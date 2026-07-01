#!/usr/bin/env python3
"""FULL LIVE RUN — GTP as an additive proposer, corpus macro-F1 delta vs frozen s21.

Design (see ../PROPOSAL.md, gtp/FINDINGS.md): GTP is an LLM/structure/context
proposer. Here it runs over the WHOLE corpus, its proposals are grounded to the real
component roster, routed by the mode GTP emits, and judged by the mode-matched judge.
Survivors UNION into the frozen s21 final links; we rescore model-doc macro-F1
(sentence, component) against SAD-SAM gold. This isolates GTP's marginal contribution
and reuses the frozen baseline (cheaper + cleaner than re-running the stochastic
pipeline). Everything reasoning-off, gpt-5.4.

Marginal-judging: a GTP proposal already in s21's union-final is an accepted link
(s21's own gate passed it in some run) -> kept without re-judging. Only genuinely
NEW (s,component) candidates are routed + judged. This is where recall is won/lost.

Baseline sanity: `--baseline-only` scores frozen s21 with NO LLM calls and must
reproduce the reported macro-F1 (~0.936) before any spend.

Run:  python3 live_run.py --baseline-only     # free, sanity gate
      python3 live_run.py                      # live GTP augmentation (spends)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FN = HERE.parent / "fn_judge"
APPROACH = HERE.parents[1]
sys.path.insert(0, str(APPROACH / "src"))
sys.path.insert(0, str(FN))
sys.path.insert(0, str(HERE))

import build_cases as BC
from llm_sad_sam.pcm_parser import parse_pcm_repository

PROJECTS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]
RUNS = ["run1", "run2", "run3"]
MODEL_FILE = {
    "mediastore": "model_2016/pcm/ms.repository",
    "teastore": "model_2020/pcm/teastore.repository",
    "teammates": "model_2021/pcm/teammates.repository",
    "bigbluebutton": "model_2021/pcm/bbb.repository",
    "jabref": "model_2021/pcm/jabref.repository",
}
EXTRACTS = APPROACH / "results/v2.6.6_extracts_s21/gpt"
CORPUS_PCACHE = HERE / "corpus_proposer_cache.json"
CORPUS_JCACHE = HERE / "corpus_judge_cache.json"


# ── data loaders ─────────────────────────────────────────────────────────────

def roster(project):
    comps = parse_pcm_repository(BC.BENCH / project / MODEL_FILE[project])
    return {c.name: c.id for c in comps}          # name -> component_id


def s21_final(project, run):
    d = json.loads((EXTRACTS / run / f"{project}.json").read_text())
    return {(lk["s"], lk["c"]) for lk in d["final"]["links"]}


def ambiguous_names(project):
    out = set()
    for run in RUNS:
        d = json.loads((EXTRACTS / run / f"{project}.json").read_text())
        for nm in d["knowledge"]["model_knowledge"].get("ambiguous_names", []):
            out.add(nm)
    return out


def gold_ids(project):
    return BC.load_gold(project)                    # {(sentence, modelElementID)}


# ── scoring (model-doc level: (sentence, component_id) ) ─────────────────────

def prf(links, gold):
    tp = len(links & gold)
    fp = len(links - gold)
    fn = len(gold - links)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1, tp, fp, fn


def macro_over_runs(link_fn, golds):
    """link_fn(project, run) -> set of (s,id). Returns per-run macro P/R/F1 lists."""
    per_run = []
    for run in RUNS:
        f1s, ps, rs = [], [], []
        for p in PROJECTS:
            P, R, F1, *_ = prf(link_fn(p, run), golds[p])
            f1s.append(F1); ps.append(P); rs.append(R)
        per_run.append((sum(ps) / 5, sum(rs) / 5, sum(f1s) / 5))
    return per_run


def _avg(rows, i):
    return sum(r[i] for r in rows) / len(rows)


# ── baseline sanity (no LLM) ─────────────────────────────────────────────────

def baseline():
    golds = {p: gold_ids(p) for p in PROJECTS}
    per = macro_over_runs(s21_final, golds)
    print("=" * 78)
    print("BASELINE s21 (frozen extracts) — model-doc macro (sentence,component), gpt-5.4")
    print("=" * 78)
    for run, (P, R, F1) in zip(RUNS, per):
        print(f"  {run}: P={P:.4f} R={R:.4f} F1={F1:.4f}")
    print(f"  MEAN: P={_avg(per,0):.4f} R={_avg(per,1):.4f} F1={_avg(per,2):.4f}"
          "   (reported s21 gpt-5.4 macro-F1 ~0.936)")
    return per, golds


# ── GTP live augmentation ────────────────────────────────────────────────────

def run_gtp(catalog_mode="name"):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    import proposer as PR
    import run_judges as RJ
    import router_judge as RJU

    golds = {p: gold_ids(p) for p in PROJECTS}
    prof = json.loads((FN / "profiles.json").read_text())   # role text (name mode ignores)
    rosters = {p: roster(p) for p in PROJECTS}
    sents_by = {p: BC.sentences(p) for p in PROJECTS}

    # 1) propose over the whole corpus IN PARALLEL (cached, grounded to the real roster)
    cache = json.loads(CORPUS_PCACHE.read_text()) if CORPUS_PCACHE.exists() else {}
    tasks = []
    for project in PROJECTS:
        names = list(rosters[project].keys())
        roles = prof.get(project) if catalog_mode == "role" else None
        for s in sorted(sents_by[project]):
            ck = f"{catalog_mode}|{project}|{s}"
            if ck not in cache:
                tasks.append((ck, sents_by[project][s], sents_by[project].get(s - 1, ""),
                              names, roles))
    if tasks:
        client = PR.make_client()
        lock = Lock()
        done = [0]

        def _one(t):
            ck, sent, prev, names, roles = t
            resp = client.query(PR.build_prompt(sent, prev, names, roles), timeout=180)
            raw = PR._parse(resp.text if resp.success else "")
            with lock:
                cache[ck] = raw
                done[0] += 1
                if done[0] % 25 == 0:
                    print(f"    proposed {done[0]}/{len(tasks)}", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(_one, tasks))
        CORPUS_PCACHE.write_text(json.dumps(cache, indent=1))

    proposals = defaultdict(list)     # project -> [(s, name, id, mode, quote)]
    dropped = 0
    for project in PROJECTS:
        rost = rosters[project]
        names = list(rost.keys())
        for s in sorted(sents_by[project]):
            raw = cache.get(f"{catalog_mode}|{project}|{s}", [])
            grounded, drop = PR.ground(raw, names)
            dropped += len(drop)
            for r in grounded:
                proposals[project].append((s, r["component"], rost[r["component"]],
                                           r["mode"], r.get("quote", "")))
        print(f"  proposed {project}: {len(proposals[project])} grounded", file=sys.stderr)
    print(f"  total hallucinated (ungrounded) refs dropped: {dropped}", file=sys.stderr)

    # 2) marginal candidates = GTP proposals never in any s21 run's final -> judge them
    union_final = {p: set().union(*(s21_final(p, r) for r in RUNS)) for p in PROJECTS}
    jcache = json.loads(CORPUS_JCACHE.read_text()) if CORPUS_JCACHE.exists() else {}
    client = RJ._client()
    kept_new = defaultdict(set)       # project -> {(s,id)} newly judged-keep
    for project in PROJECTS:
        amb = ambiguous_names(project)
        sents = BC.sentences(project)
        # dedup marginal (s,id) keeping first mode/quote
        marg = {}
        for (s, name, cid, mode, quote) in proposals[project]:
            if (s, cid) in union_final[project]:
                continue
            marg.setdefault((s, cid), (name, mode, quote))
        # build case dicts, group by mode
        by_mode = defaultdict(list)
        for (s, cid), (name, mode, quote) in marg.items():
            case = {
                "id": f"{project}|{s}|{cid}", "project": project, "sentence_num": s,
                "component": name, "sentence": sents[s], "preceding": sents.get(s - 1, ""),
                "matched_text": quote or name, "mention_type": None,
                "is_ambiguous": name in amb, "coref": None,
                "anchors": _anchors(project, s, name, sents),
            }
            by_mode[mode if mode in RJU.ROUTES else "AFFIRMATIVE"].append(((s, cid), case))
        _judge_by_mode(project, by_mode, client, jcache, kept_new, RJ, RJU)
        print(f"  judged {project}: {sum(len(v) for v in by_mode.values())} marginal "
              f"-> {len(kept_new[project])} kept", file=sys.stderr)
    CORPUS_JCACHE.write_text(json.dumps(jcache, indent=1))

    # 3) augmented(run) = s21_final(run) UNION [ proposals in union_final ] UNION [ kept_new ]
    prop_in_union = {p: {(s, cid) for (s, name, cid, m, q) in proposals[p]
                         if (s, cid) in union_final[p]} for p in PROJECTS}

    def augmented(project, run):
        return s21_final(project, run) | prop_in_union[project] | kept_new[project]

    return golds, augmented, kept_new, prop_in_union, proposals, union_final


def _anchors(project, s, name, sents):
    out = []
    for i in sorted(sents):
        if i != s and BC.standalone(name, sents[i]):
            out.append(f"S{i}: {sents[i]}")
        if len(out) >= 4:
            break
    return out


def _judge_by_mode(project, by_mode, client, jcache, kept_new, RJ, RJU):
    for mode, cases in by_mode.items():
        # cache check per case; only query uncached
        need = [(key, c) for (key, c) in cases if f"{mode}|{c['id']}" not in jcache]
        if mode == "AFFIRMATIVE":
            p1 = _run(need, lambda b: RJ.prompt_entity_pass(b, RJ.P1_FOCUS, True), client, RJ)
            p2 = _run(need, lambda b: RJ.prompt_entity_pass(b, RJ.P2_FOCUS, True), client, RJ)
            for (key, c) in need:
                jcache[f"{mode}|{c['id']}"] = bool(p1.get(c["id"]) and p2.get(c["id"]))
        elif mode == "CONTRAST":
            r = _run(need, RJU.prompt_contrast, client, RJ)
            for (key, c) in need:
                jcache[f"{mode}|{c['id']}"] = bool(r.get(c["id"]))
        elif mode == "IMPLICIT":
            r = _run(need, RJU.prompt_context, client, RJ)
            for (key, c) in need:
                jcache[f"{mode}|{c['id']}"] = bool(r.get(c["id"]))
        elif mode == "ANAPHORA":
            r = _run(need, RJ.prompt_coref_pass, client, RJ)
            for (key, c) in need:
                jcache[f"{mode}|{c['id']}"] = bool(r.get(c["id"]))
        else:                                          # CODEPATH -> reject
            for (key, c) in need:
                jcache[f"{mode}|{c['id']}"] = False
        for (key, c) in cases:
            if jcache.get(f"{mode}|{c['id']}"):
                kept_new[project].add(key)


def _run(cases, build, client, RJ):
    """batched judge over case dicts -> {case_id: bool}."""
    out = {}
    for k in range(0, len(cases), RJ.BATCH):
        sub = [c for (_key, c) in cases[k:k + RJ.BATCH]]
        if not sub:
            continue
        idxsub = list(enumerate(sub, start=1))
        verd = RJ.parse(client.query(build(idxsub), timeout=180).text)
        for i, c in enumerate(sub, start=1):
            out[c["id"]] = bool(verd.get(i, False))
    return out


def live():
    base_per, golds = baseline()
    print("\nrunning GTP over the corpus (live, cached)...", file=sys.stderr)
    golds, augmented, kept_new, prop_in_union, proposals, union_final = run_gtp("name")
    aug_per = macro_over_runs(augmented, golds)

    print("\n" + "=" * 78)
    print("GTP-AUGMENTED (s21 UNION grounded/typed/judged proposals) — gpt-5.4, reasoning-off")
    print("=" * 78)
    for run, (P, R, F1) in zip(RUNS, aug_per):
        print(f"  {run}: P={P:.4f} R={R:.4f} F1={F1:.4f}")
    bP, bR, bF = _avg(base_per, 0), _avg(base_per, 1), _avg(base_per, 2)
    aP, aR, aF = _avg(aug_per, 0), _avg(aug_per, 1), _avg(aug_per, 2)
    print(f"\n  {'':<10}{'P':>9}{'R':>9}{'F1':>9}")
    print(f"  {'baseline':<10}{bP:>9.4f}{bR:>9.4f}{bF:>9.4f}")
    print(f"  {'augmented':<10}{aP:>9.4f}{aR:>9.4f}{aF:>9.4f}")
    print(f"  {'DELTA':<10}{aP-bP:>+9.4f}{aR-bR:>+9.4f}{aF-bF:>+9.4f}")

    # marginal breakdown
    print("\nMarginal GTP contribution (kept_new = judged-keep, never in any s21 run):")
    print(f"  {'project':<14}{'kept_new':>9}{'new_TP':>8}{'new_FP':>8}{'cross_run+':>11}")
    tot_tp = tot_fp = 0
    for p in PROJECTS:
        kn = kept_new[p]
        tp = len(kn & golds[p]); fp = len(kn - golds[p])
        tot_tp += tp; tot_fp += fp
        cross = len(prop_in_union[p])
        print(f"  {p:<14}{len(kn):>9}{tp:>8}{fp:>8}{cross:>11}")
    print(f"  {'TOTAL':<14}{'':<9}{tot_tp:>8}{tot_fp:>8}")

    summary = {"baseline": {"P": bP, "R": bR, "F1": bF},
               "augmented": {"P": aP, "R": aR, "F1": aF},
               "delta_F1": aF - bF, "new_TP": tot_tp, "new_FP": tot_fp,
               "per_run_baseline": base_per, "per_run_augmented": aug_per}
    (HERE / "live_summary.json").write_text(json.dumps(summary, indent=1))
    print("\nwrote live_summary.json")


def main():
    if "--baseline-only" in sys.argv:
        baseline()
    else:
        live()


if __name__ == "__main__":
    main()
