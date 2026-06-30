#!/usr/bin/env python3
"""Evaluate the router + direct route against the canonical doc-code gold.

Answers the open feasibility question: does adding the DIRECT route raise recall
without wrecking precision? Reports, per backend slot, macro-averaged over the 5
projects and 3 runs:

  baseline   : transitive only (composed, as shipped)
  +direct/rule : transitive UNION direct, router = rule_route over ALL sentences
                 (fully offline; includes FP exposure from non-gold sentences)
  +direct/llm  : transitive UNION direct, router = cached gpt-5.4 decisions
                 (gold sentences only; the LLM-router config from the pilot)

Plus: direct-linker standalone precision, new TP added, new FP added.

Matching == paper: file-level enrolled gold via mini-src/metrics.py.
"""
import csv, glob, importlib.util, json, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
ARDOCO_HOME = Path("/mnt/hostshare/ardoco-home")
MINI = ARDOCO_HOME / "mono/evaluation/mini-src/metrics.py"
spec = importlib.util.spec_from_file_location("metrics", MINI)
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)

from llm_sad_sam.linkers.experimental.router_direct import (
    load_code_units, CodeIndex, DirectCodeLinker, rule_route, augment_doc_code, CODE)

REC = ARDOCO_HOME / "sota/recovered-links"
BENCH = M.BENCHMARK
PROJECTS = M.PROJECTS
RUNS = ["run1", "run2", "run3"]
SLOT = "gpt-5.4_s21"
ROUTER_CACHE = Path("/tmp/claude-1001/-mnt-hostshare-ardoco-home-mono/"
                    "137c09cf-a9bc-44df-87a7-a81672c330e4/scratchpad/router_cache.json")

ACM = {p: BENCH / M.ACM_FILES[p] for p in PROJECTS}


def sentences(project):
    h = glob.glob(str(BENCH / project / "text_*" / f"{project}.txt"))
    d = {}
    if h:
        for i, l in enumerate(open(h[0], errors="replace"), 1):
            d[str(i)] = l.strip()
    return d


def transitive(project, run):
    p = REC / "doc-code/aalinker-composed" / SLOT / run / f"{project}.csv"
    return M.load_result(p, "sad-code")          # set[(sentence, normpath)]


def prf(gold, res):
    if not res:
        return 0.0, 0.0, 0.0
    tp = len(gold & res)
    p = tp / len(res); r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    cache = json.loads(ROUTER_CACHE.read_text()) if ROUTER_CACHE.exists() else {}
    # per project: prepared once
    prep = {}
    for proj in PROJECTS:
        units = load_code_units(ACM[proj])
        idx = CodeIndex(units)
        dl = DirectCodeLinker(idx, include_test=True, max_files_per_package=None)
        code_files = M.load_code_model_files(proj)
        gold = M.enroll(M.load_gs_sad_code_raw(proj), code_files)
        sents = sentences(proj)
        # router maps (restricted to text-available sentences)
        rule_map = {sid: rule_route(t, dl) for sid, t in sents.items()}
        llm_map = {sid: (CODE if cache.get(f"{proj}:{sid}", {}).get("route") == CODE else "ARCH")
                   for sid in sents}
        prep[proj] = dict(idx=idx, dl=dl, gold=gold, sents=sents,
                          rule=rule_map, llm=llm_map)

    def eval_config(label, route_key=None):
        rows = []
        dnew_tp = dnew_fp = demit = dhit = 0
        for proj in PROJECTS:
            pr = prep[proj]; gold = pr["gold"]; sents = pr["sents"]; dl = pr["dl"]
            route = pr[route_key] if route_key else None
            pp = rr = ff = 0.0
            for run in RUNS:
                base = transitive(proj, run)
                if route_key:
                    aug = augment_doc_code(base, sents, dl, route)
                    # direct-only contribution diagnostics (run-invariant; count once on run1)
                    if run == "run1":
                        direct = set()
                        for sid, t in sents.items():
                            if route.get(sid) == CODE:
                                for path in dl.link_sentence(t):
                                    direct.add((sid, path))
                        demit += len(direct); dhit += len(direct & gold)
                        dnew_tp += len((direct & gold) - base)
                        dnew_fp += len((direct - gold) - base)
                else:
                    aug = base
                p, r, f = prf(gold, aug)
                pp += p; rr += r; ff += f
            rows.append((proj, pp/3, rr/3, ff/3))
        macro = (sum(x[1] for x in rows)/5, sum(x[2] for x in rows)/5, sum(x[3] for x in rows)/5)
        print(f"\n[{label}]  macro  P={macro[0]:.4f}  R={macro[1]:.4f}  F1={macro[2]:.4f}")
        for proj, p, r, f in rows:
            print(f"    {proj:<14} P={p:.4f} R={r:.4f} F1={f:.4f}")
        if route_key:
            prec = dhit/demit if demit else 0.0
            print(f"    direct-linker: emitted={demit} gold-hits={dhit} "
                  f"precision={prec:.3f} | NEW TP added={dnew_tp} NEW FP added={dnew_fp}")
        return macro

    print("=" * 70); print(f"SLOT = {SLOT}  (file-level, macro over 5 projects x 3 runs)")
    print("=" * 70)
    base = eval_config("baseline transitive")
    rule = eval_config("+direct  (router = rule, ALL sentences)", "rule")
    llm = eval_config("+direct  (router = LLM cached, gold sentences)", "llm")
    print("\n" + "=" * 70)
    print(f"Delta R  rule: {rule[1]-base[1]:+.4f}   llm: {llm[1]-base[1]:+.4f}")
    print(f"Delta P  rule: {rule[0]-base[0]:+.4f}   llm: {llm[0]-base[0]:+.4f}")
    print(f"Delta F1 rule: {rule[2]-base[2]:+.4f}   llm: {llm[2]-base[2]:+.4f}")


if __name__ == "__main__":
    main()
