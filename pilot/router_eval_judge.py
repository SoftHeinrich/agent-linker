#!/usr/bin/env python3
"""Add a JUDGE to the direct route (mirrors s_linker21's validation pass) and
measure precision recovery.

Pipeline: direct-link every sentence that resolves an identifier (the rule-route
universe), then an LLM judge keeps/rejects each (sentence, identifier) candidate
via claim-before-verdict. Judged direct links = union of kept candidates' paths,
then UNION with transitive. The judge subsumes the router as the precision gate.

Compares, macro over 5 projects x 3 runs (gpt-5.4_s21):
  baseline transitive | + direct (no judge) | + direct + judge
Reports direct-linker precision before/after the judge and how many FP/TP it drops.
"""
import glob, importlib.util, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve(); REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
ARDOCO_HOME = Path("/mnt/hostshare/ardoco-home")
MINI = ARDOCO_HOME / "mono/evaluation/mini-src/metrics.py"
spec = importlib.util.spec_from_file_location("metrics", MINI)
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
from llm_sad_sam.linkers.experimental.router_direct import (
    load_code_units, CodeIndex, DirectCodeLinker, DirectLinkJudge)

REC = ARDOCO_HOME / "sota/recovered-links"; BENCH = M.BENCHMARK
PROJECTS = M.PROJECTS; RUNS = ["run1", "run2", "run3"]; SLOT = "gpt-5.4_s21"
SCRATCH = Path("/tmp/claude-1001/-mnt-hostshare-ardoco-home-mono/"
               "137c09cf-a9bc-44df-87a7-a81672c330e4/scratchpad")
JUDGE_CACHE = SCRATCH / "judge_cache_v2.json"
ACM = {p: BENCH / M.ACM_FILES[p] for p in PROJECTS}


def sentences(project):
    h = glob.glob(str(BENCH / project / "text_*" / f"{project}.txt")); d = {}
    if h:
        for i, l in enumerate(open(h[0], errors="replace"), 1):
            if l.strip():
                d[str(i)] = l.strip()
    return d


def prf(gold, res):
    if not res:
        return 0.0, 0.0, 0.0
    tp = len(gold & res); p = tp / len(res); r = tp / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def transitive(project, run):
    return M.load_result(REC / "doc-code/aalinker-composed" / SLOT / run / f"{project}.csv", "sad-code")


def main():
    # 1) collect all direct candidates across the corpus
    prep = {}; all_cases = []   # global judge cases: (proj, sid, ident, kind, paths)
    for proj in PROJECTS:
        idx = CodeIndex(load_code_units(ACM[proj]))
        dl = DirectCodeLinker(idx, include_test=True)
        gold = M.enroll(M.load_gs_sad_code_raw(proj), M.load_code_model_files(proj))
        sents = sentences(proj)
        cands = []   # (sid, ident, kind, frozenset paths)
        for sid, text in sents.items():
            for ident, kind, paths in dl.candidates(text):
                cands.append((sid, ident, kind, paths))
                all_cases.append((proj, sid, ident, kind, text))
        prep[proj] = dict(gold=gold, sents=sents, cands=cands)

    # 2) judge (cached) -- key = proj|sid|ident|kind
    cache = json.loads(JUDGE_CACHE.read_text()) if JUDGE_CACHE.exists() else {}
    todo = [(i, c) for i, c in enumerate(all_cases)
            if f"{c[0]}|{c[1]}|{c[2]}|{c[3]}" not in cache]
    if todo:
        print(f"  judging {len(todo)} candidates via gpt-5.4 ...", file=sys.stderr)
        os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.4")
        judge = DirectLinkJudge(model="gpt-5.4", batch=10)
        STEP = 30
        for k in range(0, len(todo), STEP):
            sub = todo[k:k + STEP]
            verdicts = judge.judge([{"text": c[4], "identifier": c[2], "kind": c[3]}
                                    for _, c in sub])
            for j, (_, c) in enumerate(sub):
                cache[f"{c[0]}|{c[1]}|{c[2]}|{c[3]}"] = bool(verdicts.get(j, True))
            JUDGE_CACHE.write_text(json.dumps(cache))
            print(f"    {min(k+STEP,len(todo))}/{len(todo)} judged", file=sys.stderr)

    def kept(proj, sid, ident, kind):
        return cache.get(f"{proj}|{sid}|{ident}|{kind}", True)

    # 3) build direct link sets (no-judge and judged) per project
    for proj in PROJECTS:
        pr = prep[proj]
        nj, jd = set(), set()
        for sid, ident, kind, paths in pr["cands"]:
            for p in paths:
                nj.add((sid, p))
                if kept(proj, sid, ident, kind):
                    jd.add((sid, p))
        pr["direct_nojudge"], pr["direct_judged"] = nj, jd

    def run_cfg(label, dkey=None):
        rows = []; emit = hit = ntp = nfp = 0
        for proj in PROJECTS:
            pr = prep[proj]; gold = pr["gold"]
            direct = pr[dkey] if dkey else set()
            pp = rr = ff = 0.0
            for run in RUNS:
                base = transitive(proj, run)
                aug = (base | direct) if dkey else base
                p, r, f = prf(gold, aug); pp += p; rr += r; ff += f
            rows.append((proj, pp/3, rr/3, ff/3))
            if dkey:
                b1 = transitive(proj, "run1")
                emit += len(direct); hit += len(direct & gold)
                ntp += len((direct & gold) - b1); nfp += len((direct - gold) - b1)
        macro = tuple(sum(x[i] for x in rows)/5 for i in (1, 2, 3))
        print(f"\n[{label}]  macro  P={macro[0]:.4f}  R={macro[1]:.4f}  F1={macro[2]:.4f}")
        for proj, p, r, f in rows:
            print(f"    {proj:<14} P={p:.4f} R={r:.4f} F1={f:.4f}")
        if dkey:
            print(f"    direct: emitted={emit} hits={hit} precision={hit/emit if emit else 0:.3f}"
                  f"  NEW TP=+{ntp}  NEW FP=+{nfp}")
        return macro

    kept_n = sum(1 for c in all_cases if kept(c[0], c[1], c[2], c[3]))
    print("=" * 72)
    print(f"SLOT={SLOT}  direct + JUDGE   (macro 5proj x 3runs)")
    print(f"candidates judged: {len(all_cases)}  kept: {kept_n}  rejected: {len(all_cases)-kept_n}")
    print("=" * 72)
    b = run_cfg("baseline transitive")
    nj = run_cfg("+direct (no judge)", "direct_nojudge")
    jd = run_cfg("+direct + JUDGE", "direct_judged")
    print("\n" + "=" * 72)
    print(f"Delta vs baseline   no-judge: R{nj[1]-b[1]:+.4f} P{nj[0]-b[0]:+.4f} F1{nj[2]-b[2]:+.4f}")
    print(f"                    +judge  : R{jd[1]-b[1]:+.4f} P{jd[0]-b[0]:+.4f} F1{jd[2]-b[2]:+.4f}")


if __name__ == "__main__":
    main()
