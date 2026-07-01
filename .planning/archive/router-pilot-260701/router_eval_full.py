#!/usr/bin/env python3
"""Close the FP gap: run the LLM router over FULL documents (all sentences),
not just gold-bearing ones, and re-score doc-code end-to-end.

The earlier LLM-router eval forced every non-gold sentence to ARCH, hiding the
router's false-positive exposure on sentences that have no doc-code gold link.
Here the router decides for ALL sentences; a CODE call on a no-gold sentence that
the direct linker then resolves is a genuine FP. Reuses the gold-sentence cache;
only calls gpt-5.4 on the remaining sentences.
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
    load_code_units, CodeIndex, DirectCodeLinker, rule_route, augment_doc_code,
    SentenceRouter, CODE, ARCH)

REC = ARDOCO_HOME / "sota/recovered-links"; BENCH = M.BENCHMARK
PROJECTS = M.PROJECTS; RUNS = ["run1", "run2", "run3"]; SLOT = "gpt-5.4_s21"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
GOLD_CACHE = CACHE_DIR / "router_cache.json"
FULL_CACHE = CACHE_DIR / "router_cache_full.json"
ACM = {p: BENCH / M.ACM_FILES[p] for p in PROJECTS}


def sentences(project):
    h = glob.glob(str(BENCH / project / "text_*" / f"{project}.txt"))
    d = {}
    if h:
        for i, l in enumerate(open(h[0], errors="replace"), 1):
            t = l.strip()
            if t:
                d[str(i)] = t
    return d


def prf(gold, res):
    if not res:
        return 0.0, 0.0, 0.0
    tp = len(gold & res); p = tp / len(res); r = tp / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def build_full_router_map():
    """proj -> {sid: ARCH|CODE} for ALL sentences, using cache + gpt-5.4."""
    cache = json.loads(FULL_CACHE.read_text()) if FULL_CACHE.exists() else {}
    if not cache and GOLD_CACHE.exists():          # seed from gold-sentence cache
        for k, v in json.loads(GOLD_CACHE.read_text()).items():
            cache[k] = {"route": v.get("route", ARCH)}
    all_sents = {p: sentences(p) for p in PROJECTS}
    missing = {f"{p}:{s}": all_sents[p][s] for p in PROJECTS for s in all_sents[p]
               if f"{p}:{s}" not in cache}
    if missing:
        print(f"  routing {len(missing)} uncached sentences via gpt-5.4 ...", file=sys.stderr)
        os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.4")
        router = SentenceRouter(model="gpt-5.4", batch=12)
        ids = list(missing)
        for k in range(0, len(ids), 36):           # checkpoint every 3 batches
            sub = {gid: missing[gid] for gid in ids[k:k + 36]}
            dec = router.route(sub)
            for gid, r in dec.items():
                cache[gid] = {"route": r}
            FULL_CACHE.write_text(json.dumps(cache))
            print(f"    {min(k+36,len(ids))}/{len(ids)} done", file=sys.stderr)
    rmap = {p: {} for p in PROJECTS}
    for p in PROJECTS:
        for s in all_sents[p]:
            rmap[p][s] = CODE if cache.get(f"{p}:{s}", {}).get("route") == CODE else ARCH
    return rmap, all_sents


def transitive(project, run):
    return M.load_result(REC / "doc-code/aalinker-composed" / SLOT / run / f"{project}.csv", "sad-code")


def main():
    llm_full, all_sents = build_full_router_map()
    prep = {}
    for proj in PROJECTS:
        idx = CodeIndex(load_code_units(ACM[proj]))
        dl = DirectCodeLinker(idx, include_test=True)
        gold = M.enroll(M.load_gs_sad_code_raw(proj), M.load_code_model_files(proj))
        sents = all_sents[proj]
        rule_map = {s: rule_route(t, dl) for s, t in sents.items()}
        prep[proj] = dict(dl=dl, gold=gold, sents=sents, rule=rule_map, llm=llm_full[proj])

    def run_cfg(label, key=None):
        rows = []; emit = hit = ntp = nfp = nfp_nogold = code_n = 0
        for proj in PROJECTS:
            pr = prep[proj]; gold = pr["gold"]; sents = pr["sents"]; dl = pr["dl"]
            route = pr[key] if key else {}
            pp = rr = ff = 0.0
            for run in RUNS:
                base = transitive(proj, run)
                aug = augment_doc_code(base, sents, dl, route) if key else base
                p, r, f = prf(gold, aug); pp += p; rr += r; ff += f
            rows.append((proj, pp/3, rr/3, ff/3))
            if key:
                direct = set()
                for s, t in sents.items():
                    if route.get(s) == CODE:
                        code_n += 1
                        for path in dl.link_sentence(t):
                            direct.add((s, path))
                gold_sents = {gs for gs, _ in gold}
                emit += len(direct); hit += len(direct & gold)
                b1 = transitive(proj, "run1")
                ntp += len((direct & gold) - b1)
                nfp += len((direct - gold) - b1)
                nfp_nogold += len({(s, p) for (s, p) in (direct - gold) - b1 if s not in gold_sents})
        macro = tuple(sum(x[i] for x in rows)/5 for i in (1, 2, 3))
        print(f"\n[{label}]  macro  P={macro[0]:.4f}  R={macro[1]:.4f}  F1={macro[2]:.4f}")
        for proj, p, r, f in rows:
            print(f"    {proj:<14} P={p:.4f} R={r:.4f} F1={f:.4f}")
        if key:
            print(f"    CODE-routed sentences={code_n}  direct emitted={emit} hits={hit} "
                  f"precision={hit/emit if emit else 0:.3f}")
            print(f"    NEW TP=+{ntp}  NEW FP=+{nfp}  (of which from NO-GOLD sentences=+{nfp_nogold})")
        return macro

    print("=" * 72); print(f"SLOT={SLOT}  full-document routing  (macro 5proj x 3runs)"); print("=" * 72)
    b = run_cfg("baseline transitive")
    rl = run_cfg("+direct  rule router (all sentences)", "rule")
    lf = run_cfg("+direct  LLM router (ALL sentences)", "llm")
    print("\n" + "=" * 72)
    print(f"Delta vs baseline   rule: R{rl[1]-b[1]:+.4f} P{rl[0]-b[0]:+.4f} F1{rl[2]-b[2]:+.4f}")
    print(f"                    LLM : R{lf[1]-b[1]:+.4f} P{lf[0]-b[0]:+.4f} F1{lf[2]-b[2]:+.4f}")


if __name__ == "__main__":
    main()
