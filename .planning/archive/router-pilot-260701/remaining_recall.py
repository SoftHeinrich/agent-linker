#!/usr/bin/env python3
"""Empirically categorize the REMAINING doc-code recall gaps after the best
config (transitive UNION direct-judged), gpt-5.4_s21.

The composed pipeline has three stages, all sharing one architecture-element id
space (verified): model-doc (sentence->component) o ArCoTL bridge (component->
code), plus the new direct route (sentence->code). Each remaining miss is
attributed to the stage that failed:

  BRIDGE GAP        model-doc found the component, but ArCoTL never maps it to
                    the gold file(s) -> component->code recall hole.
  MODEL-DOC (impl)  model-doc missed the sentence->component link AND the sentence
                    names no code identifier -> implicit/functional; cascades,
                    direct route cannot help.
  MODEL-DOC (dir?)  model-doc missed it BUT the sentence names a code identifier
                    -> direct route had a shot and still missed (naming drift /
                    unresolved).
  NO-COMPONENT      gold file owned by no (non-interface) SAM component.

Reported at the (sentence, component) level (honest unit; file level is
enrolment-inflated) with file-level totals alongside.
"""
import glob, importlib.util, json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "src")
ARD = Path("/mnt/hostshare/ardoco-home"); MINI = ARD/"mono/evaluation/mini-src/metrics.py"
spec = importlib.util.spec_from_file_location("metrics", MINI)
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
from llm_sad_sam.linkers.experimental.router_direct import (
    load_code_units, CodeIndex, DirectCodeLinker)

REC = ARD/"sota/recovered-links"; BENCH = M.BENCHMARK
PROJECTS = M.PROJECTS; RUNS = ["run1", "run2", "run3"]; SLOT = "gpt-5.4_s21"
judge_cache = json.loads(
    (Path(__file__).resolve().parent / "cache" / "judge_cache_v2.json").read_text())
ACM = {p: BENCH/M.ACM_FILES[p] for p in PROJECTS}


def sents(p):
    h = glob.glob(str(BENCH/p/"text_*"/f"{p}.txt")); d = {}
    for i, l in enumerate(open(h[0], errors="replace"), 1):
        if l.strip():
            d[str(i)] = l.strip()
    return d


def load_pairs(path, a, b):
    out = set()
    import csv
    for r in csv.DictReader(open(path)):
        out.add((r[a], r[b]))
    return out


def main():
    catsc = defaultdict(int)      # (sentence,component) level
    catfile = defaultdict(int)    # file level
    examples = defaultdict(list)
    for proj in PROJECTS:
        names = {c: n for (c, n) in []}  # filled below
        S = sents(proj)
        code_files = M.load_code_model_files(proj)
        gold = M.enroll(M.load_gs_sad_code_raw(proj), code_files)
        f2c = M.load_file_to_comps(proj, code_files)        # file -> {ae_id}
        # ae_id -> name (from sam-code gold)
        import csv
        for r in csv.DictReader(open(REC/"model-code/gold"/f"{proj}.raw.csv")):
            pass
        aen = {}
        with open(BENCH/M.GS_SAM_CODE[proj]) as f:
            for r in csv.DictReader(f):
                aen[r["ae_id"]] = r["ae_name"]

        # model-doc recovered union (comp_id, sentence)
        md_union = set()
        for run in RUNS:
            md_union |= load_pairs(REC/"model-doc/aalinker"/SLOT/run/f"{proj}.csv",
                                   "target_id", "sentence_id")
        md_comps_by_s = defaultdict(set)
        for c, s in md_union:
            md_comps_by_s[s].add(c)
        # ArCoTL bridge comp_id -> files
        bridge = defaultdict(set)
        for src, tgt in load_pairs(REC/"model-code/arcotl"/f"{proj}.csv", "source_id", "target_id"):
            bridge[src].add(tgt)
        # transitive recovered union (sentence, file)
        trans = set()
        for run in RUNS:
            trans |= M.load_result(REC/"doc-code/aalinker-composed"/SLOT/run/f"{proj}.csv", "sad-code")
        # direct judged links
        idx = CodeIndex(load_code_units(ACM[proj])); dl = DirectCodeLinker(idx)
        direct = set(); has_codeid = {}
        for sid, t in S.items():
            cand = dl.candidates(t); has_codeid[sid] = bool(cand)
            for ident, kind, paths in cand:
                if judge_cache.get(f"{proj}|{sid}|{ident}|{kind}", True):
                    for fp in paths:
                        direct.add((sid, fp))
        recovered = trans | direct

        # gold (sentence, component) -> gold files
        gold_sc = defaultdict(set)
        for s, fp in gold:
            for c in f2c.get(fp, ()):
                gold_sc[(s, c)].add(fp)
            if not f2c.get(fp):
                if (s, fp) not in recovered:
                    catfile["NO-COMPONENT"] += 1

        for (s, c), gfiles in gold_sc.items():
            rec_files = {fp for fp in gfiles if (s, fp) in recovered}
            missed_files = gfiles - rec_files
            if not missed_files:
                continue
            # classify this (s,c)
            md_hit = (c, s) in md_union
            if md_hit:
                if bridge.get(c, set()) & gfiles:
                    cat = "ANOMALY (run-variance)"
                else:
                    cat = "BRIDGE GAP"
            else:
                cat = ("MODEL-DOC (direct-addressable)" if has_codeid.get(s)
                       else "MODEL-DOC (implicit)")
            if not rec_files:                 # fully-missed (s,component)
                catsc[cat] += 1
                if len(examples[cat]) < 6:
                    examples[cat].append(f"{proj} s{s}->{aen.get(c,c)}: {S.get(s,'')[:74]}")
            catfile[cat] += len(missed_files)

    print("=" * 74)
    print("REMAINING doc-code recall gaps  (best config: transitive U direct-judged)")
    print("=" * 74)
    print(f"\n{'category':<34}{'(sent,comp) misses':>20}{'file-FN':>10}")
    order = ["BRIDGE GAP", "MODEL-DOC (implicit)", "MODEL-DOC (direct-addressable)",
             "ANOMALY (run-variance)", "NO-COMPONENT"]
    for k in order:
        print(f"  {k:<32}{catsc.get(k,0):>20}{catfile.get(k,0):>10}")
    print(f"  {'TOTAL':<32}{sum(catsc.values()):>20}{sum(catfile.values()):>10}")
    for k in order:
        if examples.get(k):
            print(f"\n--- {k} ---")
            for e in examples[k]:
                print(f"    {e}")


if __name__ == "__main__":
    main()
