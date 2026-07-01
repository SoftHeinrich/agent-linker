#!/usr/bin/env python3
"""Concrete precision/recall impact of a 'second-chance re-judge' deployment.

Deployment simulated: run s21 as-is, then take every candidate its validator REJECTED and
re-judge it with approach X; add back the ones X approves. Score P/R/F1 vs the s21 baseline.
This is the honest reject-pool deployment (the 16 never-proposed FN are NOT in the reject pool,
so they are not recoverable this way — they need a proposer; shown separately as a ceiling).

Approaches: J0_amb (strict+context, one judge for all), J2_recover (global lenient),
router (LLM judge-router). Verdicts from verdicts.json / router_cache.json.
"""
import csv, json, os
from collections import defaultdict
from pathlib import Path

ARD = Path("/mnt/hostshare/ardoco-home")
BENCH = Path(os.environ.get("TRANSARC_BENCHMARK",
             ARD / "ardoco/core/tests-base/src/main/resources/benchmark"))
EX = ARD / "agent-linker/results/v2.6.6_extracts_s21/gpt"
PROJECTS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]
RUNS = ["run1", "run2", "run3"]
GS = {"mediastore":"mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
      "teastore":"teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
      "teammates":"teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
      "bigbluebutton":"bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
      "jabref":"jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv"}
HERE = Path(__file__).resolve().parent


def gold(p):
    g=set()
    for r in csv.DictReader(open(BENCH/GS[p])):
        g.add((int(r["sentence"]), r["modelElementID"]))
    return g


def cid(proj,s,c): return f"{proj}|{s}|{c}"


def load_verdicts():
    v=json.load(open(HERE/"verdicts.json"))
    rc=json.load(open(HERE/"router_cache.json"))
    r=rc["verdict"]; routing=rc["routing"]
    tags=json.load(open(HERE/"grade_cache.json"))["tags"]
    LEN={"CONTRAST","IMPLICIT","ANAPHORA"}
    def router_named(key):
        # router, but lenient routes keep only NAMED-evidence approvals
        if routing.get(key) in LEN:
            return tags.get(key,"NONE")=="NAMED"
        return bool(r.get(key))
    def get(app, key):
        if app=="router": return bool(r.get(key))
        if app=="router_named": return router_named(key)
        vv=v.get(f"{app}|{key}")
        if isinstance(vv,dict): return bool(vv.get("majority"))
        return bool(vv)
    return get


def prf(tp,fp,g):
    p=tp/(tp+fp) if tp+fp else 0; rc=tp/g if g else 0
    return p,rc,(2*p*rc/(p+rc) if p+rc else 0)


def main():
    get=load_verdicts()
    apps=["baseline","J0_amb","router_named","router","J2_recover"]
    # macro accumulators: per app -> [sumP,sumR,sumF] over projects, per run then mean
    agg={a:[0.0,0.0,0.0] for a in apps}
    addfp={a:0 for a in apps}; addtp={a:0 for a in apps}
    for proj in PROJECTS:
        G=gold(proj); ng=len(G)
        perrun={a:[0.0,0.0,0.0] for a in apps}
        for run in RUNS:
            d=json.load(open(EX/run/f"{proj}.json"))
            final={(l["s"],l["c"]) for l in d["final"]["links"]}
            ev={(e["s"],e["c"]) for e in d["entity"]["validated"]}
            cv={(e["s"],e["c"]) for e in d["coref"]["validated"]}
            rej=set()
            for e in d["entity"]["candidates"]:
                k=(e["s"],e["c"])
                if k not in ev: rej.add(k)
            for e in d["coref"]["raw"]:
                k=(e["s"],e["c"])
                if k not in cv: rej.add(k)
            rej-=final   # only things not already emitted by the other stage
            for a in apps:
                if a=="baseline":
                    nf=final
                else:
                    added={k for k in rej if get(a, cid(proj,k[0],k[1]))}
                    nf=final|added
                    addtp[a]+=len(added&G); addfp[a]+=len(added-G)
                tp=len(nf&G); fp=len(nf-G)
                p,rc,f=prf(tp,fp,ng)
                for i,x in enumerate((p,rc,f)): perrun[a][i]+=x
        for a in apps:
            for i in range(3): agg[a][i]+=perrun[a][i]/3   # mean over runs -> per project

    print("="*78)
    print("SECOND-CHANCE RE-JUDGE DEPLOYMENT — macro P/R/F1 (5 proj x 3 runs), reject-pool only")
    print("="*78)
    print(f"  {'approach':<14}{'macro P':>10}{'macro R':>10}{'macro F1':>10}   {'reject-pool added (TP/FP)':>26}")
    base=agg["baseline"]
    for a in apps:
        P,R,F=[agg[a][i]/5 for i in range(3)]
        d="" if a=="baseline" else f"  (dF1 {F-base[2]/5:+.4f})"
        added="" if a=="baseline" else f"   +{addtp[a]} TP / +{addfp[a]} FP"
        print(f"  {a:<14}{P:>10.4f}{R:>10.4f}{F:>10.4f}{added:>26}{d}")
    print("\nNotes:")
    print("  * 'reject-pool only': recovers the seen-and-rejected FN; the 16 never-proposed FN")
    print("    are NOT reachable here (they were never candidates) -> a separate proposer ceiling.")
    print("  * added TP/FP are summed over all 5 proj x 3 runs (the union re-judged each run).")


if __name__=="__main__":
    main()
