#!/usr/bin/env python3
"""Router recall CEILING: feed ALL false negatives to the router (perfect-proposer assumption).

If a proposer surfaced every gold FN and routed it through the judge-router, how many does the
router approve? Since every input is gold, approvals add TP with NO new FP -> precision holds,
recall rises. This is the ceiling the router+proposer path can reach.

Reports, macro over 5 proj x 3 runs:
  - baseline s21 P/R/F1
  - router-ceiling P/R/F1 (all per-run FN fed to router)
  - split: FN reachable NOW via the reject pool vs FN that need a proposer (never-proposed)
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


def prf(tp,fp,g):
    P=tp/(tp+fp) if tp+fp else 0; R=tp/g if g else 0
    return P,R,(2*P*R/(P+R) if P+R else 0)


def main():
    rc=json.loads((HERE/"router_cache.json").read_text())
    router=rc["verdict"]                                    # (proj|s|cid) -> approve
    gcache=json.loads((HERE/"grade_cache.json").read_text())
    routing=rc["routing"]; tags=gcache["tags"]
    LEN={"CONTRAST","IMPLICIT","ANAPHORA"}
    def rid(proj,s,c): return f"{proj}|{s}|{c}"
    def r_named(key):
        if routing.get(key) in LEN: return tags.get(key,"NONE")=="NAMED"
        return bool(router.get(key))

    agg={k:[0.0,0.0,0.0] for k in ("baseline","router_all","named_all")}
    fed=appr=appr_named=0
    reach_now=appr_now=needprop=appr_prop=0
    for proj in PROJECTS:
        G=gold(proj); ng=len(G)
        per={k:[0.0,0.0,0.0] for k in agg}
        for run in RUNS:
            d=json.load(open(EX/run/f"{proj}.json"))
            final={(l["s"],l["c"]) for l in d["final"]["links"]}
            ec={(e["s"],e["c"]) for e in d["entity"]["candidates"]}
            cc={(e["s"],e["c"]) for e in d["coref"]["raw"]}
            proposed=ec|cc
            fn=G-final                                     # this run's false negatives
            # feed ALL fn to router
            for k in fn:
                fed+=1
                key=rid(proj,k[0],k[1])
                a=bool(router.get(key)); an=r_named(key)
                appr+=a; appr_named+=an
                if k in proposed:  reach_now+=1; appr_now+=a     # already a (rejected) candidate
                else:              needprop+=1; appr_prop+=a     # never proposed
            add={k for k in fn if router.get(rid(proj,k[0],k[1]))}
            add_named={k for k in fn if r_named(rid(proj,k[0],k[1]))}
            tp0=len(final&G); fp0=len(final-G)
            for name,extra in (("baseline",set()),("router_all",add),("named_all",add_named)):
                nf=final|extra
                P,R,F=prf(len(nf&G),len(nf-G),ng)          # extra are all gold -> fp unchanged
                for i,x in enumerate((P,R,F)): per[name][i]+=x/3
        for k in agg:
            for i in range(3): agg[k][i]+=per[k][i]/5

    b=agg["baseline"]; r=agg["router_all"]; n=agg["named_all"]
    print("="*80)
    print("ROUTER RECALL CEILING — feed ALL false negatives to the judge-router")
    print("  (perfect-proposer assumption: every input is gold, so precision only holds/rises)")
    print("="*80)
    print(f"\n  {'config':<28}{'macro P':>10}{'macro R':>10}{'macro F1':>10}")
    print(f"  {'baseline s21':<28}{b[0]:>10.4f}{b[1]:>10.4f}{b[2]:>10.4f}")
    print(f"  {'router: all FN in (v1)':<28}{r[0]:>10.4f}{r[1]:>10.4f}{r[2]:>10.4f}"
          f"   dR {r[1]-b[1]:+.4f}  dF1 {r[2]-b[2]:+.4f}")
    print(f"  {'router: all FN in (NAMED)':<28}{n[0]:>10.4f}{n[1]:>10.4f}{n[2]:>10.4f}"
          f"   dR {n[1]-b[1]:+.4f}  dF1 {n[2]-b[2]:+.4f}")
    print(f"\n  FN fed to router (summed 5x3): {fed}")
    print(f"    router v1 approves : {appr}/{fed} = {100*appr/fed:.0f}%   (the recall potential)")
    print(f"    router NAMED-only  : {appr_named}/{fed} = {100*appr_named/fed:.0f}%")
    print(f"\n  Split of that potential by how the FN can REACH the router:")
    print(f"    reachable NOW (reject pool, already a candidate): approves {appr_now}/{reach_now}"
          f" = {100*appr_now/reach_now if reach_now else 0:.0f}%")
    print(f"    needs a PROPOSER (never proposed)              : approves {appr_prop}/{needprop}"
          f" = {100*appr_prop/needprop if needprop else 0:.0f}%")


if __name__ == "__main__":
    main()
