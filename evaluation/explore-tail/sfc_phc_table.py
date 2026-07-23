#!/usr/bin/env python3
"""SFC / PHC distributed RQ2-style: systems x projects, mean of 3 runs, both tasks.
SFC = gold components with 0 correct links (silent failures).
PHC = touched components with 0 correct links (phantom components)."""
import sys, csv
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mini-src"))
import metrics as m

SOTA = Path("/mnt/hostshare/ardoco-home/sota-recovered-links")
PROJ = ["mediastore","teastore","teammates","bigbluebutton","jabref"]
SHORT = {"mediastore":"MS","teastore":"TS","teammates":"TM","bigbluebutton":"BBB","jabref":"JR"}
SYSTEMS = [
    ("approach S21","gpt-5.4_s21",["run1","run2","run3"]),
    ("Artemis","__artemis__",[None]),
    ("TransArC","__transarc__",[None]),
]
def path_for(slot,run,p,task):
    if slot=="__artemis__": return SOTA/(f"model-doc/artemis-{p}-gpt-5.4.csv" if task=="sad-sam" else f"doc-code/artemis-{p}-gpt-5.4.csv")
    if slot=="__transarc__": return SOTA/(f"model-doc/swattr-{p}.csv" if task=="sad-sam" else f"doc-code/transarc-{p}.csv")
    base="model-doc/aalinker" if task=="sad-sam" else "doc-code/aalinker-composed"
    return SOTA/f"{base}/{slot}/{run}/{p}.csv"

def sfc_phc(project,res,task):
    if task=="sad-sam":
        gold=m.load_gs_sad_sam(project); gb=defaultdict(set); rb=defaultdict(set)
        for c,s in gold: gb[c].add(s)
        for c,s in res:  rb[c].add(s)
    else:
        cf=m.load_code_model_files(project); gold=m.enroll(m.load_gs_sad_code_raw(project),cf)
        f2c=m.load_file_to_comps(project,cf)
        def tc(pairs):
            o=defaultdict(set)
            for s,c in pairs:
                for comp in f2c.get(c,()): o[comp].add(s)
            return o
        gb=tc(gold); rb=tc(res)
    G=set(gb); P=set(rb); Rplus={c for c in G if gb[c]&rb.get(c,set())}
    return len(G)-len(Rplus), len(P)-len(Rplus), len(G)

def build(task):
    ncomp={}; rows=[]
    for name,slot,runs in SYSTEMS:
        sfc={}; phc={}
        for p in PROJ:
            S=[]; H=[]
            for run in runs:
                res=m.load_result(path_for(slot,run,p,task),task)
                if not res: continue
                s,h,n=sfc_phc(p,res,task); S.append(s); H.append(h); ncomp[p]=n
            sfc[p]=sum(S)/len(S) if S else float('nan')
            phc[p]=sum(H)/len(H) if H else float('nan')
        rows.append((name,sfc,phc))
    return rows, ncomp

def fmt(x): return f"{x:.2f}".rstrip('0').rstrip('.') if x==x else "—"

def show(task,label):
    rows,ncomp=build(task)
    tot=sum(ncomp[p] for p in PROJ)
    print(f"\n================ {label}  (component universe per project: "
          + " ".join(f"{SHORT[p]}={ncomp[p]}" for p in PROJ) + f"; ΣG={tot}) ================")
    for metric,idx in (("SFC  (silent-failure count)",1),("PHC  (phantom-component count)",2)):
        print(f"\n  {metric}")
        print(f"  {'system':<22}"+"".join(f"{SHORT[p]:>6}" for p in PROJ)+f"{'Σ':>7}")
        print("  "+"-"*(22+6*5+7))
        for r in rows:
            d=r[idx]; tot_s=sum(d[p] for p in PROJ if d[p]==d[p])
            print(f"  {r[0]:<22}"+"".join(f"{fmt(d[p]):>6}" for p in PROJ)+f"{fmt(tot_s):>7}")
    # CSV
    out=Path(__file__).resolve().parent/"reports"/f"sfc_phc_{task}.csv"
    with open(out,"w",newline="") as f:
        w=csv.writer(f,lineterminator="\n"); w.writerow(["metric","system"]+[SHORT[p] for p in PROJ]+["total"])
        for metric,idx in (("SFC",1),("PHC",2)):
            for r in rows:
                d=r[idx]; w.writerow([metric,r[0]]+[f"{d[p]:.3f}" for p in PROJ]+[f"{sum(d[p] for p in PROJ if d[p]==d[p]):.3f}"])

show("sad-sam","DOC-MODEL (SAD-SAM)")
show("sad-code","DOC-CODE (composed)")
print("\n[wrote reports/sfc_phc_{sad-sam,sad-code}.csv]  (approach rows = mean of 3 runs; baselines single)")
