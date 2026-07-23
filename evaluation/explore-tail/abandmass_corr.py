#!/usr/bin/env python3
"""Correlation of AbandMass% to link-F1 (the independence check). 30 cells
(6 systems x 5 projects), both tasks. SFC, WeakMass, worst/harmonic for contrast.
AbandMass/SFC/WeakMass are 'higher=worse' so a NEGATIVE rho vs link-F1 is expected;
|rho| near 0 = independent."""
import sys, math
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mini-src"))
import metrics as m
from rq2_corr import spearman

SOTA = Path("/mnt/hostshare/ardoco-home/sota-recovered-links")
PROJ = ["mediastore","teastore","teammates","bigbluebutton","jabref"]
SYS = [("gpt-5.4_s21",["run1","run2","run3"]),("sonnet_s21",["run1","run2","run3"]),
       ("gpt-5.4_full",["run1","run2","run3"]),("sonnet_full",["run1","run2","run3"]),
       ("__artemis__",[None]),("__transarc__",[None])]
def path_for(slot,run,p,task):
    if slot=="__artemis__": return SOTA/(f"model-doc/artemis-{p}-gpt-5.4.csv" if task=="sad-sam" else f"doc-code/artemis-{p}-gpt-5.4.csv")
    if slot=="__transarc__": return SOTA/(f"model-doc/swattr-{p}.csv" if task=="sad-sam" else f"doc-code/transarc-{p}.csv")
    base="model-doc/aalinker" if task=="sad-sam" else "doc-code/aalinker-composed"
    return SOTA/f"{base}/{slot}/{run}/{p}.csv"

def cell(project,res,task):
    if task=="sad-sam":
        gold={(c,s) for (c,s) in m.load_gs_sad_sam(project)}; res_c={(c,s) for (c,s) in res}
        link=m.prf(m.load_gs_sad_sam(project),res)[2]
    else:
        cf=m.load_code_model_files(project); goldf=m.enroll(m.load_gs_sad_code_raw(project),cf)
        f2c=m.load_file_to_comps(project,cf); link=m.prf(goldf,res)[2]
        def tc(pairs):
            o=set()
            for s,c in pairs:
                for comp in f2c.get(c,()): o.add((comp,s))
            return o
        gold=tc(goldf); res_c=tc(res)
    gb=defaultdict(set)
    for c,s in gold: gb[c].add(s)
    corr=defaultdict(set)
    for c,s in (gold & res_c): corr[c].add(s)
    tot=sum(len(v) for v in gb.values())
    rec={c:len(corr.get(c,set()))/len(gb[c]) for c in gb}
    aband=sum(len(gb[c]) for c in gb if rec[c]==0)/tot*100
    weak =sum(len(gb[c])*(1-rec[c]) for c in gb)/tot*100
    sfc=sum(1 for c in rec if rec[c]==0)
    per=list(rec.values())
    worst=min(per); harm=(len(per)/sum(1/x for x in per)) if all(x>0 for x in per) else 0.0
    return link,aband,weak,sfc,worst,harm

for task in ("sad-sam","sad-code"):
    L=[];A=[];W=[];S=[];Wc=[];H=[]
    for slot,runs in SYS:
        for p in PROJ:
            a=defaultdict(list)
            for run in runs:
                res=m.load_result(path_for(slot,run,p,task),task)
                if not res: continue
                link,ab,wk,sfc,wo,ha=cell(p,res,task)
                for k,v in zip("LAWSWcH".split() if False else ['L','A','W','S','Wc','H'],[link,ab,wk,sfc,wo,ha]): a[k].append(v)
            if not a['L']: continue
            L.append(sum(a['L'])/len(a['L'])); A.append(sum(a['A'])/len(a['A']))
            W.append(sum(a['W'])/len(a['W'])); S.append(sum(a['S'])/len(a['S']))
            Wc.append(sum(a['Wc'])/len(a['Wc'])); H.append(sum(a['H'])/len(a['H']))
    print(f"\n=== {task} ({len(L)} cells) — Spearman rho vs link-F1 ===")
    print(f"  AbandMass%   : {spearman(L,A):+.2f}   (independent target)")
    print(f"  SFC (count)  : {spearman(L,S):+.2f}")
    print(f"  WeakMass%    : {spearman(L,W):+.2f}   (=1-massRecall; should be ~ -1, redundant)")
    print(f"  worst-comp F1: {spearman(L,Wc):+.2f}   (magnitude tail)")
    print(f"  harmonic   F1: {spearman(L,H):+.2f}")
