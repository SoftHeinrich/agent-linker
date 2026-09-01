#!/usr/bin/env python3
"""Head-to-head on DOC-MODEL: CMR vs worst-component F1. Which is the better
size-aware metric here? Axes: independence from link-F1, discrimination, sparsity/
resolution, run-stability, and do they agree on the ranking.

CMR (component miss rate) is the metric the paper reports; it was called SFM
(silent-failure mass) when this study ran. The rename is not the only change:
this script used to re-derive the quantity with a DISTINCT-SENTENCE denominator,
while the shipped CMR divides by gold (sentence, component) ASSIGNMENTS. On
projects where a sentence is documented for more than one component the old
figure was therefore inflated. It now calls ``metrics.compute_sad_sam``, the sole
metrics implementation, so the numbers below match the paper's. The verdict the
study reached is unchanged."""
import sys; from collections import defaultdict; from pathlib import Path
from _roots import MINI_SRC, SOTA    # shared roots (see _roots.py)
sys.path.insert(0, str(MINI_SRC))
import metrics as m
from metrics import spearman
PROJ=["mediastore","teastore","teammates","bigbluebutton","jabref"]
SYS=[("S21 GPT","gpt-5.4_s21",["run1","run2","run3"]),
     ("S21 Claude","sonnet_s21",["run1","run2","run3"]),
     ("s20u GPT","gpt-5.4_full",["run1","run2","run3"]),
     ("s20u Claude","sonnet_full",["run1","run2","run3"]),
     ("Artemis","__artemis__",[None]),("TransArC","__transarc__",[None])]
def pf(slot,run,p):
    if slot=="__artemis__": return SOTA/f"model-doc/artemis-{p}-gpt-5.4.csv"
    if slot=="__transarc__": return SOTA/f"model-doc/swattr-{p}.csv"
    return SOTA/f"model-doc/aalinker/{slot}/{run}/{p}.csv"
def cell(p,res):
    gold=m.load_gs_sad_sam(p); gb=defaultdict(set); rb=defaultdict(set)
    for c,s in gold: gb[c].add(s)
    for c,s in res:  rb[c].add(s)
    rec={c:len(gb[c]&rb.get(c,set()))/len(gb[c]) for c in gb}
    panel=m.compute_sad_sam(p,res)          # link P/R/F1+F2 and CMR%, one source
    link=panel["link_f1"]
    worst=min(rec.values())                 # recall-side worst comp: this study's contrast
    return link,worst,panel["component_miss_rate"]
rows=[]; macro=defaultdict(lambda:defaultdict(list)); runstab=defaultdict(lambda:defaultdict(list))
for name,slot,runs in SYS:
    for p in PROJ:
        L=[];W=[];S=[]
        for run in runs:
            res=m.load_result(pf(slot,run,p),"sad-sam")
            if not res: continue
            l,w,s=cell(p,res); L.append(l);W.append(w);S.append(s)
        if not L: continue
        rows.append((name,sum(L)/len(L),sum(W)/len(W),sum(S)/len(S)))
        macro[name]["link"].append(sum(L)/len(L)); macro[name]["worst"].append(sum(W)/len(W)); macro[name]["cmr"].append(sum(S)/len(S))
        if len(W)>1:  # run stability (approach rows)
            mu=sum(W)/len(W); runstab[name]["worst"].append((sum((x-mu)**2 for x in W)/len(W))**.5)
            mu2=sum(S)/len(S); runstab[name]["cmr"].append((sum((x-mu2)**2 for x in S)/len(S))**.5)
L=[r[1] for r in rows];W=[r[2] for r in rows];S=[r[3] for r in rows]
print("=== INDEPENDENCE (Spearman across %d cells) ==="%len(rows))
print(f"  worst-comp F1 vs link-F1 : {spearman(L,W):+.2f}   (high => redundant)")
print(f"  CMR           vs link-F1 : {spearman(L,S):+.2f}   (low  => adds info)")
print(f"  CMR vs worst-comp F1     : {spearman(W,S):+.2f}   (do they measure the same?)")
print("\n=== DISCRIMINATION (macro per system) ===")
print(f"  {'system':<12}{'link-F1':>9}{'worstF1':>9}{'CMR%':>8}")
for name in [s[0] for s in SYS]:
    d=macro[name]; a=lambda k:sum(d[k])/len(d[k])
    print(f"  {name:<12}{a('link'):>9.3f}{a('worst'):>9.3f}{a('cmr'):>8.1f}")
print("\n=== RANKING: which baseline is WORSE? (Artemis vs TransArC) ===")
av=lambda n,k:sum(macro[n][k])/len(macro[n][k])
print(f"  worst-comp F1: Artemis {av('Artemis','worst'):.3f} vs TransArC {av('TransArC','worst'):.3f}  -> {'Artemis worse' if av('Artemis','worst')<av('TransArC','worst') else 'TransArC worse'}")
print(f"  CMR%         : Artemis {av('Artemis','cmr'):.1f} vs TransArC {av('TransArC','cmr'):.1f}  -> {'Artemis worse' if av('Artemis','cmr')>av('TransArC','cmr') else 'TransArC worse'}")
print("\n=== RESOLUTION / SPARSITY (distinct values across 30 cells) ===")
print(f"  worst-comp F1: {len(set(round(x,3) for x in W))} distinct values; #zeros={sum(1 for x in W if x==0)}")
print(f"  CMR          : {len(set(round(x,3) for x in S))} distinct values; #zeros={sum(1 for x in S if x==0)}")
print("\n=== RUN STABILITY (mean within-cell std over 3-run systems) ===")
for name in ["S21 GPT","S21 Claude","s20u GPT","s20u Claude"]:
    rs=runstab[name]
    if rs["worst"]:
        print(f"  {name:<12} worstF1 std={sum(rs['worst'])/len(rs['worst']):.3f}   CMR std={sum(rs['cmr'])/len(rs['cmr']):.3f}")
