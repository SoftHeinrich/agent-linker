#!/usr/bin/env python3
"""Is SFC too sparse? Compare the dead-only count against denser variants, for the
3 body systems, doc-model. Variants:
  SFC          = # comps recall==0                      (current; sparse)
  weak@.5      = # comps recall<0.5                      (graded bar)
  weak@.75     = # comps recall<0.75                     (graded bar)
  AbandMass%   = gold sentences in recall==0 comps / all gold sentences   (mass-weighted dead)
  WeakMass%    = sentence-weighted shortfall = sum_k size_k*(1-recall_k)/sum size_k  (continuous)
Per project + macro. WeakMass% == 1 - (sentence-weighted recall) -> approaches link recall."""
import sys
from collections import defaultdict
from pathlib import Path
from _roots import MINI_SRC, SOTA    # shared roots (see _roots.py)
sys.path.insert(0, str(MINI_SRC))
import metrics as m

PROJ = ["mediastore","teastore","teammates","bigbluebutton","jabref"]
SHORT = {"mediastore":"MS","teastore":"TS","teammates":"TM","bigbluebutton":"BBB","jabref":"JR"}
SYS = [("approach S21","model-doc/aalinker/gpt-5.4_s21/{run}/{p}.csv",["run1","run2","run3"]),
       ("Artemis","model-doc/artemis-{p}-gpt-5.4.csv",[None]),
       ("TransArC","model-doc/swattr-{p}.csv",[None])]

def variants(project,res):
    gold=m.load_gs_sad_sam(project); gb=defaultdict(set); rb=defaultdict(set)
    for c,s in gold: gb[c].add(s)
    for c,s in res:  rb[c].add(s)
    rec={c:len(gb[c]&rb.get(c,set()))/len(gb[c]) for c in gb}
    size={c:len(gb[c]) for c in gb}; tot=sum(size.values())
    sfc=sum(1 for c in rec if rec[c]==0)
    w5 =sum(1 for c in rec if rec[c]<0.5)
    w75=sum(1 for c in rec if rec[c]<0.75)
    abmass=sum(size[c] for c in rec if rec[c]==0)/tot*100
    weakmass=sum(size[c]*(1-rec[c]) for c in rec)/tot*100
    return dict(SFC=sfc, weak5=w5, weak75=w75, abmass=abmass, weakmass=weakmass)

def run():
    cols=[("SFC","SFC(=0)"),("weak5","weak<.5"),("weak75","weak<.75"),
          ("abmass","AbandMass%"),("weakmass","WeakMass%")]
    for key,label in cols:
        print(f"\n  {label}")
        print(f"  {'system':<14}"+"".join(f"{SHORT[p]:>7}" for p in PROJ)+f"{'macro':>8}")
        for name,pat,runs in SYS:
            vals=[]
            for p in PROJ:
                acc=[]
                for run in runs:
                    res=m.load_result(SOTA/(pat.format(run=run,p=p) if run else pat.format(p=p)),"sad-sam")
                    if res: acc.append(variants(p,res)[key])
                vals.append(sum(acc)/len(acc) if acc else float('nan'))
            macro=sum(vals)/len(vals)
            f=lambda x:(f"{x:.1f}" if key in("abmass","weakmass") else f"{x:.2f}".rstrip('0').rstrip('.'))
            print(f"  {name:<14}"+"".join(f"{f(v):>7}" for v in vals)+f"{f(macro):>8}")

print("DOC-MODEL, 3 body systems (approach=mean of 3 runs)")
run()
