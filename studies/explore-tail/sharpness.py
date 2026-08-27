#!/usr/bin/env python3
"""Sharpness vs independence. For each tail/coverage metric report BOTH:
  - rho   : Spearman vs link-F1 (low = independent, adds info)
  - spread: macro(best system) - macro(worst system)  (high = sharp separator)
  - ratio : macro(worst sys) / macro(best sys)         (low = dramatic separation)
plus SHARPER coverage variants that amplify misses (worst-recall, abandoned-link
mass, harsher thresholds). doc-model focus (the task where coverage is not
saturated). 30 cells, 6 systems x 5 projects."""
import math, sys
from collections import defaultdict
from pathlib import Path
from _roots import MINI_SRC, SOTA    # shared roots (see _roots.py)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(MINI_SRC))
import metrics as m
from metrics import spearman

PROJECTS = ["mediastore","teastore","teammates","bigbluebutton","jabref"]
SYSTEMS = [
    ("S21 GPT","gpt-5.4_s21",["run1","run2","run3"]),
    ("S21 Claude","sonnet_s21",["run1","run2","run3"]),
    ("s20u GPT","gpt-5.4_full",["run1","run2","run3"]),
    ("s20u Claude","sonnet_full",["run1","run2","run3"]),
    ("Artemis","__artemis__",[None]),
    ("TransArC","__transarc__",[None]),
]
def path_for(slot,run,project):
    if slot=="__artemis__": return SOTA/f"model-doc/artemis-{project}-gpt-5.4.csv"
    if slot=="__transarc__": return SOTA/f"model-doc/swattr-{project}.csv"
    return SOTA/f"model-doc/aalinker/{slot}/{run}/{project}.csv"

def cell_metrics(project,res):
    gold=m.load_gs_sad_sam(project)             # (comp,sentence)
    link=m.prf(gold,res)[2]
    gb,rb=defaultdict(set),defaultdict(set)
    for c,s in gold: gb[c].add(s)
    for c,s in res:  rb[c].add(s)
    def pr(c):
        g={(x,c) for x in gb.get(c,set())}; r={(x,c) for x in rb.get(c,set())}
        p,rec,f1=m.prf(g,r); return p,rec,f1
    comps=list(gb)
    F1=[pr(c)[2] for c in comps]
    REC=[pr(c)[1] for c in comps]
    n=len(comps); ssort=sorted(F1)
    sizes={c:len(gb[c]) for c in comps}; total=sum(sizes.values())
    missed=[c for c in comps if pr(c)[1]==0.0]   # no correct link recovered at all
    abandoned_mass=sum(sizes[c] for c in missed)/total if total else 0.0
    k25=max(1,math.ceil(0.25*n))
    return {
        "link": link,
        # --- independent-but-flat coverage ---
        "comp_coverage": sum(1 for x in REC if x>0)/n,
        "neg_n_missed": -float(len(missed)),
        # --- SHARPER candidates ---
        "worst_recall(min)": min(REC) if REC else 0.0,
        "abandoned_link_mass": abandoned_mass,           # higher=worse
        "kept_link_mass": 1-abandoned_mass,              # higher=better (for #1 check)
        "worst_F1(min)": ssort[0],
        "harmonic_F1": (n/sum(1/x for x in F1)) if all(x>0 for x in F1) else 0.0,
        "CVaR25_F1": sum(ssort[:k25])/k25,
        "strict_comp_cov": sum(1 for x in F1 if x>=0.999)/n,
        "frac_rec_ge.8": sum(1 for x in REC if x>=0.8)/n, # harsher reach bar
    }

KEYS=["link","comp_coverage","neg_n_missed","worst_recall(min)","kept_link_mass",
      "worst_F1(min)","harmonic_F1","CVaR25_F1","strict_comp_cov","frac_rec_ge.8"]

cells=[]; macro=defaultdict(lambda:defaultdict(list))
for sname,slot,runs in SYSTEMS:
    for p in PROJECTS:
        acc=defaultdict(list)
        for run in runs:
            res=m.load_result(path_for(slot,run,p),"sad-sam")
            if not res: continue
            cm=cell_metrics(p,res)
            for k,v in cm.items(): acc[k].append(v)
        if not acc["link"]: continue
        cell={"sys":sname,"proj":p}
        for k in KEYS: cell[k]=sum(acc[k])/len(acc[k])
        cells.append(cell)
        for k in KEYS: macro[sname][k].append(cell[k])

link=[c["link"] for c in cells]
M={s:{k:sum(macro[s][k])/len(macro[s][k]) for k in KEYS} for s in macro}
print("=== doc-model: independence (rho) vs sharpness (macro spread) — sorted by spread ===")
print(f"{'metric':<20}{'rho':>7}{'best':>8}{'worst':>8}{'spread':>8}{'worst/best':>11}{'top sys':>13}")
rows=[]
for k in KEYS:
    if k=="link":
        ranked=sorted(M,key=lambda s:-M[s][k]); rho=1.0
    else:
        rho=spearman(link,[c[k] for c in cells]); ranked=sorted(M,key=lambda s:-M[s][k])
    vals=[M[s][k] for s in M]; best=max(vals); worst=min(vals)
    ratio=worst/best if best else float('nan')
    rows.append((k,rho,best,worst,best-worst,ratio,ranked[0]))
for k,rho,best,worst,spread,ratio,top in sorted(rows,key=lambda r:-r[4]):
    print(f"{k:<20}{rho:>+7.2f}{best:>8.3f}{worst:>8.3f}{spread:>8.3f}{ratio:>11.2f}{top:>13}")

print("\n=== per-system macro (sharper candidates) ===")
order=sorted(M,key=lambda s:-M[s]["link"])
hdr=["link","worst_recall(min)","kept_link_mass","strict_comp_cov","frac_rec_ge.8"]
print(f"{'sys':<12}"+"".join(f"{h:>18}" for h in hdr))
for s in order:
    print(f"{s:<12}"+"".join(f"{M[s][h]:>18.3f}" for h in hdr))
