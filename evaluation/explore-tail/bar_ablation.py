#!/usr/bin/env python3
"""Ablation on the COVERAGE BAR. A component k is 'covered' if its per-component
RECALL >= theta. theta -> 0+ is the '>=1 correct link' bar (loosest); theta = 1.0
is strict full recovery. Sweep theta and watch, on doc-model (30 cells):
  coverage  = frac gold components with recall >= theta   (bounded, higher=better)
  failures  = COUNT of components with recall < theta      (unbounded, sharp)
  spread    = macro(best system) - macro(worst system)     (sharpness of coverage)
  rho       = Spearman(coverage, link-F1)                  (low = independent)
to see whether ANY bar is both sharp and independent, or whether tightening the
bar just trades independence for sharpness. Stdlib only."""
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mini-src"))
import metrics as m
from rq2_corr import spearman

SOTA = Path("/mnt/hostshare/ardoco-home/sota-recovered-links")
PROJECTS = ["mediastore","teastore","teammates","bigbluebutton","jabref"]
SYSTEMS = [
    ("S21 GPT","gpt-5.4_s21",["run1","run2","run3"]),
    ("S21 Claude","sonnet_s21",["run1","run2","run3"]),
    ("s20u GPT","gpt-5.4_full",["run1","run2","run3"]),
    ("s20u Claude","sonnet_full",["run1","run2","run3"]),
    ("Artemis","__artemis__",[None]),
    ("TransArC","__transarc__",[None]),
]
THETAS = [1e-9, 0.25, 0.5, 0.75, 1.0]   # 1e-9 == ">=1 correct link"

def path_for(slot,run,p):
    if slot=="__artemis__": return SOTA/f"model-doc/artemis-{p}-gpt-5.4.csv"
    if slot=="__transarc__": return SOTA/f"model-doc/swattr-{p}.csv"
    return SOTA/f"model-doc/aalinker/{slot}/{run}/{p}.csv"

def recalls(proj,res):
    gold=m.load_gs_sad_sam(proj); gb,rb=defaultdict(set),defaultdict(set)
    for c,s in gold: gb[c].add(s)
    for c,s in res:  rb[c].add(s)
    out={}
    for c in gb:
        hit=len(gb[c]&rb[c]); out[c]=hit/len(gb[c])
    return out, m.prf(gold,res)[2]

# gather per-cell recall vectors + link
cells=[]  # (sys, proj, link, {comp:recall})
for sname,slot,runs in SYSTEMS:
    for p in PROJECTS:
        accR=defaultdict(list); links=[]
        for run in runs:
            res=m.load_result(path_for(slot,run,p),"sad-sam")
            if not res: continue
            rec,link=recalls(p,res); links.append(link)
            for c,v in rec.items(): accR[c].append(v)
        if not links: continue
        rmean={c:sum(v)/len(v) for c,v in accR.items()}
        cells.append((sname,p,sum(links)/len(links),rmean))

link=[c[2] for c in cells]
print("=== Coverage-bar ablation (doc-model, 30 cells) ===")
print(f"{'theta(bar)':>11}{'rho_vs_link':>12}{'cov_best':>9}{'cov_worst':>10}{'cov_spread':>11}{'fail_best':>10}{'fail_worst':>11}")
for th in THETAS:
    # per cell: coverage fraction + failure count at this bar
    covs=[]; macroC=defaultdict(list); macroF=defaultdict(list)
    for sname,p,lk,rmean in cells:
        n=len(rmean); cov=sum(1 for v in rmean.values() if v>=th)/n
        fail=sum(1 for v in rmean.values() if v<th)
        covs.append(cov); macroC[sname].append(cov); macroF[sname].append(fail)
    rho=spearman(link,covs)
    cb={s:sum(v)/len(v) for s,v in macroC.items()}
    fb={s:sum(v)/len(v) for s,v in macroF.items()}  # avg failures/project
    bestC=max(cb.values()); worstC=min(cb.values())
    # failures: best system = fewest, worst = most (per project avg)
    bestF=min(fb.values()); worstF=max(fb.values())
    bar = ">=1 link" if th<1e-6 else f"R>={th:.2f}"
    print(f"{bar:>11}{rho:>+12.2f}{bestC:>9.3f}{worstC:>10.3f}{bestC-worstC:>11.3f}{bestF:>10.2f}{worstF:>11.2f}")

print("\n=== failure COUNT per system at each bar (avg components/project below bar) ===")
order=[s[0] for s in SYSTEMS]
print(f"{'system':<12}"+"".join(f"{('>=1' if th<1e-6 else f'R>={th:.2f}'):>9}" for th in THETAS))
per=defaultdict(dict)
for th in THETAS:
    for sname,p,lk,rmean in cells:
        per[sname].setdefault(th,[]).append(sum(1 for v in rmean.values() if v<th))
for s in order:
    print(f"{s:<12}"+"".join(f"{sum(per[s][th])/len(per[s][th]):>9.2f}" for th in THETAS))
