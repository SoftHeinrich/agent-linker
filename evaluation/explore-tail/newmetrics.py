#!/usr/bin/env python3
"""NEW set-only size-aware metrics (NO ranked list — only predicted link SETS).

Design constraint: our systems emit a binary set of (sentence, component) links,
no scores. So ranking metrics (Lag, MAP, success@k, one-error, CVaR) are out. We
operationalize the literature's per-group / coverage / worst-case views with the
COMPONENT as the retrieval unit, using only set membership.

Per gold component k: recall_k = |correct links in k| / |gold links in k|.
Component-grain set retrieval (volume-INDEPENDENT — each component counts once):
  G   = gold components (>=1 gold link)
  R+  = gold components the system reaches with >=1 CORRECT link        (TP)
  P   = components the system touches with >=1 predicted link
  SFC = |G| - |R+|        silent-failure count   (FN; abandoned gold comps)
  PHC = |P| - |R+|        phantom-component count (FP; touched, 0 correct)

NEW metrics:
  comp_coverage   = |R+|/|G|          (recall half; = 1 - SFC/|G|)
  comp_precision  = |R+|/|P|          (precision half; = 1 - phantom rate)
  comp_set_F1     = harmonic(cov, prec)   <-- "did you find the RIGHT components"
  SFC, PHC        = raw counts (sharp, unbounded)
Set-only survivors of the tail camp (for comparison):
  macro_recall, worst_recall(min), gmean_recall
Reference: link_f1 (the metric we want independence from)."""
import sys, math
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
def path_for(slot,run,p,task):
    if slot=="__artemis__":
        return SOTA/(f"model-doc/artemis-{p}-gpt-5.4.csv" if task=="sad-sam" else f"doc-code/artemis-{p}-gpt-5.4.csv")
    if slot=="__transarc__":
        return SOTA/(f"model-doc/swattr-{p}.csv" if task=="sad-sam" else f"doc-code/transarc-{p}.csv")
    base = "model-doc/aalinker" if task=="sad-sam" else "doc-code/aalinker-composed"
    return SOTA/f"{base}/{slot}/{run}/{p}.csv"

def comp_view(project, res, task):
    """Return per-component recall dict, gold-comp set, predicted-comp set, link_f1.
    For sad-code we map files->SAM-CODE components (enrolled gold, D-12)."""
    if task=="sad-sam":
        gold=m.load_gs_sad_sam(project); link=m.prf(gold,res)[2]
        gb=defaultdict(set); rb=defaultdict(set)
        for c,s in gold: gb[c].add(s)
        for c,s in res:  rb[c].add(s)        # predicted comp -> sentences
    else:
        cf=m.load_code_model_files(project)
        gold=m.enroll(m.load_gs_sad_code_raw(project),cf)
        f2c=m.load_file_to_comps(project,cf); link=m.prf(gold,res)[2]
        def tc(pairs):
            o=defaultdict(set)
            for s,c in pairs:
                for comp in f2c.get(c,()): o[comp].add(s)
            return o
        gb=tc(gold); rb=tc(res)
    rec={c:(len(gb[c]&rb.get(c,set()))/len(gb[c])) for c in gb}
    Gset=set(gb); Pset=set(rb)                          # gold comps, touched comps
    Rplus={c for c in Gset if rec[c]>0}                 # correctly reached gold comps
    return rec, Gset, Pset, Rplus, link

def metrics_for(rec, Gset, Pset, Rplus):
    K=len(Gset); P=len(Pset); tp=len(Rplus)
    sfc=K-tp; phc=P-tp
    cov=tp/K if K else 0.0
    prec=tp/P if P else 0.0
    csf1=(2*cov*prec/(cov+prec)) if (cov+prec) else 0.0
    rvals=list(rec.values())
    macro=sum(rvals)/K if K else 0.0
    worst=min(rvals) if rvals else 0.0
    gmean=math.exp(sum(math.log(x) for x in rvals)/K) if rvals and all(x>0 for x in rvals) else 0.0
    return {"SFC":sfc,"PHC":phc,"comp_coverage":cov,"comp_precision":prec,
            "comp_set_F1":csf1,"macro_recall":macro,"worst_recall":worst,"gmean_recall":gmean}

KEYS=["SFC","PHC","comp_coverage","comp_precision","comp_set_F1","macro_recall","worst_recall","gmean_recall"]

def run(task):
    cells=[]; macro=defaultdict(lambda:defaultdict(list))
    for sname,slot,runs in SYSTEMS:
        for p in PROJECTS:
            acc=defaultdict(list); links=[]
            for run in runs:
                res=m.load_result(path_for(slot,run,p,task),task)
                if not res: continue
                rec,G,P,R,link=comp_view(p,res,task); links.append(link)
                mv=metrics_for(rec,G,P,R)
                for k,v in mv.items(): acc[k].append(v)
            if not links: continue
            cell={"sys":sname,"proj":p,"link":sum(links)/len(links)}
            for k in KEYS: cell[k]=sum(acc[k])/len(acc[k])
            cells.append(cell)
            macro[sname]["link"].append(cell["link"])
            for k in KEYS: macro[sname][k].append(cell[k])
    link=[c["link"] for c in cells]
    M={s:{k:sum(macro[s][k])/len(macro[s][k]) for k in (["link"]+KEYS)} for s in macro}
    print(f"\n================ {task} ({len(cells)} cells) ================")
    # for SFC/PHC higher=worse: correlate with link-F1 directly, and report spread as count
    print(f"{'metric':<15}{'rho_vs_link':>12}{'best':>9}{'worst':>9}{'spread':>9}{'top(by val)':>13}")
    order=sorted(M,key=lambda s:-M[s]["link"])
    for k in KEYS:
        vals=[c[k] for c in cells]; rho=spearman(link,vals)
        mv={s:M[s][k] for s in M}
        if k in ("SFC","PHC"):  # lower better
            best=min(mv.values()); worst=max(mv.values()); top=min(mv,key=lambda s:mv[s])
        else:
            best=max(mv.values()); worst=min(mv.values()); top=max(mv,key=lambda s:mv[s])
        print(f"{k:<15}{rho:>+12.2f}{best:>9.3f}{worst:>9.3f}{abs(best-worst):>9.3f}{top:>13}")
    print(f"\n  per-system macro:  "+"  ".join(f"{s}" for s in order))
    for k in KEYS:
        print(f"  {k:<14}"+"".join(f"{M[s][k]:>9.3f}" for s in order))

for t in ("sad-sam","sad-code"):
    run(t)
