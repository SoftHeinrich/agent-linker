#!/usr/bin/env python3
"""AbandMass% — RQ2-style table. AbandMass% = share of gold (sentence,component)
decisions that fall in FULLY-ABANDONED components (component with 0 correct links).
Enrollment-free: mass is counted in (sentence,component) decisions (the suite grain
used by worst/harmonic), NOT files. Both tasks, body 3 systems, per project + macro.
Emits ASCII + a LaTeX float."""
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mini-src"))
import metrics as m

SOTA = Path("/mnt/hostshare/ardoco-home/sota-recovered-links")
PROJ = ["mediastore","teastore","teammates","bigbluebutton","jabref"]
SHORT = {"mediastore":"MS","teastore":"TS","teammates":"TM","bigbluebutton":"BBB","jabref":"JR"}
SYS = [("\\approach{}","gpt-5.4_s21",["run1","run2","run3"]),
       ("\\Artemis{}","__artemis__",[None]),
       ("\\TransArc{}","__transarc__",[None])]
NAME = {"\\approach{}":"approach S21","\\Artemis{}":"Artemis","\\TransArc{}":"TransArC"}

def path_for(slot,run,p,task):
    if slot=="__artemis__": return SOTA/(f"model-doc/artemis-{p}-gpt-5.4.csv" if task=="sad-sam" else f"doc-code/artemis-{p}-gpt-5.4.csv")
    if slot=="__transarc__": return SOTA/(f"model-doc/swattr-{p}.csv" if task=="sad-sam" else f"doc-code/transarc-{p}.csv")
    base="model-doc/aalinker" if task=="sad-sam" else "doc-code/aalinker-composed"
    return SOTA/f"{base}/{slot}/{run}/{p}.csv"

def comp_pairs(project,res,task):
    """gold_by_comp{c:set(sent)}, res correct pairs per comp; (sentence,component) grain."""
    if task=="sad-sam":
        gold={(c,s) for (c,s) in m.load_gs_sad_sam(project)}
        res_c={(c,s) for (c,s) in res}
    else:
        cf=m.load_code_model_files(project)
        goldf=m.enroll(m.load_gs_sad_code_raw(project),cf); f2c=m.load_file_to_comps(project,cf)
        def tc(pairs):
            o=set()
            for s,c in pairs:
                for comp in f2c.get(c,()): o.add((comp,s))
            return o
        gold=tc(goldf); res_c=tc(res)
    gb=defaultdict(set)
    for c,s in gold: gb[c].add(s)
    correct=defaultdict(set)
    for c,s in (gold & res_c): correct[c].add(s)
    return gb, correct

def abandmass(project,res,task):
    gb,correct=comp_pairs(project,res,task)
    tot=sum(len(v) for v in gb.values())
    dead=sum(len(gb[c]) for c in gb if not correct.get(c))
    return dead/tot*100 if tot else 0.0

def table(task):
    rows=[]
    for tex,slot,runs in SYS:
        vals=[]
        for p in PROJ:
            acc=[]
            for run in runs:
                res=m.load_result(path_for(slot,run,p,task),task)
                if res: acc.append(abandmass(p,res,task))
            vals.append(sum(acc)/len(acc) if acc else float('nan'))
        rows.append((tex,vals,sum(vals)/len(vals)))
    return rows

for task,lab in (("sad-sam","DOC-MODEL"),("sad-code","DOC-CODE")):
    rows=table(task)
    print(f"\n==== AbandMass% — {lab} ====")
    print(f"{'system':<14}"+"".join(f"{SHORT[p]:>7}" for p in PROJ)+f"{'macro':>8}")
    for tex,vals,mac in rows:
        print(f"{NAME[tex]:<14}"+"".join(f"{v:>7.1f}" for v in vals)+f"{mac:>8.1f}")

# LaTeX float (both tasks stacked)
print("\n%%%% LaTeX %%%%")
print(r"\begin{table}[t]\centering\small")
print(r"\caption{Documentation abandoned (\%): share of gold (sentence, component) "
      r"decisions in components recovered with \emph{no} correct link. 0 = every "
      r"documented component traced. Approach = mean of 3 runs.}")
print(r"\label{tab:abandmass}")
print(r"\setlength{\tabcolsep}{5pt}\begin{tabular}{@{}l rrrrr r@{}}")
print(r"\toprule")
print(r"System & MS & TS & TM & BBB & JR & Macro \\")
for task,lab in (("sad-sam","\\emph{doc--model}"),("sad-code","\\emph{doc--code}")):
    print(r"\midrule \multicolumn{7}{@{}l}{"+lab+r"} \\")
    for tex,vals,mac in table(task):
        cells=" & ".join(f"{v:.1f}" for v in vals)
        print(f"{tex} & {cells} & {mac:.1f} \\\\")
print(r"\bottomrule\end{tabular}\end{table}")
