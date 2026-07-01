#!/usr/bin/env python3
"""Re-score the router by DOWNSTREAM LINK OUTCOME, not binary label.

A router-CODE firing is:
  BENEFICIAL  if the sentence still has gold doc-code links transitive MISSED
              (the direct route can recover TP) -- regardless of arch/code label.
  REDUNDANT   if transitive already recovers ALL of the sentence's gold links
              (direct route adds no TP, only potential FP risk).
  (No router-CODE sentence is "harmful by construction": every sentence in scope
   has gold doc-code links, so a correctly-named direct link is a TP, not an FP.)
Key question from the user: are the 27 'false positive' (arch-anchored) firings
actually fine because they still recover TP?
"""
import glob, importlib.util, json
from collections import defaultdict
from pathlib import Path
MINI=Path("/mnt/hostshare/ardoco-home/mono/evaluation/mini-src/metrics.py")
spec=importlib.util.spec_from_file_location("metrics",MINI); M=importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
REC=Path("/mnt/hostshare/ardoco-home/sota/recovered-links"); PROJECTS=M.PROJECTS
RUNS=["run1","run2","run3"]; SLOT="gpt-5.4_s21"
CACHE=Path(__file__).resolve().parent.parent / "cache" / "router_cache.json"
cache=json.loads(CACHE.read_text())

# per project: enrolled gold per sentence, transitive recovered (union) per sentence
def analyze():
    rowsCODE=[]   # (proj, sid, true_label, gold_files, missed_files)
    for p in PROJECTS:
        code_files=M.load_code_model_files(p)
        gold=M.enroll(M.load_gs_sad_code_raw(p), code_files)
        rec=set()
        for r in RUNS: rec|=M.load_result(REC/"doc-code/aalinker-composed"/SLOT/r/f"{p}.csv","sad-code")
        gold_by_s=defaultdict(set); rec_by_s=defaultdict(set)
        for s,fp in gold: gold_by_s[s].add(fp)
        for s,fp in rec: rec_by_s[s].add(fp)
        for s, gfiles in gold_by_s.items():
            gid=f"{p}:{s}"
            if gid not in cache: continue
            if cache[gid]["route"]!="CODE": continue
            missed=gfiles - rec_by_s.get(s,set())
            rowsCODE.append((p,s,cache[gid]["true"],len(gfiles),len(missed)))
    return rowsCODE

rows=analyze()
benef=[r for r in rows if r[4]>0]
redun=[r for r in rows if r[4]==0]
def split(rs,lab): return [r for r in rs if r[2]==lab]

print(f"Router-CODE firings: {len(rows)} sentences")
print(f"  BENEFICIAL (transitive still misses >=1 gold link)  : {len(benef)} sentences, "
      f"{sum(r[4] for r in benef)} missed gold file-links recoverable")
print(f"  REDUNDANT  (transitive already covers all gold)      : {len(redun)} sentences "
      f"(FP-risk-only firings)")
print()
print("Breakdown by true label (the user's point about the 27 arch 'FPs'):")
for lab in ("CODE","ARCH"):
    b=split(benef,lab); r=split(redun,lab)
    print(f"  true={lab:<5}: beneficial={len(b):>2} (missed_links={sum(x[4] for x in b):>4})  "
          f"redundant={len(r):>2}")
print()
print("The 27 arch-anchored 'false positives' — are they fine?")
arch_fp=[r for r in rows if r[2]=="ARCH"]
print(f"  of {len(arch_fp)} arch-anchored CODE firings: "
      f"{len([r for r in arch_fp if r[4]>0])} recover TP (beneficial), "
      f"{len([r for r in arch_fp if r[4]==0])} redundant")
print("\n  arch-anchored firings that STILL recover missed gold (beneficial over-fire):")
for p,s,_,g,m in sorted([r for r in arch_fp if r[4]>0], key=lambda x:(x[0],int(x[1]))):
    print(f"    {p:<12} s{s:>3}: missed {m}/{g} gold files")
print("\n  arch-anchored firings that are redundant (only FP risk):")
for p,s,_,g,m in sorted([r for r in arch_fp if r[4]==0], key=lambda x:(x[0],int(x[1]))):
    print(f"    {p:<12} s{s:>3}: transitive already covers all {g} gold files")
