#!/usr/bin/env python3
"""Pilot Step 1 — size the routing opportunity (pure data, no LLM).

Per sentence with a SAD-Code (doc-code) gold link, label:
  arch-anchored : also appears in SAD-SAM gold -> transitive route CAN reach it
  direct-only   : NOT in SAD-SAM gold          -> transitive is structurally blind
Then measure how much doc-code recall is currently locked behind direct-only
sentences (approach recovery in the gpt-5.4_s21 union of 3 runs).
"""
import csv, glob, importlib.util
from collections import defaultdict
from pathlib import Path
MINI=Path("/mnt/hostshare/ardoco-home/mono/evaluation/mini-src/metrics.py")
spec=importlib.util.spec_from_file_location("metrics",MINI); M=importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
REC=Path("/mnt/hostshare/ardoco-home/sota/recovered-links"); BENCH=M.BENCHMARK
PROJECTS=M.PROJECTS; RUNS=["run1","run2","run3"]; SLOT="gpt-5.4_s21"

def dc_union(p):
    u=set()
    for r in RUNS:
        u|=M.load_result(REC/"doc-code/aalinker-composed"/SLOT/r/f"{p}.csv","sad-code")
    return u

print(f"{'project':<14}{'SC_sents':>9}{'arch':>6}{'direct':>8}{'SC_links':>9}{'links@direct':>13}{'R@arch':>8}{'R@direct':>9}")
print("-"*82)
T=defaultdict(float); allrows={}
for p in PROJECTS:
    code_files=M.load_code_model_files(p)
    sc_raw=M.load_gs_sad_code_raw(p)                    # (sent, normpath) pre-enroll
    ss=M.load_gs_sad_sam(p)                             # (elem, sent)
    sc_sents={s for s,_ in sc_raw}
    ss_sents={s for _,s in ss}
    direct={s for s in sc_sents if s not in ss_sents}
    arch=sc_sents & ss_sents
    links_direct=[(s,fp) for (s,fp) in sc_raw if s in direct]
    # recall (enrolled, file-level) restricted to arch vs direct sentences
    gold_enr=M.enroll(sc_raw, code_files)
    rec=dc_union(p)
    g_arch={(s,fp) for (s,fp) in gold_enr if s in arch}
    g_dir ={(s,fp) for (s,fp) in gold_enr if s in direct}
    r_arch=len(g_arch & rec)/len(g_arch) if g_arch else 0.0
    r_dir =len(g_dir & rec)/len(g_dir) if g_dir else 0.0
    print(f"{p:<14}{len(sc_sents):>9}{len(arch):>6}{len(direct):>8}{len(sc_raw):>9}{len(links_direct):>13}{r_arch:>8.3f}{r_dir:>9.3f}")
    T['sc']+=len(sc_sents); T['arch']+=len(arch); T['direct']+=len(direct)
    T['sclinks']+=len(sc_raw); T['dlinks']+=len(links_direct)
    allrows[p]=dict(direct=sorted(direct,key=lambda v:int(v) if v.isdigit() else 0))
print("-"*82)
print(f"{'TOTAL':<14}{int(T['sc']):>9}{int(T['arch']):>6}{int(T['direct']):>8}{int(T['sclinks']):>9}{int(T['dlinks']):>13}")
print(f"\nDirect-only sentences = {int(T['direct'])}/{int(T['sc'])} "
      f"({100*T['direct']/T['sc']:.0f}% of doc-code gold sentences), "
      f"carrying {int(T['dlinks'])}/{int(T['sclinks'])} raw gold links "
      f"({100*T['dlinks']/T['sclinks']:.0f}%).")
print("\nDirect-only sentence ids per project:")
for p in PROJECTS:
    d=allrows[p]['direct']
    print(f"  {p}: n={len(d)}  {d}")
