#!/usr/bin/env python3
"""Cascade overlap: are doc-code consistent FN inherited from model-doc FN?

Builds, per project:
  A = model-doc consistent misses as {(sentence, component_name)}
  B = doc-code consistent component-misses as {(sentence, component_name)}
      where a component-miss = a gold (sentence,component) for which NO gold
      file of that component was recovered in ANY run.
Then reports |A|, |B|, overlap, and B-only (doc-code-specific, intra-component).
"""
import csv, glob, importlib.util, os, re
from collections import defaultdict
from pathlib import Path

MINI = Path("/mnt/hostshare/ardoco-home/mono/evaluation/mini-src/metrics.py")
spec = importlib.util.spec_from_file_location("metrics", MINI)
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
REC = Path("/mnt/hostshare/ardoco-home/sota/recovered-links")
BENCH = M.BENCHMARK; PROJECTS = M.PROJECTS; RUNS = ["run1","run2","run3"]; SLOT="gpt-5.4_s21"

def elem_names(project):
    names = {}
    for run in RUNS:
        p = REC/"model-doc/aalinker"/SLOT/run/f"{project}.raw.csv"
        if p.exists():
            for r in csv.DictReader(open(p)):
                if r.get("component_id") and r.get("component_name"):
                    names[r["component_id"]] = r["component_name"]
    return names

def rec(task, project, run):
    base = REC/("model-doc/aalinker" if task=="sad-sam" else "doc-code/aalinker-composed")/SLOT/run
    return M.load_result(base/f"{project}.csv", task)

print(f"{'project':<14}{'mdoc-miss(A)':>14}{'dcode-cmiss(B)':>16}{'A∩B':>8}{'A-only':>8}{'B-only':>8}")
print("-"*68)
tot=defaultdict(int)
detail={}
for proj in PROJECTS:
    nm = elem_names(proj)
    # A: model-doc consistent misses -> (sentence, comp_name)
    gold_m = M.load_gs_sad_sam(proj)
    union_m = set().union(*[rec("sad-sam",proj,r) for r in RUNS])
    A = {(s, nm.get(c,c)) for (c,s) in (gold_m - union_m)}
    # B: doc-code consistent component-misses
    code_files = M.load_code_model_files(proj)
    gold_c = M.enroll(M.load_gs_sad_code_raw(proj), code_files)
    f2c = M.load_file_to_comps(proj, code_files)
    union_c = set().union(*[rec("sad-code",proj,r) for r in RUNS])
    # gold (sentence, comp) pairs and which were fully missed across all runs
    gold_sc = defaultdict(set)      # (s,comp) -> gold files
    found_sc = defaultdict(set)     # (s,comp) -> recovered-in-union files
    cname = {}
    for (s,fp) in gold_c:
        for ae in f2c.get(fp,()):
            gold_sc[(s,ae)].add(fp); cname[ae]=nm.get(ae,ae)
            if (s,fp) in union_c:
                found_sc[(s,ae)].add(fp)
    B = {(s, cname[ae]) for (s,ae) in gold_sc if not found_sc.get((s,ae))}
    inter=A&B; aonly=A-B; bonly=B-A
    print(f"{proj:<14}{len(A):>14}{len(B):>16}{len(inter):>8}{len(aonly):>8}{len(bonly):>8}")
    for k,v in (("A",A),("B",B),("inter",inter),("aonly",aonly),("bonly",bonly)):
        tot[k]+=len(v)
    detail[proj]=(A,B,inter,aonly,bonly)
print("-"*68)
print(f"{'TOTAL':<14}{tot['A']:>14}{tot['B']:>16}{tot['inter']:>8}{tot['aonly']:>8}{tot['bonly']:>8}")

print("\n=== B-only (doc-code component misses NOT caused by a model-doc miss) ===")
for proj in PROJECTS:
    bonly = detail[proj][4]
    if bonly:
        print(f"  [{proj}] " + ", ".join(f"s{s}->{c}" for s,c in sorted(bonly, key=lambda x:(x[1],int(x[0]) if x[0].isdigit() else 0))))

print("\n=== A-only (model-doc misses that did NOT surface as a doc-code component miss) ===")
for proj in PROJECTS:
    aonly = detail[proj][3]
    if aonly:
        print(f"  [{proj}] " + ", ".join(f"s{s}->{c}" for s,c in sorted(aonly, key=lambda x:(x[1],int(x[0]) if x[0].isdigit() else 0))))
