#!/usr/bin/env python3
import csv, glob, importlib.util
from collections import defaultdict
from pathlib import Path
MINI=Path("/mnt/hostshare/ardoco-home/mono/evaluation/mini-src/metrics.py")
spec=importlib.util.spec_from_file_location("metrics",MINI); M=importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
REC=Path("/mnt/hostshare/ardoco-home/sota/recovered-links"); BENCH=M.BENCHMARK; RUNS=["run1","run2","run3"]; SLOT="gpt-5.4_s21"

def sents(p):
    h=glob.glob(str(BENCH/p/"text_*"/f"{p}.txt"))
    d={}
    if h:
        for i,l in enumerate(open(h[0],errors="replace"),1): d[str(i)]=l.strip()
    return d
def names(p):
    nm={}
    for r in RUNS:
        f=REC/"model-doc/aalinker"/SLOT/r/f"{p}.raw.csv"
        if f.exists():
            for row in csv.DictReader(open(f)):
                if row.get("component_id") and row.get("component_name"): nm[row["component_id"]]=row["component_name"]
    return nm
def md(p,r): return M.load_result(REC/"model-doc/aalinker"/SLOT/r/f"{p}.csv","sad-sam")
def dc(p,r): return M.load_result(REC/"doc-code/aalinker-composed"/SLOT/r/f"{p}.csv","sad-code")

proj="teammates"; S=sents(proj); nm=names(proj)
md_union=set().union(*[md(proj,r) for r in RUNS])     # (comp,sent)
md_by_s=defaultdict(set)
for c,s in md_union: md_by_s[s].add(nm.get(c,c))
dc_union=set().union(*[dc(proj,r) for r in RUNS])
dc_by_s=defaultdict(set)
for s,fp in dc_union: dc_by_s[s].add(fp)

for s in ["75","100","22","84","125","195","172"]:
    print(f"\n--- teammates s{s}: {S.get(s,'')[:130]}")
    print(f"    model-doc recovered comps: {sorted(md_by_s.get(s,[])) or '∅'}")
    rec_files = sorted(dc_by_s.get(s,[]))
    print(f"    doc-code recovered files ({len(rec_files)}): {[f.split('/')[-1] for f in rec_files][:8]}")

# what does gold say for these (sentence -> components via gold)
code_files=M.load_code_model_files(proj); f2c=M.load_file_to_comps(proj,code_files)
gold=M.enroll(M.load_gs_sad_code_raw(proj),code_files)
gold_by_s=defaultdict(set)
for s,fp in gold:
    for ae in f2c.get(fp,()): gold_by_s[s].add(nm.get(ae,ae))
for s in ["75","100","22","84","125","195","172"]:
    print(f"  gold s{s} components: {sorted(gold_by_s.get(s,[]))}")
