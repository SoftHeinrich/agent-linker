#!/usr/bin/env python3
"""Remaining-FN analysis for the agent-linker (a-linker) GPT-5.4 S21 slot.

Reuses mini-src/metrics.py loaders/enrollment verbatim so matching == paper.
Tasks: sad-sam (model-doc) and sad-code (doc-code, file level + component view).
3 runs (gpt-5.4_s21). Reports per-run FN and CONSISTENT FN (missed in all 3 runs).
"""
import csv
import glob
import importlib.util
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

MINI = Path("/mnt/hostshare/ardoco-home/mono/evaluation/mini-src/metrics.py")
spec = importlib.util.spec_from_file_location("metrics", MINI)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

REC = Path("/mnt/hostshare/ardoco-home/sota/recovered-links")
BENCH = M.BENCHMARK
PROJECTS = M.PROJECTS
RUNS = ["run1", "run2", "run3"]
SLOT = "gpt-5.4_s21"

# ---- enrichment helpers -------------------------------------------------------
_sent_cache = {}
def sentences(project):
    if project in _sent_cache:
        return _sent_cache[project]
    hits = glob.glob(str(BENCH / project / "text_*" / f"{project}.txt"))
    lines = {}
    if hits:
        with open(hits[0], encoding="utf-8", errors="replace") as f:
            for i, ln in enumerate(f, 1):
                lines[str(i)] = ln.strip()
    _sent_cache[project] = lines
    return lines

_name_cache = {}
def elem_names(project):
    """model-element GUID -> display name, from recovered raw + PCM model XML."""
    if project in _name_cache:
        return _name_cache[project]
    names = {}
    # 1) recovered raw (sad-sam) gives component_name for predicted ids
    for run in RUNS:
        p = REC / "model-doc/aalinker" / SLOT / run / f"{project}.raw.csv"
        if p.exists():
            with open(p) as f:
                for r in csv.DictReader(f):
                    cid, nm = r.get("component_id"), r.get("component_name")
                    if cid and nm:
                        names[cid] = nm
    # 2) parse all model XML for id="..." entityName="..." fallback
    for mp in glob.glob(str(BENCH / project / "model_*" / "**" / "*"), recursive=True):
        if not os.path.isfile(mp):
            continue
        try:
            txt = open(mp, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        for m in re.finditer(r'id="(_[^"]+)"[^>]*entityName="([^"]*)"', txt):
            names.setdefault(m.group(1), m.group(2))
        for m in re.finditer(r'entityName="([^"]*)"[^>]*id="(_[^"]+)"', txt):
            names.setdefault(m.group(2), m.group(1))
    _name_cache[project] = names
    return names

# ---- loaders for recovered slot ----------------------------------------------
def rec_path(task, project, run):
    base = REC / ("model-doc/aalinker" if task == "sad-sam"
                  else "doc-code/aalinker-composed") / SLOT / run
    return base / f"{project}.csv"

def load_runs(task, project):
    return [M.load_result(rec_path(task, project, run), task) for run in RUNS]

# ---- sad-sam analysis ---------------------------------------------------------
def analyze_sadsam():
    print("\n" + "=" * 78)
    print("TASK 1: doc-to-model (sad-sam)  — exact (element,sentence) matching")
    print("=" * 78)
    total_gold = 0
    consistent = []   # (project, elem, sent)
    perrun_counts = defaultdict(int)
    for proj in PROJECTS:
        gold = M.load_gs_sad_sam(proj)            # set[(elementID, sentence)]
        runs = load_runs("sad-sam", proj)
        union = set().union(*runs) if runs else set()
        miss_all = gold - union                   # missed in EVERY run
        total_gold += len(gold)
        for i, r in enumerate(runs):
            perrun_counts[i] += len(gold - r)
        recalls = [len(gold & r) / len(gold) for r in runs]
        print(f"\n  {proj}: gold={len(gold)}  per-run R={[round(x,3) for x in recalls]}"
              f"  per-run FN={[len(gold - r) for r in runs]}  consistent-FN={len(miss_all)}")
        for (elem, sent) in sorted(miss_all, key=lambda x: int(x[1]) if x[1].isdigit() else 0):
            consistent.append((proj, elem, sent))
    print(f"\n  TOTAL gold(model-doc) = {total_gold}")
    print(f"  per-run FN totals: " + ", ".join(f"{RUNS[i]}={perrun_counts[i]}" for i in range(3)))
    print(f"  CONSISTENT FN (missed in all 3 runs) = {len(consistent)}")
    return consistent

def dump_sadsam(consistent):
    print("\n  --- CONSISTENT model-doc FN detail (elem -> name | sentence) ---")
    for proj in PROJECTS:
        rows = [c for c in consistent if c[0] == proj]
        if not rows:
            continue
        nm = elem_names(proj)
        sents = sentences(proj)
        print(f"\n  [{proj}]")
        for _, elem, sent in rows:
            ename = nm.get(elem, "??")
            stext = sents.get(sent, "")
            print(f"    s{sent:>3} -> {ename:<28} | {stext[:96]}")

# ---- sad-code analysis --------------------------------------------------------
def analyze_sadcode():
    print("\n" + "=" * 78)
    print("TASK 2: doc-to-code (sad-code) — enrolled gold (file level) + component view")
    print("=" * 78)
    total_gold = 0
    consistent = []   # (project, sent, filepath)
    perrun_counts = defaultdict(int)
    file_to_comps_all = {}
    for proj in PROJECTS:
        code_files = M.load_code_model_files(proj)
        gold = M.enroll(M.load_gs_sad_code_raw(proj), code_files)   # enrolled file set
        f2c = M.load_file_to_comps(proj, code_files)
        file_to_comps_all[proj] = f2c
        runs = load_runs("sad-code", proj)
        union = set().union(*runs) if runs else set()
        miss_all = gold - union
        total_gold += len(gold)
        for i, r in enumerate(runs):
            perrun_counts[i] += len(gold - r)
        recalls = [len(gold & r) / len(gold) for r in runs]
        print(f"\n  {proj}: gold(enrolled files)={len(gold)}  per-run fileR={[round(x,3) for x in recalls]}"
              f"  per-run FN={[len(gold - r) for r in runs]}  consistent-FN={len(miss_all)}")
        for (sent, fp) in miss_all:
            consistent.append((proj, sent, fp))
    print(f"\n  TOTAL enrolled gold(doc-code files) = {total_gold}")
    print(f"  per-run file-FN totals: " + ", ".join(f"{RUNS[i]}={perrun_counts[i]}" for i in range(3)))
    print(f"  CONSISTENT file-FN (missed in all 3 runs) = {len(consistent)}")
    return consistent, file_to_comps_all

def dump_sadcode(consistent, file_to_comps_all):
    print("\n  --- CONSISTENT doc-code FN: which COMPONENTS lose files, per project ---")
    nm_cache = {}
    for proj in PROJECTS:
        rows = [c for c in consistent if c[0] == proj]
        if not rows:
            continue
        f2c = file_to_comps_all[proj]
        nm = elem_names(proj)
        # group missed files by component (ae_id) and by sentence
        comp_missed = defaultdict(set)     # ae_id -> set(files)
        sent_missed = defaultdict(set)     # sent  -> set(files)
        nocomp = 0
        for _, sent, fp in rows:
            comps = f2c.get(fp, set())
            if not comps:
                nocomp += 1
            for ae in comps:
                comp_missed[ae].add((sent, fp))
            sent_missed[sent].add(fp)
        print(f"\n  [{proj}]  consistent file-FN={len(rows)}  distinct missed sentences={len(sent_missed)}")
        for ae, files in sorted(comp_missed.items(), key=lambda x: -len(x[1])):
            print(f"      {nm.get(ae, ae):<34} missed_file_links={len(files)}"
                  f"  sentences={sorted({s for s,_ in files}, key=lambda v:int(v) if v.isdigit() else 0)}")
        if nocomp:
            print(f"      (files with no SAM-CODE component: {nocomp})")

if __name__ == "__main__":
    c1 = analyze_sadsam()
    dump_sadsam(c1)
    c2, f2c = analyze_sadcode()
    dump_sadcode(c2, f2c)
