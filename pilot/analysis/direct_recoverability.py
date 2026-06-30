#!/usr/bin/env python3
"""Pilot Step 3 — is the DIRECT route actionable?

For each direct-only gold link (sentence, code_path), check whether the sentence
text explicitly names the target's package or class. If so, a direct linker
(NER over code identifiers + path match) can recover it -> the direct route is
not just decidable but executable. This bounds the payoff of adding the route.
"""
import glob, importlib.util, re
from collections import defaultdict
from pathlib import Path
MINI=Path("/mnt/hostshare/ardoco-home/mono/evaluation/mini-src/metrics.py")
spec=importlib.util.spec_from_file_location("metrics",MINI); M=importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
BENCH=M.BENCHMARK

def sentences(p):
    h=glob.glob(str(BENCH/p/"text_*"/f"{p}.txt")); d={}
    if h:
        for i,l in enumerate(open(h[0],errors="replace"),1): d[str(i)]=l.strip()
    return d

def path_tokens(path):
    """package segments + class stem from a normalized code path."""
    # e.g. teammates-logic/.../logic/api/AccountsLogic.java
    parts=re.split(r'[/\\]', path)
    cls=parts[-1].rsplit('.',1)[0] if parts else ""
    segs=[s for s in parts[:-1] if s and s.lower() not in
          ("src","main","java","test","teammates","com","edu","kit","ipd","sdq")]
    return segs, cls

p="teammates"
S=sentences(p)
sc_raw=M.load_gs_sad_code_raw(p); ss_sents={s for _,s in M.load_gs_sad_sam(p)}
direct_links=[(s,fp) for (s,fp) in sc_raw if s not in ss_sents]
by_s=defaultdict(list)
for s,fp in direct_links: by_s[s].append(fp)

named=0; total=len(direct_links); sent_named=0
examples=[]
for s in sorted(by_s, key=lambda v:int(v) if v.isdigit() else 0):
    text=S.get(s,""); low=text.lower()
    # build dotted-package candidates present in sentence (e.g. logic.api, storage.entity, x.util)
    s_named_any=False
    for fp in by_s[s]:
        segs,cls=path_tokens(fp)
        hit=False
        # class name appears verbatim
        if cls and len(cls)>3 and re.search(r'\b'+re.escape(cls)+r'\b', text):
            hit=True
        # any package segment appears (as word or in a dotted identifier)
        for seg in segs:
            if len(seg)>2 and re.search(r'\b'+re.escape(seg.lower())+r'\b', low):
                hit=True; break
        if hit: named+=1; s_named_any=True
    if s_named_any: sent_named+=1
    if len(examples)<10:
        examples.append((s, text[:88], [fp.split('/')[-1] for fp in by_s[s][:3]]))

print(f"teammates direct-only: {len(by_s)} sentences, {total} raw gold links")
print(f"  gold links whose target package/class is NAMED in the sentence: {named}/{total} ({100*named/total:.0f}%)")
print(f"  direct-only sentences with >=1 nameable target: {sent_named}/{len(by_s)} ({100*sent_named/len(by_s):.0f}%)")
print("\n  examples (sentence -> gold target files):")
for s,t,fs in examples:
    print(f"    s{s:>3}: {t}\n          -> {fs}")
