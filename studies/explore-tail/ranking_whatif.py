#!/usr/bin/env python3
"""WHAT-IF: turn the set-output ArchLinker into a RANKING system.

Today the linker emits a SET: validators hard-DROP candidate links. A ranking
system would instead keep every proposed link and ORDER them — kept links in a top
tier, validator-killed links in a lower tier (recoverable by lowering the cutoff).
Question: the recall/coverage we sacrifice by dropping — is it RECOVERABLE (the gold
link sits in the killed pile, just needs a lower rank) or ABSENT (never proposed, no
rank position exists)? And at what precision cost?

Decompose each gold MISS (FN of the Full set output):
  recoverable_TP = FN ∩ killed     (proposed then validator-killed -> ranked tier 2)
  absent_TP      = FN − killed     (never proposed -> no rank, unrecoverable)
Tiers: tier1 = final (kept); tier2 = killed = ent_rejected ∪ cor_rejected.
Also: do any SILENT-FAILURE components (0 correct in Full) have killed gold
candidates — i.e. would ranking 'un-abandon' a component the set output gave up on?

GPT-5.4 S21, mean of run1/run2/run3, doc-model (SAD-SAM)."""
import os, sys
from collections import defaultdict
from pathlib import Path
from _roots import MINI_RQ34, SOTA    # shared roots (see _roots.py)

os.environ.setdefault("RQ34_OPENAI_SLOT", str(SOTA / "phase-cache-s21/v2.6.6_s21_gpt"))
sys.path.insert(0, str(MINI_RQ34))
import rq34
rq34.install_unpickler()

BACKEND="openai"; RUNS=["run1","run2","run3"]
SLOT=rq34.SLOTS[BACKEND]

def comp(k): return k[1]   # LinkKey = (sentence_number, component_id)

print("doc-model, S21 GPT, mean of 3 runs\n")
hdr=("proj","gold","kept_TP","rec_Full","killed_TP","killed_FP","rec_CEIL","absentTP","killedComp_recov")
print(f"{hdr[0]:<14}{hdr[1]:>6}{hdr[2]:>8}{hdr[3]:>9}{hdr[4]:>10}{hdr[5]:>10}{hdr[6]:>9}{hdr[7]:>9}{hdr[8]:>17}")
agg=defaultdict(float); n=0
sf_total=0; sf_recoverable=0   # silent-failure components; of those, recoverable via killed pile
for proj in rq34.PROJECTS:
    accs=defaultdict(list); sfc=[]; sfr=[]
    for run in RUNS:
        cell=rq34.compute_cell(SLOT,run,BACKEND,proj)
        gold=cell.gold; final=cell.final
        killed=cell.ent_rejected | cell.cor_rejected
        fn = gold - final
        recoverable = fn & killed         # gold misses that sit in the killed pile
        absent = fn - killed              # gold misses never proposed
        killed_tp = killed & gold
        killed_fp = killed - gold
        rec_full = len(final & gold)/len(gold)
        rec_ceil = len((final|killed) & gold)/len(gold)
        accs["gold"].append(len(gold)); accs["kept_tp"].append(len(final&gold))
        accs["rec_full"].append(rec_full); accs["killed_tp"].append(len(killed_tp))
        accs["killed_fp"].append(len(killed_fp)); accs["rec_ceil"].append(rec_ceil)
        accs["absent"].append(len(absent))
        # silent-failure components: gold comps with 0 correct link in final
        gold_comps=defaultdict(set); reached=defaultdict(set)
        for s,c in gold: gold_comps[c].add(s)
        for k in (final & gold): reached[k[1]].add(k[0])
        sf=[c for c in gold_comps if not reached.get(c)]
        # of those, how many have a killed gold candidate (recoverable by ranking)?
        killed_comp_gold=defaultdict(set)
        for k in recoverable: killed_comp_gold[k[1]].add(k[0])
        recov=[c for c in sf if killed_comp_gold.get(c)]
        sfc.append(len(sf)); sfr.append(len(recov))
    a=lambda k: sum(accs[k])/len(accs[k])
    print(f"{proj:<14}{a('gold'):>6.0f}{a('kept_tp'):>8.1f}{a('rec_full'):>9.3f}"
          f"{a('killed_tp'):>10.1f}{a('killed_fp'):>10.1f}{a('rec_ceil'):>9.3f}{a('absent'):>9.1f}"
          f"{(sum(sfr)/len(sfr)):>8.2f}/{(sum(sfc)/len(sfc)):<.2f}")
    for k in ("rec_full","rec_ceil","killed_tp","killed_fp","absent"): agg[k]+=a(k)
    n+=1
print(f"\nMACRO rec_Full={agg['rec_full']/n:.3f}  rec_CEILING(if rank+keep)={agg['rec_ceil']/n:.3f}"
      f"  (recoverable recall headroom = +{(agg['rec_ceil']-agg['rec_full'])/n*100:.1f}pp)")
print(f"MACRO per project: killed_TP={agg['killed_tp']/n:.1f} recoverable, "
      f"killed_FP={agg['killed_fp']/n:.1f} noise, absent_TP={agg['absent']/n:.1f} unrecoverable")
