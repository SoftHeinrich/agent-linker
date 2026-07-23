#!/usr/bin/env python3
"""DOC-MODEL (SAD-SAM) gold link inequality: by SENTENCE vs by COMPONENT.
Links are (sentence, component) directly -- NO enrollment.
  by component : #sentences each component is linked to
  by sentence  : #components each sentence is linked to
Gini + top-1 share, per project + macro + pooled. Contrast with doc-code."""
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mini-src"))
import metrics as m

PROJ = ["mediastore","teastore","teammates","bigbluebutton","jabref"]
SHORT = {"mediastore":"MediaStore","teastore":"TeaStore","teammates":"Teammates",
         "bigbluebutton":"BigBlueButton","jabref":"JabRef"}

def gini(xs):
    xs = sorted(xs); n=len(xs); tot=sum(xs)
    if n==0 or tot==0: return 0.0
    cum=sum((i+1)*v for i,v in enumerate(xs))
    return (2*cum)/(n*tot)-(n+1)/n
def top1(c):
    v=sorted(c.values(),reverse=True); t=sum(v); return 100*v[0]/t if t else 0.0

print(f"{'project':<14}{'#links':>8}{'#sent':>7}{'#comp':>7}"
      f"{'GINI_sent':>11}{'GINI_comp':>11}{'top1_s%':>9}{'top1_c%':>9}{'more_unequal':>14}")
gs_l=[]; gc_l=[]; allS=Counter(); allC=Counter()
for p in PROJ:
    gold=m.load_gs_sad_sam(p)              # (component, sentence)
    by_comp=Counter(c for c,s in gold)
    by_sent=Counter(s for c,s in gold)
    gs,gc=gini(list(by_sent.values())),gini(list(by_comp.values()))
    gs_l.append(gs); gc_l.append(gc)
    for k,v in by_sent.items(): allS[f"{p}:{k}"]+=v
    for k,v in by_comp.items(): allC[f"{p}:{k}"]+=v
    w="SENTENCE" if gs>gc else ("COMPONENT" if gc>gs else "tie")
    print(f"{SHORT[p]:<14}{len(gold):>8}{len(by_sent):>7}{len(by_comp):>7}"
          f"{gs:>11.3f}{gc:>11.3f}{top1(by_sent):>9.1f}{top1(by_comp):>9.1f}{w:>14}")
ms=sum(gs_l)/len(gs_l); mc=sum(gc_l)/len(gc_l)
print(f"\n{'MACRO Gini':<14}{'':>22}{ms:>11.3f}{mc:>11.3f}"
      f"   -> {'SENTENCE more unequal' if ms>mc else 'COMPONENT more unequal'} (by {abs(ms-mc):.3f})")
print(f"{'POOLED Gini':<14}{'':>22}{gini(list(allS.values())):>11.3f}{gini(list(allC.values())):>11.3f}")
print("\n(for contrast, doc-CODE macro was: sentence 0.484, component 0.524 -> component more unequal)")
