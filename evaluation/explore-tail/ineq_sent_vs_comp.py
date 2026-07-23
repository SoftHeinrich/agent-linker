#!/usr/bin/env python3
"""DOC-CODE gold link inequality: by SENTENCE vs by COMPONENT.
Same enrolled (sentence,file) gold link set, grouped two ways:
  by component : how many links each architecture component owns
  by sentence  : how many links each documentation sentence owns
Report Gini + top-1/top-3 share for each, per project + pooled. Higher Gini = more
unequal. Enrolled gold; component universe = D-12 (interfaces dropped, suite grain)."""
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mini-src"))
import metrics as m

PROJ = ["mediastore","teastore","teammates","bigbluebutton","jabref"]
SHORT = {"mediastore":"MediaStore","teastore":"TeaStore","teammates":"Teammates",
         "bigbluebutton":"BigBlueButton","jabref":"JabRef"}

def gini(xs):
    xs = sorted(xs)
    n = len(xs); tot = sum(xs)
    if n == 0 or tot == 0: return 0.0
    cum = sum((i+1)*v for i, v in enumerate(xs))
    return (2*cum)/(n*tot) - (n+1)/n

def topk(counter, k):
    vals = sorted(counter.values(), reverse=True); tot = sum(vals)
    return 100*sum(vals[:k])/tot if tot else 0.0

def links_for(project):
    cf = m.load_code_model_files(project)
    enrolled = m.enroll(m.load_gs_sad_code_raw(project), cf)   # (sentence, file)
    f2c = m.load_file_to_comps(project, cf)                    # file -> {component}
    valid = [(s, f) for (s, f) in enrolled if f in f2c]        # file maps to a kept comp
    by_sent = Counter(s for (s, f) in valid)
    by_comp = Counter(c for (s, f) in valid for c in f2c[f])
    return valid, by_sent, by_comp

print(f"{'project':<14}{'#links':>8}{'#sent':>7}{'#comp':>7}"
      f"{'GINI_sent':>11}{'GINI_comp':>11}{'top1_s%':>9}{'top1_c%':>9}{'more_unequal':>14}")
agg = {"gs":[], "gc":[]}
all_by_sent, all_by_comp = Counter(), Counter()
for p in PROJ:
    valid, bs, bc = links_for(p)
    gs, gc = gini(list(bs.values())), gini(list(bc.values()))
    agg["gs"].append(gs); agg["gc"].append(gc)
    # pool across projects (namespace ids so they don't collide)
    for s,c in [(f"{p}:{k}",v) for k,v in bs.items()]: all_by_sent[s]+=c
    for s,c in [(f"{p}:{k}",v) for k,v in bc.items()]: all_by_comp[s]+=c
    winner = "SENTENCE" if gs > gc else ("COMPONENT" if gc > gs else "tie")
    print(f"{SHORT[p]:<14}{len(valid):>8}{len(bs):>7}{len(bc):>7}"
          f"{gs:>11.3f}{gc:>11.3f}{topk(bs,1):>9.1f}{topk(bc,1):>9.1f}{winner:>14}")

macro_s = sum(agg["gs"])/len(agg["gs"]); macro_c = sum(agg["gc"])/len(agg["gc"])
print(f"\n{'MACRO Gini':<14}{'':>22}{macro_s:>11.3f}{macro_c:>11.3f}"
      f"   -> {'SENTENCE more unequal' if macro_s>macro_c else 'COMPONENT more unequal'} "
      f"(by {abs(macro_s-macro_c):.3f})")
print(f"{'POOLED Gini':<14}{'':>22}{gini(list(all_by_sent.values())):>11.3f}"
      f"{gini(list(all_by_comp.values())):>11.3f}")
