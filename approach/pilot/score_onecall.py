"""RQ4's total floor, scored per project before the macro.

`s_linker110_onecall` sends every document whole, and `s_linker27` measured that
accuracy tracks document length (jabref 13 sentences 100.0, teammates 198 84.1). A
macro number therefore mixes the workflow's worth with a length effect, and the
per-project column is what separates them. This prints both, with the document length
beside each project so the confound is visible in the table rather than in a footnote.

**The control is cross-set by decision** -- `s_linker110` as run today in
`noevidence_e2e_{model}_r{1,2,3}_20260902`, a different invocation from the arm's. That
arm read macro F1 93.85 in one of today's sets and 92.90 in another, so ~1 F1 of drift
sits on top of every delta below. Do not quote these as in-set results.

No LLM calls.

    ../.venv/bin/python pilot/score_onecall.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import PROJECTS, load_gold                        # noqa: E402
from llm_sad_sam.core.document_loader_v2 import load_sentences      # noqa: E402
import design_audit as DA                                           # noqa: E402

RESULTS = Path("../results")
CONTROL = "noevidence_e2e_{model}_r{i}_20260902"
CONTROL_ARM = "s_linker110"
ARM_DIR = "onecall_e2e_{model}_r{i}_20260902"
ARM = "s_linker110_onecall"


def links(run_dir: Path, arm: str, project: str):
    f = run_dir / f"{arm}_{project}_links.csv"
    if not f.exists():
        return None
    return {(int(r["sentence"]), r["component_id"]) for r in csv.DictReader(open(f))}


def prf(pred, gold):
    tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    f2 = 5 * p * r / (4 * p + r) if p + r else 0.0
    return tp, fp, fn, p, r, f1, f2


def doc_length(project: str) -> int:
    """Sentence count, from the same path table every pilot on this branch reads."""
    try:
        return len(load_sentences(str(DA.BENCH / DA.PROJECTS[project][0])))
    except Exception:
        return -1


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


for model in ("terra", "luna"):
    print(f"\n{'=' * 78}\n{model}\n{'=' * 78}")
    print(f"{'project':<15}{'sents':>6}{'  ':>2}"
          f"{'head F1':>9}{'floor F1':>10}{'dF1':>8}   "
          f"{'head F2':>9}{'floor F2':>10}{'dF2':>8}")
    macro = {"head_f1": [], "arm_f1": [], "head_f2": [], "arm_f2": []}
    for project in PROJECTS:
        gold = {(s, c) for s, c in load_gold(project)}
        hf1, af1, hf2, af2 = [], [], [], []
        for i in (1, 2, 3):
            h = links(RESULTS / CONTROL.format(model=model, i=i), CONTROL_ARM, project)
            a = links(RESULTS / ARM_DIR.format(model=model, i=i), ARM, project)
            if h is None or a is None:
                continue
            hf1.append(prf(h, gold)[5]); hf2.append(prf(h, gold)[6])
            af1.append(prf(a, gold)[5]); af2.append(prf(a, gold)[6])
        if not hf1 or not af1:
            print(f"{project:<15}{'':>6}  (incomplete)")
            continue
        h1, a1, h2, a2 = mean(hf1), mean(af1), mean(hf2), mean(af2)
        macro["head_f1"].append(h1); macro["arm_f1"].append(a1)
        macro["head_f2"].append(h2); macro["arm_f2"].append(a2)
        print(f"{project:<15}{doc_length(project):>6}  "
              f"{h1:>9.3f}{a1:>10.3f}{(a1 - h1) * 100:>+8.1f}   "
              f"{h2:>9.3f}{a2:>10.3f}{(a2 - h2) * 100:>+8.1f}")
    if macro["head_f1"]:
        h1, a1 = mean(macro["head_f1"]), mean(macro["arm_f1"])
        h2, a2 = mean(macro["head_f2"]), mean(macro["arm_f2"])
        print(f"{'-' * 78}")
        print(f"{'MACRO':<15}{'':>6}  "
              f"{h1:>9.3f}{a1:>10.3f}{(a1 - h1) * 100:>+8.1f}   "
              f"{h2:>9.3f}{a2:>10.3f}{(a2 - h2) * 100:>+8.1f}")

print("\ncontrol is CROSS-SET (noevidence_*_20260902); ~1 F1 of invocation drift "
      "sits on every delta above.")
