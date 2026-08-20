"""Two offline questions about the full-name stage, answered on recorded runs.

No LLM calls. Reads the `linker_full_name` checkpoints of runs already on disk.

  claim   Does the claim-first instruction override the lenient rubric? The rubric
          says "a mention that says nothing further about the component still counts
          as a valid link"; the prompt then asks for the architectural claim and says
          to decide "based on that claim". If `claim == none` is near-always a
          rejection, the two are one contradiction and it decides cases.
  morph   What does the morphology clause admit? A candidate whose sentence does not
          write a name of the component at ANY_CASE is one only that clause (or a
          misread) can license. Counted against gold.

Usage (from approach/):  ../.venv/bin/python pilot/entity_prompt_audit.py <run-glob>...
"""
from __future__ import annotations

import collections
import csv
import glob
import os
import pickle
import re
import sys

BASE = "/mnt/hostshare/ardoco-home/alinker-replication-package"
sys.path.insert(0, BASE + "/approach/src")

from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository          # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker85 import (           # noqa: E402
    SLinker85, NameForm,
)

PROJECTS = {
    "mediastore": ("mediastore/model_2016/pcm/ms.repository",
                   "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv"),
    "teastore": ("teastore/model_2020/pcm/teastore.repository",
                 "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv"),
    "teammates": ("teammates/model_2021/pcm/teammates.repository",
                  "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "bigbluebutton": ("bigbluebutton/model_2021/pcm/bbb.repository",
                      "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "jabref": ("jabref/model_2021/pcm/jabref.repository",
               "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
}


def gold(path):
    with open(os.path.join(BASE, "benchmark", path)) as fh:
        return {(int(r["sentence"]), r["modelElementID"]) for r in csv.DictReader(fh)}


def names_of(component, knowledge):
    """The component's set of names: the model name plus its approved aliases."""
    out = {component.name}
    if knowledge is not None:
        out |= {term for term, comp in knowledge.aliases.items()
                if comp == component.name}
    return out


def written(text, names):
    """Does the sentence write any of these names at ANY_CASE?"""
    return any(SLinker85._name_spans(text, n, NameForm.ANY_CASE) for n in names)


def main(globs):
    runs = sorted(d for g in globs for d in glob.glob(os.path.join(BASE, "results", g)))
    claim = collections.Counter()
    morph = collections.Counter()
    morph_examples = []
    for run in runs:
        for variant in sorted(glob.glob(os.path.join(run, "phase_states", "*"))):
            vname = os.path.basename(variant)
            for proj, (model, gpath) in PROJECTS.items():
                fn = os.path.join(variant, "openai", proj, "linker_full_name.pkl")
                if not os.path.exists(fn):
                    continue
                state = pickle.load(open(fn, "rb"))
                kfn = os.path.join(variant, "openai", proj, "knowledge.pkl")
                knowledge = (pickle.load(open(kfn, "rb"))["doc_knowledge"]
                             if os.path.exists(kfn) else None)
                comps = parse_pcm_repository(os.path.join(BASE, "benchmark", model))
                by_id = {c.id: c for c in comps}
                by_name = {c.name: c for c in comps}
                g = gold(gpath)
                text_of = {(c["sentence"], c["component"]): c["text"]
                           for c in state["feedback"]["candidates"]}

                for d in state["feedback"]["judge_decisions"]:
                    if d.get("stage") != "full_name_judge":
                        continue
                    key = (d["sentence"], d["component_id"])
                    c = by_id.get(d["component_id"])
                    if c is None:
                        continue
                    none = (d.get("claim") or "").strip().lower() in ("", "none")
                    claim[(vname, none, bool(d["approved"]), key in g)] += 1

                for cand in state["feedback"]["candidates"]:
                    c = by_name.get(cand["component"])
                    if c is None:
                        continue
                    key = (cand["sentence"], c.id)
                    exact = written(cand["text"], names_of(c, knowledge))
                    morph[(vname, exact, key in g)] += 1
                    if not exact and len(morph_examples) < 40:
                        morph_examples.append(
                            (proj, cand["sentence"], cand["component"],
                             key in g, cand["text"][:110]))
    print(f"runs read: {len(runs)}")
    variants = sorted({k[0] for k in claim})
    for v in variants:
        n_runs = max(1, len(runs))
        print(f"\n=== {v} — claim-first, over {len(runs)} runs "
              f"(per-run averages in brackets) ===")
        print(f"{'claim':>10} {'verdict':>8} {'gold':>5} {'n':>6} {'per run':>8}")
        for none in (True, False):
            for appr in (True, False):
                for isg in (True, False):
                    n = claim[(v, none, appr, isg)]
                    if not n:
                        continue
                    print(f"{'none' if none else 'quoted':>10} "
                          f"{'approve' if appr else 'reject':>8} "
                          f"{'gold' if isg else '-':>5} {n:>6} {n / n_runs:>8.1f}")
        nq_rej_gold = claim[(v, True, False, True)]
        nq_tot = sum(claim[(v, True, a, g_)] for a in (0, 1) for g_ in (0, 1))
        nq_rej = sum(claim[(v, True, False, g_)] for g_ in (0, 1))
        if nq_tot:
            print(f"  claim=none: {nq_tot} cases, rejected {nq_rej} "
                  f"({100 * nq_rej / nq_tot:.1f}%), of which gold {nq_rej_gold} "
                  f"({nq_rej_gold / n_runs:.1f} per run)")
        print(f"\n=== {v} — candidates whose sentence writes no name at ANY_CASE ===")
        for exact in (False, True):
            for isg in (True, False):
                n = morph[(v, exact, isg)]
                if n:
                    print(f"{'writes a name' if exact else 'writes none':>14} "
                          f"{'gold' if isg else '-':>5} {n:>6} {n / n_runs:>8.1f}/run")
    if morph_examples:
        print("\nexamples of candidates with no ANY_CASE name in the sentence:")
        for proj, s, comp, isg, text in morph_examples[:20]:
            print(f"  {proj:<14} S{s:<4} {comp:<22} {'GOLD' if isg else '    '} {text}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["s85_e2e_terra_r*_20260820"])
