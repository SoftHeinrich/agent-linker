#!/usr/bin/env python3
"""FN taxonomy for s_linker21 (model-doc / SAD-SAM), gpt-5.4 slot.

Splits every remaining false negative into the stage that lost it, so we know how
much recall is *judge-recoverable in principle* (a validator saw the candidate and
rejected it) vs *proposal-limited* (extraction + coref never surfaced it, so no
judge ever saw it).

Buckets per FN (sentence, component_id):
  ENTITY-REJECTED   in entity.candidates, not in entity.validated  -> judge said no
  COREF-REJECTED    in coref.raw,        not in coref.validated    -> coref judge said no
  NEVER-PROPOSED    in neither candidate pool                      -> extraction/coref miss
  KEPT-NOT-FINAL    kept by a validator but absent from final      -> should be ~0 (merge bug)

Pickle-free: reads the neutral extract JSONs (results/v2.6.6_extracts_s21/gpt/run*/<proj>.json).
Gold via the benchmark GS_SAD_SAM csv, keyed (int sentence, component_id) to match the extracts.
"""
import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

ARD = Path("/mnt/hostshare/ardoco-home")
BENCH = Path(os.environ.get(
    "TRANSARC_BENCHMARK",
    ARD / "ardoco/core/tests-base/src/main/resources/benchmark"))
EXTRACTS = ARD / "agent-linker/results/v2.6.6_extracts_s21/gpt"
PROJECTS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]
RUNS = ["run1", "run2", "run3"]

GS_SAD_SAM = {
    "mediastore":    "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    "teastore":      "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    "teammates":     "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "bigbluebutton": "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "jabref":        "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
}


def load_gold(project):
    """set[(int sentence, component_id)]."""
    gold = set()
    with (BENCH / GS_SAD_SAM[project]).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gold.add((int(r["sentence"]), r["modelElementID"]))
    return gold


def sentences(project):
    hits = glob.glob(str(BENCH / project / "text_*" / f"{project}.txt"))
    d = {}
    if hits:
        with open(hits[0], encoding="utf-8", errors="replace") as f:
            for i, ln in enumerate(f, 1):
                d[i] = ln.strip()
    return d


def load_extract(project, run):
    p = EXTRACTS / run / f"{project}.json"
    return json.load(open(p))


def pairset(items):
    return {(it["s"], it["c"]) for it in items}


def name_map(project):
    """component_id -> display name, from the extract candidate/final rosters."""
    nm = {}
    for run in RUNS:
        d = load_extract(project, run)
        for grp in (d["entity"]["candidates"], d["final"]["links"],
                    d["coref"]["raw"]):
            for it in grp:
                if it.get("c") and it.get("component_name"):
                    nm.setdefault(it["c"], it["component_name"])
    return nm


def classify(project, run, gold):
    d = load_extract(project, run)
    final = pairset(d["final"]["links"])
    ent_cand = pairset(d["entity"]["candidates"])
    ent_kept = pairset(d["entity"]["validated"])
    cor_cand = pairset(d["coref"]["raw"])
    cor_kept = pairset(d["coref"]["validated"])
    fn = gold - final
    out = {}
    for g in fn:
        if g in ent_cand and g not in ent_kept:
            cat = "ENTITY-REJECTED"
        elif g in cor_cand and g not in cor_kept:
            cat = "COREF-REJECTED"
        elif g not in ent_cand and g not in cor_cand:
            cat = "NEVER-PROPOSED"
        else:
            cat = "KEPT-NOT-FINAL"
        out[g] = cat
    return out, final, fn


def main():
    per_run_fn = defaultdict(int)
    per_run_cat = defaultdict(lambda: defaultdict(int))
    # consistent = missed in all 3 runs; its category = category in run1 (stable enough)
    consistent = {}
    consistent_cat = {}
    for proj in PROJECTS:
        gold = load_gold(proj)
        run_fn = []
        run_cls = []
        for run in RUNS:
            cls, final, fn = classify(proj, run, gold)
            run_fn.append(fn)
            run_cls.append(cls)
            per_run_fn[run] += len(fn)
            for g, c in cls.items():
                per_run_cat[run][c] += 1
        cons = run_fn[0] & run_fn[1] & run_fn[2]
        for g in cons:
            # category: prefer a REJECTED verdict if any run rejected it, else run1's
            cats = [run_cls[i].get(g) for i in range(3)]
            rej = [c for c in cats if c in ("ENTITY-REJECTED", "COREF-REJECTED")]
            consistent[(proj, g)] = cats
            consistent_cat[(proj, g)] = rej[0] if rej else cats[0]

    nm = {p: name_map(p) for p in PROJECTS}
    sn = {p: sentences(p) for p in PROJECTS}

    print("=" * 84)
    print("s21 model-doc FN taxonomy (gpt-5.4)  — where each false negative was lost")
    print("=" * 84)
    print("\nPer-run FN totals and category split:")
    cats = ["ENTITY-REJECTED", "COREF-REJECTED", "NEVER-PROPOSED", "KEPT-NOT-FINAL"]
    hdr = f"  {'run':<8}{'FN':>5}   " + "".join(f"{c:>18}" for c in cats)
    print(hdr)
    for run in RUNS:
        row = f"  {run:<8}{per_run_fn[run]:>5}   " + "".join(
            f"{per_run_cat[run][c]:>18}" for c in cats)
        print(row)

    print(f"\nCONSISTENT FN (missed in ALL 3 runs) = {len(consistent)}")
    cc = defaultdict(int)
    for k, c in consistent_cat.items():
        cc[c] += 1
    for c in cats:
        if cc[c]:
            print(f"    {c:<18} {cc[c]}")

    print("\n--- CONSISTENT FN detail (proj  s# -> component | category | runs | sentence) ---")
    order = {"ENTITY-REJECTED": 0, "COREF-REJECTED": 1, "NEVER-PROPOSED": 2, "KEPT-NOT-FINAL": 3}
    rows = sorted(consistent.keys(),
                  key=lambda k: (k[0], order.get(consistent_cat[k], 9), k[1][0]))
    for (proj, (s, cid)) in rows:
        cat = consistent_cat[(proj, (s, cid))]
        name = nm[proj].get(cid, cid)
        txt = sn[proj].get(s, "")[:78]
        runcats = consistent[(proj, (s, cid))]
        tag = "".join({"ENTITY-REJECTED": "E", "COREF-REJECTED": "C",
                       "NEVER-PROPOSED": ".", "KEPT-NOT-FINAL": "K", None: "?"}[x]
                      for x in runcats)
        print(f"  {proj:<14} s{s:<4} {name:<22} {cat:<16} [{tag}] {txt}")


if __name__ == "__main__":
    main()
