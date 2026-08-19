"""Why does the merged reading (s26) lose 5 links and gain 6 false ones?

The claim to test is a mechanism, not a score: the per-passage reading builds a
table that (a) misses names a document-wide pass finds, costing recall, and
(b) admits names nothing judged, costing precision -- and because the full-name
linker's output is locked into the union by the earlier-wins merge and subtracted
from the two stricter linkers, both errors compound instead of being caught later.

Everything is read off the checkpoints the two variants already wrote. No LLM
call.

  D1  TABLE DIFF        which names each variant's table has that the other's
                        does not.
  D2  WHERE THE LINKS GO per linker, TP and FP for both variants, so the loss can
                        be attributed to a stage rather than to the pipeline.
  D3  ALIAS-ATTRIBUTED  of s26's extra false positives, how many are admitted
      ERRORS            only because of a name only s26's table has; of s25's
                        extra true positives, how many needed a name only s25 has.
  D4  SUBTRACTION       how many pairs the full-name linker took in one variant
      SPILLOVER         were left to a later, stricter linker in the other.

Usage: ../.venv/bin/python pilot/s26_diagnosis.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import PROJECTS, load_project, load_gold
import pickle

S25_RUNS = [Path("../results/s25_simplified_e2e_r1_20260810"),
            Path("../results/s25_simplified_e2e_r2_20260810"),
            Path("../results/s25_simplified_e2e_r3_20260810")]
S26_RUNS = [Path(f"../results/s26_unified_e2e_r{i}_20260812") for i in (1, 2, 3)]
LINKERS = ("full_name", "partial_name", "coreference")


def phase(run, variant, project, name):
    path = (run / "phase_states" / variant / "openai" / project / f"{name}.pkl")
    with path.open("rb") as handle:
        return pickle.load(handle)


def table(run, variant, project):
    knowledge = phase(run, variant, project, "knowledge")
    return {t: getattr(e, "component", e)
            for t, e in knowledge["doc_knowledge"].aliases.items()}


def links_by_linker(run, variant, project):
    out = {}
    for linker in LINKERS:
        state = phase(run, variant, project, f"linker_{linker}")
        out[linker] = {(l.sentence_number, l.component_id) for l in state["links"]}
    return out


def candidates_of(run, variant, project, linker):
    state = phase(run, variant, project, f"linker_{linker}")
    key = "candidates" if "candidates" in state["feedback"] else "proposed"
    return state["feedback"].get(key, [])


# ── D1 ───────────────────────────────────────────────────────────────────────

def diff_tables():
    print("\n### D1 what each reading finds that the other does not")
    totals = Counter()
    for project in PROJECTS:
        a = set()
        b = set()
        for run in S25_RUNS:
            a |= {t.casefold() for t in table(run, "s_linker25", project)}
        for run in S26_RUNS:
            b |= {t.casefold() for t in table(run, "s_linker26", project)}
        only25, only26 = sorted(a - b), sorted(b - a)
        print(f"  {project:14s} s25 {len(a):2d} | s26 {len(b):2d} | shared "
              f"{len(a & b):2d}")
        if only25:
            print(f"      only the document-wide pass finds: {only25}")
        if only26:
            print(f"      only the per-passage reading finds: {only26}")
        totals.update(s25=len(a), s26=len(b), shared=len(a & b),
                      only25=len(only25), only26=len(only26))
    print(f"  TOTAL          s25 {totals['s25']} | s26 {totals['s26']} | shared "
          f"{totals['shared']} | only-s25 {totals['only25']} | only-s26 "
          f"{totals['only26']}")
    return dict(totals)


# ── D2 ───────────────────────────────────────────────────────────────────────

def per_linker():
    print("\n### D2 where the links are won and lost, per linker")
    print(f"  {'':14s} {'s25 TP / FP per linker':>34s}   "
          f"{'s26 TP / FP per linker':>34s}")
    grand = {"s_linker25": Counter(), "s_linker26": Counter()}
    for project in PROJECTS:
        gold = load_gold(project)
        cells = {}
        for variant, runs in (("s_linker25", S25_RUNS), ("s_linker26", S26_RUNS)):
            per = {linker: [0, 0] for linker in LINKERS}
            for run in runs:
                got = links_by_linker(run, variant, project)
                for linker in LINKERS:
                    tp = len(got[linker] & gold)
                    per[linker][0] += tp
                    per[linker][1] += len(got[linker]) - tp
            for linker in LINKERS:
                per[linker] = [round(v / len(runs), 1) for v in per[linker]]
                grand[variant][f"{linker}_tp"] += per[linker][0]
                grand[variant][f"{linker}_fp"] += per[linker][1]
            cells[variant] = "  ".join(
                f"{linker[:4]} {per[linker][0]:5.1f}/{per[linker][1]:4.1f}"
                for linker in LINKERS)
        print(f"  {project:14s} {cells['s_linker25']}   {cells['s_linker26']}")
    print()
    for linker in LINKERS:
        a, b = grand["s_linker25"], grand["s_linker26"]
        print(f"  {linker:14s} s25 TP {a[f'{linker}_tp']:6.1f} FP "
              f"{a[f'{linker}_fp']:5.1f} | s26 TP {b[f'{linker}_tp']:6.1f} FP "
              f"{b[f'{linker}_fp']:5.1f} | delta TP "
              f"{b[f'{linker}_tp'] - a[f'{linker}_tp']:+5.1f} FP "
              f"{b[f'{linker}_fp'] - a[f'{linker}_fp']:+5.1f}")
    return {v: dict(c) for v, c in grand.items()}


# ── D3 ───────────────────────────────────────────────────────────────────────

def attribute_to_names():
    """Are the extra errors and the missing links explained by the table diff?"""
    print("\n### D3 how much of the difference the table diff explains")
    from llm_sad_sam.linkers.experimental.s_linker25 import SLinker25
    totals = Counter()
    for project in PROJECTS:
        gold = load_gold(project)
        info = load_project(project)
        sent_map = info["sent_map"]
        id_to_name = {c.id: c.name for c in info["components"]}

        t25 = {}
        for run in S25_RUNS:
            t25.update(table(run, "s_linker25", project))
        t26 = {}
        for run in S26_RUNS:
            t26.update(table(run, "s_linker26", project))
        only26 = {t: c for t, c in t26.items() if t.casefold() not in
                  {x.casefold() for x in t25}}
        only25 = {t: c for t, c in t25.items() if t.casefold() not in
                  {x.casefold() for x in t26}}

        def final(runs, variant):
            counted = Counter()
            for run in runs:
                state = phase(run, variant, project, "final")
                for link in state["final"]:
                    counted[(link.sentence_number, link.component_id)] += 1
            return {k for k, v in counted.items() if v >= 2}   # in most runs

        f25, f26 = final(S25_RUNS, "s_linker25"), final(S26_RUNS, "s_linker26")
        extra_fp = [k for k in f26 - f25 if k not in gold]
        missing_tp = [k for k in f25 - f26 if k in gold]

        def needs(pairs, names):
            hits = 0
            for snum, cid in pairs:
                comp = id_to_name.get(cid)
                sent = sent_map.get(snum)
                if not comp or not sent:
                    continue
                if SLinker25._find_exact_form(sent.text, comp):
                    continue          # the name itself is there; no alias needed
                if any(c == comp and SLinker25._find_exact_form(sent.text, t)
                       for t, c in names.items()):
                    hits += 1
            return hits

        fp_by_name = needs(extra_fp, only26)
        tp_by_name = needs(missing_tp, only25)
        print(f"  {project:14s} s26-only false positives {len(extra_fp):2d} "
              f"(explained by an s26-only name: {fp_by_name:2d}) | s25-only true "
              f"positives {len(missing_tp):2d} (explained by an s25-only name: "
              f"{tp_by_name:2d})")
        totals.update(extra_fp=len(extra_fp), fp_by_name=fp_by_name,
                      missing_tp=len(missing_tp), tp_by_name=tp_by_name)
    print(f"  TOTAL          extra FP {totals['extra_fp']} (name-explained "
          f"{totals['fp_by_name']}) | missing TP {totals['missing_tp']} "
          f"(name-explained {totals['tp_by_name']})")
    return dict(totals)


# ── D4 ───────────────────────────────────────────────────────────────────────

def subtraction_spillover():
    """Pairs one variant settles in the full-name linker and the other passes on."""
    print("\n### D4 what the subtraction hands to the stricter linkers")
    totals = Counter()
    for project in PROJECTS:
        gold = load_gold(project)
        per = Counter()
        for variant, runs in (("s_linker25", S25_RUNS), ("s_linker26", S26_RUNS)):
            for run in runs:
                got = links_by_linker(run, variant, project)
                per[f"{variant}_full"] += len(got["full_name"])
                per[f"{variant}_later"] += len(got["partial_name"]) + len(
                    got["coreference"])
                per[f"{variant}_later_tp"] += len(
                    (got["partial_name"] | got["coreference"]) & gold)
        n = len(S25_RUNS)
        print(f"  {project:14s} full-name links s25 {per['s_linker25_full']/n:5.1f} "
              f"s26 {per['s_linker26_full']/n:5.1f} | links from the two stricter "
              f"linkers s25 {per['s_linker25_later']/n:4.1f} "
              f"(TP {per['s_linker25_later_tp']/n:4.1f}) s26 "
              f"{per['s_linker26_later']/n:4.1f} "
              f"(TP {per['s_linker26_later_tp']/n:4.1f})")
        totals.update(per)
    n = len(S25_RUNS)
    print(f"  TOTAL          full-name s25 {totals['s_linker25_full']/n:.1f} vs "
          f"s26 {totals['s_linker26_full']/n:.1f} | later linkers s25 "
          f"{totals['s_linker25_later']/n:.1f} (TP "
          f"{totals['s_linker25_later_tp']/n:.1f}) vs s26 "
          f"{totals['s_linker26_later']/n:.1f} (TP "
          f"{totals['s_linker26_later_tp']/n:.1f})")
    return {k: round(v / n, 1) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("../results/s26_unified_e2e_r1_20260812/diagnosis.json"))
    args = parser.parse_args()
    report = {
        "D1_table_diff": diff_tables(),
        "D2_per_linker": per_linker(),
        "D3_attribution": attribute_to_names(),
        "D4_spillover": subtraction_spillover(),
    }
    with args.out.open("w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
