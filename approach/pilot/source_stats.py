"""Split a paired A/B by the linker that produced each link, and test each part.

Motivated by a result that cannot be taken at face value. `s_linker50` changes one
prompt constant, `COREF_RULES`, which is read by the *last* of three linkers. Six
paired runs scored it TP -3.0 at p = 0.01 — and the link-level diff put 3.2 of
those lost true positives on the **full-name** linker, which runs first and whose
every prompt byte is identical between the two arms. A change cannot lose links in
a stage it does not reach.

The explanation is in the harness, not the linker: each arm runs the whole pipeline
from scratch, including the stages it does not modify, and those stages are LLM
calls with their own run-to-run spread. Pairing arms inside one invocation controls
the model, the day and the ordering; it does not control the *upstream* sampling.
So a late-stage arm is scored partly on noise injected before it — and this branch
has been reading whole-pipeline p values for twenty variants.

This script bounds that. It restricts both arms' link sets to the links a given
linker produced and re-runs the same exact permutation test per source, so a change
can be judged on the stages it can actually reach and the rest can be read as what
it is: the spread of the stages both arms share.

    ../.venv/bin/python pilot/source_stats.py \
        --arm s_linker49 ../results/s5051_e2e_r*_20260813 \
        --arm s_linker50 ../results/s5051_e2e_r*_20260813

`--reachable-from full_name|partial_name|coreference` marks which sources the arm's
change can reach, and the report says so per source rather than leaving the reader
to work it out.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from ab_stats import permutation_report                             # noqa: E402
from design_audit import PROJECTS, load_gold                        # noqa: E402

ORDER = ("full_name", "partial_name", "coreference")


def gold_all():
    out = set()
    for project in PROJECTS:
        out |= {(project, snum, cid) for snum, cid in load_gold(project)}
    return out


GOLD = gold_all()
GOLD_BY_PROJECT = {p: {k for k in GOLD if k[0] == p} for p in PROJECTS}


def load_run(run: Path, variant: str, sources=None):
    links = set()
    found = 0
    for project in PROJECTS:
        path = run / f"{variant}_{project}_links.csv"
        if not path.exists():
            continue
        found += 1
        with path.open() as handle:
            for row in csv.DictReader(handle):
                source = (row.get("source") or "").split("_variant")[0]
                if sources is not None and source not in sources:
                    continue
                links.add((project, int(row["sentence"]), row["component_id"]))
    return links if found == len(PROJECTS) else None


def scores(links):
    tp = len(links & GOLD)
    fp = len(links - GOLD)
    f1s, f2s = [], []
    for project, gold in GOLD_BY_PROJECT.items():
        got = {k for k in links if k[0] == project}
        hit = len(got & gold)
        precision = hit / len(got) if got else 0.0
        recall = hit / len(gold) if gold else 0.0
        f1s.append(0.0 if not (precision + recall) else
                   2 * precision * recall / (precision + recall))
        f2s.append(0.0 if not (4 * precision + recall) else
                   5 * precision * recall / (4 * precision + recall))
    return {"TP": tp, "FP": fp,
            "macro F1": 100 * sum(f1s) / len(f1s),
            "macro F2": 100 * sum(f2s) / len(f2s)}


QUALITY = {name: (lambda links, key=name: scores(links)[key])
           for name in ("TP", "FP")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", nargs="+", action="append", required=True,
                        metavar=("VARIANT", "RUN"))
    parser.add_argument("--reachable-from", default=None,
                        choices=ORDER,
                        help="the earliest linker the arm's change can reach")
    args = parser.parse_args()
    if len(args.arm) != 2:
        parser.error("exactly two --arm groups")

    reach_from = ORDER.index(args.reachable_from) if args.reachable_from else None

    arms = {}
    for arm in args.arm:
        variant, runs = arm[0], [Path(p) for p in arm[1:]]
        arms[variant] = (variant, runs)

    names = list(arms)
    for label, sources in [("ALL", None)] + [(s, {s}) for s in ORDER]:
        sets = {}
        for variant, (_, runs) in arms.items():
            loaded = [load_run(r, variant, sources) for r in runs]
            loaded = [x for x in loaded if x is not None]
            sets[variant] = loaded
        if len({len(v) for v in sets.values()}) != 1 or not all(sets.values()):
            print(f"\n{label}: incomplete runs — skipped")
            continue
        reachable = ""
        if reach_from is not None and label != "ALL":
            index = ORDER.index(label)
            reachable = ("  [the change CAN reach this stage]" if index >= reach_from
                         else "  [the change CANNOT reach this stage — "
                              "any difference here is shared-stage spread]")
        print(f"\n{'=' * 78}\n{label}{reachable}")
        for variant, runs in sets.items():
            mean = {k: sum(scores(r)[k] for r in runs) / len(runs)
                    for k in ("TP", "FP")}
            print(f"  {variant:14s} links/run "
                  f"{sum(len(r) for r in runs) / len(runs):6.1f}   "
                  f"TP {mean['TP']:6.1f}   FP {mean['FP']:5.1f}")
        permutation_report(sets, quality=QUALITY,
                           title=f"{names[1]} minus {names[0]} — {label}")


if __name__ == "__main__":
    main()
