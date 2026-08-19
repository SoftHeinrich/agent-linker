"""Where the partial-name linker's error budget actually is. No LLM calls.

The merged-alias round established that the two projects whose partial-name linker
fires (teammates, bigbluebutton) carry essentially all of the pipeline's false
positives, so the partial-name linker is the remaining frontier. Before changing it,
this splits its error budget three ways, off the recorded checkpoints of the paired
runs:

  * PROPOSER CEILING -- of the gold pairs no earlier linker produced, how many does
    `_name_word_candidates` even offer? A gold pair never proposed cannot be
    recovered by any judge.
  * JUDGE RECALL     -- of the gold pairs that were proposed, how many did the
    denotation judge approve?
  * JUDGE PRECISION  -- of the non-gold pairs proposed, how many did it approve?

and reports, per project, what a perfect judge over the current proposer would score
(the stage's headroom) against what the current judge scores.

Usage, from the approach/ directory:
    ../.venv/bin/python pilot/partial_audit.py
    ../.venv/bin/python pilot/partial_audit.py --arm s_linker59 --runs '../results/s5960_e2e_r*'
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import BENCH, PROJECTS, load_gold                  # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository           # noqa: E402


def id_by_name(project):
    model = BENCH / PROJECTS[project][1]
    return {c.name: c.id for c in parse_pcm_repository(str(model))}


def pairs(view, names):
    """`_link_view` rows -> {(sentence, component_id)}, dropping unknown names."""
    out = set()
    for row in view:
        cid = names.get(row.get("component"))
        if cid is not None:
            out.add((int(row["sentence"]), cid))
    return out


def audit(runs, arm):
    gold = {p: set(load_gold(p)) for p in PROJECTS}
    names = {p: id_by_name(p) for p in PROJECTS}
    per = {p: defaultdict(list) for p in PROJECTS}
    missed_terms = {p: Counter() for p in PROJECTS}
    approved_fp = {p: Counter() for p in PROJECTS}
    rejected_gold = {p: Counter() for p in PROJECTS}

    for run in runs:
        for project in PROJECTS:
            path = (run / "phase_states" / arm / "openai" / project
                    / "linker_partial_name.pkl")
            full = (run / "phase_states" / arm / "openai" / project
                    / "linker_full_name.pkl")
            if not path.exists() or not full.exists():
                continue
            state = pickle.load(path.open("rb"))
            earlier = {(l.sentence_number, l.component_id)
                       for l in pickle.load(full.open("rb"))["links"]}
            proposed = pairs(state["feedback"]["proposed"], names[project])
            accepted = {(l.sentence_number, l.component_id) for l in state["links"]}
            g = gold[project]
            # Gold still open when this linker runs: the coreference linker may
            # also reach some of these, so this is the stage's own opportunity,
            # not the pipeline's remaining loss.
            open_gold = g - earlier
            per[project]["open_gold"].append(len(open_gold))
            per[project]["proposed"].append(len(proposed))
            per[project]["proposed_gold"].append(len(proposed & g))
            per[project]["accepted"].append(len(accepted))
            per[project]["tp"].append(len(accepted & g))
            per[project]["fp"].append(len(accepted - g))
            for key in open_gold - proposed:
                missed_terms[project][key] += 1
            for key in accepted - g:
                approved_fp[project][key] += 1
            for key in (proposed & g) - accepted:
                rejected_gold[project][key] += 1
    return per, missed_terms, approved_fp, rejected_gold


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="../results/s5960_e2e_r*_20260813")
    ap.add_argument("--arm", default="s_linker49")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    runs = sorted(Path().glob(args.runs))
    if not runs:
        sys.exit(f"no runs matched {args.runs}")
    per, missed, fps, rejgold = audit(runs, args.arm)
    print(f"\n{args.arm} over {len(runs)} runs: {', '.join(r.name for r in runs)}\n")
    head = ("project", "open gold", "proposed", "of which gold", "approved",
            "TP", "FP", "judge rec", "judge prec")
    print(f"{head[0]:<15}" + "".join(f"{h:>14}" for h in head[1:]))
    totals = Counter()
    for project in PROJECTS:
        d = per[project]
        if not d["proposed"]:
            continue
        pg, tp, fp = mean(d["proposed_gold"]), mean(d["tp"]), mean(d["fp"])
        prop, acc = mean(d["proposed"]), mean(d["accepted"])
        jr = tp / pg * 100 if pg else float("nan")
        jp = tp / acc * 100 if acc else float("nan")
        print(f"{project:<15}{mean(d['open_gold']):>14.1f}{prop:>14.1f}"
              f"{pg:>14.1f}{acc:>14.1f}{tp:>14.1f}{fp:>14.1f}"
              f"{jr:>13.0f}%{jp:>13.0f}%")
        for k, v in (("open_gold", mean(d["open_gold"])), ("proposed", prop),
                     ("proposed_gold", pg), ("accepted", acc), ("tp", tp),
                     ("fp", fp)):
            totals[k] += v
    print(f"{'TOTAL':<15}{totals['open_gold']:>14.1f}{totals['proposed']:>14.1f}"
          f"{totals['proposed_gold']:>14.1f}{totals['accepted']:>14.1f}"
          f"{totals['tp']:>14.1f}{totals['fp']:>14.1f}"
          f"{totals['tp'] / totals['proposed_gold'] * 100:>13.0f}%"
          f"{totals['tp'] / totals['accepted'] * 100:>13.0f}%")

    print("\nheadroom of the current proposer (a perfect judge over the same "
          "candidates)")
    print(f"    TP {totals['proposed_gold']:.1f} instead of {totals['tp']:.1f} "
          f"(+{totals['proposed_gold'] - totals['tp']:.1f}), FP 0 instead of "
          f"{totals['fp']:.1f}")
    print(f"    gold the proposer never offers: "
          f"{totals['open_gold'] - totals['proposed_gold']:.1f}")

    for title, table in (("false positives the judge approves", fps),
                         ("gold the judge rejects", rejgold),
                         ("open gold the proposer never offers", missed)):
        print(f"\n{title} (count = runs out of {len(runs)})")
        for project in PROJECTS:
            items = table[project].most_common(args.top)
            if not items:
                continue
            print(f"  {project}")
            for (snum, cid), n in items:
                name = next((k for k, v in id_by_name(project).items() if v == cid),
                            cid)
                print(f"    {n}x  s{snum:<5} {name}")


if __name__ == "__main__":
    main()
