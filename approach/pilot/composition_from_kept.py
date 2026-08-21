"""Composition risk for a stage arm whose change is a verdict, not a predicate.

`pilot/composition_check.py` answers this for changes that are deterministic
predicates -- it can recompute what the change adds and removes. A prompt arm's
change is the judge's verdicts, so what it adds and removes is read from the arm's
own recorded stage output instead: the `kept_<group>_<model>_<arm>.json` dumps that
`pilot/compaction_pilots.py` writes.

The question is the branch's step-3 gate, unchanged: are the pairs the arm adds or
removes pairs a LATER stage would otherwise have proposed, or pairs that are in the
recorded final link set? If both are empty the stage arm is the pipeline answer and
an end-to-end batch would measure drift instead of the change.

    ../.venv/bin/python pilot/composition_from_kept.py \
        --group fullname6 --model terra --arm anchorunion
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from compaction_pilots import PROJECTS, RECORDED, gold                # noqa: E402

BASE = "/mnt/hostshare/ardoco-home/alinker-replication-package"

#: Which stages run after each stage this round changes, in pipeline order, and which
#: feedback view holds what they PROPOSED (not what they kept).
LATER = {
    "fullname5": ("linker_partial_name", "linker_coreference"),
    "fullname6": ("linker_partial_name", "linker_coreference"),
    "denot2": ("linker_coreference",),
    "coref5": (),
    "resolve3": (),
}
VIEW = {"linker_partial_name": "proposed", "linker_coreference": "candidates"}


def proposed(path, by_name):
    if not os.path.exists(path):
        return set()
    data = pickle.load(open(path, "rb"))
    view = data["feedback"].get(VIEW[Path(path).stem], [])
    out = set()
    for row in view:
        cid = by_name.get(row.get("component"))
        if cid is not None:
            out.add((int(row["sentence"]), cid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", default="../results/compaction_round")
    args = ap.parse_args()

    pattern, variant = RECORDED[args.model]
    run_dirs = sorted(glob.glob(os.path.join(BASE, "results", pattern)))
    ctl = json.load(open(Path(args.out) /
                         f"kept_{args.group}_{args.model}_ctl.json"))
    arm = json.load(open(Path(args.out) /
                         f"kept_{args.group}_{args.model}_{args.arm}.json"))
    agg = Counter()
    n_runs = len(ctl)

    for i, run_key in enumerate(sorted(ctl)):
        run = run_dirs[i % len(run_dirs)]
        for proj, (_t, model_path, gold_path) in PROJECTS.items():
            from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
            comps = parse_pcm_repository(os.path.join(BASE, "benchmark", model_path))
            by_name = {c.name: c.id for c in comps}
            g = gold(gold_path)
            base = os.path.join(run, "phase_states", variant, "openai", proj)
            a = {tuple(x) for x in ctl[run_key].get(proj, [])}
            b = {tuple(x) for x in arm[run_key].get(proj, [])}
            added, removed = b - a, a - b
            agg["added"] += len(added)
            agg["removed"] += len(removed)
            agg["added, gold"] += len(added & g)
            agg["removed, gold"] += len(removed & g)
            final_fn = os.path.join(base, "final.pkl")
            final = set()
            if os.path.exists(final_fn):
                final = {(l.sentence_number, l.component_id)
                         for l in pickle.load(open(final_fn, "rb"))["final"]}
            for stage in LATER[args.group]:
                later = proposed(os.path.join(base, f"{stage}.pkl"), by_name)
                agg[f"added, also proposed by {stage}"] += len(added & later)
            agg["added, already in the recorded final link set"] += len(added & final)
            agg["removed, but in the recorded final link set"] += len(removed & final)

    print(f"{args.group}/{args.model}: {args.arm} against ctl, "
          f"{n_runs} runs, per five-project run\n")
    for key, value in agg.items():
        print(f"  {key:>48}: {value / max(1, n_runs):6.1f}")
    risk = sum(v for k, v in agg.items()
               if k.startswith("added, also proposed")
               or k == "removed, but in the recorded final link set")
    print(f"\n  composition risk (the branch's step-3 gate): "
          f"{risk / max(1, n_runs):.1f} pairs per run")
    if not risk:
        print("  -> structurally vacuous: the stage arm IS the pipeline answer")


if __name__ == "__main__":
    main()
