"""Does a stage change reach past its own stage? Deterministic, no LLM calls.

The reason this branch's rule used to be "confirm every adopted arm end-to-end" is one
episode: dropping `_keep_stated_names` was F2-positive on its own stage and quadrupled
false positives end-to-end. The mechanism was never mysterious -- every linker subtracts
what earlier ones produced (`_unlinked`), so a link admitted early is both locked into
the union and **stolen from the later, stricter linkers** that would have judged the same
pair.

That mechanism is deterministic, and so is its precondition. A stage change can only
compose badly if the pairs it adds or removes are pairs some *later* stage would
otherwise have proposed. This reads that straight off the recorded checkpoints:

  * for each pair the change ADDS, is it in a later stage's candidate set? is it in the
    final link set already?
  * for each pair the change REMOVES, was it in the final link set?

If both answers are empty, the stage arm *is* the pipeline answer for that change, and a
five-project end-to-end run measures the model's run-to-run drift rather than the change.
Where they are not empty, the count is the size of the composition risk and says how much
end-to-end evidence the arm needs.

    ../.venv/bin/python pilot/composition_check.py --change infl
    ../.venv/bin/python pilot/composition_check.py --change statednet
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import PROJECTS, load_gold                        # noqa: E402
from partial_screen import Probe, project_cache                     # noqa: E402

#: Which candidate sets each stage's change could be stolen from, in run order.
LATER = {"full_name": ("linker_partial_name", "linker_coreference"),
         "partial_name": ("linker_coreference",),
         "coreference": ()}
VIEW = {"linker_partial_name": "proposed", "linker_coreference": "candidates"}


def keys(view, by_name):
    out = set()
    for row in view:
        cid = by_name.get(row.get("component"))
        if cid is not None:
            out.add((int(row["sentence"]), cid))
    return out


def infl_change(probe, sentences, components, state):
    """s_linker62: the partial-name proposer's prefix bounded to inflections."""
    base = set(probe.candidates(sentences, components, "base")) - state["linked"]
    new = set(probe.candidates(sentences, components, "infl")) - state["linked"]
    return "partial_name", new - base, base - new


def statednet_change(probe, sentences, components, state):
    """s_linker64: the case-sensitive stated-name net at the full-name proposer."""
    net = {(s.number, c.id) for s in sentences for c in components
           if re.search(rf"(?<!\w){re.escape(c.name)}(?!\w)", s.text)}
    return "full_name", net - state["proposed"] - state["linked"], set()


CHANGES = {"infl": infl_change, "statednet": statednet_change}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--change", required=True, choices=sorted(CHANGES))
    ap.add_argument("--runs", default="../results/s5960_e2e_r*_20260813")
    ap.add_argument("--arm", default="s_linker49")
    args = ap.parse_args()
    runs = sorted(Path().glob(args.runs))

    agg, detail = Counter(), Counter()
    for run in runs:
        for project in PROJECTS:
            base = run / "phase_states" / args.arm / "openai" / project
            if not (base / "linker_coreference.pkl").exists():
                continue
            sentences, components = project_cache(project)
            by_name = {c.name: c.id for c in components}
            knowledge = pickle.load((base / "knowledge.pkl").open("rb"))
            probe = Probe(getattr(knowledge.get("doc_knowledge"), "aliases", {}) or {})
            full = pickle.load((base / "linker_full_name.pkl").open("rb"))
            state = {
                "linked": {(l.sentence_number, l.component_id) for l in full["links"]},
                "proposed": keys(full["feedback"]["candidates"], by_name),
            }
            stage, added, removed = CHANGES[args.change](
                probe, sentences, components, state)
            final = {(l.sentence_number, l.component_id)
                     for l in pickle.load((base / "final.pkl").open("rb"))["final"]}
            gold = set(load_gold(project))
            later = {}
            for name in LATER[stage]:
                data = pickle.load((base / f"{name}.pkl").open("rb"))
                later[name] = keys(data["feedback"][VIEW[name]], by_name)
            agg["added"] += len(added)
            agg["removed"] += len(removed)
            for key in added:
                if key in final:
                    agg["added, already in the final link set"] += 1
                for name, candidates in later.items():
                    if key in candidates:
                        agg[f"added, also proposed by {name}"] += 1
                        detail[(project, key, name, key in gold)] += 1
            for key in removed:
                if key in final:
                    agg["removed, but was in the final link set"] += 1
                    detail[(project, key, "removed-from-final", key in gold)] += 1

    n = len(runs) or 1
    print(f"\n{args.change} at the {stage} stage, per run over {len(runs)} runs\n")
    for label in ("added", "removed"):
        print(f"    {agg[label] / n:6.1f}   candidates {label}")
    risk = 0
    for label, count in agg.items():
        if label in ("added", "removed"):
            continue
        risk += count
        print(f"    {count / n:6.1f}   {label}")
    for key, count in detail.most_common():
        print(f"        {count}x  {key}")
    print(f"\n    composition risk: {risk / n:.1f} pairs per run"
          + ("  -- the stage arm is the pipeline answer for this change"
             if not risk else "  -- end-to-end evidence needed"))


if __name__ == "__main__":
    main()
