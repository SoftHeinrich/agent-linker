"""Does the union's change reach past its own stage? Deterministic, no LLM calls.

The measurement policy's level 3: a stage arm **is** the pipeline answer when the pairs
it adds or removes are not pairs a later stage would otherwise propose, and are not
already in the final link set. The union changes which pairs the two name streams emit;
the only stage after them is the coreference linker, so the question has an exact form:

  * every pair the union ADDS — does the recorded coreference linker propose it anyway,
    and is it already in the recorded final link set? An added pair the pipeline already
    had is worth nothing end to end.
  * every pair the union REMOVES — was it in the recorded final link set, and would
    coreference have re-proposed it? A removed pair coreference re-proposes costs the
    pipeline nothing; a removed *gold* pair nothing re-proposes is a real loss.

Read off the six recorded head runs and the union pilot's own dumps.

    ../.venv/bin/python pilot/union_composition.py ../results/union_round/dump_terra_v13.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from simmerge_audit import E2E_RUNS                                   # noqa: E402
from union_diff import project_context                                # noqa: E402
from union_pilots import DEFAULT_RUN                                  # noqa: E402


def recorded(project):
    """Per recorded run: the coreference candidates, and the final link set."""
    out = []
    for _model, runs in E2E_RUNS.items():
        for run in runs:
            base = (Path(run) / "phase_states" / "s_linker110" / "openai" / project)
            coref, final = base / "linker_coreference.pkl", base / "final.pkl"
            if not (coref.exists() and final.exists()):
                continue
            with coref.open("rb") as handle:
                feedback = pickle.load(handle)["feedback"]
            with final.open("rb") as handle:
                links = pickle.load(handle)["final"]
            out.append((
                {(int(row["sentence"]), row["component"])
                 for row in feedback.get("candidates", [])},
                {(link.sentence_number, link.component_name) for link in links},
            ))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dumps", nargs="+")
    parser.add_argument("--arms", nargs=2, default=["control", "union"])
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    left, right = args.arms

    for path in args.dumps:
        with open(path) as handle:
            dump = json.load(handle)
        contexts, runs = {}, {}
        totals = collections.Counter()
        losses = []
        for sample in sorted(dump):
            for project, arms in sorted(dump[sample].items()):
                if left not in arms or right not in arms:
                    continue
                if project not in contexts:
                    contexts[project] = project_context(project, args.run)
                    runs[project] = recorded(project)
                context, recordings = contexts[project], runs[project]
                gold, name_of = context["gold"], context["name_of"]
                kept = {arm: {tuple(p) for p in arms[arm]} for arm in (left, right)}
                added = kept[right] - kept[left]
                removed = kept[left] - kept[right]
                for pair in added:
                    key = (pair[0], name_of[pair[1]])
                    totals["added"] += 1
                    totals["added_gold"] += pair in gold
                    totals["added_reproposed"] += any(
                        key in candidates for candidates, _ in recordings)
                    totals["added_already_final"] += any(
                        key in final for _, final in recordings)
                for pair in removed:
                    key = (pair[0], name_of[pair[1]])
                    totals["removed"] += 1
                    totals["removed_gold"] += pair in gold
                    rescued = any(key in candidates for candidates, _ in recordings)
                    in_final = sum(1 for _, final in recordings if key in final)
                    totals["removed_reproposed"] += rescued
                    totals["removed_in_final"] += bool(in_final)
                    if pair in gold and not rescued:
                        losses.append(
                            f"{project} S{pair[0]} {name_of[pair[1]]} "
                            f"(in {in_final}/{len(recordings)} recorded finals)")

        print(f"\n{Path(path).name}: `{right}` against `{left}`, "
              f"summed over every sample")
        print(f"  added   {totals['added']:4d} pairs "
              f"({totals['added_gold']} gold); coreference proposes "
              f"{totals['added_reproposed']} of them anyway, "
              f"{totals['added_already_final']} are already in a recorded final set")
        print(f"  removed {totals['removed']:4d} pairs "
              f"({totals['removed_gold']} gold); coreference re-proposes "
              f"{totals['removed_reproposed']}, "
              f"{totals['removed_in_final']} were in a recorded final set")
        unrescued = collections.Counter(losses)
        print(f"\n  gold the union removes that nothing downstream re-proposes: "
              f"{len(unrescued)} distinct")
        for line, count in unrescued.most_common(12):
            print(f"    {line}  [{count} sample(s)]")
        risk = len(unrescued)
        print(f"\n  composition risk: {risk} distinct gold pairs. "
              + ("The stage arm is the pipeline answer for this change."
                 if risk == 0 else
                 "An end-to-end batch would measure this; below the recorded TP "
                 "floor of 4.8 it cannot see it."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
