"""The last recall bucket: sentences that state a name the extractor never proposed.

After the partial-name proposer's ownership defect is repaired (`s_linker62`), the
workflow's residual recall loss is almost entirely one bucket, and
`pilot/partial_gap.py` named it: 5.8 gold pairs per run sit in a sentence that states a
whole name of the component, so the partial-name proposer defers to the full-name
linker -- which never produced them. `pilot/partial_hole.py` splits it:

    3.0/run  the extraction call never proposed the pair
    2.7/run  it did, and a full-name judge rejected it

and refutes routing the bucket to the *denotation* judge (+0.7 gold, +10.0 spurious),
because the whole-name test is also the alias table's suppression role.

This screens the other route for the first half: a deterministic whole-name scan that
proposes every (sentence, component) whose sentence states a name of the component and
that the extractor did not propose, to be judged by the **full-name** two-pass judge,
which is the strict one. Reported deterministically, per project, before any call is
paid for: how many pairs, how many gold, and how they compare with the extractor's own
proposal set -- because a net this size only pays if the judge behind it is precise
enough, and the judge's measured precision on its current mix is the yardstick.

    ../.venv/bin/python pilot/statednet_screen.py
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import PROJECTS, load_gold                        # noqa: E402
from partial_screen import Probe, project_cache                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="../results/s5960_e2e_r*_20260813")
    ap.add_argument("--arm", default="s_linker49")
    ap.add_argument("--aliases", action="store_true",
                    help="widen the net from the model name to every discovered alias")
    args = ap.parse_args()
    runs = sorted(Path().glob(args.runs))

    agg = Counter()
    added = Counter()
    for run in runs:
        for project in PROJECTS:
            base = run / "phase_states" / args.arm / "openai" / project
            if not (base / "linker_full_name.pkl").exists():
                continue
            sentences, components = project_cache(project)
            by_name = {c.name: c.id for c in components}
            knowledge = pickle.load((base / "knowledge.pkl").open("rb"))
            probe = Probe(getattr(knowledge.get("doc_knowledge"), "aliases", {}) or {})
            fn = pickle.load((base / "linker_full_name.pkl").open("rb"))
            linked = {(l.sentence_number, l.component_id) for l in fn["links"]}
            proposed = {(int(r["sentence"]), by_name[r["component"]])
                        for r in fn["feedback"]["candidates"]
                        if r["component"] in by_name}
            gold = set(load_gold(project))
            # `_states_a_name` counts the discovered aliases too. The extractor is
            # already given the alias table, so an alias-based net re-offers a weak
            # signal the extraction call has seen and declined; the model name is the
            # strongest lexical evidence there is. `--aliases` measures the wider net.
            net = {(s.number, c.id) for s in sentences for c in components
                   if (probe._states_a_name(s.text, c.name) if args.aliases
                       else probe._find_exact_form(s.text, c.name))}
            fresh = net - proposed - linked
            agg[(project, "runs")] += 1
            agg[(project, "extractor")] += len(proposed)
            agg[(project, "extractor_gold")] += len(proposed & gold)
            agg[(project, "net")] += len(net)
            agg[(project, "fresh")] += len(fresh)
            agg[(project, "fresh_gold")] += len(fresh & gold)
            for key in fresh & gold:
                name = next(c.name for c in components if c.id == key[1])
                added[(project, True, key[0], name)] += 1

    print(f"\n{args.arm}, a deterministic whole-name net behind the full-name judge, "
          f"per run over {len(runs)} runs\n")
    print(f"{'project':<16}{'extractor':>11}{'gold':>7}{'stated':>9}"
          f"{'new pairs':>11}{'gold':>7}{'spurious':>10}{'gold/pair':>11}")
    total = Counter()
    for project in PROJECTS:
        n = agg[(project, "runs")] or 1
        if not agg[(project, "net")]:
            continue
        fresh = agg[(project, "fresh")] / n
        fresh_gold = agg[(project, "fresh_gold")] / n
        print(f"{project:<16}{agg[(project, 'extractor')] / n:>11.1f}"
              f"{agg[(project, 'extractor_gold')] / n:>7.1f}"
              f"{agg[(project, 'net')] / n:>9.1f}{fresh:>11.1f}{fresh_gold:>7.1f}"
              f"{fresh - fresh_gold:>10.1f}"
              f"{(fresh_gold / fresh if fresh else 0):>11.2f}")
        for key in ("extractor", "extractor_gold", "net", "fresh", "fresh_gold"):
            total[key] += agg[(project, key)] / n
    fresh, fresh_gold = total["fresh"], total["fresh_gold"]
    print(f"{'TOTAL':<16}{total['extractor']:>11.1f}{total['extractor_gold']:>7.1f}"
          f"{total['net']:>9.1f}{fresh:>11.1f}{fresh_gold:>7.1f}"
          f"{fresh - fresh_gold:>10.1f}"
          f"{(fresh_gold / fresh if fresh else 0):>11.2f}")
    print(f"\n    the extractor's own proposals run at "
          f"{total['extractor_gold'] / total['extractor']:.2f} gold per pair; "
          f"this net runs at {fresh_gold / fresh:.2f}")
    print("\n    gold the net would reach")
    for (project, _, snum, name), count in added.most_common():
        print(f"        {count}x  {project:<14} s{snum:<5} {name}")


if __name__ == "__main__":
    main()
