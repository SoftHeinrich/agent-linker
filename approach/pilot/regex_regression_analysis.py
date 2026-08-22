"""Where the scan LOSES to the LLM extraction pass, and what the difference is made of.

The E2E batch is F2-positive on luna and flat on terra, and the flatness is not
uniform: mediastore falls 7.1 macro F2 on terra and teammates 2.0, while
bigbluebutton rises 6.8. This asks what those regressions are made of, pair by pair,
against the control's own runs.

Every pair in the symmetric difference of the two arms' final link sets is classified:

    FP gained   the scan links it, the control does not, it is not gold
    TP lost     the control links it, the scan does not, it is gold
    FP avoided / TP gained -- the two directions that go the other way

and each `FP gained` is then asked what the scan matched on, read from that run's own
phase states: the **surface** the sentence writes, whether it is the catalog **name**
or a discovered **alias**, and whether it is **lowercase** (which is what
`STRICTER_CLAUSE` calls evidence against a name). Each `TP lost` is asked whether the
scan proposed it at all, and which stage held it in the control.

The control's phase states are not readable (two variants wrote that namespace), so
the control side is read from its per-variant link CSVs, which carry `source`.

No LLM calls.

    ../.venv/bin/python pilot/regex_regression_analysis.py --model terra
    ../.venv/bin/python pilot/regex_regression_analysis.py --model terra --project mediastore
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

import regex_proposer_pilots as PILOT                                 # noqa: E402
from llm_sad_sam.core.document_loader_v2 import (                     # noqa: E402
    build_sent_map, load_sentences)
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository            # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker92a import SLinker92a   # noqa: E402

ARMS = {
    "ctl": ("../results/solo_e2e_%s_r*_20260821", "s_linker92", "s_linker89"),
    "scan": ("../results/regex_e2e_%s_r*_20260822", "s_linker92a", "s_linker92a"),
}


def links_of(run_dir, variant, project):
    """(pair -> source) from a run's per-variant link CSV."""
    path = os.path.join(run_dir, f"{variant}_{project}_links.csv")
    with open(path) as handle:
        return {(int(r["sentence"]), r["component_id"]): r["source"]
                for r in csv.DictReader(handle)}


def scan_detail(run_dir, namespace, project, name_to_id):
    """What the scan proposed in this run, and what surface it matched on."""
    stage = PILOT.state(os.path.join(run_dir, "phase_states"), namespace,
                        project, "linker_full_name")
    knowledge = PILOT.state(os.path.join(run_dir, "phase_states"), namespace,
                            project, "knowledge")
    aliases = dict(knowledge["doc_knowledge"].aliases) if knowledge else {}
    proposed = {}
    for item in stage["feedback"]["candidates"] if stage else []:
        if item["component"] in name_to_id:
            proposed[(item["sentence"], name_to_id[item["component"]])] = item
    return proposed, aliases, stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(PILOT.RECORDED), required=True)
    parser.add_argument("--project", default=None)
    parser.add_argument("--top", type=int, default=14)
    args = parser.parse_args()

    projects = {k: v for k, v in PILOT.PROJECTS.items()
                if not args.project or k == args.project}
    ctl_dirs = sorted(glob.glob(ARMS["ctl"][0] % args.model))
    scan_dirs = sorted(glob.glob(ARMS["scan"][0] % args.model))
    runs = min(len(ctl_dirs), len(scan_dirs))

    tally = collections.Counter()
    per_project = collections.defaultdict(collections.Counter)
    fp_gained = collections.Counter()
    tp_lost = collections.Counter()
    fp_shape = collections.Counter()
    fp_surface = collections.Counter()
    per_stage = collections.Counter()

    #: The proposer itself, so a matched surface is recomputed by the code that
    #: produced it rather than by a second copy of the rule.
    scanner = SLinker92a.__new__(SLinker92a)
    sent_text = {}
    for project, (text, _m, _g) in projects.items():
        sent_text[project] = {s.number: s.text for s in load_sentences(
            os.path.join(PILOT.BASE, "benchmark", text))}

    for index in range(runs):
        for project, (_t, model_path, gold_path) in projects.items():
            components = parse_pcm_repository(
                os.path.join(PILOT.BASE, "benchmark", model_path))
            name_to_id = {c.name: c.id for c in components}
            id_to_name = {c.id: c.name for c in components}
            truth = PILOT.gold(gold_path)

            control = links_of(ctl_dirs[index], ARMS["ctl"][1], project)
            scan = links_of(scan_dirs[index], ARMS["scan"][1], project)
            proposed, aliases, _stage = scan_detail(
                scan_dirs[index], ARMS["scan"][2], project, name_to_id)
            by_component = collections.defaultdict(list)
            for term, component in aliases.items():
                by_component[component].append(term)
            scanner.doc_knowledge = type("K", (), {"aliases": aliases})()

            for pair in set(scan) - set(control):
                label = "TP gained" if pair in truth else "FP gained"
                tally[label] += 1
                per_project[project][label] += 1
                per_stage[(scan[pair], label)] += 1
                if pair in truth:
                    continue
                name = id_to_name.get(pair[1], pair[1])
                fp_gained[(project, pair[0], name, scan[pair])] += 1
                if scan[pair] == "full_name":
                    # The phase state's candidate view drops `matched_text`, so the
                    # surface is recomputed from the scan itself -- the same call the
                    # proposer made, on the same sentence and the same alias table.
                    matched, via = "", "name"
                    text = sent_text[project][pair[0]]
                    for candidate_name in (name, *by_component.get(name, ())):
                        found = scanner._writes_name(text, candidate_name)
                        if found:
                            matched = found
                            via = "name" if candidate_name == name else "alias"
                            break
                    case = ("lowercase" if matched and matched.islower()
                            else "capitalised")
                    fp_shape[(via, case)] += 1
                    fp_surface[(via, matched, name)] += 1

            for pair in set(control) - set(scan):
                label = "TP lost" if pair in truth else "FP avoided"
                tally[label] += 1
                per_project[project][label] += 1
                per_stage[(control[pair], label)] += 1
                if pair not in truth:
                    continue
                tp_lost[(project, pair[0], id_to_name.get(pair[1], pair[1]),
                         control[pair], "proposed" if pair in proposed
                         else "not proposed")] += 1

    print(f"{args.model}, {runs} paired runs, pooled over "
          f"{len(projects)} project(s):\n")
    for label in ("TP gained", "TP lost", "FP gained", "FP avoided"):
        print(f"  {label:<12}{tally[label] / runs:7.1f} a run")

    print(f"\n  per project (a run):")
    print(f"  {'project':<16}{'TP+':>6}{'TP-':>6}{'FP+':>6}{'FP-':>6}{'net TP':>8}"
          f"{'net FP':>8}")
    for project in projects:
        row = per_project[project]
        print(f"  {project:<16}{row['TP gained'] / runs:>6.1f}"
              f"{row['TP lost'] / runs:>6.1f}{row['FP gained'] / runs:>6.1f}"
              f"{row['FP avoided'] / runs:>6.1f}"
              f"{(row['TP gained'] - row['TP lost']) / runs:>+8.1f}"
              f"{(row['FP gained'] - row['FP avoided']) / runs:>+8.1f}")

    print(f"\n  the difference, by the STAGE it sits at (a run). Only `full_name`\n"
          f"  is a stage this change touches:")
    print(f"  {'stage':<16}{'TP+':>6}{'TP-':>6}{'FP+':>6}{'FP-':>6}{'net TP':>8}"
          f"{'net FP':>8}")
    for stage in ("full_name", "partial_name", "coreference"):
        g, l = per_stage[(stage, "TP gained")], per_stage[(stage, "TP lost")]
        fg, fa = per_stage[(stage, "FP gained")], per_stage[(stage, "FP avoided")]
        print(f"  {stage:<16}{g / runs:>6.1f}{l / runs:>6.1f}{fg / runs:>6.1f}"
              f"{fa / runs:>6.1f}{(g - l) / runs:>+8.1f}{(fg - fa) / runs:>+8.1f}")

    if fp_shape:
        print(f"\n  what the full-name FPs the scan ADDS matched on (a run):")
        for (via, case), count in fp_shape.most_common():
            print(f"    matched the {via:<6} {case:<12}{count / runs:6.1f}")
        print(f"\n  the surfaces themselves (a run):")
        for (via, matched, name), count in fp_surface.most_common(args.top):
            print(f"    {matched!r:<24} -> {name:<22}via {via:<6}{count / runs:5.1f}")

    print(f"\n  false positives the scan adds, most persistent first:")
    for (project, snum, name, source), count in fp_gained.most_common(args.top):
        print(f"    {project:<15}S{snum:<5}{name:<24}{source:<14}x{count}")

    print(f"\n  true positives the scan loses, most persistent first:")
    for (project, snum, name, source, state), count in tp_lost.most_common(args.top):
        print(f"    {project:<15}S{snum:<5}{name:<24}"
              f"was {source:<14}{state:<14}x{count}")


if __name__ == "__main__":
    main()
