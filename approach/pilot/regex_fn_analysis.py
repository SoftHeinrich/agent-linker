"""Where the false negatives are, before and after the extraction pass becomes a scan.

The branch's standing error-shape result is that 95% of false negatives never reach a
judge -- the proposer is the bottleneck, not the gate. `s_linker92a` replaces the
proposer this result is mostly about, so the question is whether the shape survives
the change or moves.

Every false negative of a run is classified by **the furthest it got**, over all three
linkers, from the recorded phase states of that run:

    linked-elsewhere  not an FN at all -- another linker holds it (reported, not counted)
    fn/rejected       some stage proposed it and its judge said no -- a JUDGING failure
    fn/unproposed     no stage proposed it -- a PROPOSING failure

and each `fn/unproposed` is then asked what a scan could ever have reached it at:

    whole-name    the sentence writes a whole name of the component (ANY_CASE or
                  the spelling-variant fidelity), catalog name or discovered alias
    one-word      the sentence writes one word of the name, at any inflection --
                  the partial-name linker's row
    no surface    the sentence writes nothing of the name; only the coreference
                  linker can reach it, and only from context

The `ctl` column is the recorded LLM extraction; the `scan` column swaps in
`s_linker92a`'s candidate set and the stage arm's recorded verdicts for it. The other
two linkers are the same recorded stages in both columns, which is exactly how
`pilot/regex_proposer_pilots.py` composed them.

No LLM calls.

    ../.venv/bin/python pilot/regex_fn_analysis.py --model terra
    ../.venv/bin/python pilot/regex_fn_analysis.py --model terra --examples 20
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

import regex_extract_audit as AUDIT                                   # noqa: E402
import regex_proposer_pilots as PILOT                                 # noqa: E402
from llm_sad_sam.core.document_loader_v2 import (                     # noqa: E402
    build_sent_map, load_sentences)
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository            # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker92 import (             # noqa: E402
    NameForm, SLinker92)

ROUND = Path("../results/regex_round")

#: Ordered: a pair is labelled by the FIRST bucket it falls in.
BUCKETS = ("linked", "fn/rejected", "fn/unproposed")
REACH = ("whole-name", "one-word", "no surface")


def surfaces(sentences, components, aliases):
    """Per (sentence, component): the loosest row of the relation that reaches it.

    Reads the head's own `_name_spans`, so this is the module's relation and not a
    second copy of it. `whole-name` is the union of the two whole-name fidelities,
    over the catalog name and the run's discovered aliases alike.
    """
    by_component = collections.defaultdict(list)
    for term, component in (aliases or {}).items():
        by_component[component].append(term)
    out = {}
    for sentence in sentences:
        text = sentence.text
        for component in components:
            names = [component.name] + by_component.get(component.name, [])
            key = (sentence.number, component.id)
            if any(AUDIT.spans(text, n, "any_case")
                   or AUDIT.spans(text, n, "any_spelling") for n in names):
                out[key] = "whole-name"
            elif SLinker92._name_spans(text, component.name, NameForm.ANY_WORD):
                out[key] = "one-word"
            else:
                out[key] = "no surface"
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(PILOT.RECORDED), required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--examples", type=int, default=0)
    args = parser.parse_args()

    run_dirs, variant = PILOT.recorded_runs(args.model)
    kept = {arm: json.loads((ROUND / f"kept_{args.model}_{arm}.json").read_text())
            for arm in ("ctl", "scan")}

    counts = {arm: collections.defaultdict(list) for arm in kept}
    examples = {arm: collections.Counter() for arm in kept}

    for run in range(args.runs):
        run_dir = run_dirs[run % len(run_dirs)]
        per_run = {arm: collections.Counter() for arm in kept}
        for project, (text, model_path, gold_path) in PILOT.PROJECTS.items():
            components = parse_pcm_repository(
                os.path.join(PILOT.BASE, "benchmark", model_path))
            sentences = load_sentences(os.path.join(PILOT.BASE, "benchmark", text))
            sent_map = build_sent_map(sentences)
            name_to_id = {c.name: c.id for c in components}
            id_to_name = {c.id: c.name for c in components}
            truth = PILOT.gold(gold_path)

            knowledge = PILOT.state(run_dir, variant, project, "knowledge")
            aliases = dict(knowledge["doc_knowledge"].aliases) if knowledge else {}
            reach = surfaces(sentences, components, aliases)

            full = PILOT.state(run_dir, variant, project, "linker_full_name")
            partial = PILOT.state(run_dir, variant, project, "linker_partial_name")
            coref = PILOT.state(run_dir, variant, project, "linker_coreference")

            def pairs(stage, field):
                return {(i["sentence"], name_to_id[i["component"]])
                        for i in stage["feedback"].get(field, [])
                        if i["component"] in name_to_id}

            other_proposed = (pairs(partial, "proposed") | pairs(coref, "candidates"))
            other_links = {(l.sentence_number, l.component_id)
                           for stage in (partial, coref) for l in stage["links"]}

            ctl_candidates = pairs(full, "candidates")
            scan_candidates = set(AUDIT.regex_keys(
                project, sentences, components, aliases,
                form="any_case", use_aliases=True, skip_dotted=False))

            for arm in kept:
                proposed = (ctl_candidates if arm == "ctl" else scan_candidates)
                approved = {tuple(x) for x in kept[arm][f"run{run + 1}"][project]}
                links = approved | other_links
                for pair in truth:
                    if pair in links:
                        per_run[arm]["linked"] += 1
                        continue
                    if pair in proposed or pair in other_proposed:
                        bucket = "fn/rejected"
                    else:
                        bucket = "fn/unproposed"
                    per_run[arm][bucket] += 1
                    per_run[arm][f"{bucket} @ {reach[pair]}"] += 1
                    if args.examples:
                        examples[arm][(bucket, reach[pair], project, pair[0],
                                       id_to_name.get(pair[1], pair[1]))] += 1
        for arm in kept:
            counts[arm]["_"].append(per_run[arm])

    def mean(arm, key):
        return st.mean(c[key] for c in counts[arm]["_"])

    gold_total = sum(len(PILOT.gold(g)) for _t, _m, g in PILOT.PROJECTS.values())
    print(f"{args.model}, {args.runs} runs, per five-project run "
          f"(gold {gold_total}):\n")
    print(f"{'bucket':<28}{'ctl':>8}{'scan':>8}{'delta':>8}")
    print("-" * 52)
    rows = ["linked", "fn/rejected", "fn/unproposed"]
    rows += [f"fn/unproposed @ {r}" for r in REACH]
    rows += [f"fn/rejected @ {r}" for r in REACH]
    for row in rows:
        c, s = mean("ctl", row), mean("scan", row)
        lead = "  " if "@" in row else ""
        print(f"{lead + row:<28}{c:>8.1f}{s:>8.1f}{s - c:>+8.1f}")

    if args.examples:
        print(f"\nthe scan arm's false negatives, most persistent first:")
        for (bucket, reach_row, project, snum, name), n in \
                examples["scan"].most_common(args.examples):
            print(f"  {bucket:<15}{reach_row:<12}{project:<14}S{snum:<5}"
                  f"{name:<26}x{n}")


if __name__ == "__main__":
    main()
