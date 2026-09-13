"""Level-2 stage pilot for the union judge: both name streams, fixed candidates.

The two proposers in front of this measurement are deterministic scans and the alias
table is pinned from a recorded run, so **the candidate set is byte-identical across
arms and samples** — the only thing that varies is how the cases are judged, and the
only calls spent are the name streams' judging calls (the coreference linker never
runs here).

    arm `control`   `s_linker110`: two prompts, two rubrics, two reply schemas
    arm `union`     `s_linker120`: one prompt, one rule, the default stated per
                    `naming` row, and an evidence line computed from the match

Read **per row, not in the total**: the rows have base rates 0.98 / 0.49 / 0.31
(`pilot/unijudge_audit.py`), so a union that trades them against each other reads
neutral in a sum and is not neutral. Every row's kept/gold/spurious is printed, per
arm, per sample.

    OPENAI_API_KEY="$OAI_KEY" LLM_BACKEND=openai \\
    OPENAI_MODEL_NAME=gpt-5.6-terra OPENAI_REASONING_EFFORT=none \\
      ../.venv/bin/python pilot/union_pilots.py --samples 3
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker110 import SLinker110  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker120 import (  # noqa: E402
    MentionType, SLinker120,
)
from llm_sad_sam.linkers.experimental.union_iterations import ITERATIONS  # noqa: E402
from llm_sad_sam.llm_client import LLMBackend  # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository  # noqa: E402

from reading_pilots import BENCH, DATASETS, gold_pairs  # noqa: E402

DEFAULT_RUN = ROOT.parent / "results/consolidation_e2e_terra_r1_20260825"
ARMS = {"control": SLinker110, "union": SLinker120}

#: Any iteration of the union rule can be run as its own arm: `--arms control v3 v9`
#: builds `SLinker120` with that iteration pinned, in the same invocation, so two
#: versions of the rule are comparable without trusting two invocations.
ARMS.update({name: SLinker120 for name in ITERATIONS})

#: Which computed mention labels a case prints. The head keeps two of the five
#: `MentionType` values on the ground that the judge cannot re-derive them from the
#: sentence it is shown; these arms are that ground, measured rather than asserted.
#: Same rule, same candidates, same call count -- only the `mention=` line differs
#: (`pilot/mention_label_audit.py` counts the cases each arm changes).
#:
#:   alllabels  every label the classifier can compute, on every case that has one
#:   aliasmute  drop `via known alias`, which restates the case's own `writes` line
#:   nomention  print no computed label at all: the floor of the field
LABEL_ARMS = {
    "alllabels": frozenset(MentionType),
    "aliasmute": frozenset({MentionType.CODE_TOKEN}),
    "nomention": frozenset(),
}
ARMS.update({
    name: type(f"SLinker120_{name}", (SLinker120,),
               {"RETAINED_MENTION_TYPES": retained,
                "_VARIANT_NAME": f"s_linker120_{name}"})
    for name, retained in LABEL_ARMS.items()
})
ROWS = ("whole name", "alias", "word only")


def pinned_knowledge(run: Path, project: str):
    path = run / "phase_states" / "s_linker110" / "openai" / project / "knowledge.pkl"
    with open(path, "rb") as handle:
        return pickle.load(handle)["doc_knowledge"]


def rows_of(linker, sentences, components, sent_map, name_to_id):
    """{(sentence, component id): naming row} over the merged candidate stream.

    Computed with `s_linker120`'s own evidence function on a head instance, so both
    arms are bucketed by exactly the same fact.
    """
    reader = SLinker120.__new__(SLinker120)
    reader.doc_knowledge = linker.doc_knowledge
    candidates = reader._name_candidates(
        sentences, components, name_to_id, sent_map)
    return {(c.sentence_number, c.component_id):
            reader._union_evidence(c, components, sent_map)["naming"]
            for c in candidates}, candidates


def run_arm(arm, linker, sentences, components, name_to_id, sent_map):
    """The arm's kept pairs over both name streams, and its candidate count."""
    if arm != "control":
        links, feedback = linker._run_name_linker(
            sentences, components, name_to_id, sent_map)
        proposed = len(feedback["candidates"])
        return {(l.sentence_number, l.component_id) for l in links}, proposed
    full_links, full_feedback = linker._run_full_name_linker(
        sentences, components, name_to_id, sent_map)
    partial_links, partial_feedback = linker._run_partial_name_linker(
        sentences, components, sent_map)
    kept = {(l.sentence_number, l.component_id)
            for l in list(full_links) + list(partial_links)}
    proposed = len(full_feedback["candidates"]) + len(partial_feedback["proposed"])
    return kept, proposed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=["control", "union"],
                        help="control, union (the active iteration), or any iteration "
                             "name from union_iterations.ITERATIONS")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--datasets", nargs="+", default=sorted(DATASETS))
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--dump")
    args = parser.parse_args()

    backend = LLMBackend.OPENAI
    model = os.environ.get("OPENAI_MODEL_NAME", "")
    totals = collections.defaultdict(collections.Counter)
    per_row = collections.defaultdict(collections.Counter)
    dump: dict = {}

    for project in args.datasets:
        text, repo, gold_path = DATASETS[project]
        components = parse_pcm_repository(str(BENCH / repo))
        sentences = load_sentences(str(BENCH / text))
        sent_map = build_sent_map(sentences)
        name_to_id = {c.name: c.id for c in components}
        gold = gold_pairs(BENCH / gold_path)
        knowledge = pinned_knowledge(args.run, project)

        probe = SLinker110.__new__(SLinker110)
        probe.doc_knowledge = knowledge
        row_of, candidates = rows_of(
            probe, sentences, components, sent_map, name_to_id)
        counts = collections.Counter(row_of.values())
        print(f"\n=== {project}: {len(sentences)} sentences, {len(components)} "
              f"components, {len(gold)} gold, {len(candidates)} candidates "
              f"({', '.join(f'{r} {counts[r]}' for r in ROWS)}) ===", flush=True)

        sizes = set()
        for sample in range(1, args.samples + 1):
            for arm in args.arms:
                linker = ARMS[arm](backend=backend, model=model)
                linker.doc_knowledge = knowledge
                if arm in ITERATIONS:
                    linker.iteration_name = arm
                kept, proposed = run_arm(
                    arm, linker, sentences, components, name_to_id, sent_map)
                sizes.add(proposed)
                good = kept & gold
                totals[arm]["kept"] += len(kept)
                totals[arm]["gold"] += len(good)
                totals[arm]["spurious"] += len(kept) - len(good)
                totals[arm]["candidates"] += proposed
                totals[arm]["calls"] += len(linker._llm_calls)
                for pair in kept:
                    row = row_of.get(pair, "?")
                    per_row[(arm, row)]["kept"] += 1
                    per_row[(arm, row)]["gold" if pair in gold else "spurious"] += 1
                for row in ROWS:
                    per_row[(arm, row)]["cases"] += counts[row]
                    per_row[(arm, row)]["case_gold"] += sum(
                        1 for pair, value in row_of.items()
                        if value == row and pair in gold)
                dump.setdefault(f"sample{sample}", {}).setdefault(project, {})[arm] = \
                    sorted([list(p) for p in kept])
                print(f"  {arm:<8} sample {sample}: {len(kept):4d} kept, "
                      f"{len(good):4d} gold, {len(kept) - len(good):4d} spurious "
                      f"(of {proposed} candidates, {len(linker._llm_calls)} calls)",
                      flush=True)
        if len(sizes) > 1:
            print(f"  !! candidate set differs across arms: {sorted(sizes)}")

    runs = args.samples
    print(f"\nname streams on {model or 'unset'}, {runs} samples, "
          f"per five-project run:")
    print(f"  {'arm':<9}{'candidates':>11}{'kept':>8}{'gold':>8}{'spurious':>10}"
          f"{'precision':>11}{'calls':>7}")
    for arm in args.arms:
        row = totals[arm]
        precision = row["gold"] / row["kept"] if row["kept"] else 0.0
        print(f"  {arm:<9}{row['candidates'] / runs:>11.1f}{row['kept'] / runs:>8.1f}"
              f"{row['gold'] / runs:>8.1f}{row['spurious'] / runs:>10.1f}"
              f"{precision:>11.3f}{row['calls'] / runs:>7.1f}")

    print(f"\nper naming row (cases are fixed; kept/gold/spurious are per run):")
    print(f"  {'row':<12}{'cases':>7}{'gold in':>9}" + "".join(
        f"{arm[:7]:>9}{'gold':>7}{'sp':>6}" for arm in args.arms))
    for row in ROWS:
        first = per_row[(args.arms[0], row)]
        line = f"  {row:<12}{first['cases'] / runs:>7.0f}{first['case_gold'] / runs:>9.0f}"
        for arm in args.arms:
            bucket = per_row[(arm, row)]
            line += (f"{bucket['kept'] / runs:>9.1f}{bucket['gold'] / runs:>7.1f}"
                     f"{bucket['spurious'] / runs:>6.1f}")
        print(line)

    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dump, open(args.dump, "w"))
        print("kept pairs written to", args.dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
