"""Level-2 stage pilots for the judge round: one gate, fixed candidates, four arms.

Both proposers in front of the two gates measured here are **deterministic scans**
(`s_linker92a`'s named scan, `s_linker109`'s partial scan), so with a recorded alias
table pinned the candidate set is byte-identical across arms and across samples. That
is the cleanest single-step measurement on this branch: the only thing that varies is
the judging prompt, and the only calls spent are that gate's.

    gate `lenient`   the full-name gate            control = the head
                     `readings`  s111: the reply enumerates the surface's readings
    gate `sortal`    the partial-name denotation gate
                     `order`     s112: the quote is demanded before the verdict
                     `readings`  s113: the readings are enumerated between the two

Why these two gates and not the third: `pilot/judge_census.py` puts the F2-weighted
headroom at 27.3/43.3 (terra/luna) for the lenient gate and 20.0/27.7 for the sortal
one, against 15.7/13.0 for the strict coreference gate -- and most of the strict gate's
is gold it loses, which no reply template recovers.

Put every arm a claim rests on in the SAME invocation: absolute levels drift between
invocation sets, so `--arms` in one run is the comparison, not two runs compared after.

    OPENAI_API_KEY="$OAI_KEY" LLM_BACKEND=openai \\
    OPENAI_MODEL_NAME=gpt-5.6-terra OPENAI_REASONING_EFFORT=none \\
      ../.venv/bin/python pilot/nextgen_pilots.py --gate sortal --samples 3
"""
from __future__ import annotations

import argparse
import collections
import csv
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
from llm_sad_sam.linkers.experimental.s_linker111 import SLinker111  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker112 import SLinker112  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker113 import SLinker113  # noqa: E402
from llm_sad_sam.llm_client import LLMBackend  # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository  # noqa: E402

from reading_pilots import BENCH, DATASETS, gold_pairs  # noqa: E402

#: A recorded run of the head, read only for its approved alias table. Pinning it
#: holds the alias module's own 2.8-term run-to-run swing out of every delta.
DEFAULT_RUN = ROOT.parent / "results/consolidation_e2e_terra_r1_20260825"

GATES = {
    "lenient": {"control": SLinker110, "readings": SLinker111},
    "sortal": {"control": SLinker110, "order": SLinker112, "readings": SLinker113},
}


def pinned_knowledge(run: Path, project: str):
    """The alias table a recorded run's judge approved, as the head stored it."""
    path = run / "phase_states" / "s_linker110" / "openai" / project / "knowledge.pkl"
    with open(path, "rb") as handle:
        return pickle.load(handle)["doc_knowledge"]


def run_gate(gate, linker, sentences, components, name_to_id, sent_map):
    """One gate over its own scan's candidates. Returns (kept pairs, candidates)."""
    if gate == "lenient":
        links, feedback = linker._run_full_name_linker(
            sentences, components, name_to_id, sent_map)
        proposed = feedback["candidates"]
    else:
        links, feedback = linker._run_partial_name_linker(
            sentences, components, sent_map)
        proposed = feedback["proposed"]
    return ({(l.sentence_number, l.component_id) for l in links}, len(proposed))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=sorted(GATES), required=True)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--datasets", nargs="+", default=sorted(DATASETS))
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--dump")
    args = parser.parse_args()

    arms = args.arms or list(GATES[args.gate])
    unknown = [a for a in arms if a not in GATES[args.gate]]
    if unknown:
        parser.error(f"unknown arm(s) for gate {args.gate}: {unknown}")

    backend = LLMBackend.OPENAI
    model = os.environ.get("OPENAI_MODEL_NAME", "")
    totals = collections.defaultdict(collections.Counter)
    dump: dict = {}

    for project in args.datasets:
        text, repo, gold_path = DATASETS[project]
        components = parse_pcm_repository(str(BENCH / repo))
        sentences = load_sentences(str(BENCH / text))
        sent_map = build_sent_map(sentences)
        name_to_id = {c.name: c.id for c in components}
        gold = gold_pairs(BENCH / gold_path)
        knowledge = pinned_knowledge(args.run, project)
        print(f"\n=== {project}: {len(sentences)} sentences, {len(components)} "
              f"components, {len(gold)} gold, "
              f"{len(getattr(knowledge, 'aliases', {}) or {})} aliases held fixed ===",
              flush=True)

        sizes = set()
        for sample in range(1, args.samples + 1):
            for arm in arms:
                linker = GATES[args.gate][arm](backend=backend, model=model)
                linker.doc_knowledge = knowledge
                kept, proposed = run_gate(
                    args.gate, linker, sentences, components, name_to_id, sent_map)
                sizes.add(proposed)
                good = len(kept & gold)
                totals[arm]["kept"] += len(kept)
                totals[arm]["gold"] += good
                totals[arm]["spurious"] += len(kept) - good
                totals[arm]["candidates"] += proposed
                dump.setdefault(f"sample{sample}", {}).setdefault(project, {})[arm] = \
                    sorted([list(p) for p in kept])
                print(f"  {arm:<9} sample {sample}: {len(kept):4d} kept, "
                      f"{good:4d} gold, {len(kept) - good:4d} spurious "
                      f"(of {proposed} candidates)", flush=True)
        if len(sizes) > 1:
            print(f"  !! candidate set differs across arms: {sorted(sizes)}")

    runs = args.samples
    print(f"\n{args.gate} gate on {model or 'unset'}, {runs} samples, "
          f"per five-project run:")
    print(f"  {'arm':<10}{'candidates':>12}{'kept':>8}{'gold':>8}{'spurious':>10}"
          f"{'precision':>11}")
    for arm in arms:
        row = totals[arm]
        precision = row["gold"] / row["kept"] if row["kept"] else 0.0
        print(f"  {arm:<10}{row['candidates'] / runs:>12.1f}{row['kept'] / runs:>8.1f}"
              f"{row['gold'] / runs:>8.1f}{row['spurious'] / runs:>10.1f}"
              f"{precision:>11.3f}")
    if args.dump:
        json.dump(dump, open(args.dump, "w"))
        print("kept pairs written to", args.dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
