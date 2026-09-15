"""Error analysis for the union-judge pilot. No LLM calls — reads a pilot's dump.

`pilot/union_pilots.py --dump` writes the kept pairs of every arm in every sample. This
splits the disagreement between the arms by the fact the union is graded on
(`naming`, capitalization, the alternative set), and prints the cases themselves, so an
iteration changes a clause for a reason that is in the data.

    ../.venv/bin/python pilot/union_diff.py ../results/union_round/dump_terra_v1.json
    ../.venv/bin/python pilot/union_diff.py <dump> --show 25 --row "word only"
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

from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker120 import SLinker120  # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository  # noqa: E402

from reading_pilots import BENCH, DATASETS, gold_pairs  # noqa: E402
from union_pilots import DEFAULT_RUN, pinned_knowledge  # noqa: E402


def project_context(project, run):
    text, repo, gold_path = DATASETS[project]
    components = parse_pcm_repository(str(BENCH / repo))
    sentences = load_sentences(str(BENCH / text))
    sent_map = build_sent_map(sentences)
    reader = SLinker120.__new__(SLinker120)
    reader.doc_knowledge = pinned_knowledge(run, project)
    candidates = reader._name_candidates(
        sentences, components, {c.name: c.id for c in components}, sent_map)
    evidence = {(c.sentence_number, c.component_id):
                reader._union_evidence(c, components, sent_map) for c in candidates}
    return {
        "gold": gold_pairs(BENCH / gold_path),
        "evidence": evidence,
        "name_of": {c.id: c.name for c in components},
        "text_of": {s.number: s.text for s in sentences},
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump")
    parser.add_argument("--arms", nargs=2, default=["control", "union"])
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--show", type=int, default=20)
    parser.add_argument("--row", default=None)
    args = parser.parse_args()

    with open(args.dump) as handle:
        dump = json.load(handle)
    left, right = args.arms
    samples = sorted(dump)
    contexts = {}
    counts = collections.defaultdict(collections.Counter)
    cases = collections.defaultdict(list)

    for sample in samples:
        for project, arms in dump[sample].items():
            if left not in arms or right not in arms:
                continue
            if project not in contexts:
                contexts[project] = project_context(project, args.run)
            context = contexts[project]
            gold = context["gold"]
            kept = {arm: {tuple(pair) for pair in arms[arm]} for arm in (left, right)}
            for pair in kept[left] ^ kept[right]:
                evidence = context["evidence"].get(pair)
                if evidence is None:
                    continue
                row = evidence["naming"]
                shape = "capitalized" if evidence["span"][:1].isupper() else "lowercase"
                direction = ("only " + left) if pair in kept[left] else ("only " + right)
                kind = "gold" if pair in gold else "spurious"
                counts[(row, shape)][f"{direction}, {kind}"] += 1
                cases[(row, direction, kind)].append((project, pair, evidence, sample))

    print(f"disagreement between `{left}` and `{right}`, "
          f"{len(samples)} samples, counted per case-sample\n")
    print(f"  {'row':<12}{'shape':<13}" + "".join(
        f"{key:>26}" for key in (f"only {left}, gold", f"only {left}, spurious",
                                 f"only {right}, gold", f"only {right}, spurious")))
    for key in sorted(counts):
        row, shape = key
        bucket = counts[key]
        print(f"  {row:<12}{shape:<13}"
              + "".join(f"{bucket[k]:>26}" for k in
                        (f"only {left}, gold", f"only {left}, spurious",
                         f"only {right}, gold", f"only {right}, spurious")))

    print("\nthe cases, by bucket (deduplicated across samples):")
    for key in sorted(cases):
        row, direction, kind = key
        if args.row and row != args.row:
            continue
        seen = {}
        for project, pair, evidence, _sample in cases[key]:
            seen.setdefault((project, pair), evidence)
        print(f"\n  [{row}] {direction}, {kind} — {len(seen)} distinct cases")
        for (project, pair), evidence in list(seen.items())[:args.show]:
            context = contexts[project]
            alternatives = (", alternatives=" + ", ".join(evidence["alternatives"])
                            if evidence["alternatives"] else "")
            print(f"    {project} S{pair[0]} {context['name_of'][pair[1]]}  "
                  f"span=\"{evidence['span']}\"{alternatives}")
            print(f"      {context['text_of'][pair[0]][:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
