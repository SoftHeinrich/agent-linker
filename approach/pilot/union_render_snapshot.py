"""Every prompt and every case `s_linker120` renders, hashed. No LLM calls.

A refactor of the union judge is only allowed to change how the file reads. This
snapshots what it *says*: for all five projects, under an empty alias table and a
recorded one, for every iteration in the trail, it renders every case and every
judging prompt and writes one digest per (project, alias-table, iteration).

    ../.venv/bin/python pilot/union_render_snapshot.py --write before.json
    ...refactor...
    ../.venv/bin/python pilot/union_render_snapshot.py --check before.json

`--check` is the equivalence test the compaction round's lesson demands: write it
before adopting a refactor, not after.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from llm_sad_sam.core.data_types_v2 import DocumentKnowledge  # noqa: E402
from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker120 import SLinker120  # noqa: E402
from llm_sad_sam.linkers.experimental.union_iterations import ITERATIONS  # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository  # noqa: E402

from reading_pilots import BENCH, DATASETS  # noqa: E402

RUN = ROOT.parent / "results/consolidation_e2e_terra_r1_20260825"


def pinned(project):
    path = RUN / "phase_states" / "s_linker110" / "openai" / project / "knowledge.pkl"
    with open(path, "rb") as handle:
        return pickle.load(handle)["doc_knowledge"]


def render(project, knowledge, iteration):
    """Every case and every prompt this iteration renders for this project."""
    text, repo, _ = DATASETS[project]
    components = parse_pcm_repository(str(BENCH / repo))
    sentences = load_sentences(str(BENCH / text))
    sent_map = build_sent_map(sentences)
    name_to_id = {c.name: c.id for c in components}

    linker = SLinker120.__new__(SLinker120)
    linker.doc_knowledge = knowledge
    linker.iteration_name = iteration
    spec = linker.iteration

    out = []
    candidates = linker._name_candidates(sentences, components, name_to_id, sent_map)
    for index, candidate in enumerate(candidates, 1):
        evidence = linker._union_evidence(candidate, components, sent_map)
        out.append(json.dumps(evidence, sort_keys=True))
        out.append(linker._format_union_case(index, candidate, evidence, sent_map))
        out.append(linker._format_union_case(index, candidate, evidence, sent_map,
                                             shown_in=1))
    names = [c.name for c in components]
    table = [{"sentence": s.number, "text": s.text} for s in sentences[:8]]
    for named in (True, False):
        for sentence_table in ([], table):
            out.append(linker._prompt_union(names, sentence_table, out[:2], named))
    out.append(repr((spec.fields, spec.blind_word_only, spec.batch_by_evidence,
                     spec.contract_follows_batch, spec.verdict, spec.clauses)))
    return "\n\x00\n".join(out)


class _Stub:
    """A judge that answers every case, alternating the verdict.

    The uniform round's lesson: an equivalence test whose stub answers nothing
    compares two empty kept-sets and passes on a variant that differs. This one
    answers both contracts, so the parse path and the kept set are both exercised.
    """

    def __init__(self, log):
        self.log = log

    def set_phase(self, phase):
        self.log.append(f"phase={phase}")

    def __call__(self, prompt, **kwargs):
        self.log.append(prompt)
        count = prompt.count("\nCase ")
        return {"validations": [
            {"case": i,
             "claim": "" if i % 4 == 3 else f"quote {i}",
             "approve": i % 2 == 0,
             "denotation": "participant" if i % 3 else "associated"}
            for i in range(1, count + 1)
        ]}


def judged(project, knowledge, iteration):
    """Every prompt `_judge_union` sends and every decision it records. No calls."""
    text, repo, _ = DATASETS[project]
    components = parse_pcm_repository(str(BENCH / repo))
    sentences = load_sentences(str(BENCH / text))
    sent_map = build_sent_map(sentences)
    name_to_id = {c.name: c.id for c in components}

    linker = SLinker120.__new__(SLinker120)
    linker.doc_knowledge = knowledge
    linker.iteration_name = iteration
    log: list[str] = []
    stub = _Stub(log)
    linker.llm = stub
    linker._ask = stub

    candidates = linker._name_candidates(sentences, components, name_to_id, sent_map)
    approved, decisions = linker._judge_union(
        candidates, components, sentences, sent_map)
    log.append(repr(sorted((c.sentence_number, c.component_id) for c in approved)))
    log.append(json.dumps({f"{k[0]}:{k[1]}": v for k, v in decisions.items()},
                          sort_keys=True))
    return "\n\x00\n".join(log)


def snapshot():
    digests = {}
    for project in sorted(DATASETS):
        for label, knowledge in (("empty", DocumentKnowledge()),
                                 ("recorded", pinned(project))):
            for iteration in sorted(ITERATIONS):
                key = f"{project}/{label}/{iteration}"
                body = render(project, knowledge, iteration)
                digests[key] = hashlib.sha256(body.encode()).hexdigest()
                body = judged(project, knowledge, iteration)
                digests[key + "/judge"] = hashlib.sha256(body.encode()).hexdigest()
    return digests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write")
    parser.add_argument("--check")
    args = parser.parse_args()

    digests = snapshot()
    if args.write:
        json.dump(digests, open(args.write, "w"), indent=1, sort_keys=True)
        print(f"{len(digests)} digests written to {args.write}")
        return 0
    if args.check:
        before = json.load(open(args.check))
        drift = sorted(k for k in set(before) | set(digests)
                       if before.get(k) != digests.get(k))
        for key in drift:
            print(f"  CHANGED {key}")
        print(f"{len(digests) - len(drift)}/{len(digests)} renderings identical")
        return 1 if drift else 0
    print(f"{len(digests)} digests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
