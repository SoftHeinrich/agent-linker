"""Level-2 stage pilot: the coreference shortlist, annotated with the name linker's
verdicts.

`s_linker122`'s resolver prints a per-case shortlist of the components the sentences
above it NAME -- a lexical fact `_states_a_name` computes. The name linker has already
judged every one of those mentions by then and `_run_linker` withholds its verdicts on
purpose. This pilot prices handing them over.

    arm `head`          the head's shortlist: `Facade (S3)`
    arm `annot`         the same entries, each carrying the earlier judge's verdict:
                        `Facade (S3, linked)` / `Storage (S130, named only)`.
                        The mark and nothing else -- no rule speaks about it.
    arm `annot_clause`  the same mark, plus one sentence saying what it means and that
                        a `named only` entry is the weaker antecedent. The fact is in
                        the case, the weighing is in the prompt (the branch's design
                        law), so this arm is the one the law predicts should work.
    arm `annot_only`    the suppressing extreme: entries the name judge rejected are
                        not listed at all. `pilot/coref_shortlist_audit.py` prices its
                        ceiling with no calls (0.3 net links a run on both models), so
                        this arm is here to bound the RISK, not to win.

WHAT IS PINNED. Both the alias table and the name linker's link set come from the
recorded run's own checkpoints, so every arm is annotated against the SAME verdicts and
nothing about the name stage is resampled into the comparison. Only the resolver and
its own judge are re-run.

WHAT IS SCORED. A stage read on the resolver alone would miss the thing that decides:
`link` merges by pair and an earlier linker wins, so a resolution for a pair the name
linker already carries changes nothing. The pilot therefore reports the NET
contribution (kept, and not already a name link) and scores the composed link set
`pinned name links | kept coreference` with `score_runs.scores` -- the same TP / FP /
macro F1 / macro F2 the E2E batches are read with. Because the name half is pinned
rather than resampled, the composed number here carries the resolver's variance only.

Usage:

    PY=/path/to/.venv/bin/python
    $PY pilot/coref_annot_pilots.py --verify          # no LLM calls

    OPENAI_API_KEY=... LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.6-terra \\
    OPENAI_REASONING_EFFORT=none OPENAI_SERVICE_TIER=default \\
      $PY pilot/coref_annot_pilots.py --samples 3 --dump dump_terra.json
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
from llm_sad_sam.linkers.experimental.s_linker122 import SLinker122  # noqa: E402
from llm_sad_sam.llm_client import LLMBackend  # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository  # noqa: E402

from reading_pilots import BENCH, DATASETS, gold_pairs  # noqa: E402
from score_runs import scores  # noqa: E402

DEFAULT_RUN = ROOT.parent / "results/noanchor_e2e_terra_r1_20260914"

#: The two marks. Deliberately not a judgement in themselves -- `linked` is what the
#: earlier stage kept, `named only` is what it saw and did not keep. Naming them after
#: the verdict rather than after a quality ("strong"/"weak") keeps the case a statement
#: of fact; whether a `named only` entry is worth less is the clause's business.
MARK_LINKED = "linked"
MARK_NAMED = "named only"

#: The weighing `annot_clause` adds, and the only sentence any arm here adds anywhere.
#: The mark is a fact the earlier stage produced; what to do with it is not derivable
#: from the case, so by the design law it belongs in the prompt and not in the mark.
ANNOT_CLAUSE = (
    "Each entry is marked `linked` where an earlier reading of that sentence found it "
    "asserts something of the component, and `named only` where it found the name "
    "doing some other job there. A `named only` entry is the weaker antecedent."
)

#: Where the clause goes: in the head's own paragraph about the list, after it says
#: what the list is and before it says what to do with it, so the mark is explained
#: where the list is introduced and nowhere else. The anchor is a span of the head's
#: prompt, not a re-typing of it, and the placement is asserted -- if the paragraph is
#: reworded this arm fails loudly instead of silently measuring the head.
ANCHOR_SENTENCE = "Quote the referring expression first,"


class _Annotated(SLinker122):
    """The head with a name-verdict-aware shortlist. Subclasses choose what to do.

    `name_links` is set by the caller from the recorded run, as {(sentence, name)}.
    Only `_named_before` is touched: its result is read nowhere but the shortlist line,
    so decorating the sentence number there changes the case and nothing else.
    """

    name_links: set = frozenset()

    def _mark(self, name, number):
        return (MARK_LINKED if (number, name) in self.name_links else MARK_NAMED)

    def _named_before(self, comp_names, sentence_table, target):
        near = super()._named_before(comp_names, sentence_table, target)
        return [(name, f"{number}, {self._mark(name, number)}")
                for name, number in near]


class Annot(_Annotated):
    """The mark alone. No rule in the prompt speaks about it."""


class AnnotClause(_Annotated):
    """The mark, plus the one sentence that says how to weigh it."""

    def _prompt_coref(self, comp_names, sentence_table, targets) -> str:
        prompt = super()._prompt_coref(comp_names, sentence_table, targets)
        placed = prompt.replace(
            ANCHOR_SENTENCE, f"{ANNOT_CLAUSE} {ANCHOR_SENTENCE}", 1)
        assert placed != prompt, "the clause was not placed"
        return placed


class AnnotOnly(SLinker122):
    """The suppressing extreme: a rejected mention is not offered as an antecedent."""

    name_links: set = frozenset()

    def _named_before(self, comp_names, sentence_table, target):
        return [(name, number)
                for name, number in super()._named_before(
                    comp_names, sentence_table, target)
                if (number, name) in self.name_links]


ARMS = {"head": SLinker122, "annot": Annot, "annot_clause": AnnotClause,
        "annot_only": AnnotOnly}


def phase(run: Path, project: str, name: str, variant="s_linker122"):
    path = run / "phase_states" / variant / "openai" / project / f"{name}.pkl"
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load(project, run):
    text, repo, gold_path = DATASETS[project]
    components = parse_pcm_repository(str(BENCH / repo))
    sentences = load_sentences(str(BENCH / text))
    name_state = phase(run, project, "linker_name")
    return {
        "components": components,
        "sentences": sentences,
        "sent_map": build_sent_map(sentences),
        "name_to_id": {c.name: c.id for c in components},
        "gold": gold_pairs(BENCH / gold_path),
        "knowledge": phase(run, project, "knowledge")["doc_knowledge"],
        # The name half, pinned: the pairs it kept, and the same set keyed by name for
        # the shortlist mark. Nothing here is re-judged by any arm.
        "name_links": {(l.sentence_number, l.component_id)
                       for l in name_state["links"]},
        "name_links_by_name": {(l.sentence_number, l.component_name)
                               for l in name_state["links"]},
    }


def build(arm, data, backend=None, model=""):
    linker = (ARMS[arm].__new__(ARMS[arm]) if backend is None
              else ARMS[arm](backend=backend, model=model))
    linker.doc_knowledge = data["knowledge"]
    if arm != "head":
        linker.name_links = data["name_links_by_name"]
    return linker


def prompts_of(arm, data):
    """Every resolver prompt the arm would send, with `_ask` stubbed. No calls."""
    linker = build(arm, data)
    sent = []

    class _Recorder:
        def set_phase(self, phase_name):
            pass

    linker.llm = _Recorder()
    linker._ask = lambda prompt, **_: sent.append(prompt) or {"resolutions": []}
    linker._resolve_references(
        data["sentences"], data["components"],
        data["name_to_id"], data["sent_map"])
    return sent


def verify(projects, run):
    """The zero-call half: does each arm change the bytes, and by how much?"""
    print("=== resolver prompts: does the arm differ from the head, and how far? ===")
    for project in projects:
        data = load(project, run)
        base = prompts_of("head", data)
        base_bytes = sum(len(p) for p in base)
        line = (f"  {project:<14} head {len(base):2d} calls, "
                f"{base_bytes:6d} B")
        for arm in ("annot", "annot_clause", "annot_only"):
            other = prompts_of(arm, data)
            same = "IDENTICAL" if other == base else "differs"
            delta = sum(len(p) for p in other) - base_bytes
            line += f" | {arm} {sum(len(p) for p in other):6d} B ({delta:+5d}) {same}"
        print(line)

    print("\n=== one annotated case per project, so the mark is readable ===")
    for project in projects:
        data = load(project, run)
        shown = False
        for prompt in prompts_of("annot_clause", data):
            for chunk in prompt.split("--- Case ")[1:]:
                line = next((l for l in chunk.splitlines()
                             if l.startswith("NAMED BEFORE")), "")
                if MARK_NAMED in line:
                    print(f"  {project}: {line[:300]}")
                    shown = True
                    break
            if shown:
                break
        if not shown:
            print(f"  {project}: no `{MARK_NAMED}` entry anywhere in this run")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+",
                        default=["head", "annot", "annot_clause", "annot_only"])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--datasets", nargs="+", default=sorted(DATASETS))
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--dump")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify:
        verify(args.datasets, args.run)
        return 0

    backend = LLMBackend.OPENAI
    model = os.environ.get("OPENAI_MODEL_NAME", "")
    totals = collections.defaultdict(collections.Counter)
    composed: dict = collections.defaultdict(set)
    dump: dict = {}

    for project in args.datasets:
        data = load(project, args.run)
        gold = data["gold"]
        pinned = data["name_links"]
        print(f"\n=== {project}: {len(data['sentences'])} sentences, "
              f"{len(data['components'])} components, {len(gold)} gold, "
              f"{len(pinned)} name links pinned "
              f"({len(pinned & gold)} gold) ===", flush=True)

        for sample in range(1, args.samples + 1):
            for arm in args.arms:
                linker = build(arm, data, backend, model)
                approved, feedback = linker._run_coreference_linker(
                    data["sentences"], data["components"],
                    data["name_to_id"], data["sent_map"])
                proposed = {(row["sentence"],
                             data["name_to_id"].get(row["component"]))
                            for row in feedback["candidates"]}
                kept = {(l.sentence_number, l.component_id) for l in approved}
                net = kept - pinned
                final = pinned | kept

                totals[arm]["proposed"] += len(proposed)
                totals[arm]["proposed_gold"] += len(proposed & gold)
                totals[arm]["kept"] += len(kept)
                totals[arm]["kept_gold"] += len(kept & gold)
                totals[arm]["net"] += len(net)
                totals[arm]["net_gold"] += len(net & gold)
                totals[arm]["calls"] += len(linker._llm_calls)
                composed[(arm, sample)] |= {(project, s, c) for s, c in final}

                print(f"  {arm:<13} sample {sample}: proposed {len(proposed):3d} "
                      f"({len(proposed & gold):3d} gold), kept {len(kept):3d} "
                      f"({len(kept & gold):3d} gold), NET {len(net):3d} "
                      f"({len(net & gold):3d} gold), {len(linker._llm_calls)} calls",
                      flush=True)
                dump.setdefault(f"sample{sample}", {}).setdefault(project, {})[arm] = \
                    sorted([list(p) for p in final])

    runs = args.samples
    print(f"\n{model or 'unset'}, {runs} samples, per run over "
          f"{len(args.datasets)} projects — the coreference stage:")
    print(f"  {'arm':<13}{'proposed':>10}{'gold':>7}{'kept':>7}{'gold':>7}"
          f"{'NET':>7}{'gold':>7}{'spurious':>10}{'calls':>7}")
    for arm in args.arms:
        row = totals[arm]
        print(f"  {arm:<13}{row['proposed'] / runs:>10.1f}"
              f"{row['proposed_gold'] / runs:>7.1f}{row['kept'] / runs:>7.1f}"
              f"{row['kept_gold'] / runs:>7.1f}{row['net'] / runs:>7.1f}"
              f"{row['net_gold'] / runs:>7.1f}"
              f"{(row['net'] - row['net_gold']) / runs:>10.1f}"
              f"{row['calls'] / runs:>7.1f}")

    print("\ncomposed with the pinned name links "
          "(TP / FP / macro F1 / macro F2, mean over samples):")
    print(f"  {'arm':<13}{'TP':>8}{'FP':>8}{'macro F1':>11}{'macro F2':>11}")
    per_sample = {}
    for arm in args.arms:
        values = [scores(composed[(arm, s)]) for s in range(1, runs + 1)]
        per_sample[arm] = values
        print(f"  {arm:<13}"
              + "".join(f"{sum(v[k] for v in values) / runs:>8.1f}" for k in ("TP", "FP"))
              + "".join(f"{sum(v[k] for v in values) / runs:>11.2f}"
                        for k in ("macro F1", "macro F2")))

    print("\nper sample, so a reader can see the spread:")
    for arm in args.arms:
        for i, value in enumerate(per_sample[arm], 1):
            print(f"  {arm:<13} sample {i}: TP {value['TP']:3d} FP {value['FP']:3d} "
                  f"F1 {value['macro F1']:.2f} F2 {value['macro F2']:.2f}")

    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump, "w") as handle:
            json.dump(dump, handle)
        print("\ncomposed link sets written to", args.dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
