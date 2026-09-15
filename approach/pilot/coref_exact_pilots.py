"""Level-2 stage pilot: the antecedent's own surface, as a mark and as a refusal.

`s_linker124` annotated the shortlist with the union judge's VERDICT and the doc-code
gate refused it. `pilot/coref_written_audit.py` then priced the alternative this pilot
measures -- the antecedent's `written` value, the deterministic lexical fact
`s_linker123` already computes for that judge -- and found a clean separation over six
recorded runs on both models:

    antecedent written as      gold     FP   precision
    whole name                  142     14      91.0%
    short form (alias)           39      3      92.9%
    whole name (qualified)        0     10       0.0%
    one word / not on the list    0      7       0.0%

**17 false positives and 0 gold** sit in the two rows where the antecedent sentence does
not write the component's name AS A NAME. The manual reading
(`pilot/coref_antecedent_dump.py`) says what they are: a package path read as a mention
(`Package overview contains storage.api, storage.entity, storage.search.` offered as the
antecedent for `Storage`), and a sentence that never writes the name at all
(`... bbb-html5 uses 2 "frontend" and two "backend" processes` offered for
`HTML5 Server`, six times). Every gold antecedent, without exception, writes the name as
a free-standing noun phrase -- `The Storage component performs CRUD ...`, then `It` in
the next sentence.

WHY THIS FACT AND NOT s124's. A judge's verdict is DISCOVERED -- another stage's output,
resampled every run -- and `s_linker109` already ruled that such a fact may open a case
and may not close one. `written` is GIVEN: catalog plus document, identical every run.
s109's own nesting predicate is the precedent for a given fact closing a case.

    arm `head`        `s_linker123` untouched.
    arm `annotexact`  the mark, in the shortlist, on the entries whose mention is only
                      inside a longer identifier: `Storage (S130, in identifier only)`.
                      The annotation form of the fact -- it informs, it does not decide.
    arm `refuse`      the refusal, in code, before the judge: a resolution whose cited
                      antecedent sentence does not write the name as a name is not put
                      to the judge at all. This EXTENDS a guard `_resolve_references`
                      already has (it drops a resolution citing no antecedent sentence);
                      here the sentence must also mention the component.

The two are the same fact on the two sides of the design law, which is the comparison
the round is for: `annotexact` tells the resolver, `refuse` decides in code.

WHAT IS PINNED: the alias table and the name-link set, from the recorded run's own
checkpoints, so only the resolver and its judge are resampled -- the s124 E2E's defect
(an unpinned name stage whose noise exceeded the effect) cannot recur here.

Usage:

    PY=/path/to/.venv/bin/python
    $PY pilot/coref_exact_pilots.py --verify          # no LLM calls

    OPENAI_API_KEY=... LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.6-terra \\
    OPENAI_REASONING_EFFORT=none OPENAI_SERVICE_TIER=default \\
      $PY pilot/coref_exact_pilots.py --samples 3 --dump dump_terra.json
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
from llm_sad_sam.linkers.experimental.s_linker123 import SLinker123  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker126 import SLinker126  # noqa: E402
from llm_sad_sam.llm_client import LLMBackend  # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository  # noqa: E402

from reading_pilots import BENCH, DATASETS, gold_pairs  # noqa: E402
from score_runs import scores  # noqa: E402

DEFAULT_RUN = ROOT.parent / "results/shortlistmark_e2e_terra_r1_20260914"

#: The two `written` values that ARE a mention of the component: the name itself, and a
#: short form the document established for it. The other two are the refused rows --
#: `whole name (qualified)` (every writing sits inside a longer joined or dotted
#: identifier) and `one word` (the sentence does not write the name at all).
NAMES_IT = ("whole name", "short form")

#: What the mark says. Not a quality ("weak"), a statement of what the sentence does:
#: the name is there, inside a longer identifier, and nowhere else in that sentence.
MARK_QUALIFIED = "in identifier only"


class AnnotExact(SLinker123):
    """The shortlist, with the entries that only name the component inside an identifier.

    Only `_named_before` is touched, and only the sentence-number field of each entry --
    the name stays the catalog's own string and the reply contract is untouched, which
    is the same surface `s_linker124` used and the same one `test_s124.py` pinned.
    """

    def _named_before(self, comp_names, sentence_table, target):
        text_of = {row.get("sentence"): row.get("text", "")
                   for row in sentence_table}
        marked = []
        for name, number in super()._named_before(
                comp_names, sentence_table, target):
            written = self._written_as(text_of.get(number, ""), name)
            marked.append((name, f"{number}, {MARK_QUALIFIED}")
                          if written not in NAMES_IT else (name, number))
        return marked


class RefuseAntecedent(SLinker123):
    """A resolution whose cited antecedent does not write the name as a name is dropped.

    Placed before the judge rather than after it, for the reason `s_linker109` placed
    its refusal before one: a case the code can already answer should not be spent on a
    call. It is the last linker, so nothing downstream can be starved by the removal
    (the measurement policy's level 3, structurally).
    """

    def _antecedent_names_it(self, link, sent_map, metadata):
        """Does the sentence this resolution cites write the component's name as such?"""
        record = metadata.get((link.sentence_number, link.component_id), {})
        number = record.get("antecedent_sentence")
        sentence = sent_map.get(number) if number is not None else None
        if sentence is None:
            # `_resolve_references` already refuses a resolution citing no antecedent;
            # this predicate answers a different question and does not re-answer that one.
            return True
        return self._written_as(
            sentence.text, link.component_name) in NAMES_IT

    def _validate_coref_links(self, coref_links, sent_map, components, metadata):
        return super()._validate_coref_links(
            [link for link in coref_links
             if self._antecedent_names_it(link, sent_map, metadata)],
            sent_map, components, metadata)


ARMS = {"head": SLinker123, "annotexact": AnnotExact, "refuse": RefuseAntecedent,
        # The shipped composition: the shortlist GONE and the contract it used to
        # assert enforced in code instead. Run beside `refuse` (which keeps the
        # list) so the question "does the list still earn its place once the
        # predicate exists?" is answered inside ONE invocation.
        "s126": SLinker126}


def phase(run: Path, project: str, name: str, variant="s_linker123"):
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
        "name_links": {(l.sentence_number, l.component_id)
                       for l in name_state["links"]},
    }


def build(arm, data, backend=None, model=""):
    linker = (ARMS[arm].__new__(ARMS[arm]) if backend is None
              else ARMS[arm](backend=backend, model=model))
    linker.doc_knowledge = data["knowledge"]
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


def shortlist_lines(prompts):
    return [line for p in prompts for line in p.splitlines()
            if line.startswith("NAMED BEFORE THIS CASE:")]


def verify(projects, run):
    """No calls: what each arm changes, and what it must not change."""
    print("=== `refuse` sends the HEAD's resolver prompts, byte for byte ===")
    for project in projects:
        data = load(project, run)
        base = prompts_of("head", data)
        same = prompts_of("refuse", data) == base
        print(f"  {project:<14} identical: {same}")
        assert same, "the refusal must not touch the resolver's case"

    print("\n=== `annotexact` differs only in the shortlist lines, and marks a minority ===")
    for project in projects:
        data = load(project, run)
        base = prompts_of("head", data)
        arm = prompts_of("annotexact", data)

        def strip(ps):
            return ["\n".join(l for l in p.splitlines()
                              if not l.startswith("NAMED BEFORE THIS CASE:"))
                    for p in ps]
        assert strip(arm) == strip(base), "annotexact moved something else"
        entries = sum(line.count("(S") for line in shortlist_lines(base))
        marked = sum(line.count(MARK_QUALIFIED) for line in shortlist_lines(arm))
        share = marked / entries * 100 if entries else 0.0
        print(f"  {project:<14} {entries:5d} entries, {marked:4d} marked "
              f"({share:4.1f}%), everything else identical")

    print("\n=== one marked case per project, so the mark is readable ===")
    for project in projects:
        data = load(project, run)
        line = next((l for l in shortlist_lines(prompts_of("annotexact", data))
                     if MARK_QUALIFIED in l), "")
        print(f"  {project}: {line[:220] if line else 'no marked entry in this run'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+",
                        default=["head", "annotexact", "refuse"])
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

                print(f"  {arm:<12} sample {sample}: proposed {len(proposed):3d} "
                      f"({len(proposed & gold):3d} gold), kept {len(kept):3d} "
                      f"({len(kept & gold):3d} gold), NET {len(net):3d} "
                      f"({len(net & gold):3d} gold), {len(linker._llm_calls)} calls",
                      flush=True)
                dump.setdefault(f"sample{sample}", {}).setdefault(project, {})[arm] = \
                    sorted([list(p) for p in final])

    runs = args.samples
    print(f"\n{model or 'unset'}, {runs} samples, per run over "
          f"{len(args.datasets)} projects — the coreference stage:")
    print(f"  {'arm':<12}{'proposed':>10}{'gold':>7}{'kept':>7}{'gold':>7}"
          f"{'NET':>7}{'gold':>7}{'spurious':>10}{'calls':>7}")
    for arm in args.arms:
        row = totals[arm]
        print(f"  {arm:<12}{row['proposed'] / runs:>10.1f}"
              f"{row['proposed_gold'] / runs:>7.1f}{row['kept'] / runs:>7.1f}"
              f"{row['kept_gold'] / runs:>7.1f}{row['net'] / runs:>7.1f}"
              f"{row['net_gold'] / runs:>7.1f}"
              f"{(row['net'] - row['net_gold']) / runs:>10.1f}"
              f"{row['calls'] / runs:>7.1f}")

    print("\ncomposed with the pinned name links "
          "(TP / FP / macro F1 / macro F2, mean over samples):")
    print(f"  {'arm':<12}{'TP':>8}{'FP':>8}{'macro F1':>11}{'macro F2':>11}")
    per_sample = {}
    for arm in args.arms:
        values = [scores(composed[(arm, s)]) for s in range(1, runs + 1)]
        per_sample[arm] = values
        print(f"  {arm:<12}"
              + "".join(f"{sum(v[k] for v in values) / runs:>8.1f}" for k in ("TP", "FP"))
              + "".join(f"{sum(v[k] for v in values) / runs:>11.2f}"
                        for k in ("macro F1", "macro F2")))

    print("\nper sample, so a reader can see the spread:")
    for arm in args.arms:
        for i, value in enumerate(per_sample[arm], 1):
            print(f"  {arm:<12} sample {i}: TP {value['TP']:3d} FP {value['FP']:3d} "
                  f"F1 {value['macro F1']:.2f} F2 {value['macro F2']:.2f}")

    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump, "w") as handle:
            json.dump(dump, handle)
        print("\ncomposed link sets written to", args.dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
