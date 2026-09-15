"""Level-2 stage pilot: is the antecedent shortlist itself an attachment prior?

`s_linker124` marked the shortlist with the name judge's verdicts and the doc-code gate
refused it (`../results/coref_annot_round/README.md`). The diagnosis that came out of the
refusal is what this pilot tests, and it is NOT "the mark picked the wrong antecedent":

  * The gold component was among the resolver's own `candidates` in **0** of the
    surviving false positives, on both arms. These are not mis-picked antecedents.
  * 10 of the mark's 15 extra false positives landed in cases where the resolver listed
    exactly ONE candidate -- no competition at all, only the decision whether to attach.
    Single-candidate precision fell 95.5% -> 77.4% while multi-candidate precision barely
    moved (84.9% -> 81.5%).

So the mark did not mis-discriminate; it **over-attached**. That makes the shortlist's own
framing the thing under suspicion, because the resolver prompt already asserts it:

    That list has already been checked against the document, so it is where the
    antecedent will be if there is one.

That sentence tells the model an antecedent exists and where. The three arms separate the
list's EVIDENCE from its ENDORSEMENT:

    arm `head`         `s_linker123` untouched -- the old computed shortlist, and the
                       control. This is also the "roll the mark back" arm: deleting
                       `s_linker124`'s two overrides lands exactly here.
    arm `noprior`      the same list, with the endorsing sentence removed and the
                       procedure kept. The model still sees the candidates; it is no
                       longer told the answer is among them.
    arm `noshortlist`  no `NAMED BEFORE THIS CASE` line and no paragraph about it. The
                       resolver reads the sentences and decides unaided.

WHAT IS PINNED, and it is the E2E's defect. Both the alias table and the name linker's
link set come from the recorded run's checkpoints, so the name stage is not resampled into
the comparison. The s124 E2E did resample it, and its `full_name` noise (terra +12 FP a
batch) exceeded the effect under test -- that is the round's methodological finding, and
this pilot is built not to repeat it.

WHAT IS SCORED. `link` merges by pair and an earlier linker wins, so the pilot reports the
NET contribution (kept, and not already a name link) and scores the composed set
`pinned name links | kept coreference` with `score_runs.scores` -- the same TP / FP /
macro F1 / macro F2 the E2E batches are read with.

Usage:

    PY=/path/to/.venv/bin/python
    $PY pilot/coref_shortlist_pilots.py --verify          # no LLM calls

    OPENAI_API_KEY=... LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.6-terra \\
    OPENAI_REASONING_EFFORT=none OPENAI_SERVICE_TIER=default \\
      $PY pilot/coref_shortlist_pilots.py --samples 3 --dump dump_terra.json
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
from llm_sad_sam.llm_client import LLMBackend  # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository  # noqa: E402

from reading_pilots import BENCH, DATASETS, gold_pairs  # noqa: E402
from score_runs import scores  # noqa: E402

DEFAULT_RUN = ROOT.parent / "results/shortlistmark_e2e_terra_r1_20260914"

#: The per-case line the shortlist is rendered on. `noshortlist` drops it.
CASE_LINE = "NAMED BEFORE THIS CASE:"

#: The sentence that turns the list from evidence into an endorsement: it asserts the
#: antecedent EXISTS and that it is in the list. Removed whole by `noprior`, because a
#: half-removal would leave the prompt asserting one of the two.
PRIOR_SENTENCE = (
    "That list has\nalready been checked against the document, so it is where the "
    "antecedent will be if\nthere is one. ")

#: The whole paragraph, which `noshortlist` replaces -- a prompt that no longer prints
#: the list must not keep explaining it.
SHORTLIST_PARA = (
    "Each case lists NAMED BEFORE THIS CASE: the components the sentences above it\n"
    "actually name, with the sentence that names each, nearest first. That list has\n"
    "already been checked against the document, so it is where the antecedent will be if\n"
    "there is one. Quote the referring expression first, then say which entries of that\n"
    "list could be what it points to, then name the one it does point to.")

#: What replaces it. The procedural half of the instruction survives -- quoting the
#: referring expression first is what makes the reply auditable and is not about the
#: list -- and nothing is added that the head does not also say.
NO_LIST_PARA = (
    "Quote the referring expression first, then name the component it points to.")


class NoPrior(SLinker123):
    """The list, stripped of the claim that the answer is in it."""

    def _prompt_coref(self, comp_names, sentence_table, targets) -> str:
        prompt = super()._prompt_coref(comp_names, sentence_table, targets)
        out = prompt.replace(PRIOR_SENTENCE, "", 1)
        assert out != prompt, "the endorsing sentence was not found in the head's prompt"
        return out


class NoShortlist(SLinker123):
    """No list and no paragraph about one. The resolver reads the sentences unaided."""

    def _prompt_coref(self, comp_names, sentence_table, targets) -> str:
        prompt = super()._prompt_coref(comp_names, sentence_table, targets)
        out = "\n".join(line for line in prompt.splitlines()
                        if not line.startswith(CASE_LINE))
        assert out != prompt, "no shortlist line was found to remove"
        assert SHORTLIST_PARA in out, "the head's shortlist paragraph moved"
        return out.replace(SHORTLIST_PARA, NO_LIST_PARA, 1)


ARMS = {"head": SLinker123, "noprior": NoPrior, "noshortlist": NoShortlist}


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
        # The name half, pinned. No arm here re-judges it and no arm reads its verdicts:
        # that was s124's mechanism and it is not on trial in this pilot.
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


def verify(projects, run):
    """The zero-call half: does each arm change the bytes, and does it change only that?"""
    print("=== resolver prompts: how far each arm moves from the head ===")
    for project in projects:
        data = load(project, run)
        base = prompts_of("head", data)
        base_bytes = sum(len(p) for p in base)
        line = f"  {project:<14} head {len(base):2d} calls, {base_bytes:6d} B"
        for arm in ("noprior", "noshortlist"):
            other = prompts_of(arm, data)
            assert len(other) == len(base), f"{arm} changed the call count"
            total = sum(len(p) for p in other)
            line += f" | {arm} {total:6d} B ({total - base_bytes:+6d})"
        print(line)

    print("\n=== the shortlist is gone from `noshortlist`, and only that ===")
    for project in projects:
        data = load(project, run)
        base = prompts_of("head", data)
        none_arm = prompts_of("noshortlist", data)
        listed = sum(1 for p in base for l in p.splitlines()
                     if l.startswith(CASE_LINE))
        left = sum(1 for p in none_arm for l in p.splitlines()
                   if l.startswith(CASE_LINE))
        # every non-shortlist, non-paragraph line must be untouched
        def strip(ps):
            return ["\n".join(l for l in p.splitlines()
                              if not l.startswith(CASE_LINE))
                    .replace(SHORTLIST_PARA, "").replace(NO_LIST_PARA, "")
                    for p in ps]
        same = strip(base) == strip(none_arm)
        print(f"  {project:<14} head listed {listed:3d} shortlists, "
              f"noshortlist has {left:3d}; everything else identical: {same}")
        assert left == 0 and same

    print("\n=== `noprior` differs from the head by exactly the one sentence ===")
    for project in projects:
        data = load(project, run)
        base = prompts_of("head", data)
        arm = prompts_of("noprior", data)
        ok = all(a == b.replace(PRIOR_SENTENCE, "", 1) for a, b in zip(arm, base))
        delta = sum(len(b) - len(a) for a, b in zip(arm, base))
        print(f"  {project:<14} exact single-sentence removal: {ok}  "
              f"({delta} B removed over {len(base)} calls)")
        assert ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+",
                        default=["head", "noprior", "noshortlist"])
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
