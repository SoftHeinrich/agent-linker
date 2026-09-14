"""Level-2 stage pilot for two `s_linker121` ablations: the anchors block, and the scan's
one refusal. Both arms and the head run in the same invocation, on the same pinned alias
table, so nothing is compared across invocation sets.

    arm `head`        `s_linker121` as it stands
    arm `noanchor`    the case carries no `anchors` block and the rule carries no
                      `anchors` line (the line is *sliced* off `_FIELD_LINES`, not
                      retyped, so a drift in the constant is a drift in the arm)
    arm `norefusal`   `_only_inside_another_name` never fires, so the one-word scan
                      keeps every pair it finds

The two ablations are not symmetric and the pilot does not pretend they are:

  * `noanchor` changes what the judge is shown for **every** candidate on every project.
  * `norefusal` changes the **candidate set**, and only where the refusal fires. On four
    of the five projects the arm is the head by construction — same candidates, same
    prompt bytes — which `--verify` checks with no calls. Calls are spent for it only
    where it differs.

`--verify` (no calls) prints, per project: what the refusal drops and how much of that
is gold, and whether each arm's prompts differ from the head's at all.

    ../.venv/bin/python pilot/s121_ablations.py --verify

    OPENAI_API_KEY=... LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.6-terra \\
    OPENAI_REASONING_EFFORT=none OPENAI_SERVICE_TIER=default \\
      ../.venv/bin/python pilot/s121_ablations.py --samples 3 --dump dump_terra.json

The dump is `union_stats.py`'s format, so the paired sign-flip test is the same one the
union round read its deltas with:

    ../.venv/bin/python pilot/union_stats.py dump_terra.json --arms head noanchor
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
from llm_sad_sam.linkers.experimental import s_linker121 as head_module  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker121 import NameForm, SLinker121  # noqa: E402
from llm_sad_sam.llm_client import LLMBackend  # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository  # noqa: E402

from reading_pilots import BENCH, DATASETS, gold_pairs  # noqa: E402
from union_pilots import DEFAULT_RUN, ROWS  # noqa: E402


#: The rule's anchors line, computed off the head's own constant rather than retyped:
#: it is the last line of `_FIELD_LINES`, and removing it is what "no anchors" means in
#: the rule. If the constant is reordered this slice fails loudly instead of silently
#: removing nothing (`_check_arms` asserts the prompt actually changed).
ANCHOR_RULE_LINE = "\n  anchors -- " + head_module._FIELD_LINES.split(
    "  anchors -- ", 1)[1]


class NoAnchors(SLinker121):
    """The head, minus the `anchors` evidence: not computed, not printed, not ruled on.

    Both halves go together on purpose. Dropping the block but keeping the rule's line
    about it would leave the judge a line about a field no case carries, which measures
    a different thing than removing the evidence.
    """

    def _union_evidence(self, candidate, components, sent_map):
        evidence = super()._union_evidence(candidate, components, sent_map)
        evidence["anchors"] = []
        return evidence

    def _prompt_union(self, comp_names, sentence_table, cases) -> str:
        return super()._prompt_union(comp_names, sentence_table, cases).replace(
            ANCHOR_RULE_LINE, "")


class NoRefusal(SLinker121):
    """The head, minus the one-word scan's single refusal.

    `_only_inside_another_name` is the one place the deterministic layer ends a case
    rather than opening one. With it off, every pair the scan finds reaches the judge.
    """

    def _only_inside_another_name(self, text, name, components) -> bool:
        return False


class NoRefusalSplit(NoRefusal):
    """`norefusal`, with the head's batches held byte-identical.

    The plain `norefusal` arm changes two things at once: the judge sees 12 more cases,
    and those cases change where the batch boundaries fall (bigbluebutton goes 4 calls
    to 5). The union round measured the batch boundary as an effect in its own right,
    so an arm that moves both cannot say which one it measured.

    This arm judges the head's candidates in the head's own batches — the same call
    bytes, verified by `--verify` — and then judges the refused pairs in one extra call
    of their own. Whatever it differs from the head by is the added cases and nothing
    else.
    """

    def _judge_union(self, candidates, components, sentences, sent_map):
        base, extra = [], []
        for candidate in candidates:
            refused = SLinker121._only_inside_another_name(
                self, candidate.sentence_text, candidate.component_name, components)
            (extra if refused else base).append(candidate)
        approved, decisions = super()._judge_union(
            base, components, sentences, sent_map)
        if extra:
            more_approved, more_decisions = super()._judge_union(
                extra, components, sentences, sent_map)
            approved = list(approved) + list(more_approved)
            decisions = {**decisions, **more_decisions}
        return approved, decisions


ARMS = {"head": SLinker121, "noanchor": NoAnchors, "norefusal": NoRefusal,
        "norefusal_split": NoRefusalSplit}


def pinned_knowledge(run: Path, project: str):
    path = run / "phase_states" / "s_linker110" / "openai" / project / "knowledge.pkl"
    with open(path, "rb") as handle:
        return pickle.load(handle)["doc_knowledge"]


def load(project, run):
    text, repo, gold_path = DATASETS[project]
    components = parse_pcm_repository(str(BENCH / repo))
    sentences = load_sentences(str(BENCH / text))
    return {
        "components": components,
        "sentences": sentences,
        "sent_map": build_sent_map(sentences),
        "name_to_id": {c.name: c.id for c in components},
        "gold": gold_pairs(BENCH / gold_path),
        "knowledge": pinned_knowledge(run, project),
    }


def probe(arm, data):
    """An arm instance with no LLM behind it, for the deterministic questions."""
    linker = ARMS[arm].__new__(ARMS[arm])
    linker.doc_knowledge = data["knowledge"]
    return linker


def rows_of(data):
    """{(sentence, component id): naming row}, read with the head's own evidence."""
    reader = probe("head", data)
    candidates = reader._name_candidates(
        data["sentences"], data["components"], data["name_to_id"], data["sent_map"])
    return {(c.sentence_number, c.component_id):
            reader._union_evidence(c, data["components"], data["sent_map"])["naming"]
            for c in candidates}, candidates


def prompts_of(arm, data):
    """Every prompt the arm would send, with `_ask` stubbed. No calls.

    This is the identity check: two arms that send the same bytes over the same
    candidates are the same arm on this project, whatever their code says.
    """
    linker = probe(arm, data)
    sent = []

    class _Recorder:
        def set_phase(self, phase):
            pass

    linker.llm = _Recorder()
    linker._ask = lambda prompt, **_: sent.append(prompt) or {"validations": []}
    candidates = linker._name_candidates(
        data["sentences"], data["components"], data["name_to_id"], data["sent_map"])
    linker._judge_union(
        candidates, data["components"], data["sentences"], data["sent_map"])
    return sent, candidates


def refused_pairs(data):
    """The pairs the refusal drops on this project, as {(sentence, component): surface}."""
    linker = probe("head", data)
    everything = {}
    for sentence in data["sentences"]:
        for component in data["components"]:
            if linker._states_a_name(sentence.text, component.name):
                continue
            spans = linker._name_spans(
                sentence.text, component.name, NameForm.ANY_WORD)
            if spans:
                start, end = spans[-1]
                everything[(sentence.number, component.id)] = (
                    component.name, sentence.text[start:end], sentence.text)
    kept = {(c.sentence_number, c.component_id)
            for c in linker._scan(data["sentences"], data["components"])}
    return {pair: everything[pair] for pair in everything if pair not in kept}


def verify(projects, run):
    """The zero-call half: what the refusal drops, and which arms differ at all."""
    print("=== the refusal: what it drops, and whether any of it is gold ===")
    total = gold_total = 0
    for project in projects:
        data = load(project, run)
        dropped = refused_pairs(data)
        golden = [pair for pair in dropped if pair in data["gold"]]
        total += len(dropped)
        gold_total += len(golden)
        print(f"  {project:<14} refused {len(dropped):3d}   gold among them "
              f"{len(golden)}")
        for pair in sorted(dropped):
            name, surface, text = dropped[pair]
            mark = "GOLD" if pair in data["gold"] else "    "
            print(f"     {mark} S{pair[0]} [{name}] {surface!r} :: {text[:90]}")
    print(f"  TOTAL refused {total}, gold among them {gold_total}\n")

    print("=== prompt identity: does the arm differ from the head on this project? ===")
    for project in projects:
        data = load(project, run)
        base, base_candidates = prompts_of("head", data)
        line = f"  {project:<14} head: {len(base_candidates):3d} cases, {len(base)} calls"
        for arm in ("noanchor", "norefusal", "norefusal_split"):
            other, other_candidates = prompts_of(arm, data)
            if other == base:
                verdict = "IDENTICAL"
            elif other[:len(base)] == base:
                verdict = f"head's {len(base)} calls verbatim +{len(other) - len(base)}"
            else:
                verdict = "differs"
            line += (f" | {arm}: {len(other_candidates):3d} cases, {len(other)} calls, "
                     f"{verdict}")
        print(line)
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=["head", "noanchor", "norefusal"])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--datasets", nargs="+", default=sorted(DATASETS))
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--dump")
    parser.add_argument("--verify", action="store_true",
                        help="the deterministic half only: no LLM calls")
    args = parser.parse_args()

    if args.verify:
        verify(args.datasets, args.run)
        return 0

    backend = LLMBackend.OPENAI
    model = os.environ.get("OPENAI_MODEL_NAME", "")
    totals = collections.defaultdict(collections.Counter)
    per_row = collections.defaultdict(collections.Counter)
    added_verdicts = collections.defaultdict(collections.Counter)
    dump: dict = {}

    for project in args.datasets:
        data = load(project, args.run)
        gold = data["gold"]
        row_of, candidates = rows_of(data)
        counts = collections.Counter(row_of.values())
        dropped = refused_pairs(data)

        # Which arms are worth calls here: an arm whose prompts are the head's, over
        # the head's candidates, is the head on this project and is skipped rather
        # than paid for. The skip is printed, so a reader sees where the arm is silent.
        base, _ = prompts_of("head", data)
        arms = []
        for arm in args.arms:
            if arm != "head" and prompts_of(arm, data)[0] == base:
                print(f"  ({project}: {arm} is byte-identical to head — skipped)")
                continue
            arms.append(arm)

        print(f"\n=== {project}: {len(data['sentences'])} sentences, "
              f"{len(data['components'])} components, {len(gold)} gold, "
              f"{len(candidates)} candidates "
              f"({', '.join(f'{r} {counts[r]}' for r in ROWS)}), "
              f"refusal drops {len(dropped)} "
              f"({sum(1 for p in dropped if p in gold)} gold) ===", flush=True)

        for sample in range(1, args.samples + 1):
            for arm in arms:
                linker = ARMS[arm](backend=backend, model=model)
                linker.doc_knowledge = data["knowledge"]
                links, feedback = linker._run_name_linker(
                    data["sentences"], data["components"],
                    data["name_to_id"], data["sent_map"])
                kept = {(l.sentence_number, l.component_id) for l in links}
                good = kept & gold
                proposed = len(feedback["candidates"])
                totals[arm]["kept"] += len(kept)
                totals[arm]["gold"] += len(good)
                totals[arm]["spurious"] += len(kept) - len(good)
                totals[arm]["candidates"] += proposed
                totals[arm]["calls"] += len(linker._llm_calls)
                for pair in kept:
                    row = row_of.get(pair, "refused-by-head")
                    per_row[(arm, row)]["kept"] += 1
                    per_row[(arm, row)]["gold" if pair in gold else "spurious"] += 1
                for row in ROWS:
                    per_row[(arm, row)]["cases"] += counts[row]
                    per_row[(arm, row)]["case_gold"] += sum(
                        1 for pair, value in row_of.items()
                        if value == row and pair in gold)
                if arm.startswith("norefusal"):
                    for pair in dropped:
                        added_verdicts[project][
                            "approved" if pair in kept else "rejected"] += 1
                dump.setdefault(f"sample{sample}", {}).setdefault(project, {})[arm] = \
                    sorted([list(p) for p in kept])
                print(f"  {arm:<10} sample {sample}: {len(kept):4d} kept, "
                      f"{len(good):4d} gold, {len(kept) - len(good):4d} spurious "
                      f"(of {proposed} candidates, {len(linker._llm_calls)} calls)",
                      flush=True)

    runs = args.samples
    print(f"\n{model or 'unset'}, {runs} samples, per run over "
          f"{len(args.datasets)} projects:")
    print(f"  {'arm':<10}{'candidates':>11}{'kept':>8}{'gold':>8}{'spurious':>10}"
          f"{'precision':>11}{'calls':>7}")
    for arm in args.arms:
        row = totals[arm]
        if not row["kept"] and not row["calls"]:
            continue
        precision = row["gold"] / row["kept"] if row["kept"] else 0.0
        print(f"  {arm:<10}{row['candidates'] / runs:>11.1f}{row['kept'] / runs:>8.1f}"
              f"{row['gold'] / runs:>8.1f}{row['spurious'] / runs:>10.1f}"
              f"{precision:>11.3f}{row['calls'] / runs:>7.1f}")

    print("\nper naming row (cases are fixed; kept/gold/spurious are per run):")
    header = f"  {'row':<12}{'cases':>7}{'gold in':>9}"
    print(header + "".join(f"{arm[:8]:>10}{'gold':>7}{'sp':>6}" for arm in args.arms))
    for row in ROWS:
        first = per_row[(args.arms[0], row)]
        line = f"  {row:<12}{first['cases'] / runs:>7.0f}{first['case_gold'] / runs:>9.0f}"
        for arm in args.arms:
            bucket = per_row[(arm, row)]
            line += (f"{bucket['kept'] / runs:>10.1f}{bucket['gold'] / runs:>7.1f}"
                     f"{bucket['spurious'] / runs:>6.1f}")
        print(line)

    if added_verdicts:
        print("\nthe pairs the refusal drops, once the judge is allowed to see them:")
        for project, counter in sorted(added_verdicts.items()):
            seen = counter["approved"] + counter["rejected"]
            print(f"  {project:<14} {counter['approved']} approved, "
                  f"{counter['rejected']} rejected, of {seen} case-samples "
                  f"({counter['approved'] / runs:.1f} spurious a run)")

    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump, "w") as handle:
            json.dump(dump, handle)
        print("\nkept pairs written to", args.dump)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
