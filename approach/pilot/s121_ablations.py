"""Level-2 stage pilot for two `s_linker121` ablations: the anchors block, and the scan's
one refusal. Both arms and the head run in the same invocation, on the same pinned alias
table, so nothing is compared across invocation sets.

    arm `head`        `s_linker121` as it stands
    arm `noanchor`    the case carries no `anchors` block and the rule carries no
                      `anchors` line (the line is *sliced* off `_FIELD_LINES`, not
                      retyped, so a drift in the constant is a drift in the arm)
    arm `refusal`     `s_linker109`'s nesting refusal put back: a one-word pair whose
                      word is written only inside another component's whole name is
                      dropped. It ran as `norefusal`/`norefusal_split` against a head
                      that still had it; the head lost it on this round's result, so the
                      arm is now stated in the direction that changes something

The two ablations are not symmetric and the pilot does not pretend they are:

  * `noanchor` changes what the judge is shown for **every** candidate on every project.
  * `refusal` changes the **candidate set**, and only where the predicate fires. On four
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
from llm_sad_sam.linkers.experimental.s_linker121 import (  # noqa: E402
    NameForm, SLinker121, UNION_DEMAND,
)
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


def nests_inside_another_name(linker, text, name, components) -> bool:
    """`s_linker109`'s nesting refusal, as it stood in `s_linker121` before this round.

    True when EVERY writing of ``name``'s word lies inside a span where ``text`` writes
    some other component's whole name. It lives here, in the pilot, because the head no
    longer carries it: an arm that prices a removed predicate has to own it, or the
    round stops being reproducible the moment the head moves.

    Catalog names only, never discovered aliases — the asymmetry the predicate was
    designed with, since this is the one thing that would END a case rather than open
    one (the alias form cost 3 gold links in one recorded run).
    """
    mine = linker._name_spans(text, name, NameForm.ANY_WORD)
    if not mine:
        return False
    covering = []
    for component in components:
        if component.name != name:
            covering.extend(
                linker._name_spans(text, component.name, NameForm.ANY_CASE))
    if not covering:
        return False
    return all(any(start <= a and b <= end for start, end in covering)
               for a, b in mine)


class Refusal(SLinker121):
    """The head with the nesting refusal put back, for the arm that prices it."""

    def _scan(self, sentences, components):
        return [candidate for candidate in super()._scan(sentences, components)
                if not nests_inside_another_name(
                    self, candidate.sentence_text, candidate.component_name,
                    components)]


class RefusalSplit(Refusal):
    """`refusal`, with the head's batches held byte-identical.

    The plain `refusal` arm changes two things at once: the judge sees 12 fewer cases,
    and that changes where the batch boundaries fall (bigbluebutton goes 5 calls to 4).
    The union round measured the batch boundary as an effect in its own right, so an arm
    that moves both cannot say which one it measured.

    This arm judges the head's full candidate set in the head's own batches — the same
    call bytes — and discards the refused pairs' verdicts afterwards. Whatever it
    differs from the head by is the refusal and nothing else.
    """

    def _scan(self, sentences, components):
        return SLinker121._scan(self, sentences, components)

    def _judge_union(self, candidates, components, sentences, sent_map):
        approved, decisions = super()._judge_union(
            candidates, components, sentences, sent_map)
        return ([c for c in approved
                 if not nests_inside_another_name(
                     self, c.sentence_text, c.component_name, components)],
                decisions)


#: The clause `noanchor_clause` puts in place of the anchor block — the FIRST attempt,
#: kept as an arm because it is what the tight clause is measured against. 456 B.
#:
#: It is too long and it says three things where the mechanism is one. Its enumeration
#: of readings (a platform, a technology, a broader product, an ordinary English sense)
#: is the shape s71/s72 priced at ~0.8 F1 and the general round caught twice; its last
#: sentence — "Approve only when this sentence uses that surface for this component" —
#: restates `STRICTER_CLAUSE`'s own last sentence, and a restatement at the lenient gate
#: is the branch's standing redundancy (s86, s87, three instances).
ALIAS_NOT_AUTHORITY = (
    "Where this sentence does not write the component's name in full, the surface it "
    "does write is what the case reports of it, not something the document has "
    "certified: the same letters can belong to a platform, a technology, a broader "
    "product or an ordinary English sense as easily as to the component named here, "
    "and a short form established elsewhere is not established in this sentence. "
    "Approve only when this sentence uses that surface for this component."
)

#: The same weighing with everything that is not the mechanism removed. 99 B.
#:
#: What the anchors were doing, in one sentence of the error analysis: they let the judge
#: tell "this surface names this component in this document" from "this surface could".
#: With the block gone the judge cannot check the first, so the only honest thing to say
#: is that the second is not the first. Nothing else in the round's data is load-bearing.
#:
#: It is NOT a restatement of `STRICTER_CLAUSE`, which is about an ordinary English word
#: coinciding with a name. An established short form is not an ordinary English word, and
#: it is the row the error analysis found leaking (22 of 30 spurious on terra, 18 of them
#: one component). This generalizes that clause's principle to the surface the alias
#: stage supplied, which is the one surface no rule in the module speaks about.
SURFACE_NOT_EVIDENCE = (
    "That a surface can name this component is not evidence that it does here."
)

#: The line `anchor_count` puts in place of the rule's anchors line. Same field, reduced
#: to its cardinality, so the arm asks whether what restrains the judge is the anchors'
#: CONTENT or merely the news that the document names this component elsewhere.
ANCHOR_COUNT_LINE = (
    "\n  anchors -- how many other sentences of this document name this component. "
    "They fix that the name is the document's; they do not decide this sentence."
)


class _ClausedNoAnchors(NoAnchors):
    """`noanchor`, plus one clause in place of the block. Subclasses set `CLAUSE`.

    The arm family that answers "is the prompt not enough?" If a clause recovers what
    the anchor block was buying, the anchors were patching an under-specified rule and
    are removable; if it does not, they are carrying a fact no clause can state.
    """

    CLAUSE = ""

    def _prompt_union(self, comp_names, sentence_table, cases) -> str:
        prompt = super()._prompt_union(comp_names, sentence_table, cases)
        placed = prompt.replace(
            UNION_DEMAND, f"{self.CLAUSE}\n\n{UNION_DEMAND}", 1)
        assert placed != prompt, "the clause was not placed"
        return placed


class NoAnchorClause(_ClausedNoAnchors):
    """The first, verbose clause. 456 B."""

    CLAUSE = ALIAS_NOT_AUTHORITY


class NoAnchorTight(_ClausedNoAnchors):
    """The same weighing at 99 B: only what the error analysis showed is load-bearing."""

    CLAUSE = SURFACE_NOT_EVIDENCE


class AnchorCount(SLinker121):
    """The anchors reduced to their count: the fact stays, its content goes.

    Between `head` and `noanchor` this is the arm that separates the two readings of the
    same result — that the judge is restrained by seeing HOW the document writes this
    name, or merely by learning THAT it writes it elsewhere.
    """

    def _format_union_case(self, index, candidate, evidence, sent_map, shown_in=0):
        counted = dict(evidence, anchors=[])
        case = super()._format_union_case(index, candidate, counted, sent_map, 0)
        if evidence["anchors"]:
            case += f"\n  Anchors (other sentences naming it): {len(evidence['anchors'])}"
        return case

    def _prompt_union(self, comp_names, sentence_table, cases) -> str:
        prompt = super()._prompt_union(comp_names, sentence_table, cases)
        counted = prompt.replace(ANCHOR_RULE_LINE, ANCHOR_COUNT_LINE, 1)
        assert counted != prompt, "the anchors line was not replaced"
        return counted


class AnchorTrim(SLinker121):
    """One anchor instead of `ANCHOR_LIMIT`: the content stays, the volume goes.

    `anchor_count` moved the two models in opposite directions and `pilot/anchor_why.py`
    says the cases that flip are the dotted-identifier ones (31% of the changed cases
    against 5.6-9.2% of the agreed ones) and that terra's losses carry the LONGEST
    anchor blocks (538 chars against a 353 population mean). Two readings fit that:
    the judge is using what the anchors SAY, or it is being crowded by how much they
    say. This arm separates them — a judge that only needed one example keeps its
    verdicts, a judge that needed the evidence loses them the way it lost them to the
    count.

    **An arm, not an adoption candidate.** `ANCHOR_LIMIT` and `CONTEXT_SENTENCES` are
    deliberately one value, and a value chosen by search is not defensible on this
    branch; a win here would have to be argued into a unification, not read off.
    """

    ANCHOR_LIMIT = 1


ARMS = {"head": SLinker121, "noanchor": NoAnchors, "refusal": Refusal,
        "refusal_split": RefusalSplit, "noanchor_clause": NoAnchorClause,
        "anchor_count": AnchorCount, "anchor_trim": AnchorTrim,
        "noanchor_tight": NoAnchorTight}


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
    return {pair: everything[pair] for pair in everything
            if nests_inside_another_name(
                linker, everything[pair][2], everything[pair][0], data["components"])}


def verify(projects, run):
    """The zero-call half: what the refusal drops, and which arms differ at all."""
    print("=== the nesting refusal: what it would drop, and is any of it gold ===")
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
        for arm in ("noanchor", "noanchor_clause", "noanchor_tight", "anchor_count",
                    "anchor_trim", "refusal", "refusal_split"):
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
    parser.add_argument("--arms", nargs="+", default=["head", "noanchor", "refusal"])
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
                if arm.startswith("refusal"):
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
        print("\nthe pairs the refusal would drop, as the head's judge answers them:")
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
