"""Why the partial-name proposer never offers 22.8 gold pairs per run. No LLM calls.

`pilot/partial_audit.py` shows the denotation judge is not the bottleneck (95% recall,
83% precision over the candidates it sees, headroom +1.0 TP / -3.5 FP). The bottleneck
is upstream: 41.5 gold pairs are still open when the partial-name linker runs and only
18.7 are proposed.

`_name_word_candidates` can decline a pair for exactly four reasons, and they are all
deterministic, so each open-gold pair is attributable without any LLM call:

  states_a_name   the sentence states a whole name of the component (or a discovered
                  alias of it), so the pair belongs to the full-name linker -- which
                  either never proposed it or had it rejected;
  no_hook         no word of the sentence begins with any word of the component's
                  name, so there is no partial name to find (coreference territory);
  ambiguous       a hook exists but the word prefix-matches more than one component,
                  and the proposer only offers uniquely-owned words;
  qualified       the only hook sits inside a dotted/qualified identifier.

Each is then cross-referenced with the final link set: a pair the coreference linker
recovers is not a loss, it is a division of labour.

    ../.venv/bin/python pilot/partial_gap.py
    ../.venv/bin/python pilot/partial_gap.py --arm s_linker59 --detail
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import BENCH, PROJECTS, load_gold                  # noqa: E402
from llm_sad_sam.core.document_loader_v2 import load_sentences       # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository           # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker59 import SLinker59    # noqa: E402

WORD = r"[A-Za-z]+[A-Za-z0-9]*|\d+"


class Probe(SLinker59):
    """The real predicates, with no LLM client and an injected alias table."""

    def __init__(self, aliases):                    # noqa: D107 - no super().__init__
        self.doc_knowledge = type("K", (), {"aliases": aliases})()


def project_data(project):
    text, model, _ = PROJECTS[project]
    return (load_sentences(str(BENCH / text)),
            parse_pcm_repository(str(BENCH / model)))


def reason(probe, sentence, component, components):
    """Why `_name_word_candidates` does not offer (sentence, component)."""
    if probe._states_a_name(sentence.text, component.name):
        return "states_a_name", ""
    words = {w.casefold() for w in re.findall(WORD, component.name)}
    hooks, qualified, ambiguous = [], [], []
    for match in re.finditer(WORD, sentence.text):
        surface = match.group(0).casefold()
        if not any(surface.startswith(w) for w in words):
            continue
        if probe._inside_qualified_identifier(sentence.text, match.start(),
                                             match.end()):
            qualified.append(match.group(0))
            continue
        owners = [
            c for c in components
            if any(surface.startswith(w)
                   for w in {x.casefold() for x in re.findall(WORD, c.name)})
        ]
        if len(owners) != 1:
            ambiguous.append(f"{match.group(0)}~{'/'.join(c.name for c in owners)}")
            continue
        hooks.append(match.group(0))
    if hooks:
        return "proposed_elsewhere", "/".join(hooks)
    if ambiguous:
        return "ambiguous", "; ".join(ambiguous)
    if qualified:
        return "qualified", "/".join(qualified)
    return "no_hook", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="../results/s5960_e2e_r*_20260813")
    ap.add_argument("--arm", default="s_linker49")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    runs = sorted(Path().glob(args.runs))
    tally = Counter()
    saved = Counter()
    detail = defaultdict(Counter)
    for run in runs:
        for project in PROJECTS:
            base = run / "phase_states" / args.arm / "openai" / project
            if not (base / "linker_partial_name.pkl").exists():
                continue
            sentences, components = project_data(project)
            by_number = {s.number: s for s in sentences}
            by_id = {c.id: c for c in components}
            knowledge = pickle.load((base / "knowledge.pkl").open("rb"))
            aliases = getattr(knowledge.get("doc_knowledge"), "aliases", {}) or {}
            probe = Probe(aliases)
            state = pickle.load((base / "linker_partial_name.pkl").open("rb"))
            earlier = {(l.sentence_number, l.component_id)
                       for l in pickle.load(
                           (base / "linker_full_name.pkl").open("rb"))["links"]}
            proposed = {(int(r["sentence"]), r["component"])
                        for r in state["feedback"]["proposed"]}
            final = {(l.sentence_number, l.component_id)
                     for l in pickle.load((base / "final.pkl").open("rb"))["final"]}
            for snum, cid in load_gold(project) - earlier:
                component = by_id.get(cid)
                sentence = by_number.get(snum)
                if component is None or sentence is None:
                    continue
                if (snum, component.name) in proposed:
                    continue
                why, evidence = reason(probe, sentence, component, components)
                recovered = (snum, cid) in final
                tally[why] += 1
                saved[(why, recovered)] += 1
                if args.detail:
                    detail[why][f"{project} s{snum} {component.name}"
                                f"{' [recovered]' if recovered else ''}"
                                f"{'  ' + evidence if evidence else ''}"] += 1

    n = len(runs)
    print(f"\n{args.arm}, gold open at the partial-name stage and not proposed, "
          f"per run over {n} runs\n")
    print(f"{'reason':<20}{'per run':>10}{'recovered later':>18}{'lost':>8}")
    for why, count in tally.most_common():
        rec = saved[(why, True)] / n
        print(f"{why:<20}{count / n:>10.1f}{rec:>18.1f}{count / n - rec:>8.1f}")
    total = sum(tally.values()) / n
    rec = sum(v for (_, r), v in saved.items() if r) / n
    print(f"{'TOTAL':<20}{total:>10.1f}{rec:>18.1f}{total - rec:>8.1f}")

    if args.detail:
        for why in tally:
            print(f"\n{why}")
            for label, count in detail[why].most_common():
                print(f"    {count}x  {label}")


if __name__ == "__main__":
    main()
