"""The two deterministic holes in the partial-name proposer, sized. No LLM calls.

`pilot/partial_gap.py` splits the 22.8 open-gold pairs the proposer never offers into
15.0 with no lexical hook at all (every one of which the coreference linker recovers --
correct division of labour), 5.8 excluded because the sentence states a whole name, and
2.0 excluded as ambiguously owned. The last two are lost outright: 7.7 gold pairs per
run that no stage of the workflow ever sees again.

This sizes both, and each has a candidate repair that needs no LLM call to evaluate:

  HOLE 1  the whole-name exclusion assumes the full-name linker handled the pair. It
          did not always: the exclusion fires on `_states_a_name`, but the full-name
          linker only links what its *extractor proposed* and its *judges approved*.
          Split the 5.8 into (a) the extractor proposed it and a judge rejected it --
          re-proposing there second-guesses a judge that saw more -- and (b) the
          extractor never proposed it at all, which is an extraction miss with no
          safety net.

  HOLE 2  owner uniqueness uses a bare prefix test in both directions, so a sentence
          word matches a name word whenever it *starts with* it. `WebRTC` therefore
          matches both `WebRTC-SFU` (exactly) and `BBB web` (as a prefix of `web`),
          the pair is dropped as ambiguous, and two gold links go with it. Repair:
          when at least one component matches a word exactly, prefix-only matches
          lose. Scored here over every sentence of all five documents: gold gained,
          spurious gained, and candidate-set growth.

    ../.venv/bin/python pilot/partial_hole.py
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import BENCH, PROJECTS, load_gold                  # noqa: E402
from llm_sad_sam.core.document_loader_v2 import load_sentences       # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository           # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker59 import SLinker59    # noqa: E402

WORD = r"[A-Za-z]+[A-Za-z0-9]*|\d+"


class Probe(SLinker59):
    def __init__(self, aliases):                    # noqa: D107
        self.doc_knowledge = type("K", (), {"aliases": aliases})()

    def _name_word_candidates_exact_wins(self, sentences, components):
        """`_name_word_candidates` with one change: an exact word match outranks a
        prefix match when deciding whether the owner is unique."""
        words = {c.id: {w.casefold() for w in re.findall(WORD, c.name)}
                 for c in components}
        out = {}
        for sentence in sentences:
            for match in re.finditer(WORD, sentence.text):
                if self._inside_qualified_identifier(sentence.text, match.start(),
                                                    match.end()):
                    continue
                surface = match.group(0).casefold()
                exact = [c for c in components if surface in words[c.id]]
                prefix = [c for c in components
                          if any(surface.startswith(w) for w in words[c.id])]
                owners = exact or prefix
                if len(owners) != 1:
                    continue
                component = owners[0]
                if self._states_a_name(sentence.text, component.name):
                    continue
                out[(sentence.number, component.id)] = match.group(0)
        return out

    def _name_word_candidates_keys(self, sentences, components, considered=None,
                                   exact_wins=False):
        """`_name_word_candidates` as a {(sentence, component_id): matched_text} map.

        `considered` switches the whole-name exclusion from "the sentence states a
        name" to "the sentence states a name **and** the full-name stage ruled on
        this pair", so an extraction miss falls through to this linker instead of
        being dropped by both.
        """
        words = {c.id: {w.casefold() for w in re.findall(WORD, c.name)}
                 for c in components}
        out = {}
        for sentence in sentences:
            for match in re.finditer(WORD, sentence.text):
                if self._inside_qualified_identifier(sentence.text, match.start(),
                                                    match.end()):
                    continue
                surface = match.group(0).casefold()
                if exact_wins:
                    exact = [c for c in components if surface in words[c.id]]
                    prefix = [c for c in components
                              if any(surface.startswith(w) for w in words[c.id])]
                    owners = exact or prefix
                else:
                    owners = [c for c in components
                              if any(surface.startswith(w) for w in words[c.id])]
                if len(owners) != 1:
                    continue
                component = owners[0]
                if self._states_a_name(sentence.text, component.name) and (
                        considered is None
                        or (sentence.number, component.id) in considered):
                    continue
                out[(sentence.number, component.id)] = match.group(0)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="../results/s5960_e2e_r*_20260813")
    ap.add_argument("--arm", default="s_linker49")
    args = ap.parse_args()
    runs = sorted(Path().glob(args.runs))

    print("\nHOLE 1 -- gold excluded by the whole-name test, split by what the "
          "full-name stage did with it\n")
    split = Counter()
    detail = Counter()
    for run in runs:
        for project in PROJECTS:
            base = run / "phase_states" / args.arm / "openai" / project
            if not (base / "linker_partial_name.pkl").exists():
                continue
            sentences, components = project_cache(project)
            by_id = {c.id: c for c in components}
            by_num = {s.number: s for s in sentences}
            knowledge = pickle.load((base / "knowledge.pkl").open("rb"))
            probe = Probe(getattr(knowledge.get("doc_knowledge"), "aliases", {}) or {})
            fn = pickle.load((base / "linker_full_name.pkl").open("rb"))
            earlier = {(l.sentence_number, l.component_id) for l in fn["links"]}
            fn_proposed = {(int(r["sentence"]), r["component"])
                           for r in fn["feedback"]["candidates"]}
            final = {(l.sentence_number, l.component_id)
                     for l in pickle.load((base / "final.pkl").open("rb"))["final"]}
            pn = pickle.load((base / "linker_partial_name.pkl").open("rb"))
            pn_proposed = {(int(r["sentence"]), r["component"])
                           for r in pn["feedback"]["proposed"]}
            for snum, cid in load_gold(project) - earlier:
                component, sentence = by_id.get(cid), by_num.get(snum)
                if component is None or sentence is None:
                    continue
                if (snum, component.name) in pn_proposed:
                    continue
                if not probe._states_a_name(sentence.text, component.name):
                    continue
                if (snum, cid) in final:
                    split["recovered by coreference"] += 1
                    continue
                where = ("full-name judge rejected it"
                         if (snum, component.name) in fn_proposed
                         else "the extractor never proposed it")
                split[where] += 1
                detail[f"{where:<32} {project} s{snum} {component.name}"] += 1
    n = len(runs)
    for label, count in split.most_common():
        print(f"    {count / n:>5.1f}/run  {label}")
    for label, count in detail.most_common():
        print(f"        {count}x  {label}")

    print("\nHOLE 2 -- owner uniqueness: exact word match outranks prefix match\n")
    print(f"{'project':<16}{'candidates':>12}{'gold':>8}"
          f"{'candidates':>14}{'gold':>8}{'new gold':>10}{'new spurious':>14}")
    print(f"{'':<16}{'(current)':>12}{'':>8}{'(exact wins)':>14}")
    totals = Counter()
    for project in PROJECTS:
        sentences, components = project_cache(project)
        aliases = alias_union(runs, args.arm, project)
        probe = Probe(aliases)
        base = probe._name_word_candidates_keys(sentences, components)
        new = probe._name_word_candidates_exact_wins(sentences, components)
        gold = set(load_gold(project))
        gained = set(new) - set(base)
        lost = set(base) - set(new)
        print(f"{project:<16}{len(base):>12}{len(set(base) & gold):>8}"
              f"{len(new):>14}{len(set(new) & gold):>8}"
              f"{len(gained & gold):>10}{len(gained - gold):>14}")
        if gained:
            for key in sorted(gained):
                mark = "GOLD" if key in gold else "    "
                name = next(c.name for c in components if c.id == key[1])
                print(f"      + {mark}  s{key[0]:<5} {name:<18} '{new[key]}'")
        if lost:
            for key in sorted(lost):
                mark = "GOLD" if key in gold else "    "
                name = next(c.name for c in components if c.id == key[1])
                print(f"      - {mark}  s{key[0]:<5} {name:<18} '{base[key]}'")
        totals["base"] += len(base)
        totals["base_gold"] += len(set(base) & gold)
        totals["new"] += len(new)
        totals["new_gold"] += len(set(new) & gold)
    print(f"{'TOTAL':<16}{totals['base']:>12}{totals['base_gold']:>8}"
          f"{totals['new']:>14}{totals['new_gold']:>8}")

    print("\nHOLE 1 repair -- exclude only pairs the full-name stage actually ruled "
          "on\n")
    print(f"{'project':<16}{'candidates':>12}{'gold':>8}{'candidates':>14}"
          f"{'gold':>8}{'new gold':>10}{'new spurious':>14}")
    agg = Counter()
    seen = Counter()
    for run in runs:
        for project in PROJECTS:
            base_dir = run / "phase_states" / args.arm / "openai" / project
            if not (base_dir / "linker_partial_name.pkl").exists():
                continue
            sentences, components = project_cache(project)
            by_name = {c.name: c.id for c in components}
            knowledge = pickle.load((base_dir / "knowledge.pkl").open("rb"))
            probe = Probe(getattr(knowledge.get("doc_knowledge"), "aliases", {}) or {})
            fn = pickle.load((base_dir / "linker_full_name.pkl").open("rb"))
            linked = {(l.sentence_number, l.component_id) for l in fn["links"]}
            considered = {(int(r["sentence"]), by_name[r["component"]])
                          for r in fn["feedback"]["candidates"]
                          if r["component"] in by_name} | linked
            gold = set(load_gold(project))
            base = probe._name_word_candidates_keys(sentences, components)
            new = probe._name_word_candidates_keys(sentences, components,
                                                   considered=considered)
            base = {k: v for k, v in base.items() if k not in linked}
            new = {k: v for k, v in new.items() if k not in linked}
            gained = set(new) - set(base)
            agg[(project, "base")] += len(base)
            agg[(project, "base_gold")] += len(set(base) & gold)
            agg[(project, "new")] += len(new)
            agg[(project, "new_gold")] += len(set(new) & gold)
            agg[(project, "runs")] += 1
            for key in gained:
                name = next(c.name for c in components if c.id == key[1])
                seen[(project, key in gold, key[0], name, new[key])] += 1
    for project in PROJECTS:
        n = agg[(project, "runs")] or 1
        if not agg[(project, "new")]:
            continue
        print(f"{project:<16}{agg[(project, 'base')] / n:>12.1f}"
              f"{agg[(project, 'base_gold')] / n:>8.1f}"
              f"{agg[(project, 'new')] / n:>14.1f}"
              f"{agg[(project, 'new_gold')] / n:>8.1f}"
              f"{(agg[(project, 'new_gold')] - agg[(project, 'base_gold')]) / n:>10.1f}"
              f"{(agg[(project, 'new')] - agg[(project, 'base')] - agg[(project, 'new_gold')] + agg[(project, 'base_gold')]) / n:>14.1f}")
    print()
    for (project, is_gold, snum, name, text), count in seen.most_common():
        print(f"    {count}x  {'GOLD' if is_gold else '    '}  {project:<14} "
              f"s{snum:<5} {name:<18} '{text}'")


_CACHE = {}


def project_cache(project):
    if project not in _CACHE:
        text, model, _ = PROJECTS[project]
        _CACHE[project] = (load_sentences(str(BENCH / text)),
                           parse_pcm_repository(str(BENCH / model)))
    return _CACHE[project]


def alias_union(runs, arm, project):
    """Terms every run agreed on, so the deterministic screen does not depend on
    one run's alias table."""
    tables = []
    for run in runs:
        path = run / "phase_states" / arm / "openai" / project / "knowledge.pkl"
        if path.exists():
            table = getattr(pickle.load(path.open("rb")).get("doc_knowledge"),
                            "aliases", {}) or {}
            tables.append(table)
    if not tables:
        return {}
    common = set(tables[0])
    for table in tables[1:]:
        common &= set(table)
    return {t: tables[0][t] for t in common}


if __name__ == "__main__":
    main()
