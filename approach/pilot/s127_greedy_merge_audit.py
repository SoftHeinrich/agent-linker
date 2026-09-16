#!/usr/bin/env python3
"""Fixed-input audit for greedy name spans and merged ambiguous candidates.

The proposed representation has no ``competitors`` field:

* a partial-name candidate is refused when every occurrence of its matched word is
  contained in a longer whole component name written by the sentence;
* remaining candidates with the same sentence and matched surface form one judge case
  whose value is the set of component names.

This audit replays the deterministic transformation over every s122 checkpoint from
the recorded no-anchor round.  It spends no LLM calls: aliases, candidate judgments,
and gold links are fixed by those checkpoints.  Run from ``approach/``::

    ../.venv/bin/python pilot/s127_greedy_merge_audit.py
"""
from __future__ import annotations

import collections
import contextlib
import csv
import glob
import io
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APPROACH = os.path.dirname(HERE)
ROOT = os.path.dirname(APPROACH)
sys.path.insert(0, os.path.join(APPROACH, "src"))

from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences  # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository                     # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker122 import NameForm              # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker123 import SLinker123             # noqa: E402

BENCH = os.environ.get("ALINKER_BENCHMARK", os.path.join(ROOT, "benchmark"))
RUNS = os.environ.get(
    "ALINKER_S122_RUNS", os.path.join(ROOT, "results", "noanchor_e2e_*_20260914"))
PROJECTS = {
    "mediastore": ("mediastore/text_2016/mediastore.txt",
                   "mediastore/model_2016/pcm/ms.repository",
                   "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv"),
    "teastore": ("teastore/text_2020/teastore.txt",
                 "teastore/model_2020/pcm/teastore.repository",
                 "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv"),
    "teammates": ("teammates/text_2021/teammates.txt",
                  "teammates/model_2021/pcm/teammates.repository",
                  "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "bigbluebutton": ("bigbluebutton/text_2021/bigbluebutton.txt",
                      "bigbluebutton/model_2021/pcm/bbb.repository",
                      "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "jabref": ("jabref/text_2021/jabref.txt",
               "jabref/model_2021/pcm/jabref.repository",
               "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
}


def arm(knowledge):
    obj = SLinker123.__new__(SLinker123)
    obj.doc_knowledge = knowledge
    return obj


def load(project):
    doc, model, gold = PROJECTS[project]
    sentences = load_sentences(os.path.join(BENCH, doc))
    components = parse_pcm_repository(os.path.join(BENCH, model))
    gold_pairs = set()
    with open(os.path.join(BENCH, gold), newline="") as handle:
        for row in csv.DictReader(handle):
            gold_pairs.add((int(row["sentence"]), row["modelElementID"]))
    return sentences, components, build_sent_map(sentences), gold_pairs


def covering_spans(linker, text, component_name, components):
    spans = []
    for other in components:
        if other.name != component_name:
            spans.extend(linker._name_spans(text, other.name, NameForm.ANY_CASE))
    return spans


def greedily_covered(linker, candidate, components):
    """Whether all word matches are owned by a longer written component name."""
    mine = linker._name_spans(
        candidate.sentence_text, candidate.component_name, NameForm.ANY_WORD)
    covers = covering_spans(
        linker, candidate.sentence_text, candidate.component_name, components)
    return bool(mine and covers) and all(
        any(start <= a and b <= end and (end - start) > (b - a)
            for start, end in covers)
        for a, b in mine)


def main():
    totals = collections.Counter()
    by_project = collections.defaultdict(collections.Counter)
    examples = collections.Counter()
    ambiguous_verdicts = collections.Counter()
    run_deltas = []
    runs = sorted(glob.glob(RUNS))
    for run in runs:
        for project in PROJECTS:
            state_dir = os.path.join(
                run, "phase_states", "s_linker122", "openai", project)
            knowledge_path = os.path.join(state_dir, "knowledge.pkl")
            name_path = os.path.join(state_dir, "linker_name.pkl")
            if not os.path.exists(knowledge_path) or not os.path.exists(name_path):
                continue
            with open(knowledge_path, "rb") as handle:
                linker = arm(pickle.load(handle)["doc_knowledge"])
            with open(name_path, "rb") as handle:
                feedback = pickle.load(handle)["feedback"]
            sentences, components, sent_map, gold = load(project)
            name_to_id = {c.name: c.id for c in components}
            with contextlib.redirect_stdout(io.StringIO()):
                candidates = linker._name_candidates(
                    sentences, components, name_to_id, sent_map)
            decisions = {
                (d["sentence"], d["component_id"]): d
                for d in feedback["judge_decisions"]
            }
            kept = []
            accepted_before = {
                key for key, decision in decisions.items() if decision.get("approved")}
            for candidate in candidates:
                key = (candidate.sentence_number, candidate.component_id)
                if (candidate.source != "full_name"
                        and greedily_covered(linker, candidate, components)):
                    totals["greedy_removed"] += 1
                    by_project[project]["greedy_removed"] += 1
                    totals["removed_gold"] += key in gold
                    totals["removed_approved"] += bool(
                        decisions.get(key, {}).get("approved"))
                    examples[(project, candidate.sentence_number,
                              candidate.matched_text, candidate.component_name)] += 1
                else:
                    kept.append(candidate)

            groups = collections.defaultdict(list)
            for candidate in kept:
                surface = (candidate.matched_text or candidate.component_name).casefold()
                groups[(candidate.sentence_number, surface)].append(candidate)
            ambiguous = [group for group in groups.values() if len(group) > 1]
            kept_keys = {(c.sentence_number, c.component_id) for c in kept}
            accepted_after = accepted_before & kept_keys
            before_tp = len(accepted_before & gold)
            before_fp = len(accepted_before - gold)
            after_tp = len(accepted_after & gold)
            after_fp = len(accepted_after - gold)
            run_deltas.append((os.path.basename(run), project,
                               before_tp, before_fp, after_tp, after_fp))
            totals["project_runs"] += 1
            totals["candidates_before"] += len(candidates)
            totals["candidates_after"] += len(kept)
            totals["merged_cases"] += len(groups)
            totals["ambiguous_cases"] += len(ambiguous)
            by_project[project]["ambiguous_cases"] += len(ambiguous)
            for group in ambiguous:
                sentence = group[0].sentence_number
                selected = tuple(sorted(
                    c.component_name for c in group
                    if decisions.get((c.sentence_number, c.component_id), {}).get(
                        "approved")))
                ambiguous_verdicts[(project, sentence, selected)] += 1
                if project == "bigbluebutton" and sentence in {19, 27, 31, 60}:
                    names = ", ".join(sorted(c.component_name for c in group))
                    examples[(project, sentence, group[0].matched_text, names)] += 1

    print(f"CACHE {len(runs)} recorded runs; {totals['project_runs']} project-runs")
    print(f"candidates {totals['candidates_before']} -> {totals['candidates_after']}")
    print(f"greedy removed {totals['greedy_removed']}: "
          f"gold={totals['removed_gold']}, previously approved={totals['removed_approved']}")
    print(f"merged judge cases {totals['merged_cases']}; "
          f"genuinely ambiguous={totals['ambiguous_cases']}")
    print("cached accepted-link projection (greedy filter only):")
    changed = [row for row in run_deltas if (row[2], row[3]) != (row[4], row[5])]
    for run, project, btp, bfp, atp, afp in changed:
        print(f"  {run}/{project}: TP {btp}->{atp}, FP {bfp}->{afp}")
    if not changed:
        print("  no accepted links changed")
    print("by project:")
    for project in PROJECTS:
        print(f"  {project}: greedy_removed={by_project[project]['greedy_removed']}, "
              f"ambiguous_cases={by_project[project]['ambiguous_cases']}")
    print("impacted/relevant BBB examples (occurrences over cached runs):")
    for key, count in sorted(examples.items()):
        print(f"  {count:2d}x {key[0]} S{key[1]} {key[2]!r} -> {key[3]}")
    print("cached verdict combinations for merged cases:")
    for (project, sentence, selected), count in sorted(ambiguous_verdicts.items()):
        value = ", ".join(selected) if selected else "none"
        print(f"  {count:2d}x {project} S{sentence}: selected={value}")
    return 0 if totals["removed_gold"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
