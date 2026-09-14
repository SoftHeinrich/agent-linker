#!/usr/bin/env python3
"""The merged evidence field, priced at level 1 against the recorded s122 runs.

Three modes, none of which spends an LLM call.

``--verify``
    Rebuild every judging call of every recorded run from the benchmark and that
    run's own recorded alias table, with the head and with `s_linker123`. Asserts
    (1) the head's rebuilt prompts are **byte-identical to the ones actually sent**,
    which is what makes the rest of this audit evidence rather than a simulation;
    (2) the candidate sets of the two arms are equal, so nothing about the scan moves;
    (3) the arms' prompts differ only inside the evidence lines and the rule's field
    block; and (4) `written` -> `naming` reproduces the head's `naming` on every case.

``--cells``
    The four values of `written`, their gold density, and what the head's judge did
    with them, off the recorded `judge_decisions` and the benchmark gold standard.

``--bytes``
    The byte effect, per case, per call and per five-project run.

Reproduce from ``approach/``::

    ../.venv/bin/python pilot/written_field_audit.py --verify
    ../.venv/bin/python pilot/written_field_audit.py --cells --bytes
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import glob
import io
import json
import os
import pickle
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APPROACH = os.path.dirname(HERE)
ROOT = os.path.dirname(APPROACH)
sys.path.insert(0, os.path.join(APPROACH, "src"))
os.environ.setdefault("PHASE_CACHE_DIR", os.path.join(APPROACH, ".written_audit_cache"))

from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences  # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository                     # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker122 as base_mod           # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker122 import SLinker122            # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker123 import NAMING_OF, SLinker123  # noqa: E402

BENCH = os.environ.get("ALINKER_BENCHMARK", os.path.join(ROOT, "benchmark"))
#: The s122 round's own end-to-end sweep. Its runs carry both arms and, per project,
#: the alias table each one actually ran with.
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
JUDGE_PHASE = "phase_25_name_union_judge"
EVIDENCE = "  Evidence: "

#: The clause `s_linker122` carried WHEN THE RECORDED RUNS WERE MADE. The head's own
#: `SURFACE_NOT_EVIDENCE` was re-scoped afterwards ("s122 promoted: standalone, and the
#: clause gets its scope back"), so the head no longer renders the prompts its recorded
#: E2E was measured on. Substituting this back is the ONE allowed difference between a
#: rebuilt prompt and a sent one, and `verify` reports it rather than tolerating it
#: silently: with the substitution the rebuild is byte-identical on every project-run,
#: which is what makes the rest of this audit evidence instead of a simulation.
RECORDED_CLAUSE = (
    "That a surface can name this component is not evidence that it does here.")


class _Phase:
    """The one method `_judge_union` calls on the client before it asks."""

    def set_phase(self, _phase):
        return None


def _arm(cls, knowledge):
    """An arm with no LLM client, ready to render prompts and nothing else."""
    obj = cls.__new__(cls)
    obj.doc_knowledge = knowledge
    obj.llm = _Phase()
    obj._llm_calls = []
    return obj


def _prompts(arm, sentences, components, name_to_id, sent_map):
    """Every judging prompt this arm would send, in the order it would send them."""
    captured = []

    def ask(prompt, **_kwargs):
        captured.append(prompt)
        return {}

    arm._ask = ask
    with contextlib.redirect_stdout(io.StringIO()):
        candidates = arm._name_candidates(sentences, components, name_to_id, sent_map)
        arm._judge_union(candidates, components, sentences, sent_map)
    return candidates, captured


def _cached_prompts(run, project, variant):
    """The judging prompts a recorded run actually sent, in order."""
    out = []
    pattern = os.path.join(run, "llm_logs", f"{variant}_*_{project}_*_calls.json")
    for path in sorted(glob.glob(pattern)):
        for call in json.load(open(path)):
            if call.get("phase") == JUDGE_PHASE:
                out.append(call["prompt"])
    return out


def _evidence_lines(prompt):
    body = prompt.split("CASES:\n", 1)[1].split("\n\nReturn JSON:", 1)[0]
    return [l for l in body.split("\n") if l.startswith(EVIDENCE)]


def _load(project):
    doc, model, _ = PROJECTS[project]
    sentences = load_sentences(os.path.join(BENCH, doc))
    components = parse_pcm_repository(os.path.join(BENCH, model))
    return (sentences, components,
            {c.name: c.id for c in components}, build_sent_map(sentences))


def _knowledge(run, project, variant):
    path = os.path.join(run, "phase_states", variant, "openai", project,
                        "knowledge.pkl")
    if not os.path.exists(path):
        return None
    return pickle.load(open(path, "rb"))["doc_knowledge"]


def verify(runs):
    checks = failures = 0
    faithful = drifted = 0
    equal_candidates = 0
    naming_checked = 0
    ev_only = 0

    def check(condition, label):
        nonlocal checks, failures
        checks += 1
        if not condition:
            failures += 1
            print(f"  FAIL  {label}")

    for run in runs:
        for project in PROJECTS:
            knowledge = _knowledge(run, project, "s_linker122")
            if knowledge is None:
                continue
            loaded = _load(project)
            head = _arm(SLinker122, knowledge)
            arm = _arm(SLinker123, knowledge)
            head_cands, head_prompts = _prompts(head, *loaded)
            arm_cands, arm_prompts = _prompts(arm, *loaded)
            cached = _cached_prompts(run, project, "s_linker122")

            # (1) the rebuild reproduces what was sent, modulo the re-scoped clause
            if cached:
                as_sent = [p.replace(base_mod.SURFACE_NOT_EVIDENCE, RECORDED_CLAUSE)
                           for p in head_prompts]
                same = as_sent == cached
                check(same, f"{os.path.basename(run)}/{project}: rebuilt head prompts "
                            f"!= sent prompts ({len(head_prompts)} vs {len(cached)})")
                faithful += same
                drifted += head_prompts != cached

            # (2) the scan does not move
            key = lambda c: (c.sentence_number, c.component_id, c.matched_text)
            same_cands = sorted(map(key, head_cands)) == sorted(map(key, arm_cands))
            check(same_cands, f"{os.path.basename(run)}/{project}: candidate sets differ")
            equal_candidates += same_cands
            check(len(head_prompts) == len(arm_prompts),
                  f"{os.path.basename(run)}/{project}: call count differs")

            # (3) the prompts differ only where they are meant to
            for before, after in zip(head_prompts, arm_prompts):
                head_body = before.split("CASES:\n", 1)
                arm_body = after.split("CASES:\n", 1)
                head_rule, arm_rule = head_body[0], arm_body[0]
                # outside the rule's field block the preamble is untouched
                cut = head_rule.split("  writes -- ", 1)[0]
                check(arm_rule.startswith(cut),
                      f"{os.path.basename(run)}/{project}: preamble changed")
                tail_head = before.split("\n\nReturn JSON:", 1)[1]
                tail_arm = after.split("\n\nReturn JSON:", 1)[1]
                check(tail_head == tail_arm,
                      f"{os.path.basename(run)}/{project}: reply contract changed")
                head_cases = head_body[1].split("\n\nReturn JSON:", 1)[0].split("\n")
                arm_cases = arm_body[1].split("\n\nReturn JSON:", 1)[0].split("\n")
                check(len(head_cases) == len(arm_cases),
                      f"{os.path.basename(run)}/{project}: case line count differs")
                for a, b in zip(head_cases, arm_cases):
                    if a.startswith(EVIDENCE) or b.startswith(EVIDENCE):
                        ev_only += 1
                    else:
                        check(a == b, f"{os.path.basename(run)}/{project}: a non-"
                                      f"evidence case line changed: {a[:60]!r}")

            # (4) the recorded projection is reproduced
            for candidate in arm_cands:
                ev_head = head._union_evidence(candidate, loaded[1], loaded[3])
                ev_arm = arm._union_evidence(candidate, loaded[1], loaded[3])
                naming_checked += 1
                check(ev_arm["naming"] == ev_head["naming"],
                      f"naming projection differs for {candidate.component_name}")
                check(ev_arm["competitors"] == ev_head["alternatives"],
                      f"competitors != alternatives for {candidate.component_name}")
                check(ev_arm["written"] in SLinker123.WRITTEN,
                      f"unexpected written value {ev_arm['written']!r}")
                check(NAMING_OF[ev_arm["written"]] == ev_head["naming"],
                      "NAMING_OF disagrees with the head")

    print(f"\nVERIFY  {checks - failures}/{checks} checks pass")
    print(f"  head prompts byte-identical to the sent ones, under the recorded "
          f"clause: {faithful} project-runs")
    if drifted:
        print(f"  NOTE the head's clause was re-scoped after these runs, so {drifted} "
              f"project-runs differ from what was sent by exactly that sentence:")
        print(f"       recorded: {RECORDED_CLAUSE}")
        print(f"       head    : {base_mod.SURFACE_NOT_EVIDENCE}")
    print(f"  candidate sets equal: {equal_candidates} project-runs")
    print(f"  evidence lines compared: {ev_only}")
    print(f"  evidence bundles compared: {naming_checked}")
    return failures


def _qualified(text, name):
    occ = list(re.finditer(rf"\b{re.escape(name.lower())}\b", text))
    return bool(occ) and all(
        SLinker122._in_dotted_path(text, m.start(), m.end()) for m in occ)


def cells(runs):
    docs, comps, gold = {}, {}, {}
    for project, (doc, model, gs) in PROJECTS.items():
        docs[project] = {s.number: s.text
                         for s in load_sentences(os.path.join(BENCH, doc))}
        comps[project] = {c.id: c.name
                          for c in parse_pcm_repository(os.path.join(BENCH, model))}
        gold[project] = {(int(r["sentence"]), r["modelElementID"])
                         for r in csv.DictReader(open(os.path.join(BENCH, gs)))}

    seen = collections.Counter()
    appr = collections.Counter()
    isgold = collections.Counter()
    goldappr = collections.Counter()
    by_model = collections.defaultdict(collections.Counter)
    unmapped = 0
    for run in runs:
        for variant in ("s_linker121", "s_linker122"):
            for project in PROJECTS:
                path = os.path.join(run, "phase_states", variant, "openai", project,
                                    "linker_name.pkl")
                if not os.path.exists(path):
                    continue
                state = pickle.load(open(path, "rb"))
                for d in state["feedback"]["judge_decisions"]:
                    name = comps[project].get(d["component_id"])
                    text = docs[project].get(d["sentence"])
                    if not name or text is None:
                        unmapped += 1
                        continue
                    base = {"whole name": "whole name", "alias": "short form",
                            "word only": "one word"}[d["naming"]]
                    if base == "whole name" and _qualified(text, name):
                        base = "whole name (qualified)"
                    g = (d["sentence"], d["component_id"]) in gold[project]
                    seen[base] += 1
                    appr[base] += bool(d["approved"])
                    isgold[base] += g
                    goldappr[base] += g and bool(d["approved"])
                    model = "terra" if "terra" in os.path.basename(run) else "luna"
                    cell = by_model[(model, base)]
                    cell["n"] += 1
                    cell["g"] += g
                    cell["a"] += bool(d["approved"])
                    cell["ga"] += g and bool(d["approved"])

    total = sum(seen.values())
    print(f"\nCELLS  {total} judged cases, {len(runs)} runs x 2 arms x 5 projects, "
          f"{unmapped} unmapped")
    print(f"  {'written=':<24}{'cases':>7}{'share':>8}{'gold':>7}{'gold rate':>11}"
          f"{'keep rate':>11}{'over-keep':>11}{'precision':>11}")
    for value in SLinker123.WRITTEN:
        n = seen[value]
        if not n:
            continue
        g, a, ga = isgold[value], appr[value], goldappr[value]
        print(f"  {value:<24}{n:>7}{100 * n / total:>7.1f}%{g:>7}{g / n:>11.3f}"
              f"{a / n:>11.3f}{(a - g) / n:>+11.3f}{(ga / a if a else 0):>11.3f}")
    g, a, ga = sum(isgold.values()), sum(appr.values()), sum(goldappr.values())
    print(f"  {'TOTAL':<24}{total:>7}{100.0:>7.1f}%{g:>7}{g / total:>11.3f}"
          f"{a / total:>11.3f}{(a - g) / total:>+11.3f}{ga / a:>11.3f}")

    print("\n  by model -- a value's gold rate is a property of the data, but what the "
          "judge\n  did with it is a property of the model, and the two rows that "
          "matter split:")
    print(f"  {'model':<7}{'written=':<24}{'cases':>7}{'gold rate':>11}"
          f"{'keep rate':>11}{'over-keep':>11}{'precision':>11}")
    for model in ("terra", "luna"):
        for value in SLinker123.WRITTEN:
            c = by_model.get((model, value))
            if not c:
                continue
            n, gg, aa, gga = c["n"], c["g"], c["a"], c["ga"]
            print(f"  {model:<7}{value:<24}{n:>7}{gg / n:>11.3f}{aa / n:>11.3f}"
                  f"{(aa - gg) / n:>+11.3f}{(gga / aa if aa else 0):>11.3f}")
    return 0


def byte_effect(runs):
    old = new = 0
    per_call = []
    calls = 0
    for run in runs:
        for project in PROJECTS:
            knowledge = _knowledge(run, project, "s_linker122")
            if knowledge is None:
                continue
            loaded = _load(project)
            _, head_prompts = _prompts(_arm(SLinker122, knowledge), *loaded)
            _, arm_prompts = _prompts(_arm(SLinker123, knowledge), *loaded)
            for before, after in zip(head_prompts, arm_prompts):
                calls += 1
                per_call.append(len(after) - len(before))
                old += sum(map(len, _evidence_lines(before)))
                new += sum(map(len, _evidence_lines(after)))
    print(f"\nBYTES  {calls} judging calls rebuilt over {len(runs)} runs")
    print(f"  evidence lines: {old} -> {new} B "
          f"({100 * (new - old) / old:+.1f}%)")
    print(f"  whole judging call: mean {statistics.fmean(per_call):+.0f} B, "
          f"best {min(per_call):+d}, worst {max(per_call):+d}")
    print(f"  per five-project run: {sum(per_call) / len(runs):+.0f} B")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--cells", action="store_true")
    ap.add_argument("--bytes", dest="bytes_", action="store_true")
    args = ap.parse_args()
    if not (args.verify or args.cells or args.bytes_):
        args.verify = args.cells = args.bytes_ = True

    runs = sorted(glob.glob(RUNS))
    if not runs:
        print(f"no recorded runs at {RUNS}", file=sys.stderr)
        return 2
    print(f"recorded runs: {len(runs)}")

    failures = 0
    if args.verify:
        failures += verify(runs)
    if args.cells:
        cells(runs)
    if args.bytes_:
        byte_effect(runs)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
