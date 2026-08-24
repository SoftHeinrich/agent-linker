"""Level-1 audit for the consolidation: four questions, no LLM calls.

The regex round put a scan in front of the full-name gate and the reading round put
a third blind proposer in front of all three. Both were measured against their own
base and neither was measured against the other. Composing them is only worth
building if the parts are not paying for the same gold twice, and that is decidable
off the recorded runs.

  Q1  Is the third look redundant once the named proposer is a scan?
      `s_linker101`'s gold over `s_linker90`, and how much of it the scan proposes.
  Q2  What does narrowing the resolver to no-name sentences cost?
      Coref links the scan already proposes AND whose full-name verdict was reject:
      exactly the pairs that stop existing.
  Q3  What does that narrowing save? Sentences that write no name of any component.
  Q4  Is the residual error sibling-shaped, and are the siblings enumerable in code?
      The partial-name gate's own decisions, split by whether the sentence carries a
      link to another component whose name shares a word with the one under test.

    ../.venv/bin/python pilot/consolidation_audit.py
    ../.venv/bin/python pilot/consolidation_audit.py --json ../results/consolidation_audit.json
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import BENCH, PROJECTS, load_gold                  # noqa: E402
from regex_extract_audit import regex_keys, signature, spans         # noqa: E402
from llm_sad_sam.core.document_loader_v2 import load_sentences       # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository           # noqa: E402

RESULTS = Path("../results")

#: The scan's fidelity, as `s_linker92a` runs it.
SCAN = dict(form="any_case", use_aliases=True, skip_dotted=False)

#: Recorded runs of the scan arm — the only source of a *scan-side* alias table.
SCAN_VARIANT = "s_linker92a"


def load_projects():
    projects = {}
    for name, (text, model, _) in PROJECTS.items():
        components = parse_pcm_repository(str(BENCH / model))
        projects[name] = {
            "sentences": load_sentences(str(BENCH / text)),
            "components": components,
            "name_to_id": {c.name: c.id for c in components},
            "id_to_name": {c.id: c.name for c in components},
            "gold": load_gold(name),
        }
    return projects


def scan_runs():
    """Recorded scan runs that hold all five projects, newest naming first."""
    for base in sorted(RESULTS.glob(f"*/phase_states/{SCAN_VARIANT}/openai")):
        if all((base / p / "linker_full_name.pkl").exists() for p in PROJECTS):
            yield base


def model_of(path: Path) -> str:
    text = str(path)
    return "terra" if "terra" in text else "luna" if "luna" in text else "other"


def aliases_of(base: Path, project: str) -> dict:
    with open(base / project / "knowledge.pkl", "rb") as handle:
        return dict(pickle.load(handle)["doc_knowledge"].aliases)


def stage_of(base: Path, project: str, stage: str) -> dict:
    with open(base / project / f"{stage}.pkl", "rb") as handle:
        return pickle.load(handle)


# ── Q1: is the third look redundant in front of a scan? ──────────────────────

def links_from_csv(path: Path) -> set:
    with open(path) as handle:
        return {(int(row["sentence"]), row["component_id"])
                for row in csv.DictReader(handle)}


def e2e_pairs():
    """(label, model, {project: {variant: links}}) for every recorded E2E set."""
    for directory in sorted(RESULTS.glob("*_e2e_*")):
        by_project = defaultdict(dict)
        for path in directory.glob("*_links.csv"):
            stem = path.stem[: -len("_links")]
            for project in PROJECTS:
                if stem.endswith(f"_{project}"):
                    by_project[project][stem[: -len(project) - 1]] = links_from_csv(path)
        if by_project:
            yield directory.name, model_of(directory), dict(by_project)


def question_1(projects, alias_tables):
    """What s101 adds over s90 in gold, and how much of it a scan already proposes."""
    rows = []
    for label, model, by_project in e2e_pairs():
        variants = set().union(*(set(v) for v in by_project.values()))
        if not {"s_linker90", "s_linker101"} <= variants:
            continue
        added_gold = defaultdict(set)
        for project, links in by_project.items():
            gold = projects[project]["gold"]
            added_gold[project] = gold & (links["s_linker101"] - links["s_linker90"])
        reach = {"canonical": 0, "with aliases": 0}
        total = sum(len(v) for v in added_gold.values())
        for project, pairs in added_gold.items():
            data = projects[project]
            base = regex_keys(project, data["sentences"], data["components"], {},
                              use_aliases=False, **{k: v for k, v in SCAN.items()
                                                    if k != "use_aliases"})
            reach["canonical"] += len(pairs & base)
            covered = set()
            for table in alias_tables[project]:
                covered |= pairs & regex_keys(
                    project, data["sentences"], data["components"], table, **SCAN)
            reach["with aliases"] += len(covered)
        rows.append({"set": label, "model": model, "added gold": total,
                     "scan reaches (canonical)": reach["canonical"],
                     "scan reaches (any recorded table)": reach["with aliases"]})
    return rows


# ── Q2/Q3: narrowing the resolver to sentences that write no name ────────────

def question_23(projects, runs):
    """Per run: what the narrowing removes, what it costs, what it saves.

    The filter is **per sentence** -- ask the resolver only about sentences that write
    no name of *any* component -- which is `s_linker93`'s, and the reading round
    refused it for a defect this measures directly: a sentence that names X and refers
    back to Y loses the Y case. A dropped case is only a loss when nothing else
    proposes that pair, so each dropped link is scored against the scan and the
    recorded full-name verdict that would then decide it.
    """
    rows = []
    for base in runs:
        row = {"run": base.parts[2], "model": model_of(base),
               "coref links": 0, "dropped": 0, "rescued": 0,
               "lost": 0, "lost gold": 0, "lost gold @other": 0,
               "sentences": 0, "narrowed": 0, "cases": 0, "narrowed cases": 0}
        for project, data in projects.items():
            gold = data["gold"]
            table = aliases_of(base, project)
            scan = regex_keys(project, data["sentences"], data["components"],
                              table, **SCAN)
            named_sentences = {snum for snum, _ in scan}
            coref = stage_of(base, project, "linker_coreference")
            full = stage_of(base, project, "linker_full_name")
            verdicts = {(d["sentence"], d["component_id"]): bool(d.get("approved"))
                        for d in full["feedback"].get("judge_decisions", [])}
            for link in coref["links"]:
                pair = (link.sentence_number, link.component_id)
                row["coref links"] += 1
                if link.sentence_number not in named_sentences:
                    continue                      # the narrowed resolver still sees it
                row["dropped"] += 1
                if pair in scan and verdicts.get(pair, True):
                    row["rescued"] += 1           # the scan proposes it; the gate keeps it
                    continue
                row["lost"] += 1
                if pair in gold:
                    row["lost gold"] += 1
                    if pair not in scan:
                        row["lost gold @other"] += 1   # the s93 defect, exactly
            row["sentences"] += len(data["sentences"])
            row["narrowed"] += sum(1 for s in data["sentences"]
                                   if s.number not in named_sentences)
            # cases the resolver is actually asked about: every sentence, batched
            row["cases"] += len(data["sentences"])
            row["narrowed cases"] += sum(1 for s in data["sentences"]
                                         if s.number not in named_sentences)
        rows.append(row)
    return rows


# ── Q4: is the residual error sibling-shaped? ────────────────────────────────

def siblings_of(name, all_names):
    """Catalog names sharing a signature word with ``name`` — a fact, not a list."""
    words = set(signature(name))
    return sorted(other for other in all_names
                  if other != name and words & set(signature(other)))


def question_4(projects, runs):
    """Partial-name decisions split by whether a code-enumerable sibling exists.

    A sibling group is a fact about the runtime catalog: names sharing a signature
    word. The question a discriminator would be asked is a choice inside one group,
    so each error is also labelled by whether the group is *contested in that
    sentence* -- another member of the same group is gold there. A contested error is
    one a chooser could fix; an uncontested one needs a "none of these" answer, which
    is the denotation judge's own question and not a new one.
    """
    rows = []
    for base in runs:
        row = {"run": base.parts[2], "model": model_of(base),
               "FP": 0, "FP sibling": 0, "FP contested": 0,
               "FN": 0, "FN sibling": 0, "FN contested": 0,
               "groups": 0, "cases a run": 0}
        for project, data in projects.items():
            gold = data["gold"]
            names = list(data["name_to_id"])
            sib = {n: siblings_of(n, names) for n in names}
            ids = data["name_to_id"]
            gold_names_by_sentence = {}
            for snum, cid in gold:
                gold_names_by_sentence.setdefault(snum, set()).add(
                    data["id_to_name"].get(cid, ""))
            stage = stage_of(base, project, "linker_partial_name")
            final = {(l.sentence_number, l.component_id)
                     for l in stage_of(base, project, "final")["final"]}
            groups = {frozenset([n, *sib[n]]) for n in names if sib[n]}
            row["groups"] += len(groups)
            contested_cases = set()
            for decision in stage["feedback"].get("judge_decisions", []):
                pair = (decision["sentence"], decision["component_id"])
                name = data["id_to_name"].get(decision["component_id"], "")
                family = set(sib.get(name, ()))
                here = gold_names_by_sentence.get(decision["sentence"], set())
                contested = bool(family & here)
                if contested:
                    contested_cases.add((decision["sentence"], frozenset(family | {name})))
                if decision.get("approved") and pair not in gold:
                    row["FP"] += 1
                    row["FP sibling"] += bool(family)
                    row["FP contested"] += contested
                if not decision.get("approved") and pair in gold and pair not in final:
                    row["FN"] += 1
                    row["FN sibling"] += bool(family)
                    row["FN contested"] += contested
            row["cases a run"] += len(contested_cases)
        rows.append(row)
    return rows



# ── Q5: is the sibling confusion one question asked twice? ───────────────────

def question_5(projects, runs):
    """Partial-name decisions grouped by (sentence, quoted claim).

    The denotation judge is target-blind by design: its case carries the expression
    and the sentence, never the component. When one expression reaches several
    components, the judge is therefore shown the *same case twice* and must answer it
    the same way both times -- so every member of the group is approved or none is.
    That is not a judging failure, it is a question the judge was never asked. This
    counts how often it happens and what the group contains.
    """
    rows = []
    for base in runs:
        row = {"run": base.parts[2], "model": model_of(base),
               "groups": 0, "shared groups": 0, "approved together": 0,
               "one gold in group": 0, "no gold in group": 0,
               "FP inside shared groups": 0, "TP inside shared groups": 0}
        for project, data in projects.items():
            gold = data["gold"]
            stage = stage_of(base, project, "linker_partial_name")
            by_case = {}
            for decision in stage["feedback"].get("judge_decisions", []):
                key = (decision["sentence"], decision.get("claim", ""))
                by_case.setdefault(key, []).append(decision)
            for key, members in by_case.items():
                row["groups"] += 1
                if len(members) < 2:
                    continue
                row["shared groups"] += 1
                approved = [m for m in members if m.get("approved")]
                if len(approved) == len(members):
                    row["approved together"] += 1
                hits = [m for m in approved
                        if (m["sentence"], m["component_id"]) in gold]
                row["one gold in group"] += len(hits) == 1
                row["no gold in group"] += len(hits) == 0
                row["TP inside shared groups"] += len(hits)
                row["FP inside shared groups"] += len(approved) - len(hits)
        rows.append(row)
    return rows


def mean(rows, key):
    return statistics.mean(r[key] for r in rows) if rows else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    projects = load_projects()
    runs = list(scan_runs())
    alias_tables = defaultdict(list)
    for base in runs:
        for project in PROJECTS:
            alias_tables[project].append(aliases_of(base, project))

    print(f"{len(runs)} recorded scan runs: "
          f"{', '.join(sorted({model_of(b) for b in runs}))}\n")

    print("Q1  what the third look adds, and whether the scan already reaches it")
    q1 = question_1(projects, alias_tables)
    if not q1:
        print("    no recorded set holds both s_linker90 and s_linker101\n")
    for row in q1:
        print(f"    {row['set']:<28} {row['model']:<6} added gold {row['added gold']:>3}"
              f" | scan reaches {row['scan reaches (canonical)']:>3} canonical,"
              f" {row['scan reaches (any recorded table)']:>3} with a recorded table")
    print()

    print("Q2/Q3  narrowing the resolver to sentences that write no name (s93's filter)")
    q23 = question_23(projects, runs)
    for row in q23:
        print(f"    {row['run']:<34} coref {row['coref links']:>3}"
              f" | dropped {row['dropped']:>3} = rescued {row['rescued']:>3}"
              f" + lost {row['lost']:>2} ({row['lost gold']} gold,"
              f" {row['lost gold @other']} of them on a sentence naming only another)"
              f" | sentences {row['sentences']} -> {row['narrowed']}")
    if q23:
        print(f"    mean a run: lost {mean(q23, 'lost'):.1f} pairs, "
              f"{mean(q23, 'lost gold'):.1f} gold "
              f"({mean(q23, 'lost gold @other'):.1f} to the s93 defect); "
              f"sentences {mean(q23, 'sentences'):.0f} -> {mean(q23, 'narrowed'):.0f} "
              f"({100 * (1 - mean(q23, 'narrowed') / mean(q23, 'sentences')):.0f}% fewer)")
    print()

    print("Q4  the partial-name gate's errors, split by code-enumerable siblings")
    q4 = question_4(projects, runs)
    for row in q4:
        print(f"    {row['run']:<34} FP {row['FP']:>3}"
              f" ({row['FP sibling']:>3} sibling, {row['FP contested']:>3} contested)"
              f" | FN {row['FN']:>2}"
              f" ({row['FN sibling']:>2} sibling, {row['FN contested']:>2} contested)"
              f" | {row['cases a run']:>2} contested groups")
    if q4:
        print(f"    mean a run: FP {mean(q4, 'FP'):.1f}"
              f" -> {mean(q4, 'FP sibling'):.1f} have a sibling"
              f" -> {mean(q4, 'FP contested'):.1f} sit in a group another member owns;"
              f" FN {mean(q4, 'FN'):.1f} -> {mean(q4, 'FN sibling'):.1f}"
              f" -> {mean(q4, 'FN contested'):.1f}."
              f" A chooser would be asked {mean(q4, 'cases a run'):.1f} questions a run"
              f" over {mean(q4, 'groups'):.0f} groups.")

    print()
    print("Q5  one expression, several components: the case the judge is shown twice")
    q5 = question_5(projects, runs)
    for row in q5:
        print(f"    {row['run']:<34} shared groups {row['shared groups']:>3}"
              f" of {row['groups']:>3} | approved as a block {row['approved together']:>3}"
              f" | inside them TP {row['TP inside shared groups']:>2}"
              f" FP {row['FP inside shared groups']:>3}"
              f" | exactly one gold {row['one gold in group']:>2},"
              f" none {row['no gold in group']:>3}")
    if q5:
        print(f"    mean a run: {mean(q5, 'shared groups'):.1f} shared groups,"
              f" {mean(q5, 'approved together'):.1f} approved as a block;"
              f" they hold {mean(q5, 'TP inside shared groups'):.1f} TP and"
              f" {mean(q5, 'FP inside shared groups'):.1f} FP."
              f" {mean(q5, 'one gold in group'):.1f} groups have exactly one right"
              f" answer, {mean(q5, 'no gold in group'):.1f} have none.")

    if args.json:
        args.json.write_text(json.dumps(
            {"q1": q1, "q23": q23, "q4": q4, "q5": q5}, indent=2))
        print(f"\nwritten to {args.json}")


if __name__ == "__main__":
    main()
