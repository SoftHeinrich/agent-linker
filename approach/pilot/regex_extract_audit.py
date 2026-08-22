"""Level-1 audit: could a regex scan stand in for the head's LLM extraction pass?

`ENTITY_EXTRACTION_RULES` asks the extractor for exactly one thing -- the sentences
that *write* a component's name, "spelled as the COMPONENTS list spells it or as one
of the KNOWN ALIASES", counting "different spacing, hyphenation or compound joining"
as the same name. That contract is a regex, and it is one this branch already states:
the whole-name row of the surface-realization relation (`s_linker76.NameForm`), at
one of three fidelities.

So the question is not whether a scan *can* express the contract but whether the
model, asked for it, proposes something a scan does not. That is answerable off the
recorded runs, with no call spent: replay the scan over the same documents, the same
catalogs and the same discovered aliases, and compare the candidate sets pair by pair.

  1  the candidate sets   what each scan reaches, against the LLM extraction of that
                          same run, and against the gold standard.
  2  the composition      what the run's *final* link set would have been with the
                          scan in front of the recorded judge. A pair both proposed
                          is given the verdict the run recorded; a pair only the scan
                          proposes was never put to that gate, so the arm is reported
                          as a bracket -- every such pair rejected, and every one
                          approved -- rather than as a point estimate.

No LLM calls. The LLM side is read from `../results/*/phase_states/<variant>/openai`.

    ../.venv/bin/python pilot/regex_extract_audit.py
    ../.venv/bin/python pilot/regex_extract_audit.py --json ../results/regex_round/audit.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import statistics
import sys
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import BENCH, PROJECTS, load_gold                  # noqa: E402
from llm_sad_sam.core.document_loader_v2 import load_sentences       # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository           # noqa: E402

RESULTS = Path("../results")

#: The extraction pass under test. s90, s91 and s92 inherit s89's verbatim, and all
#: four write their phase states under this name, so every recorded directory of the
#: family is a sample of the same extractor.
VARIANT = "s_linker89"


# ── the relation: whole-name extent, three fidelities ────────────────────────

@lru_cache(maxsize=None)
def signature(expression: str) -> tuple:
    """An expression's word sequence, CamelCase split, separators dropped.

    A spaced form, a hyphenated form and a run-together form of the same words share
    a signature, which is what makes a spelling variant recognizable.
    """
    normalized = unicodedata.normalize("NFKC", expression)
    normalized = normalized.replace("-", " ").replace("_", " ")
    return tuple(
        token.casefold()
        for token in re.findall(
            r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[A-Z]+|\d+", normalized
        )
    )


def spans(text: str, name: str, form: str):
    """Spans of ``text`` that write ``name`` whole, at the given fidelity.

    `any_case` and `any_spelling` do not nest -- compound splitting reaches a
    spaced writing of a run-together name, and case folding reaches a run-together
    writing whose case boundaries fall elsewhere -- so `a|b` is their union.
    """
    if "|" in form:
        found = []
        for part in form.split("|"):
            found.extend(spans(text, name, part))
        return found
    if form == "as_spelled":
        return [(m.start(), m.end()) for m in re.finditer(
            rf"(?<!\w){re.escape(name)}(?!\w)", text)]
    if form == "any_case":
        return [(m.start(), m.end()) for m in re.finditer(
            rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE)]
    if form == "any_spelling":
        target = signature(name)
        if not target:
            return []
        words = list(re.finditer(r"[A-Za-z0-9]+", text))
        found = []
        for i, first in enumerate(words):
            for j in range(i, min(len(words), i + len(target))):
                if j > i and not re.fullmatch(
                        r"[\s_-]+", text[words[j - 1].end():words[j].start()]):
                    break
                start, end = first.start(), words[j].end()
                if signature(text[start:end]) == target:
                    found.append((start, end))
        return found
    raise ValueError(form)


def in_dotted_path(text, start, end) -> bool:
    """True when text[start:end] is glued to a dot on either side, as in x.y."""
    before = start > 1 and text[start - 1] == "." and text[start - 2].isalnum()
    after = end + 1 < len(text) and text[end] == "." and text[end + 1].isalnum()
    return before or after


_CACHE: dict = {}


def regex_keys(project, sentences, components, aliases, form, *,
               use_aliases, skip_dotted):
    """(sentence, component) pairs whose sentence writes a name of the component.

    Memoized on the alias table because runs repeat it. ``aliases`` is the run's own
    discovered term -> component map: the knowledge stage is not what this question is
    about, so its output is an input here.
    """
    key = (project, form, use_aliases, skip_dotted,
           tuple(sorted(aliases.items())) if use_aliases else ())
    if key in _CACHE:
        return _CACHE[key]
    by_component = defaultdict(list)
    for term, component in (aliases or {}).items():
        by_component[component].append(term)
    keys = set()
    for sentence in sentences:
        text = sentence.text
        for component in components:
            names = [component.name]
            if use_aliases:
                names += by_component.get(component.name, [])
            for name in names:
                hit = False
                for start, end in spans(text, name, form):
                    if skip_dotted and in_dotted_path(text, start, end):
                        continue
                    hit = True
                    break
                if hit:
                    keys.add((sentence.number, component.id))
                    break
    _CACHE[key] = keys
    return keys


# ── the recorded LLM side ────────────────────────────────────────────────────

def recorded_runs(variant):
    """Every recorded run directory of ``variant`` that holds all five projects."""
    for base in sorted(RESULTS.glob(f"*/phase_states/{variant}/openai")):
        if all((base / project / "linker_full_name.pkl").exists()
               and (base / project / "final.pkl").exists()
               and (base / project / "knowledge.pkl").exists()
               for project in PROJECTS):
            yield base


def llm_side(base, project, name_to_id):
    """One project-run: what the extractor proposed, what the gate said, what linked."""
    directory = base / project
    with open(directory / "linker_full_name.pkl", "rb") as handle:
        stage = pickle.load(handle)
    with open(directory / "knowledge.pkl", "rb") as handle:
        aliases = dict(pickle.load(handle)["doc_knowledge"].aliases)
    with open(directory / "final.pkl", "rb") as handle:
        final = pickle.load(handle)["final"]
    verdicts = {
        (decision["sentence"], decision["component_id"]): bool(decision.get("approved"))
        for decision in stage["feedback"].get("judge_decisions", [])
    }
    return {
        "candidates": {(item["sentence"], name_to_id[item["component"]])
                       for item in stage["feedback"]["candidates"]
                       if item["component"] in name_to_id},
        "accepted": {(link.sentence_number, link.component_id)
                     for link in stage["links"]},
        "final": {(link.sentence_number, link.component_id) for link in final},
        "aliases": aliases,
        "verdicts": verdicts,
    }


# ── the arms ─────────────────────────────────────────────────────────────────

ARMS = [
    ("as_spelled",     dict(form="as_spelled",   use_aliases=True,  skip_dotted=False)),
    ("any_case",       dict(form="any_case",     use_aliases=True,  skip_dotted=False)),
    ("any_case+d",     dict(form="any_case",     use_aliases=True,  skip_dotted=True)),
    ("any_spelling",   dict(form="any_spelling", use_aliases=True,  skip_dotted=False)),
    ("any_spell+d",    dict(form="any_spelling", use_aliases=True,  skip_dotted=True)),
    ("both",           dict(form="any_case|any_spelling",
                            use_aliases=True,  skip_dotted=False)),
    ("both+d",         dict(form="any_case|any_spelling",
                            use_aliases=True,  skip_dotted=True)),
    ("any_case/noal",  dict(form="any_case",     use_aliases=False, skip_dotted=False)),
    ("any_spell/noal", dict(form="any_spelling", use_aliases=False, skip_dotted=False)),
]

#: What a replayed gate is assumed to do with a pair the recorded run never showed
#: it. The two policies bracket the arm; nothing in between is claimed.
POLICIES = ("reject", "approve")


def compose(side, keys, policy):
    """The final link set this arm would produce in front of the recorded judge.

    Only the full-name stage changes, so the other two linkers' links carry over
    untouched. A pair the extractor also proposed is given the verdict the run
    recorded for it; a pair only the scan proposes was never put to this gate.
    """
    admitted = {pair for pair in keys & side["candidates"]
                if side["verdicts"].get(pair, False)}
    if policy == "approve":
        admitted |= keys - side["candidates"]
    return (side["final"] - side["accepted"]) | admitted


# ── scoring, as `pilot/score_runs.py` defines it ─────────────────────────────

def scores(links, gold_by_project):
    """TP, FP, macro F1, macro F2 of one run's link set, keyed (project, snum, cid)."""
    gold = set().union(*gold_by_project.values())
    f1s, f2s = [], []
    for project, project_gold in gold_by_project.items():
        got = {k for k in links if k[0] == project}
        hit = len(got & project_gold)
        precision = hit / len(got) if got else 0.0
        recall = hit / len(project_gold) if project_gold else 0.0
        f1s.append(0.0 if not (precision + recall) else
                   2 * precision * recall / (precision + recall))
        f2s.append(0.0 if not (4 * precision + recall) else
                   5 * precision * recall / (4 * precision + recall))
    return {"TP": len(links & gold), "FP": len(links - gold),
            "macro F1": 100 * sum(f1s) / len(f1s),
            "macro F2": 100 * sum(f2s) / len(f2s)}


def model_of(base: Path) -> str:
    name = base.parts[2]
    for model in ("terra", "luna"):
        if model in name:
            return model
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    parser.add_argument("--variant", default=VARIANT)
    args = parser.parse_args()

    projects = {}
    gold_by_project = {}
    for name in PROJECTS:
        text, model, _ = PROJECTS[name]
        projects[name] = {
            "sentences": load_sentences(str(BENCH / text)),
            "components": parse_pcm_repository(str(BENCH / model)),
            "name_to_id": {c.name: c.id
                           for c in parse_pcm_repository(str(BENCH / model))},
            "gold": load_gold(name),
        }
        gold_by_project[name] = {(name, snum, cid) for snum, cid in projects[name]["gold"]}

    rows = defaultdict(list)          # arm -> one dict a run
    composed = defaultdict(list)      # (arm, policy) -> one score dict a run
    models = []
    runs = list(recorded_runs(args.variant))

    for base in runs:
        models.append(model_of(base))
        totals = defaultdict(lambda: defaultdict(int))
        arm_links = {(arm, policy): set() for arm, _ in ARMS for policy in POLICIES}
        control = set()
        for name, project in projects.items():
            gold = project["gold"]
            side = llm_side(base, name, project["name_to_id"])
            llm, final = side["candidates"], side["final"]
            control |= {(name, snum, cid) for snum, cid in final}
            totals["llm"]["pairs"] += len(llm)
            totals["llm"]["gold"] += len(llm & gold)
            totals["accepted"]["pairs"] += len(side["accepted"])
            totals["accepted"]["gold"] += len(side["accepted"] & gold)
            totals["final"]["pairs"] += len(final)
            totals["final"]["gold"] += len(final & gold)
            for arm, options in ARMS:
                keys = regex_keys(name, project["sentences"], project["components"],
                                  side["aliases"], **options)
                totals[arm]["pairs"] += len(keys)
                totals[arm]["gold"] += len(keys & gold)
                totals[arm]["missed_gold"] += len((llm & gold) - keys)
                totals[arm]["added_gold"] += len((keys & gold) - llm)
                totals[arm]["added_pairs"] += len(keys - llm)
                totals[arm]["dropped_pairs"] += len(llm - keys)
                totals[arm]["new_gold"] += len((keys & gold) - final)
                totals[arm]["at_risk_gold"] += len((llm & gold & final) - keys)
                for policy in POLICIES:
                    arm_links[(arm, policy)] |= {
                        (name, snum, cid)
                        for snum, cid in compose(side, keys, policy)}
        for arm, values in totals.items():
            rows[arm].append(dict(values))
        composed[("control", "-")].append(scores(control, gold_by_project))
        for key, links in arm_links.items():
            composed[key].append(scores(links, gold_by_project))

    def mean(values):
        return statistics.mean(values) if values else 0.0

    def column(arm, field):
        return [row.get(field, 0) for row in rows[arm]]

    gold_total = mean(column("gold", "pairs")) or sum(
        len(g) for g in gold_by_project.values())
    print(f"{len(runs)} recorded runs of {args.variant} "
          f"({models.count('terra')} terra, {models.count('luna')} luna) "
          f"x 5 projects; gold {gold_total:.0f} pairs a run\n")

    print("1. CANDIDATE SETS -- what each proposer hands the full-name gate\n")
    header = (f"{'proposer':<16}{'pairs':>8}{'gold':>8}{'prec':>7}"
              f"{'+pairs':>8}{'-pairs':>8}{'+gold':>7}{'-gold':>7}"
              f"{'newgold':>9}{'atrisk':>8}")
    print(header)
    print("-" * len(header))
    for arm in ["llm", "accepted", "final"] + [a for a, _ in ARMS]:
        pairs, gold = mean(column(arm, "pairs")), mean(column(arm, "gold"))
        print(f"{arm:<16}{pairs:>8.1f}{gold:>8.1f}{gold / pairs if pairs else 0:>7.3f}"
              f"{mean(column(arm, 'added_pairs')):>8.1f}"
              f"{mean(column(arm, 'dropped_pairs')):>8.1f}"
              f"{mean(column(arm, 'added_gold')):>7.1f}"
              f"{mean(column(arm, 'missed_gold')):>7.1f}"
              f"{mean(column(arm, 'new_gold')):>9.1f}"
              f"{mean(column(arm, 'at_risk_gold')):>8.1f}")
    print("\n  +/- columns are against that run's own LLM extraction.")
    print("  newgold: gold this scan proposes that the run's FINAL link set misses.")
    print("  atrisk:  gold the run linked, the extractor proposed, this scan does not.")

    print("\n\n2. COMPOSITION -- the same run's final link set, gate replayed\n")
    header = (f"{'arm':<16}{'policy':<9}{'TP':>7}{'FP':>7}"
              f"{'macro F1':>10}{'macro F2':>10}")
    print(header)
    print("-" * len(header))
    base_row = composed[("control", "-")]
    print(f"{'control':<16}{'-':<9}"
          f"{mean([r['TP'] for r in base_row]):>7.1f}"
          f"{mean([r['FP'] for r in base_row]):>7.1f}"
          f"{mean([r['macro F1'] for r in base_row]):>10.2f}"
          f"{mean([r['macro F2'] for r in base_row]):>10.2f}")
    for arm, _ in ARMS:
        for policy in POLICIES:
            run_rows = composed[(arm, policy)]
            print(f"{arm:<16}{policy:<9}"
                  f"{mean([r['TP'] for r in run_rows]):>7.1f}"
                  f"{mean([r['FP'] for r in run_rows]):>7.1f}"
                  f"{mean([r['macro F1'] for r in run_rows]):>10.2f}"
                  f"{mean([r['macro F2'] for r in run_rows]):>10.2f}")
    print("\n  reject/approve bracket the arm: what the recorded gate would have to do\n"
          "  with the pairs it was never shown. The true value is inside the bracket;\n"
          "  which point it sits at is a level-2 question and needs the gate run.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "variant": args.variant,
            "runs": [str(r) for r in runs],
            "models": models,
            "candidate_sets": {arm: rows[arm] for arm in rows},
            "composition": {f"{arm}|{policy}": values
                            for (arm, policy), values in composed.items()},
        }, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
