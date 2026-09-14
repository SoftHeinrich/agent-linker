"""Level-1 audit: what would annotating the coreference shortlist with the name
linker's links actually reach?

`s_linker122`'s resolver prints, per case, a line the round called the shortlist:

    NAMED BEFORE THIS CASE: Facade (S3), Media Store (S1)

`_named_before` computes it with `_states_a_name` -- a purely LEXICAL fact: which
sentences above this one write a component's name. The name linker has already run by
then and has already judged every one of those mentions, and `_run_linker` withholds
its verdicts on purpose ("No linker receives the links the earlier one produced").

The proposal is to annotate each entry with that verdict, so the resolver can tell an
antecedent the name judge ACCEPTED from one it REJECTED. This file prices it without
spending a call, off recorded runs:

  R1  reach       -- how many shortlist entries would carry a mark, and how many of
                     the entries are ones the name judge rejected (the only entries an
                     annotation can say anything new about)
  R2  ceiling     -- the resolutions the recorded runs produced, split by whether the
                     antecedent they cited was name-approved; gold on each side is the
                     most the annotation could win or lose in each direction
  R3  net         -- the same split over what coreference actually CONTRIBUTES: pairs
                     no name link already carries, since `link` merges by pair
  R4  antecedent  -- the net links whose cited antecedent the name judge rejected,
                     listed, because that is the failure the annotation is meant to
                     prevent and the gold among them is what it would cost

Usage (no LLM calls, no network):

    /path/to/.venv/bin/python pilot/coref_shortlist_audit.py \\
        --runs ../results/noanchor_e2e_terra_r1_20260914 --variant s_linker122
"""
from __future__ import annotations

import argparse
import collections
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker122 import (  # noqa: E402
    SLinker122, get_comp_names,
)
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository  # noqa: E402

from reading_pilots import BENCH, DATASETS, gold_pairs  # noqa: E402


def phase(run: Path, variant: str, project: str, name: str):
    path = run / "phase_states" / variant / "openai" / project / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def probe(knowledge):
    """A head instance with no LLM behind it, for the deterministic questions."""
    linker = SLinker122.__new__(SLinker122)
    linker.doc_knowledge = knowledge
    return linker


def shortlists(linker, sentences, components):
    """Every case's shortlist, exactly as `_prompt_coref` would build it.

    The resolver is driven with `_ask` stubbed, so the batching, the window and the
    sentence table are the variant's own and not a re-declaration of them.
    """
    comp_names = get_comp_names(components)
    out = []

    class _Recorder:
        def set_phase(self, phase_name):
            pass

    linker.llm = _Recorder()
    linker._ask = lambda prompt, **_: {"resolutions": []}
    original = linker._prompt_coref

    def spy(names, sentence_table, targets):
        for target in targets:
            out.append((target["target"],
                        linker._named_before(names, sentence_table, target["target"])))
        return original(names, sentence_table, targets)

    linker._prompt_coref = spy
    linker._resolve_references(
        sentences, components,
        {c.name: c.id for c in components},
        build_sent_map(sentences))
    assert comp_names  # the catalog the spy was called with
    return out


def audit(runs, variant, projects):
    per_project = collections.defaultdict(collections.Counter)
    rejected_antecedents = []

    for run in runs:
        for project in projects:
            text, repo, gold_path = DATASETS[project]
            know = phase(run, variant, project, "knowledge")
            name_state = phase(run, variant, project, "linker_name")
            coref_state = phase(run, variant, project, "linker_coreference")
            if not (know and name_state and coref_state):
                print(f"  (skip {run.name}/{project}: checkpoint missing)")
                continue

            components = parse_pcm_repository(str(BENCH / repo))
            sentences = load_sentences(str(BENCH / text))
            sent_map = build_sent_map(sentences)
            gold = gold_pairs(BENCH / gold_path)
            name_to_id = {c.name: c.id for c in components}
            id_to_name = {c.id: c.name for c in components}

            linker = probe(know["doc_knowledge"])

            # What the name linker decided, both ways round: `links` is what it kept,
            # `candidates` is everything its judge saw, so the difference is what it
            # was shown and refused -- the only entries an annotation can inform.
            accepted = {(l.sentence_number, l.component_id) for l in name_state["links"]}
            judged = set()
            for row in name_state["feedback"].get("candidates", []):
                cid = name_to_id.get(row.get("component"))
                if cid is not None and isinstance(row.get("sentence"), int):
                    judged.add((row["sentence"], cid))
            rejected = judged - accepted

            # R1 -- the shortlist population.
            for _target, near in shortlists(linker, sentences, components):
                per_project[project]["cases"] += 1
                for name, snum in near:
                    pair = (snum, name_to_id.get(name))
                    per_project[project]["entries"] += 1
                    if pair in accepted:
                        per_project[project]["entry_accepted"] += 1
                    elif pair in rejected:
                        per_project[project]["entry_rejected"] += 1
                    else:
                        per_project[project]["entry_unjudged"] += 1

            # R2/R3/R4 -- the resolutions the run actually made.
            meta = {(m["sentence"], m["component_id"]): m
                    for m in coref_state["feedback"].get("metadata", [])}
            approved = {(l.sentence_number, l.component_id)
                        for l in coref_state["links"]}

            for pair in meta:
                snum, cid = pair
                ant = meta[pair].get("antecedent_sentence")
                ant_pair = (ant, cid)
                mark = ("acc" if ant_pair in accepted
                        else "rej" if ant_pair in rejected else "unj")
                is_gold = pair in gold
                kept = pair in approved
                new = pair not in accepted          # coreference's own contribution
                for scope, on in (("prop", True), ("kept", kept),
                                  ("netprop", new), ("net", kept and new)):
                    if not on:
                        continue
                    per_project[project][f"{scope}_{mark}"] += 1
                    if is_gold:
                        per_project[project][f"{scope}_{mark}_gold"] += 1
                if kept and new and mark == "rej":
                    rejected_antecedents.append(
                        (project, snum, id_to_name.get(cid, cid), ant,
                         "GOLD" if is_gold else "    ",
                         meta[pair].get("reference", ""),
                         sent_map[snum].text[:80] if snum in sent_map else ""))

    totals = collections.Counter()
    for project in projects:
        totals.update(per_project[project])
    return per_project, totals, rejected_antecedents


def report(per_project, totals, rejected_antecedents, n):
    print(f"\n=== R1: the shortlist population (per run, over {n} recorded runs) ===")
    print(f"  {'project':<15}{'cases':>7}{'entries':>9}{'/case':>7}"
          f"{'accepted':>10}{'rejected':>10}{'unjudged':>10}")
    for project in sorted(per_project):
        c = per_project[project]
        print(f"  {project:<15}{c['cases'] / n:>7.0f}{c['entries'] / n:>9.1f}"
              f"{(c['entries'] / max(c['cases'], 1)):>7.1f}"
              f"{c['entry_accepted'] / n:>10.1f}{c['entry_rejected'] / n:>10.1f}"
              f"{c['entry_unjudged'] / n:>10.1f}")
    print(f"  {'TOTAL':<15}{totals['cases'] / n:>7.0f}{totals['entries'] / n:>9.1f}"
          f"{(totals['entries'] / max(totals['cases'], 1)):>7.1f}"
          f"{totals['entry_accepted'] / n:>10.1f}{totals['entry_rejected'] / n:>10.1f}"
          f"{totals['entry_unjudged'] / n:>10.1f}")

    for scope, label in (("prop", "R2a: every resolution the resolver proposed"),
                         ("kept", "R2b: the resolutions the coreference judge kept"),
                         ("netprop", "R3a: proposed pairs no name link already carries"),
                         ("net", "R3b: coreference's NET contribution (kept and new)")):
        print(f"\n=== {label} — split by the cited antecedent's name verdict "
              f"(per run) ===")
        print(f"  {'antecedent':<26}{'pairs':>8}{'gold':>8}{'spurious':>10}"
              f"{'precision':>11}")
        for mark, name in (("acc", "name-ACCEPTED"),
                           ("rej", "name-REJECTED"),
                           ("unj", "never judged (no scan)")):
            pairs = totals[f"{scope}_{mark}"]
            good = totals[f"{scope}_{mark}_gold"]
            if not pairs:
                continue
            print(f"  {name:<26}{pairs / n:>8.1f}{good / n:>8.1f}"
                  f"{(pairs - good) / n:>10.1f}{good / pairs:>11.3f}")
        allp = sum(totals[f"{scope}_{m}"] for m in ("acc", "rej", "unj"))
        allg = sum(totals[f"{scope}_{m}_gold"] for m in ("acc", "rej", "unj"))
        print(f"  {'ALL':<26}{allp / n:>8.1f}{allg / n:>8.1f}{(allp - allg) / n:>10.1f}"
              f"{(allg / allp if allp else 0):>11.3f}")

    print(f"\n=== R4: the net links whose antecedent the name judge REJECTED "
          f"({len(rejected_antecedents)} over {n} runs) ===")
    for project, snum, comp, ant, mark, ref, text in sorted(rejected_antecedents):
        print(f"  {mark} {project:<14} S{snum} -> {comp} (ant S{ant}) "
              f"ref={ref!r:<28} {text!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--variant", default="s_linker122")
    parser.add_argument("--datasets", nargs="+", default=sorted(DATASETS))
    args = parser.parse_args()

    per_project, totals, rejected = audit(args.runs, args.variant, args.datasets)
    report(per_project, totals, rejected, len(args.runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
