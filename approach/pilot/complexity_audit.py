"""Deterministic audit of the hand-coded paths left in the s25 workflow.

No LLM call. Each section answers one question: does this code change a decision,
or does it only change text that a judge then ignores? A path whose delta is zero
on all five projects can be deleted or merged without a pilot.

  T1 TWO NAME PRIMITIVES -- `has_standalone_mention` (case rules for single-word
     names, dot and hyphen boundary rules, 20 lines, still marked
     "#TODO functionality") and `_find_exact_form` (case-insensitive word
     boundary, 4 lines) both answer "does this text state this name". The first
     decides anchors and the coreference antecedent gate; the second decides the
     full-name contract filter and the partial-name suppressor. Two answers to
     one question is the complexity; the question is how often they differ.
  T2 MENTION-TYPE CLASSIFIER -- `MentionType` (5 values) plus
     `_all_occurrences_in_qualified_path` exist to put one substring in a judge
     prompt. If the distribution is degenerate, the classifier is dead weight.
  T3 SPELLING VARIANTS -- ~40 lines of nested scanning. How many links does it
     actually contribute, and are they gold?
  T4 QUALIFIED-IDENTIFIER TEST -- a four-way disjunction. Which disjuncts do any
     work?
  T5 UNREAD FIELDS -- `extraction_rationale` (constant string) and
     `antecedent_via_alias` (model self-report). Confirm nothing reads them.

Usage: ../.venv/bin/python pilot/complexity_audit.py
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import (
    PROJECTS, DEFAULT_RUN, load_project, load_phase, load_gold,
)

from llm_sad_sam.linkers.experimental import s_linker25 as L25
from llm_sad_sam.linkers.experimental.helper_v3 import has_standalone_mention
from llm_sad_sam.linkers.experimental.s_linker25 import SLinker25, MentionType


def _linker(doc_knowledge=None):
    """A linker instance without touching __init__ (which builds an LLM client)."""
    linker = SLinker25.__new__(SLinker25)
    linker.doc_knowledge = doc_knowledge
    return linker


def _aliases(knowledge):
    """Alias table as term -> component name, tolerating pre-scope checkpoints."""
    table = {}
    for term, entry in knowledge["doc_knowledge"].aliases.items():
        table[term] = getattr(entry, "component", entry)
    return table


class _Table:
    def __init__(self, mapping):
        self.aliases = mapping


# ── T1: two primitives for one question ──────────────────────────────────────

def audit_primitives(run):
    print("\n### T1 two name-matching primitives")
    print("     `has_standalone_mention` vs `_find_exact_form`, over every")
    print("     (component name, sentence) pair on all five projects")
    totals = Counter()
    gate_disagreements = []
    for name in PROJECTS:
        project = load_project(name)
        knowledge = load_phase(run, name, "knowledge")
        table = _aliases(knowledge)
        per = Counter()
        for component in project["components"]:
            for sentence in project["sentences"]:
                standalone = has_standalone_mention(component.name, sentence.text)
                exact = bool(SLinker25._find_exact_form(sentence.text, component.name))
                per["pairs"] += 1
                if standalone == exact:
                    continue
                per["disagree"] += 1
                per["exact_only" if exact else "standalone_only"] += 1

        # The one place a disagreement changes a link: the coreference gate.
        coref = load_phase(run, name, "linker_coreference")
        id_to_name = {c.id: c.name for c in project["components"]}
        for meta in coref["feedback"]["metadata"]:
            component = id_to_name.get(meta["component_id"])
            antecedent = project["sent_map"].get(meta.get("antecedent_sentence"))
            if not component or not antecedent:
                continue
            names = [component] + [t for t, c in table.items() if c == component]
            gate_standalone = has_standalone_mention(component, antecedent.text) or any(
                re.search(rf"\b{re.escape(t)}\b", antecedent.text, re.IGNORECASE)
                for t in names[1:]
            )
            gate_exact = any(
                SLinker25._find_exact_form(antecedent.text, n) for n in names
            )
            if gate_standalone != gate_exact:
                gate_disagreements.append(
                    (name, meta["sentence"], component, gate_standalone, gate_exact))
        print(f"  {name:14s} pairs {per['pairs']:6d} | disagree {per['disagree']:4d} "
              f"(exact-only {per['exact_only']:4d}, standalone-only "
              f"{per['standalone_only']:3d})")
        totals.update(per)
    print(f"  TOTAL          pairs {totals['pairs']} | disagree {totals['disagree']} "
          f"(exact-only {totals['exact_only']}, standalone-only "
          f"{totals['standalone_only']})")
    print(f"  coreference gate verdict flips on the promoted run's resolutions: "
          f"{len(gate_disagreements)}")
    for item in gate_disagreements:
        print(f"    {item[0]} S{item[1]} {item[2]}: standalone={item[3]} exact={item[4]}")
    return {"pairs": totals["pairs"], "disagree": totals["disagree"],
            "exact_only": totals["exact_only"],
            "standalone_only": totals["standalone_only"],
            "gate_flips": len(gate_disagreements)}


# ── T2: the mention-type classifier ──────────────────────────────────────────

def audit_mention_type(run):
    print("\n### T2 mention-type classifier (5 values -> one prompt substring)")
    totals = Counter()
    for name in PROJECTS:
        project = load_project(name)
        knowledge = load_phase(run, name, "knowledge")
        linker = _linker(_Table(_aliases(knowledge)))
        full = load_phase(run, name, "linker_full_name")
        per = Counter()
        for item in full["feedback"]["candidates"]:
            per[linker._classify_mention_typed(item["component"], item["text"]).value] += 1
        print(f"  {name:14s} {dict(per)}")
        totals.update(per)
    print(f"  TOTAL          {dict(totals)}")
    used = {k for k, v in totals.items() if v}
    print(f"  values ever produced: {len(used)} of {len(MentionType)} "
          f"({sorted(used)})")
    return dict(totals)


# ── T3: spelling variants ────────────────────────────────────────────────────

def audit_variants(run):
    print("\n### T3 spelling-variant generator")
    totals = Counter()
    for name in PROJECTS:
        project = load_project(name)
        gold = load_gold(name)
        name_to_id = project["name_to_id"]
        linker = _linker(_Table({}))
        generated = linker._spelling_variant_candidates(
            project["sentences"], project["components"])
        full = load_phase(run, name, "linker_full_name")
        accepted_variants = [l for l in full["links"]
                            if l.source == "full_name_variant"]
        gold_variants = [l for l in accepted_variants
                         if (l.sentence_number, l.component_id) in gold]
        # A generated pair only enters the candidate set when extraction missed
        # it (`_add_spelling_variants` uses setdefault), so an accepted
        # `full_name_variant` link is by construction one extraction did not
        # propose. Comparing against the recorded candidate list would be
        # circular -- variants are already in it.
        print(f"  {name:14s} generated {len(generated):3d} | accepted as a variant "
              f"link {len(accepted_variants):3d} (gold {len(gold_variants):3d}) "
              f"{[(l.sentence_number, l.component_name) for l in accepted_variants]}")
        totals.update(generated=len(generated), accepted=len(accepted_variants),
                      gold=len(gold_variants))
    print(f"  TOTAL          generated {totals['generated']} | accepted "
          f"{totals['accepted']} (gold {totals['gold']}) — every accepted variant "
          f"is a pair extraction missed")
    return dict(totals)


# ── T4: the qualified-identifier disjunction ─────────────────────────────────

def audit_qualified(run):
    print("\n### T4 `_inside_qualified_identifier` — which disjuncts do work")
    print("     counted over every word span the partial-name generator scans")
    totals = Counter()
    word = re.compile(r"[A-Za-z]+[A-Za-z0-9]*|\d+")
    for name in PROJECTS:
        project = load_project(name)
        per = Counter()
        for sentence in project["sentences"]:
            text = sentence.text
            for match in word.finditer(text):
                start, end = match.start(), match.end()
                before = text[start - 1] if start else ""
                after = text[end] if end < len(text) else ""
                dotted_before = before == "." and start > 1 and text[start - 2].isalnum()
                dotted_after = (after == "." and end + 1 < len(text)
                                and text[end + 1].isalnum())
                joined_before = before in "-_" or (before and before.isalnum())
                joined_after = after in "-_" or (after and after.isalnum())
                fires = dotted_before or dotted_after or joined_before or joined_after
                per["spans"] += 1
                if not fires:
                    continue
                per["suppressed"] += 1
                if dotted_before or dotted_after:
                    per["dotted"] += 1
                if joined_before or joined_after:
                    per["joined"] += 1
                if (dotted_before or dotted_after) and not (joined_before or joined_after):
                    per["dotted_only"] += 1
        print(f"  {name:14s} spans {per['spans']:5d} | suppressed "
              f"{per['suppressed']:4d} | dotted {per['dotted']:4d} | joined "
              f"{per['joined']:4d} | dotted-but-not-joined {per['dotted_only']:3d}")
        totals.update(per)
    print(f"  TOTAL          spans {totals['spans']} | suppressed "
          f"{totals['suppressed']} | dotted {totals['dotted']} | joined "
          f"{totals['joined']} | dotted-but-not-joined {totals['dotted_only']}")
    if not totals["dotted_only"]:
        print("     -> the dotted disjuncts never decide anything the joined ones")
        print("        do not already decide: the test reduces to an adjacency test")
    return dict(totals)


# ── T5: fields nothing reads ─────────────────────────────────────────────────

def audit_unread_fields(run):
    print("\n### T5 fields that reach a prompt or a trace and nothing else")
    source = inspect.getsource(L25)

    rationale_sites = re.findall(r"extraction_rationale", source)
    rationale_values = set()
    for name in PROJECTS:
        knowledge = load_phase(run, name, "knowledge")
        linker = _linker(_Table(_aliases(knowledge)))
        project = load_project(name)
        full = load_phase(run, name, "linker_full_name")
        for item in full["feedback"]["candidates"][:50]:
            pair = (item["sentence"], project["name_to_id"].get(item["component"]))
            if pair[1] is None:
                continue
            fake = type("C", (), {
                "component_name": item["component"], "sentence_number": pair[0],
                "sentence_text": item["text"], "matched_text": item.get("source", ""),
                "source": "full_name", "component_id": pair[1]})()
            rationale_values.add(
                linker._build_evidence_bundle(fake, project["sent_map"])
                .extraction_rationale)
    print(f"  extraction_rationale: {len(rationale_sites)} mentions in the module, "
          f"{len(rationale_values)} distinct value(s) across all candidates "
          f"{sorted(rationale_values)}")

    via_alias = 0
    for name in PROJECTS:
        coref = load_phase(run, name, "linker_coreference")
        via_alias += sum(1 for m in coref["feedback"]["metadata"]
                         if m.get("antecedent_via_alias"))
    reads = [line.strip() for line in source.splitlines()
             if "antecedent_via_alias" in line]
    print(f"  antecedent_via_alias: set true on {via_alias} resolutions; "
          f"{len(reads)} code line(s) mention it:")
    for line in reads:
        print(f"      {line}")
    prompt_bytes = len(L25.ANTECEDENT_ALIAS_RULES)
    print(f"      the rules block it exists for is {prompt_bytes} bytes of every "
          f"coreference prompt")
    return {"rationale_distinct": len(rationale_values),
            "via_alias_true": via_alias,
            "antecedent_rules_bytes": prompt_bytes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path,
                        default=Path("../results/s25_complexity_audit/audit.json"))
    args = parser.parse_args()
    print(f"run: {args.run}")
    report = {
        "T1_primitives": audit_primitives(args.run),
        "T2_mention_type": audit_mention_type(args.run),
        "T3_variants": audit_variants(args.run),
        "T4_qualified": audit_qualified(args.run),
        "T5_unread": audit_unread_fields(args.run),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump({"run": str(args.run), "audits": report}, handle, indent=2)
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
