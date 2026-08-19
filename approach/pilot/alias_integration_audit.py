"""Is the alias module a separate concept, or a projection of the extractor?

Both modules are the same model reading the same document against the same
component list. The alias module asks "what surface forms name component X"; the
extractor asks "which sentences reference component X" and already reports the
surface it matched (`matched_text`). If the alias table is recoverable from that
field, the module is a projection of the extractor, not a second idea, and the
paper has one stage where it currently has two.

Everything here is read off a promoted run's checkpoints and per-call traces. No
LLM call.

  A1 RECOVERABILITY -- of the aliases the module discovered, how many appear as
     an extractor `matched_text` that is not the component's own name? That is
     the recall of a derived table.
  A2 PRECISION OF A DERIVED TABLE -- of the non-name surfaces the extractor
     reports, how many are aliases the module also found, and what are the rest?
  A3 WHO NEEDS THE TABLE, AND WHEN -- each consumer of the alias table, with the
     stage it runs in. A consumer that runs after extraction can read a derived
     table; only a consumer that runs before it needs a discovered one.
  A4 DEPENDENCE -- how many final links depend on an alias at all: admitted by
     the contract filter only via an alias, labelled VIA_ALIAS, admitted through
     the antecedent gate only via an alias, or suppressed in the partial-name
     proposer only by an alias.

Usage: ../.venv/bin/python pilot/alias_integration_audit.py
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import (
    PROJECTS, DEFAULT_RUN, load_project, load_phase, load_calls,
)
from llm_sad_sam.linkers.experimental.s_linker25 import SLinker25, MentionType


def _linker(aliases):
    linker = SLinker25.__new__(SLinker25)
    linker.doc_knowledge = type("K", (), {"aliases": aliases})()
    return linker


def _aliases(knowledge):
    return {t: getattr(e, "component", e)
            for t, e in knowledge["doc_knowledge"].aliases.items()}


def _extraction_refs(run, name):
    """(sentence, component, matched_text) for every extractor reference."""
    refs = []
    for call in load_calls(run, name):
        if not str(call.get("phase", "")).startswith("phase_25_full_name_extract"):
            continue
        text = call.get("response_text") or ""
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            continue
        for ref in data.get("references", []):
            snum, comp = ref.get("sentence"), ref.get("component")
            if isinstance(snum, str) and snum.lstrip("Ss").isdigit():
                snum = int(snum.lstrip("Ss"))
            if isinstance(snum, int) and comp:
                refs.append((snum, comp, str(ref.get("matched_text", ""))))
    return refs


def _derived_table(refs, comp_names):
    """The alias table a derived design would build from extractor output.

    A surface the extractor matched that is neither the component's own name nor
    a substring-equal variant of it is, by the extractor's own report, another
    way the document names that component.
    """
    table = {}
    counts = Counter()
    for _snum, comp, matched in refs:
        if comp not in comp_names or not matched:
            continue
        surface = matched.strip()
        if not surface or surface.casefold() == comp.casefold():
            continue
        # a surface that merely contains the name adds nothing new
        if comp.casefold() in surface.casefold():
            continue
        table.setdefault(surface, comp)
        counts[(surface, comp)] += 1
    return table, counts


# ── A1 / A2 ──────────────────────────────────────────────────────────────────

def audit_recoverability(run):
    print("\n### A1/A2 can the alias table be derived from extractor output?")
    totals = Counter()
    for name in PROJECTS:
        project = load_project(name)
        comp_names = {c.name for c in project["components"]}
        discovered = _aliases(load_phase(run, name, "knowledge"))
        refs = _extraction_refs(run, name)
        derived, counts = _derived_table(refs, comp_names)

        d_keys = {k.casefold(): v for k, v in discovered.items()}
        v_keys = {k.casefold(): v for k, v in derived.items()}
        recovered = {k for k in d_keys if k in v_keys and v_keys[k] == d_keys[k]}
        missed = {k for k in d_keys if k not in recovered}
        extra = {k for k in v_keys if k not in d_keys}

        print(f"  {name:14s} discovered {len(discovered):2d} | derived "
              f"{len(derived):2d} | recovered {len(recovered):2d} | missed "
              f"{len(missed):2d} | derived-only {len(extra):2d}")
        if missed:
            print(f"      missed by a derived table: "
                  f"{sorted(discovered[k] and k for k in discovered if k.casefold() in missed)}")
        if extra:
            print(f"      derived-only surfaces: "
                  f"{sorted(k for k in derived if k.casefold() in extra)[:8]}")
        totals.update(discovered=len(discovered), derived=len(derived),
                      recovered=len(recovered), missed=len(missed),
                      extra=len(extra))
    print(f"  TOTAL          discovered {totals['discovered']} | derived "
          f"{totals['derived']} | recovered {totals['recovered']} | missed "
          f"{totals['missed']} | derived-only {totals['extra']}")
    if totals["discovered"]:
        print(f"     -> a derived table recovers "
              f"{totals['recovered'] / totals['discovered']:.0%} of the discovered "
              f"aliases and adds {totals['extra']} surfaces of its own")
    return dict(totals)


# ── A3 ───────────────────────────────────────────────────────────────────────

CONSUMERS = [
    ("extraction prompt (KNOWN ALIASES)", "full-name, before extraction",
     "needs a table that exists BEFORE extraction"),
    ("_keep_stated_names", "full-name, after extraction", "can read a derived table"),
    ("_classify_mention_typed (VIA_ALIAS)", "full-name, after extraction",
     "can read a derived table"),
    ("_name_word_candidates (whole-name suppressor)", "partial-name",
     "can read a derived table"),
    ("_review_identity_batch (anchors)", "partial-name",
     "can read a derived table"),
    ("_antecedent_states_name", "coreference", "can read a derived table"),
]


def audit_consumers(run):
    print("\n### A3 who reads the alias table, and when")
    for consumer, stage, verdict in CONSUMERS:
        print(f"  {consumer:46s} {stage:28s} {verdict}")
    print("     -> exactly one consumer runs before extraction. Every other")
    print("        consumer could read a table derived from extraction output.")
    return {"consumers": len(CONSUMERS), "pre_extraction": 1}


# ── A4 ───────────────────────────────────────────────────────────────────────

def audit_dependence(run):
    print("\n### A4 how many links depend on an alias at all")
    totals = Counter()
    for name in PROJECTS:
        project = load_project(name)
        aliases = _aliases(load_phase(run, name, "knowledge"))
        linker = _linker(aliases)
        gold = project["gold"]
        name_to_id = project["name_to_id"]
        per = Counter()

        full = load_phase(run, name, "linker_full_name")
        for link in full["links"]:
            comp = link.component_name
            text = project["sent_map"][link.sentence_number].text
            names = [t for t, c in aliases.items() if c == comp]
            if linker._find_exact_form(text, comp):
                continue
            if any(linker._find_exact_form(text, t) for t in names):
                per["full_name_via_alias"] += 1
                per["full_name_via_alias_gold"] += (
                    link.sentence_number, link.component_id) in gold

        coref = load_phase(run, name, "linker_coreference")
        id_to_name = {c.id: c.name for c in project["components"]}
        approved = {(l.sentence_number, l.component_id) for l in coref["links"]}
        for meta in coref["feedback"]["metadata"]:
            comp = id_to_name.get(meta["component_id"])
            ante = project["sent_map"].get(meta.get("antecedent_sentence"))
            if not comp or not ante:
                continue
            if linker._find_exact_form(ante.text, comp):
                continue
            names = [t for t, c in aliases.items() if c == comp]
            if any(linker._find_exact_form(ante.text, t) for t in names):
                per["coref_via_alias"] += 1
                key = (meta["sentence"], meta["component_id"])
                per["coref_via_alias_kept"] += key in approved
                per["coref_via_alias_gold"] += key in gold

        # partial-name: candidates the alias table suppressed
        suppressed = 0
        words = re.compile(r"[A-Za-z]+[A-Za-z0-9]*|\d+")
        words_by_component = {
            c.id: [w.casefold() for w in words.findall(c.name)]
            for c in project["components"]
        }
        for sentence in project["sentences"]:
            for match in words.finditer(sentence.text):
                if linker._inside_qualified_identifier(
                        sentence.text, match.start(), match.end()):
                    continue
                surface = match.group(0).casefold()
                owners = [c for c in project["components"]
                          if any(surface.startswith(w)
                                 for w in words_by_component[c.id])]
                if len(owners) != 1:
                    continue
                comp = owners[0]
                if linker._find_exact_form(sentence.text, comp.name):
                    continue
                alias_hits = [t for t, c in aliases.items() if c == comp.name
                              and linker._find_exact_form(sentence.text, t)]
                if alias_hits:
                    suppressed += 1
        per["partial_suppressed_by_alias"] = suppressed

        print(f"  {name:14s} full-name links admitted only via an alias "
              f"{per['full_name_via_alias']:2d} (gold "
              f"{per['full_name_via_alias_gold']:2d}) | coref antecedents via "
              f"alias {per['coref_via_alias']:2d} (kept "
              f"{per['coref_via_alias_kept']:2d}, gold "
              f"{per['coref_via_alias_gold']:2d}) | partial candidates "
              f"suppressed by an alias {per['partial_suppressed_by_alias']:2d}")
        totals.update(per)
    print(f"  TOTAL          full-name via alias {totals['full_name_via_alias']} "
          f"(gold {totals['full_name_via_alias_gold']}) | coref via alias "
          f"{totals['coref_via_alias']} (kept {totals['coref_via_alias_kept']}, "
          f"gold {totals['coref_via_alias_gold']}) | partial suppressed "
          f"{totals['partial_suppressed_by_alias']}")
    return dict(totals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path,
                        default=Path("../results/s25_alias_integration/audit.json"))
    args = parser.parse_args()
    print(f"run: {args.run}")
    report = {
        "A1_A2_recoverability": audit_recoverability(args.run),
        "A3_consumers": audit_consumers(args.run),
        "A4_dependence": audit_dependence(args.run),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump({"run": str(args.run), "audits": report}, handle, indent=2,
                  default=str)
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
