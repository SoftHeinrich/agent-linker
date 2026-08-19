"""Inventory of every code-driven decision in the s25 pipeline.

No LLM call. For each gate, measured off the promoted run's checkpoints: how many
candidates it admits, how many it rejects, and -- where the gold standard can say
-- how many gold links its rejections cost. A gate that rejects nothing is dead
weight; a gate that rejects only non-gold pairs is earning its place; a gate that
rejects gold pairs is buying precision with recall, which matters for F2.

The distinction that matters for the paper is not "code vs LLM" but *what kind*
of code:

  SANITY      -- the model named a sentence or component that does not exist, or
                 returned a malformed field. Rejecting that is not a heuristic
                 and no reviewer will ask about it.
  GROUNDING   -- the model's own quote is checked against the text it claims to
                 quote. Also not a domain heuristic: it verifies the model
                 against itself, and it is stated in one sentence.
  HEURISTIC   -- a hand-written linguistic rule that decides whether a link is
                 admissible: the stated-name contract filter, the partial-name
                 word test, the spelling-variant signature, the coreference
                 antecedent test, the mention-type classifier. These are what
                 make the approach read as rule-driven rather than LLM-driven,
                 and each one has to justify itself.

Usage: ../.venv/bin/python pilot/gate_audit.py
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
from llm_sad_sam.linkers.experimental.s_linker25 import SLinker25


def _linker(aliases):
    linker = SLinker25.__new__(SLinker25)
    linker.doc_knowledge = type("K", (), {"aliases": aliases})()
    return linker


def _aliases(knowledge):
    return {t: getattr(e, "component", e)
            for t, e in knowledge["doc_knowledge"].aliases.items()}


def _extraction_refs(run, name):
    """Every (sentence, component-name) the extractor proposed, from the trace."""
    refs = set()
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
            matched = ref.get("matched_text", "")
            if isinstance(snum, str) and snum.lstrip("Ss").isdigit():
                snum = int(snum.lstrip("Ss"))
            if isinstance(snum, int) and comp:
                refs.add((snum, comp, matched))
    return refs


# ── HEURISTIC 1: the stated-name contract filter ─────────────────────────────

def audit_contract_filter(run):
    """`_keep_stated_names` overrides the extractor with a lexical test."""
    print("\n### H1 stated-name contract filter (`_keep_stated_names`)")
    print("     the extractor proposes; this lexical test decides admission")
    totals = Counter()
    for name in PROJECTS:
        project = load_project(name)
        knowledge = load_phase(run, name, "knowledge")
        aliases = _aliases(knowledge)
        linker = _linker(aliases)
        name_to_id = project["name_to_id"]
        gold = project["gold"]
        proposed = _extraction_refs(run, name)
        kept = rejected = kept_gold = rejected_gold = 0
        for snum, comp, matched in proposed:
            cid = name_to_id.get(comp)
            sent = project["sent_map"].get(snum)
            if cid is None or sent is None:
                continue
            names = [comp] + [t for t, c in aliases.items() if c == comp]
            states = any(linker._find_exact_form(sent.text, n) for n in names)
            is_gold = (snum, cid) in gold
            if states:
                kept += 1
                kept_gold += is_gold
            else:
                rejected += 1
                rejected_gold += is_gold
        print(f"  {name:14s} extractor proposed {len(proposed):3d} | kept "
              f"{kept:3d} (gold {kept_gold:3d}) | REJECTED {rejected:3d} "
              f"(gold {rejected_gold:3d})")
        totals.update(proposed=len(proposed), kept=kept, kept_gold=kept_gold,
                      rejected=rejected, rejected_gold=rejected_gold)
    print(f"  TOTAL          proposed {totals['proposed']} | kept {totals['kept']} "
          f"(gold {totals['kept_gold']}) | REJECTED {totals['rejected']} "
          f"(gold {totals['rejected_gold']})")
    print(f"     -> the filter discards {totals['rejected']} extractor proposals, "
          f"{totals['rejected_gold']} of them gold")
    return dict(totals)


# ── HEURISTIC 2: the partial-name word test ──────────────────────────────────

def audit_partial_proposer(run):
    """`_name_word_candidates` is a fully hand-written proposer."""
    print("\n### H2 partial-name proposer (`_name_word_candidates`)")
    print("     prefix match on a name word, owned by exactly one component,")
    print("     outside a qualified identifier, in a sentence stating no name")
    totals = Counter()
    for name in PROJECTS:
        project = load_project(name)
        knowledge = load_phase(run, name, "knowledge")
        linker = _linker(_aliases(knowledge))
        gold = project["gold"]
        proposals = linker._name_word_candidates(
            project["sentences"], project["components"])
        reachable = [c for c in proposals
                     if (c.sentence_number, c.component_id) in gold]
        partial = load_phase(run, name, "linker_partial_name")
        accepted = [l for l in partial["links"]]
        accepted_gold = [l for l in accepted
                         if (l.sentence_number, l.component_id) in gold]
        print(f"  {name:14s} proposed {len(proposals):3d} | gold reachable "
              f"{len(reachable):3d} | judged in {len(accepted):2d} "
              f"(gold {len(accepted_gold):2d}) | precision of the proposer "
              f"{len(reachable) / max(len(proposals), 1):.0%}")
        totals.update(proposed=len(proposals), reachable=len(reachable),
                      accepted=len(accepted), accepted_gold=len(accepted_gold))
    print(f"  TOTAL          proposed {totals['proposed']} | gold reachable "
          f"{totals['reachable']} | accepted {totals['accepted']} "
          f"(gold {totals['accepted_gold']})")
    print(f"     -> the two judges throw away "
          f"{totals['proposed'] - totals['accepted']} of "
          f"{totals['proposed']} proposals; the heuristic is a wide net, the")
    print(f"        \x4cLM does the discriminating")
    return dict(totals)


# ── HEURISTIC 3: the coreference antecedent test ─────────────────────────────

def audit_antecedent_gate(run):
    """`_antecedent_states_name` rejects resolutions before the judge sees them."""
    print("\n### H3 coreference antecedent gate (`_antecedent_states_name`)")
    totals = Counter()
    for name in PROJECTS:
        project = load_project(name)
        knowledge = load_phase(run, name, "knowledge")
        aliases = _aliases(knowledge)
        linker = _linker(aliases)
        gold = project["gold"]
        name_to_id = project["name_to_id"]
        surviving = {(m["sentence"], m["component_id"])
                     for m in load_phase(run, name, "linker_coreference")
                     ["feedback"]["metadata"]}
        # Every resolution the model reported, from the trace, before the gate.
        reported = set()
        for call in load_calls(run, name):
            if call.get("phase") != "phase_25_coreference":
                continue
            text = call.get("response_text") or ""
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                continue
            try:
                data = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
            for res in data.get("resolutions", []):
                snum, comp = res.get("sentence"), res.get("component")
                if isinstance(snum, str) and snum.lstrip("Ss").isdigit():
                    snum = int(snum.lstrip("Ss"))
                cid = name_to_id.get(comp)
                if isinstance(snum, int) and cid and snum in project["sent_map"]:
                    reported.add((snum, cid))
        blocked = reported - surviving
        blocked_gold = {p for p in blocked if p in gold}
        print(f"  {name:14s} model reported {len(reported):3d} | passed the gate "
              f"{len(reported & surviving):3d} | BLOCKED {len(blocked):3d} "
              f"(gold {len(blocked_gold):2d})")
        totals.update(reported=len(reported), passed=len(reported & surviving),
                      blocked=len(blocked), blocked_gold=len(blocked_gold))
    print(f"  TOTAL          reported {totals['reported']} | passed "
          f"{totals['passed']} | BLOCKED {totals['blocked']} "
          f"(gold {totals['blocked_gold']})")
    print(f"     -> the gate is the single largest code-driven rejection in the "
          f"pipeline; it costs {totals['blocked_gold']} gold links")
    return dict(totals)


# ── GROUNDING: checks on the model's own quote ────────────────────────────────

def audit_grounding(run):
    """The partial-name judge's three evidence conditions."""
    print("\n### G grounding checks in the partial-name judge")
    print("     claim substring, anchor membership, non-empty alternative")
    totals = Counter()
    for name in PROJECTS:
        project = load_project(name)
        gold = project["gold"]
        partial = load_phase(run, name, "linker_partial_name")
        per = Counter()
        for decision in partial["feedback"]["judge_decisions"]:
            pair = (decision["sentence"], decision["component_id"])
            per["decisions"] += 1
            if decision.get("requested_keep") and not decision.get("evidence_valid"):
                per["voided"] += 1
                per["voided_gold"] += pair in gold
            if decision.get("denotation") == "associated":
                per["denotation_rejected"] += 1
                per["denotation_rejected_gold"] += pair in gold
        print(f"  {name:14s} decisions {per['decisions']:3d} | keep voided by a "
              f"failed evidence check {per['voided']:2d} (gold "
              f"{per['voided_gold']:2d}) | denotation said 'associated' "
              f"{per['denotation_rejected']:3d} (gold "
              f"{per['denotation_rejected_gold']:2d})")
        totals.update(per)
    print(f"  TOTAL          decisions {totals['decisions']} | voided "
          f"{totals['voided']} (gold {totals['voided_gold']}) | denotation "
          f"rejected {totals['denotation_rejected']} (gold "
          f"{totals['denotation_rejected_gold']})")
    return dict(totals)


# ── SANITY: rejections of impossible model output ────────────────────────────

def audit_sanity(run):
    print("\n### S sanity rejections (model named something that does not exist)")
    totals = Counter()
    for name in PROJECTS:
        project = load_project(name)
        name_to_id = project["name_to_id"]
        per = Counter()
        for snum, comp, matched in _extraction_refs(run, name):
            per["refs"] += 1
            if comp not in name_to_id:
                per["unknown_component"] += 1
            elif snum not in project["sent_map"]:
                per["unknown_sentence"] += 1
            else:
                sent = project["sent_map"][snum]
                if matched and matched.lower() not in sent.text.lower():
                    per["span_not_in_sentence"] += 1
        print(f"  {name:14s} refs {per['refs']:3d} | unknown component "
              f"{per['unknown_component']:2d} | unknown sentence "
              f"{per['unknown_sentence']:2d} | quoted span absent "
              f"{per['span_not_in_sentence']:2d}")
        totals.update(per)
    print(f"  TOTAL          refs {totals['refs']} | unknown component "
          f"{totals['unknown_component']} | unknown sentence "
          f"{totals['unknown_sentence']} | quoted span absent "
          f"{totals['span_not_in_sentence']}")
    return dict(totals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path,
                        default=Path("../results/s25_gate_audit/audit.json"))
    args = parser.parse_args()
    print(f"run: {args.run}")
    report = {
        "H1_contract_filter": audit_contract_filter(args.run),
        "H2_partial_proposer": audit_partial_proposer(args.run),
        "H3_antecedent_gate": audit_antecedent_gate(args.run),
        "G_grounding": audit_grounding(args.run),
        "S_sanity": audit_sanity(args.run),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump({"run": str(args.run), "audits": report}, handle, indent=2)
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
