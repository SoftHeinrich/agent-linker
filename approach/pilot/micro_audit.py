"""Every micro-condition in the s25 pipeline, and how often it changes an outcome.

No LLM call. The big gates were priced in `gate_audit.py`; this is the fine
print inside them -- the `if A and not B` clauses, the case rules, the
conjunctions, and the places where two nearly identical tests use different
predicates. For each one: how many times does it fire, and how many times does
firing change the result? A condition that never changes a result is deletable
without a pilot; two conditions that never disagree are one condition.

Sections:

  M1  `has_standalone_mention` -- a single-word/multi-word case-rule split plus
      four boundary `continue`s. Which branches ever change the answer?
  M2  the two qualified-path tests. `_inside_qualified_identifier` asks
      `text[end + 1].isalnum()`; `_all_occurrences_in_qualified_path` asks
      `text[e + 1].isalpha()` for the same shape. Do they ever disagree?
  M3  `_classify_mention_typed` -- a four-branch cascade. Which branches fire?
  M4  `_name_signature` -- a four-alternative regex. Which alternatives match?
  M5  `_spelling_variant_candidates` -- the separator test, the unique-owner
      test, and the "already the plain name" skip.
  M6  `_name_word_candidates` -- the prefix rule (asymmetric: a sentence word
      may extend a name word, never the reverse) and the unique-owner test.
  M7  judge-output conjunctions -- which conjunct of `evidence_valid` ever
      fails, in the denotation step and in the identity step.
  M8  parser tolerances read off the traces -- did any response ever return a
      string "true" for `approve`, an alias payload as a dict, or a sentence
      number as a string? Each tolerance is a branch kept for a case that may
      never have occurred.

Usage: ../.venv/bin/python pilot/micro_audit.py
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
from llm_sad_sam.linkers.experimental.helper_v3 import has_standalone_mention
from llm_sad_sam.linkers.experimental.s_linker25 import SLinker25, MentionType


def _linker(aliases):
    linker = SLinker25.__new__(SLinker25)
    linker.doc_knowledge = type("K", (), {"aliases": aliases})()
    return linker


def _aliases(knowledge):
    return {t: getattr(e, "component", e)
            for t, e in knowledge["doc_knowledge"].aliases.items()}


def _all_names(project, aliases):
    """Every name a component can be stated by, per component name."""
    out = {}
    for component in project["components"]:
        out[component.name] = [component.name] + [
            t for t, c in aliases.items() if c == component.name]
    return out


# ── M1: has_standalone_mention's internal branches ───────────────────────────

def _standalone_variants(comp_name, text):
    """The predicate, plus each of its branches disabled one at a time."""
    if not comp_name:
        return {}
    is_single = " " not in comp_name
    if is_single:
        lower_pattern = rf"\b{re.escape(comp_name)}\b"
        cap = comp_name[0].upper() + comp_name[1:]
        cap_pattern = rf"\b{re.escape(cap)}\b"
        pattern = lower_pattern if comp_name[0].islower() else cap_pattern
        flags = 0
    else:
        pattern = rf"\b{re.escape(comp_name)}\b"
        flags = re.IGNORECASE

    def run(skip=()):
        for m in re.finditer(pattern, text, flags):
            s, e = m.start(), m.end()
            if "dot_before" not in skip and s > 0 and text[s - 1] == ".":
                continue
            if ("dot_after" not in skip and e < len(text) and text[e] == "."
                    and e + 1 < len(text) and text[e + 1].isalpha()):
                continue
            if "hyphen_before" not in skip and s > 0 and text[s - 1] == "-":
                continue
            if ("hyphen_after" not in skip and e < len(text) and text[e] == "-"
                    and "-" not in comp_name):
                continue
            return True
        return False

    baseline = run()
    out = {"baseline": baseline}
    for branch in ("dot_before", "dot_after", "hyphen_before", "hyphen_after"):
        out[branch] = run(skip=(branch,))
    # And the case rule: what if the whole thing were case-insensitive?
    out["case_insensitive"] = bool(
        re.search(rf"\b{re.escape(comp_name)}\b", text, re.IGNORECASE))
    return out


def audit_standalone(run):
    print("\n### M1 `has_standalone_mention` branches")
    print("     each boundary rule disabled in turn, over every (name, sentence)")
    totals = Counter()
    for name in PROJECTS:
        project = load_project(name)
        per = Counter()
        for component in project["components"]:
            for sentence in project["sentences"]:
                got = _standalone_variants(component.name, sentence.text)
                if not got:
                    continue
                per["pairs"] += 1
                for branch in ("dot_before", "dot_after", "hyphen_before",
                               "hyphen_after"):
                    if got[branch] != got["baseline"]:
                        per[branch] += 1
                if got["case_insensitive"] != got["baseline"]:
                    per["case_rule_matters"] += 1
        print(f"  {name:14s} pairs {per['pairs']:5d} | dot_before {per['dot_before']:3d} "
              f"| dot_after {per['dot_after']:3d} | hyphen_before "
              f"{per['hyphen_before']:3d} | hyphen_after {per['hyphen_after']:3d} "
              f"| case rule {per['case_rule_matters']:3d}")
        totals.update(per)
    print(f"  TOTAL          pairs {totals['pairs']} | dot_before "
          f"{totals['dot_before']} | dot_after {totals['dot_after']} | "
          f"hyphen_before {totals['hyphen_before']} | hyphen_after "
          f"{totals['hyphen_after']} | case rule {totals['case_rule_matters']}")
    dead = [b for b in ("dot_before", "dot_after", "hyphen_before", "hyphen_after")
            if not totals[b]]
    if dead:
        print(f"     -> never changes the answer on any project: {dead}")
    return dict(totals)


# ── M2: the two qualified-path tests ─────────────────────────────────────────

def audit_qualified_predicates(run):
    print("\n### M2 the two qualified-path tests (isalnum vs isalpha)")
    print("     same shape `word.` followed by a character; do they disagree?")
    totals = Counter()
    word = re.compile(r"[A-Za-z0-9]+")
    for name in PROJECTS:
        project = load_project(name)
        per = Counter()
        for sentence in project["sentences"]:
            text = sentence.text
            for match in word.finditer(text):
                end = match.end()
                per["spans"] += 1
                if end >= len(text) or text[end] != "." or end + 1 >= len(text):
                    continue
                nxt = text[end + 1]
                per["dot_followed"] += 1
                if nxt.isalnum() != nxt.isalpha():
                    per["disagree"] += 1
                    per[f"only_alnum:{nxt}"] += 1
        print(f"  {name:14s} spans {per['spans']:5d} | `word.` + char "
              f"{per['dot_followed']:4d} | predicates disagree {per['disagree']:3d}")
        totals.update(per)
    print(f"  TOTAL          spans {totals['spans']} | `word.` + char "
          f"{totals['dot_followed']} | disagree {totals['disagree']}")
    if not totals["disagree"]:
        print("     -> the two predicates are the same function on this corpus:")
        print("        every character following `word.` is either a letter or")
        print("        neither a letter nor a digit. One test suffices.")
    else:
        chars = {k.split(":", 1)[1] for k in totals if k.startswith("only_alnum:")}
        print(f"     -> they differ only where a digit follows: {sorted(chars)}")
    return dict(totals)


# ── M3: the mention-type cascade ─────────────────────────────────────────────

def audit_cascade(run):
    print("\n### M3 `_classify_mention_typed` cascade branches")
    totals = Counter()
    for name in PROJECTS:
        project = load_project(name)
        knowledge = load_phase(run, name, "knowledge")
        linker = _linker(_aliases(knowledge))
        full = load_phase(run, name, "linker_full_name")
        per = Counter()
        for item in full["feedback"]["candidates"]:
            per[linker._classify_mention_typed(item["component"],
                                               item["text"]).value] += 1
        totals.update(per)
    for value in MentionType:
        print(f"  {value.name:22s} {totals[value.value]:4d}")
    dead = [v.name for v in MentionType if not totals[v.value]]
    print(f"     -> branches never taken: {dead or 'none'}")
    return {k: v for k, v in totals.items()}


# ── M4: the name-signature regex alternatives ────────────────────────────────

SIGNATURE_ALTS = [
    (r"[A-Z]+(?=[A-Z][a-z]|\b)", "acronym-run"),
    (r"[A-Z]?[a-z]+", "capitalised-or-lower"),
    (r"[A-Z]+", "bare-upper"),
    (r"\d+", "digits"),
]


def audit_signature(run):
    print("\n### M4 `_name_signature` regex alternatives")
    print("     which alternative produces each token, over component names and")
    print("     every candidate span the variant generator inspects")
    combined = re.compile("|".join(f"({p})" for p, _ in SIGNATURE_ALTS))
    totals = Counter()
    for name in PROJECTS:
        project = load_project(name)
        texts = [c.name for c in project["components"]]
        texts += [s.text for s in project["sentences"]]
        for text in texts:
            for match in combined.finditer(text):
                for index, (_pattern, label) in enumerate(SIGNATURE_ALTS, start=1):
                    if match.group(index) is not None:
                        totals[label] += 1
                        break
    for _pattern, label in SIGNATURE_ALTS:
        print(f"  {label:22s} {totals[label]:6d}")
    dead = [l for _p, l in SIGNATURE_ALTS if not totals[l]]
    print(f"     -> alternatives that never match: {dead or 'none'}")
    return dict(totals)


# ── M5 / M6: the two deterministic generators' internal tests ────────────────

def audit_generators(run):
    print("\n### M5/M6 internal tests of the two deterministic generators")
    totals = Counter()
    word_variant = re.compile(r"[A-Za-z0-9]+")
    word_partial = re.compile(r"[A-Za-z]+[A-Za-z0-9]*|\d+")
    for name in PROJECTS:
        project = load_project(name)
        knowledge = load_phase(run, name, "knowledge")
        aliases = _aliases(knowledge)
        linker = _linker(aliases)
        per = Counter()

        # M5: spelling variants
        owners = {}
        for component in project["components"]:
            signature = SLinker25._name_signature(component.name)
            if signature:
                owners.setdefault(signature, []).append(component)
        max_words = max((len(item) for item in owners), default=0)
        for sentence in project["sentences"]:
            words = list(word_variant.finditer(sentence.text))
            for start_index, first in enumerate(words):
                for end_index in range(start_index,
                                       min(len(words), start_index + max_words)):
                    last = words[end_index]
                    if end_index > start_index:
                        separator = sentence.text[
                            words[end_index - 1].end():last.start()]
                        if not re.fullmatch(r"[\s_-]+", separator):
                            per["m5_separator_break"] += 1
                            break
                    start, end = first.start(), last.end()
                    if SLinker25._inside_qualified_identifier(
                            sentence.text, start, end):
                        per["m5_qualified_skip"] += 1
                        continue
                    surface = sentence.text[start:end]
                    targets = owners.get(SLinker25._name_signature(surface), ())
                    if not targets:
                        continue
                    per["m5_signature_hit"] += 1
                    if len(targets) != 1:
                        per["m5_ambiguous_owner"] += 1
                        continue
                    if surface.casefold() == targets[0].name.casefold():
                        per["m5_plain_name_skip"] += 1

        # M6: partial-name word test
        words_by_component = {
            component.id: [w.casefold() for w in word_partial.findall(component.name)]
            for component in project["components"]
        }
        names = _all_names(project, aliases)
        for sentence in project["sentences"]:
            for match in word_partial.finditer(sentence.text):
                surface = match.group(0).casefold()
                if SLinker25._inside_qualified_identifier(
                        sentence.text, match.start(), match.end()):
                    per["m6_qualified_skip"] += 1
                    continue
                exact = [c for c in project["components"]
                         if surface in words_by_component[c.id]]
                prefix = [c for c in project["components"]
                          if any(surface.startswith(w) and surface != w
                                 for w in words_by_component[c.id])]
                owners_here = [c for c in project["components"]
                               if any(surface.startswith(w)
                                      for w in words_by_component[c.id])]
                if owners_here:
                    per["m6_word_hit"] += 1
                if len(owners_here) > 1:
                    per["m6_ambiguous_owner"] += 1
                if prefix and not exact:
                    per["m6_prefix_only"] += 1
                    if len(owners_here) == 1:
                        component = owners_here[0]
                        if not any(linker._find_exact_form(sentence.text, n)
                                   for n in names[component.name]):
                            per["m6_prefix_only_admitted"] += 1
        print(f"  {name:14s} M5 sep-break {per['m5_separator_break']:4d} "
              f"qual-skip {per['m5_qualified_skip']:4d} sig-hit "
              f"{per['m5_signature_hit']:3d} ambig {per['m5_ambiguous_owner']:3d} "
              f"plain-skip {per['m5_plain_name_skip']:3d} | M6 word-hit "
              f"{per['m6_word_hit']:4d} ambig {per['m6_ambiguous_owner']:3d} "
              f"prefix-only {per['m6_prefix_only']:3d} admitted "
              f"{per['m6_prefix_only_admitted']:3d}")
        totals.update(per)
    print(f"  TOTAL          M5 sep-break {totals['m5_separator_break']} "
          f"qual-skip {totals['m5_qualified_skip']} sig-hit "
          f"{totals['m5_signature_hit']} ambig {totals['m5_ambiguous_owner']} "
          f"plain-skip {totals['m5_plain_name_skip']}")
    print(f"                 M6 word-hit {totals['m6_word_hit']} ambig "
          f"{totals['m6_ambiguous_owner']} prefix-only "
          f"{totals['m6_prefix_only']} of which admitted "
          f"{totals['m6_prefix_only_admitted']}")
    if not totals["m6_prefix_only_admitted"]:
        print("     -> the prefix rule never admits a candidate an exact word")
        print("        match would not: `startswith` can be `==`")
    return dict(totals)


# ── M7: judge-output conjunctions ────────────────────────────────────────────

def audit_conjunctions(run):
    print("\n### M7 which conjunct of `evidence_valid` ever fails")
    totals = Counter()
    for name in PROJECTS:
        project = load_project(name)
        partial = load_phase(run, name, "linker_partial_name")
        per = Counter()
        for decision in partial["feedback"]["judge_decisions"]:
            path = decision.get("path")
            if path == "denotation":
                per["denotation_decisions"] += 1
                claim = str(decision.get("claim", ""))
                text = project["sent_map"][decision["sentence"]].text
                if decision.get("denotation") not in ("participant", "associated"):
                    per["d_bad_label"] += 1
                if not claim:
                    per["d_empty_claim"] += 1
                elif claim.casefold() not in text.casefold():
                    per["d_claim_not_substring"] += 1
            elif path == "identity":
                per["identity_decisions"] += 1
                if not decision.get("claim"):
                    per["i_empty_claim"] += 1
                if not decision.get("alternative"):
                    per["i_empty_alternative"] += 1
                if decision.get("anchor_sentence") in (0, None):
                    per["i_bad_anchor"] += 1
        totals.update(per)
    print(f"  denotation decisions {totals['denotation_decisions']:4d} | bad label "
          f"{totals['d_bad_label']:3d} | empty claim {totals['d_empty_claim']:3d} "
          f"| claim not a substring {totals['d_claim_not_substring']:3d}")
    print(f"  identity   decisions {totals['identity_decisions']:4d} | empty claim "
          f"{totals['i_empty_claim']:3d} | empty alternative "
          f"{totals['i_empty_alternative']:3d} | unusable anchor "
          f"{totals['i_bad_anchor']:3d}")
    return dict(totals)


# ── M8: parser tolerances, read off the traces ───────────────────────────────

def audit_tolerances(run):
    print("\n### M8 parser tolerances — did the case they guard ever occur?")
    totals = Counter()
    for name in PROJECTS:
        for call in load_calls(run, name):
            text = call.get("response_text") or ""
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                continue
            try:
                data = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                totals["unparsable"] += 1
                continue
            for item in data.get("validations", []):
                totals["approve_fields"] += 1
                if isinstance(item.get("approve"), str):
                    totals["approve_as_string"] += 1
            for key in ("abbreviations", "synonyms"):
                if key in data:
                    totals["alias_payloads"] += 1
                    if isinstance(data[key], dict):
                        totals["alias_payload_as_dict"] += 1
            for group in ("references", "resolutions"):
                for item in data.get(group, []):
                    totals["sentence_fields"] += 1
                    if isinstance(item.get("sentence"), str):
                        totals["sentence_as_string"] += 1
            for item in data.get("judgments", []):
                totals["case_fields"] += 1
                if isinstance(item.get("case"), str):
                    totals["case_as_string"] += 1
    print(f"  approve fields {totals['approve_fields']:5d} | returned as a string "
          f"{totals['approve_as_string']}")
    print(f"  alias payloads {totals['alias_payloads']:5d} | returned as a dict "
          f"{totals['alias_payload_as_dict']}")
    print(f"  sentence fields {totals['sentence_fields']:4d} | returned as a string "
          f"{totals['sentence_as_string']}")
    print(f"  case fields {totals['case_fields']:8d} | returned as a string "
          f"{totals['case_as_string']}")
    print(f"  unparsable responses {totals['unparsable']}")
    return dict(totals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path,
                        default=Path("../results/s25_micro_audit/audit.json"))
    args = parser.parse_args()
    print(f"run: {args.run}")
    report = {
        "M1_standalone_branches": audit_standalone(args.run),
        "M2_qualified_predicates": audit_qualified_predicates(args.run),
        "M3_mention_cascade": audit_cascade(args.run),
        "M4_signature_alternatives": audit_signature(args.run),
        "M5_M6_generators": audit_generators(args.run),
        "M7_conjunctions": audit_conjunctions(args.run),
        "M8_tolerances": audit_tolerances(args.run),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump({"run": str(args.run), "audits": report}, handle, indent=2,
                  default=str)
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
