"""Can the authored prompts state general principles instead of this corpus's shapes?

`pilot/prompt_defensibility.py` prices what is in them. Of 3645 bytes of authored
instruction in `s_linker70`, 814 are corpus-shaped and 1131 more are mixed, and the same
stipulation -- "a name inside a package/member path of the form X.Y or X.Y.Z is not the
component" -- is written **five times in five prompts**, in five different wordings. A
sixth copy exists that names no syntax at all: `QUALIFIED_CLAUSE`, written for the
denotation prompt in the fold round and measured neutral there.

So the question is not whether the distinction is needed. It is whether the *general*
statement of it does the work of the five bespoke ones -- and whether the four-condition
reject list survives at all now that two of its conditions are restated by
`STRICTER_CLAUSE` in the same prompt.

    plainrubric   the full-name judging prompt, generalized in one step: the numbered
                  reject list and the three approve-shapes are replaced by one
                  principle plus the two general clauses, and P1's trailing mention of
                  code-level identifiers goes with them. 20 of the 91 calls per run.
    plainextract  the extraction and alias prompts, same treatment: the `X.Y or X.Y.Z`
                  shape is replaced by `QUALIFIED_CLAUSE`. 14 of the 91 calls.

Each pilot is TWO arms, scored on the stage the change can reach, replayed against
recorded checkpoints. Per the measurement policy in CLAUDE.md this is step 2; an E2E is
owed only once a change is adopted.

    AB_RUNS=3 ../.venv/bin/python pilot/general_prompt_pilots.py --pilot plainrubric
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_pilots import collect                                     # noqa: E402
from bind_audit import PROJECTS, extractor_pairs                      # noqa: E402
from bind_pilots import against_first, linker                         # noqa: E402
from fold_pilots import OUT, SOURCE_RUN, SOURCE_VARIANT, inputs       # noqa: E402
from llm_sad_sam.core.data_types_v2 import CandidateLink              # noqa: E402
import llm_sad_sam.linkers.experimental.s_linker70 as L70             # noqa: E402


# ── the general statements the enumerations become ───────────────────────────

#: `LAYERED_ENTITY_RULES` without its four numbered reject-conditions and without its
#: three enumerated approve-shapes. What is kept is the part that states the stage's
#: standard of proof -- approve by default, reject on a positive ground -- and the
#: ground itself is stated once, generally: the name is doing some other job here.
#: Each of the four conditions keeps whatever general ground it has, and loses its
#: corpus-shaped form:
#:   (1) the `X.Y` syntax  -> `QUALIFIED_CLAUSE`, which this arm adds. Ground: the
#:       compositionality of qualified names, a property of every language with member
#:       access, not of these documents.
#:   (2) negation          -> the second half of the sentence below. Ground: ordinary
#:       logic -- a denied predication asserts nothing. It keeps its place even though
#:       0.0 of 90.3 recorded rejections per run cite one, because the test is
#:       defensibility, not firing rate.
#:   (3) names a different entity and (4) generic technology term -> `STRICTER_CLAUSE`,
#:       already in this prompt. Ground: the use/mention distinction.
#: What is dropped outright is the *enumeration* and the three approve-shapes ("a bare
#: mention, a heading, or a list"), which name where a name may sit in a document. Those
#: have no ground in general practice and the approve-by-default sentence in front of
#: them already licenses every one.
GENERAL_ENTITY_RULES = (
    "Approve the link by default: the component is named here and the document treats "
    "it as part of the system. Reject only on a positive ground -- that the sentence "
    "asserts nothing of this component, because the name is doing some other job here, "
    "or because the sentence denies what it would otherwise say of it."
)

#: `P1_FOCUS` without its trailing "rather than only as part of a code-level
#: identifier". The distinction is not lost; it moves to `QUALIFIED_CLAUSE`, stated
#: once for both passes instead of once inside one pass's question.
GENERAL_P1_FOCUS = (
    "Check architectural participation: does the sentence name this "
    "component as an architectural participant?"
)


def judging_arm(name, *, rules, p1_focus, clauses=()):
    """An `SLinker70` whose full-name judging prompt is built from the given parts."""
    from llm_sad_sam.linkers.experimental.s_linker70 import SLinker70

    tail = "".join(f"\n{c}\n" for c in clauses)

    def _prompt_validation(comp_names, cases, focus, strict: bool = False) -> str:
        block = L70.LAYERED_COREF_RULES if strict else rules
        extra = "" if strict else tail
        return f"""Validate components in a document. {focus}

COMPONENTS: {', '.join(comp_names)}

{block}
{extra}
For each case, first quote the EXACT words from the sentence that state the
architectural claim about the component (or write "none" if the sentence makes no
such claim), then decide approve true/false based on that claim.

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "claim": "<exact quote or none>", "approve": true}}]}}
JSON only:"""

    def _validate_with_evidence(self, candidates, bundles, components, sent_map,
                                p1_tag, p2_tag, stage_label):
        """s70's body with `P1_FOCUS` taken from the arm instead of the module."""
        from llm_sad_sam.linkers.experimental.s_linker70 import get_comp_names
        if not candidates:
            return [], {}
        comp_names = get_comp_names(components)
        decisions, approved = {}, []
        for _, batch in self._iter_batches(candidates, self.JUDGE_BATCH):
            cases = []
            for i, c in enumerate(batch):
                p = self._prev_prefix(c.sentence_number, sent_map)
                bundle = bundles.get((c.sentence_number, c.component_id))
                block = self._format_evidence(bundle) if bundle else ""
                cases.append((
                    f'Case {i+1}: "{c.matched_text}" -> {c.component_name}\n'
                    f'  {p}"{c.sentence_text}"\n{block}', c))
            strings = [ct for ct, _ in cases]
            r1 = self._run_validation_pass(comp_names, strings, p1_focus, p1_tag)
            r2 = self._run_validation_pass(comp_names, strings, L70.P2_FOCUS, p2_tag)
            for i, (_t, c) in enumerate(cases):
                ok = r1.get(i, False) and r2.get(i, False)
                decisions[(c.sentence_number, c.component_id)] = {
                    "approved": ok, "p1": r1.get(i, False), "p2": r2.get(i, False),
                    "path": f"{stage_label}_twopass", "stage": f"{stage_label}_twopass"}
                if ok:
                    approved.append(c)
        return approved, decisions

    return type(f"Arm_{name}", (SLinker70,),
                {"_VARIANT_NAME": f"s_linker70_{name}",
                 "_prompt_validation": staticmethod(_prompt_validation),
                 "_validate_with_evidence": _validate_with_evidence})


def s70_candidates(obj, info, project_name):
    """s70's own full-name candidate set: the recorded extractor plus the two scans."""
    names = {c.id: c.name for c in info["components"]}
    merged = {}
    for snum, cid in extractor_pairs(SOURCE_RUN, SOURCE_VARIANT, project_name):
        merged[(snum, cid)] = CandidateLink(
            snum, info["sent_map"][snum].text, names[cid], cid, "", source="full_name")
    for scan in (L70.SCANS["spelling"], L70.SCANS["stated_name"]):
        for cand in obj._scan(info["sentences"], info["components"], scan):
            merged.setdefault((cand.sentence_number, cand.component_id), cand)
    return list(merged.values())


def pilot_plainrubric():
    """The full-name judging prompt, with the enumerations replaced by principles.

    Both arms judge the identical candidate set -- s70's -- so every difference is the
    prompt. The general arm is *shorter* and states strictly less about this corpus; if
    it holds, the four-condition list was accretion and the two general clauses do its
    work.
    """
    arms = {
        "s70 (enumerated rubric)": judging_arm(
            "enum", rules=L70.LAYERED_ENTITY_RULES, p1_focus=L70.P1_FOCUS,
            clauses=(L70.STRICTER_CLAUSE,)),
        "general rubric": judging_arm(
            "general", rules=GENERAL_ENTITY_RULES, p1_focus=GENERAL_P1_FOCUS,
            clauses=(L70.QUALIFIED_CLAUSE, L70.STRICTER_CLAUSE)),
    }

    def unit(cls, run, project_name):
        info = inputs(project_name)
        obj = linker(cls, knowledge=info["knowledge"])
        cands = s70_candidates(obj, info, project_name)
        bundles = {(c.sentence_number, c.component_id):
                   obj._build_evidence_bundle(c, info["sent_map"]) for c in cands}
        approved, _ = obj._validate_with_evidence(
            cands, bundles, info["components"], info["sent_map"],
            "pilot_p1", "pilot_p2", "full_name_twopass")
        return {(project_name, (c.sentence_number, c.component_id))
                for c in approved}

    sets = {}
    for label, cls in arms.items():
        sets[label] = collect(label, lambda run, p, c=cls: unit(c, run, p), PROJECTS)
        print(f"  {label:28s} approved per run: {[len(s) for s in sets[label]]}")
    print(f"\n  authored bytes in the rubric: "
          f"{len(L70.LAYERED_ENTITY_RULES) + len(L70.P1_FOCUS)} enumerated vs "
          f"{len(GENERAL_ENTITY_RULES) + len(GENERAL_P1_FOCUS) + len(L70.QUALIFIED_CLAUSE)}"
          f" general\n")
    return against_first(sets, "plainrubric")


#: `ENTITY_EXTRACTION_RULES` with its code-path clause removed. The clause is not
#: dropped, it is restated: `QUALIFIED_CLAUSE` is appended to the prompt and carries the
#: same distinction on an SE-practice ground instead of by naming the shape. The
#: "even if the compound identifier is semantically related to the component" aside goes
#: with it -- that is an instruction about a confusion, not a criterion.
GENERAL_EXTRACTION_RULES = (
    "Include a reference only when the sentence itself writes the component's name or "
    "one of the KNOWN ALIASES. Exclude a component that the sentence only implies as a "
    "participant in a described interaction without naming it, and exclude a name used "
    "as ordinary English with no architectural intent. Favor inclusion among the "
    "sentences that do name it."
)

#: `LAYERED_COREF_RULES` without "or when the reference is only to a code-level
#: identifier" -- the fifth copy. Same replacement: the clause is appended generally.
GENERAL_COREF_RULES = (
    "These are coreference links: a pronoun or noun phrase in the sentence is claimed "
    "to refer back to the component, which is NOT named in the sentence itself. Approve "
    "only when the sentence contains a genuine referring expression that unambiguously "
    "points to THIS component and makes an architectural claim about it. Reject when "
    "there is no such referring expression or when the antecedent could equally be a "
    "different component. When uncertain, reject."
)


def extraction_arm(name, *, rules, extra=""):
    """An `SLinker70` whose extraction prompt is built from the given parts."""
    from llm_sad_sam.linkers.experimental.s_linker70 import SLinker70

    @staticmethod
    def _prompt_extraction(comp_names, mappings, batch) -> str:
        body = rules + (f"\n\n{extra}" if extra else "")
        return f"""Extract ALL references to components from this document.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings)}' if mappings else ''}

{body}

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence"}}]}}
JSON only:"""

    return type(f"Arm_{name}", (SLinker70,),
                {"_VARIANT_NAME": f"s_linker70_{name}",
                 "_prompt_extraction": _prompt_extraction})


def pilot_plainextract():
    """The extraction prompt's code-path clause, replaced by the general statement.

    Scored on the candidate set the full-name judge would receive -- the extractor's
    own proposals plus the two scans -- so the arm is read where its change lands.
    """
    arms = {
        "s70 (names the shape)": extraction_arm(
            "ex_base", rules=L70.ENTITY_EXTRACTION_RULES),
        "general clause instead": extraction_arm(
            "ex_gen", rules=GENERAL_EXTRACTION_RULES, extra=L70.QUALIFIED_CLAUSE),
    }

    def unit(cls, run, project_name):
        info = inputs(project_name)
        obj = linker(cls, knowledge=info["knowledge"])
        cands = list(obj._extract_named_mentions(
            info["sentences"], info["components"], info["name_to_id"],
            info["sent_map"]).values())
        for row in ("spelling", "stated_name"):
            cands = obj._add_scan(cands, info["sentences"], info["components"], row)
        return {(project_name, (c.sentence_number, c.component_id)) for c in cands}

    sets = {}
    for label, cls in arms.items():
        sets[label] = collect(label, lambda run, p, c=cls: unit(c, run, p), PROJECTS)
        print(f"  {label:28s} candidates per run: {[len(s) for s in sets[label]]}")
    return against_first(sets, "plainextract")


def pilot_plaincoref(minimal=False):
    """The coreference rubric's code-path clause, replaced by the general statement.

    Replayed on the resolutions the recorded run reported, so the proposer is fixed and
    only the judging prompt differs.
    """
    from fold_pilots import recorded_resolutions
    from llm_sad_sam.core.data_types_v2 import SadSamLink
    from llm_sad_sam.linkers.experimental.s_linker70 import SLinker70

    def coref_arm(name, *, rules, extra=""):
        def _prompt_validation(comp_names, cases, focus, strict: bool = False) -> str:
            block = (rules if strict else L70.LAYERED_ENTITY_RULES)
            tail = (f"\n{extra}\n" if (extra and strict)
                    else ("" if strict else f"\n{L70.STRICTER_CLAUSE}\n"))
            return f"""Validate components in a document. {focus}

COMPONENTS: {', '.join(comp_names)}

{block}
{tail}
For each case, first quote the EXACT words from the sentence that state the
architectural claim about the component (or write "none" if the sentence makes no
such claim), then decide approve true/false based on that claim.

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "claim": "<exact quote or none>", "approve": true}}]}}
JSON only:"""
        return type(f"Arm_{name}", (SLinker70,),
                    {"_VARIANT_NAME": f"s_linker70_{name}",
                     "_prompt_validation": staticmethod(_prompt_validation)})

    if minimal:
        # only the fifth copy of the syntax is removed; nothing is added, nothing
        # else is reworded. This separates "the phrase was load-bearing" from
        # "the replacement was wrong".
        stripped = L70.LAYERED_COREF_RULES.replace(
            ", or when the reference is only to a code-level identifier", "")
        assert stripped != L70.LAYERED_COREF_RULES
        arms = {
            "s70 (names the shape)": coref_arm(
                "cf_base2", rules=L70.LAYERED_COREF_RULES),
            "phrase removed, nothing added": coref_arm("cf_min", rules=stripped),
        }
    else:
        arms = {
            "s70 (names the shape)": coref_arm(
                "cf_base", rules=L70.LAYERED_COREF_RULES),
            "general clause instead": coref_arm(
                "cf_gen", rules=GENERAL_COREF_RULES, extra=L70.QUALIFIED_CLAUSE),
        }

    def unit(cls, run, project_name):
        info = inputs(project_name)
        obj = linker(cls, knowledge=info["knowledge"])
        links = [SadSamLink(snum, cid, comp, source="coreference")
                 for (snum, cid), (comp, _ant)
                 in recorded_resolutions(project_name).items()]
        approved, _ = obj._validate_coref_links(
            links, info["sent_map"], info["components"])
        return {(project_name, (l.sentence_number, l.component_id))
                for l in approved}

    sets = {}
    for label, cls in arms.items():
        sets[label] = collect(label, lambda run, p, c=cls: unit(c, run, p), PROJECTS)
        print(f"  {label:28s} approved per run: {[len(s) for s in sets[label]]}")
    return against_first(sets, "plaincoref_min" if minimal else "plaincoref")



#: `ALIAS_EXCLUSION_RULES` with the syntax removed but the imperative kept. The first
#: attempt swapped in `QUALIFIED_CLAUSE`, which is a *descriptive* sentence, and the
#: table grew by 12.7 terms per run -- so the old rule was doing work its wording does
#: not state: a flatly prohibitive sentence makes the extractor conservative about
#: everything, not only about what it prohibits. This form keeps the prohibition and
#: drops only the shape.
GENERAL_ALIAS_EXCLUSION = (
    "A fragment of a longer identifier is not an alias: if a term appears only as part "
    "of a compound or qualified name, do not include it."
)


def pilot_plainalias(imperative=False):
    """The last place a prompt spells out `X.Y or X.Y.Z`, and the module's own verdict.

    The comment three lines above `ALIAS_EXCLUSION_RULES` says naming that shape is "a
    rule written for one corpus" -- 62 of 198 sentences on the one benchmark that has
    dotted identifiers at all, 0-6 on the other four -- and the shape stayed in the
    string. `QUALIFIED_CLAUSE` states the same distinction on an SE-practice ground.

    Scored directly on what the rule prohibits rather than on downstream links: the
    proposed alias terms, and how many of them are fragments of a longer identifier.
    Two calls per project per arm, so the question costs a twentieth of an E2E.
    """
    import re as _re
    suffix = "2" if imperative else ""
    from llm_sad_sam.linkers.experimental.s_linker70 import SLinker70

    def alias_arm(name, *, exclusion):
        @staticmethod
        def _prompt_doc_knowledge_extract(comp_names, doc_lines) -> str:
            return f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

{L70.DOC_KNOWLEDGE_EXTRACTION_RULES}

{exclusion}

DOCUMENT:
{chr(10).join(doc_lines)}

Return JSON:
{{
  "abbreviations": [{{"term": "short_form", "component": "FullComponent"}}],
  "synonyms":      [{{"term": "specific_alternative_name", "component": "FullComponent"}}]
}}
JSON only:"""
        return type(f"Arm_{name}", (SLinker70,),
                    {"_VARIANT_NAME": f"s_linker70_{name}",
                     "_prompt_doc_knowledge_extract": _prompt_doc_knowledge_extract})

    arms = {
        "s70 (spells out X.Y)": alias_arm(
            f"al_base{suffix}", exclusion=L70.ALIAS_EXCLUSION_RULES),
        ("imperative general form" if imperative else "general clause instead"):
            alias_arm(f"al_gen{suffix}",
                      exclusion=GENERAL_ALIAS_EXCLUSION if imperative
                      else L70.QUALIFIED_CLAUSE),
    }
    dotted = _re.compile(r"[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+")
    prohibited = {label: [] for label in arms}

    def unit(cls, run, project_name, label):
        info = inputs(project_name)
        obj = linker(cls)
        knowledge = obj._learn_document_knowledge(
            info["sentences"], info["components"])
        table = dict(knowledge.aliases)
        prohibited[label].append(sum(1 for t in table if dotted.search(t)))
        return {(project_name, (term, comp)) for term, comp in table.items()}

    sets = {}
    for label, cls in arms.items():
        sets[label] = collect(label, lambda run, p, c=cls, l=label: unit(c, run, p, l),
                              PROJECTS)
        print(f"  {label:28s} aliases per run: {[len(s) for s in sets[label]]}"
              f"   fragments of a longer identifier: {prohibited[label]}")
    print("\n    The rule is a prohibition, so the number that matters is the second:")
    print("    an arm that admits no identifier fragment has enforced the rule, however")
    print("    it was worded.\n")
    return against_first(sets, "plainalias2" if imperative else "plainalias")


PILOTS = {"plainrubric": pilot_plainrubric,
          "plainextract": pilot_plainextract,
          "plaincoref": pilot_plaincoref,
          "plaincoref_min": lambda: pilot_plaincoref(minimal=True),
          "plainalias": pilot_plainalias,
          "plainalias2": lambda: pilot_plainalias(imperative=True)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", nargs="+", required=True,
                        choices=sorted(PILOTS) + ["all"])
    args = parser.parse_args()
    names = sorted(PILOTS) if "all" in args.pilot else args.pilot
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}")
        PILOTS[name]()


if __name__ == "__main__":
    main()
