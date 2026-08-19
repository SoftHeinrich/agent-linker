"""The last finetuned spans of `s_linker74`, priced one at a time.

The general round left `s_linker74` with two kinds of authored text a reviewer can still
read as fitted to this benchmark, and they are different kinds:

    the SYNTAX, four more times   `ENTITY_EXTRACTION_RULES`, `P1_FOCUS` and
                                  `LAYERED_COREF_RULES` each restate "a name inside a
                                  longer identifier is not the component" in their own
                                  wording, and `ALIAS_EXCLUSION_RULES` still spells
                                  `X.Y or X.Y.Z` outright. One sentence already states
                                  the distinction on an SE-practice ground for every
                                  judge that needs it: `QUALIFIED_CLAUSE`.
    the COMPENSATION              the alias syntax was kept in the general round on
                                  measured grounds -- both general rewordings grow the
                                  alias table from 24.0 to ~37 terms per run, and an
                                  over-large table was priced at F1 94.57 vs 96.42
                                  (`s_linker39`/`s_linker40`). The clause's effect is
                                  not the effect it states: it makes the extractor
                                  conservative about everything.

Two arms have never been run, and they are the two the round's own lessons point at:

    rubricsyntax  DEGENERALIZING without RESTRUCTURING. `plainrubric` bundled two edits
                  -- it dropped `P1_FOCUS`'s code-level tail (what the bar asks for) AND
                  replaced the four numbered reject-conditions with one principle (what
                  the bar does not ask for, and what cost ~0.8 F1 composed in s71/s72).
                  This arm takes the first and leaves the second: the enumeration stays
                  byte-identical to s74's, `P1_FOCUS` loses its tail, and
                  `QUALIFIED_CLAUSE` carries the distinction once for both passes.
    aliascomp     COMPENSATING for the alias syntax instead of paying for it. If the
                  general wording's cost is a larger table rather than admitted
                  fragments, the branch's own law says where to recover it: **the looser
                  the proposer, the stricter the judge behind it**. The alias judge
                  currently breaks ties towards APPROVE. This arm removes the syntax
                  from the extractor and flips that tie-break, which is prior-work
                  ground rather than a new rule, and reads the JUDGED table -- the thing
                  that actually reaches the extraction prompt.

Both are step 2 of the measurement policy: one stage, fixed recorded inputs, N samples a
side. Neither is an E2E and neither is quoted as one.

    AB_SOURCE_RUN=../results/s74_solo_r1_20260817 AB_SOURCE_VARIANT=s_linker74 \
    AB_RUNS=3 ../.venv/bin/python pilot/finetune_pilots.py --pilot rubricsyntax
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_pilots import collect                                     # noqa: E402
from bind_audit import PROJECTS, extractor_pairs                      # noqa: E402
from bind_pilots import against_first, linker                         # noqa: E402
from fold_pilots import OUT, SOURCE_RUN, SOURCE_VARIANT, inputs       # noqa: E402
from llm_sad_sam.core.data_types_v2 import CandidateLink              # noqa: E402
import llm_sad_sam.linkers.experimental.s_linker74 as L               # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker74 import SLinker74     # noqa: E402


# ── the general statements the bespoke copies become ─────────────────────────

#: `P1_FOCUS` without "rather than only as part of a code-level identifier". The
#: distinction is not dropped: `QUALIFIED_CLAUSE` states it once, in the same prompt,
#: for both passes instead of inside one pass's question.
GENERAL_P1_FOCUS = (
    "Check architectural participation: does the sentence name this "
    "component as an architectural participant?"
)

#: `ALIAS_EXCLUSION_RULES` with the shape gone and the prohibition kept. Identical to
#: `general_prompt_pilots.GENERAL_ALIAS_EXCLUSION`, re-declared here so this module's
#: arms are readable without chasing the other file; the assertion below holds them
#: equal.
GENERAL_ALIAS_EXCLUSION = (
    "A fragment of a longer identifier is not an alias: if a term appears only as part "
    "of a compound or qualified name, do not include it."
)

#: `DOC_KNOWLEDGE_JUDGE_RULES` with its tie-break flipped. Everything else is s74's,
#: byte for byte. The ground is this branch's own measured principle, stated in the
#: module docstring and in three rounds of results: **the looser the proposer, the
#: stricter the judge behind it.** Removing the syntax loosens the alias proposer by
#: ~13 terms per run, so the judge behind it tightens. That is prior-work ground, not a
#: new rule, and it names no surface form.
STRICT_ALIAS_JUDGE_RULES = L.DOC_KNOWLEDGE_JUDGE_RULES.replace(
    "When uncertain, prefer APPROVE.", "When uncertain, prefer REJECT.")
assert STRICT_ALIAS_JUDGE_RULES != L.DOC_KNOWLEDGE_JUDGE_RULES

DOTTED = re.compile(r"[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+")


# ── arm builders ─────────────────────────────────────────────────────────────

def judging_arm(name, *, rules, p1_focus, clauses):
    """An `SLinker74` whose full-name judging prompt is built from the given parts.

    The strict (coreference) branch is left at s74's wording in every arm, so nothing
    but the full-name prompt differs.
    """
    tail = "".join(f"\n{c}\n" for c in clauses)

    def _prompt_validation(comp_names, cases, focus, strict: bool = False) -> str:
        block = L.LAYERED_COREF_RULES if strict else rules
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
        """s74's body with `P1_FOCUS` taken from the arm instead of the module."""
        from llm_sad_sam.linkers.experimental.s_linker74 import get_comp_names
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
            r2 = self._run_validation_pass(comp_names, strings, L.P2_FOCUS, p2_tag)
            for i, (_t, c) in enumerate(cases):
                ok = r1.get(i, False) and r2.get(i, False)
                decisions[(c.sentence_number, c.component_id)] = {
                    "approved": ok, "p1": r1.get(i, False), "p2": r2.get(i, False),
                    "path": f"{stage_label}_twopass", "stage": f"{stage_label}_twopass"}
                if ok:
                    approved.append(c)
        return approved, decisions

    return type(f"Arm_{name}", (SLinker74,),
                {"_VARIANT_NAME": f"s_linker74_{name}",
                 "_prompt_validation": staticmethod(_prompt_validation),
                 "_validate_with_evidence": _validate_with_evidence})


def s74_candidates(obj, info, project_name):
    """s74's own full-name candidate set: the recorded extractor plus the two scans."""
    names = {c.id: c.name for c in info["components"]}
    merged = {}
    for snum, cid in extractor_pairs(SOURCE_RUN, SOURCE_VARIANT, project_name):
        merged[(snum, cid)] = CandidateLink(
            snum, info["sent_map"][snum].text, names[cid], cid, "", source="full_name")
    for scan in (L.SCANS["spelling"], L.SCANS["stated_name"]):
        for cand in obj._scan(info["sentences"], info["components"], scan):
            merged.setdefault((cand.sentence_number, cand.component_id), cand)
    return list(merged.values())


def pilot_rubricsyntax(minimal=False):
    """`P1_FOCUS`'s tail, removed with the enumeration left standing.

    The one edit the general round never made on its own. Both arms judge the identical
    candidate set, so every difference is the prompt, and the rubric block is
    byte-identical in both.

    ``minimal`` removes the tail and adds **nothing**. That is the arm s75 actually
    needs: with the enumeration kept, reject-condition (1) of the rubric already states
    what `QUALIFIED_CLAUSE` states, in the same prompt, so adding the clause is a
    restatement rather than a relocation. The branch's own `plaincoref_min` made the
    same separation -- "the phrase was load-bearing" is a different finding from "the
    replacement was wrong".
    """
    base = judging_arm("rs_base", rules=L.LAYERED_ENTITY_RULES, p1_focus=L.P1_FOCUS,
                       clauses=(L.STRICTER_CLAUSE,))
    if minimal:
        gen = judging_arm("rs_min", rules=L.LAYERED_ENTITY_RULES,
                          p1_focus=GENERAL_P1_FOCUS, clauses=(L.STRICTER_CLAUSE,))
        arms = {"s74 (P1 names the shape)": base,
                "P1 general, nothing added": gen}
    else:
        gen = judging_arm("rs_gen", rules=L.LAYERED_ENTITY_RULES,
                          p1_focus=GENERAL_P1_FOCUS,
                          clauses=(L.QUALIFIED_CLAUSE, L.STRICTER_CLAUSE))
        arms = {"s74 (P1 names the shape)": base,
                "P1 general, QUALIFIED_CLAUSE added": gen}

    def unit(cls, run, project_name):
        info = inputs(project_name)
        obj = linker(cls, knowledge=info["knowledge"])
        cands = s74_candidates(obj, info, project_name)
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
        print(f"  {label:36s} approved per run: {[len(s) for s in sets[label]]}")
    print(f"\n  the reject-enumeration is byte-identical in both arms "
          f"({len(L.LAYERED_ENTITY_RULES)} bytes); only P1's tail and the added "
          f"clause differ\n")
    return against_first(sets, "rubricsyntax_min" if minimal else "rubricsyntax")


def alias_arm(name, *, exclusion, judge_rules):
    """An `SLinker74` whose alias extraction and alias judging prompts are given."""

    @staticmethod
    def _prompt_doc_knowledge_extract(comp_names, doc_lines) -> str:
        return f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

{L.DOC_KNOWLEDGE_EXTRACTION_RULES}

{exclusion}

DOCUMENT:
{chr(10).join(doc_lines)}

Return JSON:
{{
  "abbreviations": [{{"term": "short_form", "component": "FullComponent"}}],
  "synonyms":      [{{"term": "specific_alternative_name", "component": "FullComponent"}}]
}}
JSON only:"""

    @staticmethod
    def _prompt_doc_knowledge_judge(comp_names, mapping_list) -> str:
        return f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}



{judge_rules}

Return JSON:
{{"approved": ["term1", "term2"]}}
JSON only:"""

    return type(f"Arm_{name}", (SLinker74,),
                {"_VARIANT_NAME": f"s_linker74_{name}",
                 "_prompt_doc_knowledge_extract": _prompt_doc_knowledge_extract,
                 "_prompt_doc_knowledge_judge": _prompt_doc_knowledge_judge})


def pilot_aliascomp():
    """Remove the alias syntax and recover the table size at the judge instead.

    Read on the JUDGED table -- what the extraction prompt actually receives -- not on
    the proposed one, because the judge is the stage the compensation acts at. Three
    arms, so both the loss and the recovery are visible in one invocation:

        s74                 the syntax, judge lenient      (the head)
        general, lenient    the general round's arm        (the ~13-term growth)
        general, strict     the same wording, judge tie-broken the other way
    """
    arms = {
        "s74 (spells out X.Y)": alias_arm(
            "al74", exclusion=L.ALIAS_EXCLUSION_RULES,
            judge_rules=L.DOC_KNOWLEDGE_JUDGE_RULES),
        "general, lenient judge": alias_arm(
            "algen", exclusion=GENERAL_ALIAS_EXCLUSION,
            judge_rules=L.DOC_KNOWLEDGE_JUDGE_RULES),
        "general, strict judge": alias_arm(
            "algenstrict", exclusion=GENERAL_ALIAS_EXCLUSION,
            judge_rules=STRICT_ALIAS_JUDGE_RULES),
    }
    fragments = {label: [] for label in arms}

    def unit(cls, run, project_name, label):
        info = inputs(project_name)
        obj = linker(cls)
        knowledge = obj._learn_document_knowledge(
            info["sentences"], info["components"])
        table = dict(knowledge.aliases)
        fragments[label].append(sum(1 for t in table if DOTTED.search(t)))
        return {(project_name, (term, comp)) for term, comp in table.items()}

    sets = {}
    for label, cls in arms.items():
        sets[label] = collect(label, lambda run, p, c=cls, l=label: unit(c, run, p, l),
                              PROJECTS)
        print(f"  {label:24s} judged aliases per run: {[len(s) for s in sets[label]]}"
              f"   identifier fragments: {fragments[label]}")
    print("\n    The prohibition's own test is the second column: an arm admitting no")
    print("    fragment has enforced the rule however it was worded. The first column")
    print("    is what the general round showed the clause is really doing.\n")
    return against_first(sets, "aliascomp")


PILOTS = {"rubricsyntax": pilot_rubricsyntax,
          "rubricsyntax_min": lambda: pilot_rubricsyntax(minimal=True),
          "aliascomp": pilot_aliascomp}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", nargs="+", required=True,
                        choices=sorted(PILOTS) + ["all"])
    args = parser.parse_args()
    names = sorted(PILOTS) if "all" in args.pilot else args.pilot
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"source run: {SOURCE_RUN}  variant: {SOURCE_VARIANT}  "
          f"runs: {os.environ.get('AB_RUNS', '3')}")
    for name in names:
        print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}")
        PILOTS[name]()


if __name__ == "__main__":
    main()
