"""The conceptual warts left in `s_linker74`, and the one the fold law says should go.

The rounds so far removed rules. What is left is not a rule problem but a *concept*
problem -- places where the design does something a reviewer can fairly call inelegant
even though every piece of it is measured:

  C1  A REGEX CLASSIFIER COMPUTES A FEATURE FOR THE LLM ABOUT TEXT THE LLM IS READING.
      `_classify_mention_typed` labels each full-name case with one of five values --
      proper case standalone / lowercase mention / lowercase inside qualified name /
      via known alias / indirect -- and the evidence line carries it as `mention=...`.
      The deterministic layer's stated job is to decide *which pairs get asked about*.
      This is it answering part of the question as well. It is the last hand-built
      feature in the workflow, and three of its five values are now restated in English
      in the same prompt: `STRICTER_CLAUSE` is the case distinction and
      `QUALIFIED_CLAUSE` is the inside-an-identifier distinction.

  C2  `unique_owner` survives only because a judge is deliberately kept ignorant. The
      ground on record is "the denotation judge is target-blind by design". That is a
      workaround defended by a measurement, not a principle, and it leaves code
      answering a question that is a judgement.

  C3  the two-pass full-name judge asks two hand-chosen facets and ANDs them.

  C4  the alias table has two opposite jobs: it admits full-name candidates and
      suppresses partial-name ones.

  C5  stage order does epistemic work never stated as a claim -- `_union` is
      earlier-wins, so a lenient linker's admission is final and a stricter one never
      revisits it.

This file attacks C1, because the fold law already predicts the answer and predicts it
unevenly. The law: **a gate or a label folds into a judge's prompt exactly when that
judge is shown the information it reads.** Four of the five mention values are computed
from the sentence alone, which the judge has in front of it; `VIA_ALIAS` is computed from
the alias table, which the judge does not have. So the prediction is not "drop the label"
but "drop the four derivable values and keep the one that carries information the judge
lacks" -- and the arm below is built to separate those two outcomes rather than confound
them.

    conceptlabel   three arms: the label as computed; the label gone entirely; the four
                   derivable values replaced by a question the judge answers itself,
                   with the alias fact kept in the evidence.

    AB_RUNS=3 ../.venv/bin/python pilot/concept_pilots.py --pilot conceptlabel
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_pilots import collect                                     # noqa: E402
from bind_audit import PROJECTS, extractor_pairs                      # noqa: E402
from bind_pilots import against_first, linker                         # noqa: E402
from fold_pilots import OUT, SOURCE_RUN, SOURCE_VARIANT, inputs       # noqa: E402
from llm_sad_sam.core.data_types_v2 import CandidateLink              # noqa: E402
import llm_sad_sam.linkers.experimental.s_linker74 as L74             # noqa: E402


#: The claim step, with the realization question folded in. The judge already quotes the
#: words that state the claim; this asks it to say how the name is written first. It
#: names no shape and no component: the four alternatives are the four values the regex
#: computed, stated as things a reader of the sentence can see.
REALIZATION_STEP = (
    "For each case, first say how the sentence writes the component's name -- as the\n"
    "catalog spells it, in a different case, or only inside a longer identifier -- and\n"
    "write \"not written\" if the sentence does not write it at all. Then quote the EXACT\n"
    "words from the sentence that state the architectural claim about the component (or\n"
    "write \"none\" if the sentence makes no such claim), then decide approve true/false\n"
    "based on that claim."
)

BASE_STEP = (
    "For each case, first quote the EXACT words from the sentence that state the\n"
    "architectural claim about the component (or write \"none\" if the sentence makes no\n"
    "such claim), then decide approve true/false based on that claim."
)


def label_arm(name, *, mode):
    """`mode`: 'computed' (s74), 'gone', or 'judged' (derivable values asked for)."""
    from llm_sad_sam.linkers.experimental.s_linker74 import SLinker74

    step = REALIZATION_STEP if mode == "judged" else BASE_STEP
    fields = ('{"validations": [{"case": 1, '
              + ('"realization": "<how the name is written>", ' if mode == "judged" else "")
              + '"claim": "<exact quote or none>", "approve": true}]}')

    def _format_evidence(self, bundle) -> str:
        head = f'  Evidence: source={bundle.source}, span="{bundle.matched_span}"'
        if mode == "computed":
            head += f", mention={bundle.mention_type}"
        elif mode == "judged" and bundle.mention_type == "via known alias":
            # the one value the judge cannot derive: it is a fact about the alias
            # table, not about the sentence in front of it.
            head += ", the sentence uses another name the document establishes for it"
        lines = [head]
        if bundle.preceding_text:
            lines.append(f'  [prev: "{bundle.preceding_text}"]')
        if bundle.anchor_sentences:
            lines.append("  Anchors (confirmed refs):")
            for a in bundle.anchor_sentences:
                lines.append(f"    {a}")
        return "\n".join(lines)

    def _prompt_validation(comp_names, cases, focus, strict: bool = False) -> str:
        rules = L74.LAYERED_COREF_RULES if strict else L74.LAYERED_ENTITY_RULES
        tail = "" if strict else f"\n{L74.STRICTER_CLAUSE}\n"
        return f"""Validate components in a document. {focus}

COMPONENTS: {', '.join(comp_names)}

{rules}
{tail}
{step}

CASES:
{chr(10).join(cases)}

Return JSON:
{fields}
JSON only:"""

    return type(f"Arm_{name}", (SLinker74,),
                {"_VARIANT_NAME": f"s_linker74_{name}",
                 "_format_evidence": _format_evidence,
                 "_prompt_validation": staticmethod(_prompt_validation)})


def s74_candidates(obj, info, project_name):
    names = {c.id: c.name for c in info["components"]}
    merged = {}
    for snum, cid in extractor_pairs(SOURCE_RUN, SOURCE_VARIANT, project_name):
        merged[(snum, cid)] = CandidateLink(
            snum, info["sent_map"][snum].text, names[cid], cid, "", source="full_name")
    for scan in (L74.SCANS["spelling"], L74.SCANS["stated_name"]):
        for cand in obj._scan(info["sentences"], info["components"], scan):
            merged.setdefault((cand.sentence_number, cand.component_id), cand)
    return list(merged.values())


def pilot_conceptlabel():
    arms = {
        "s74 (regex computes the label)": label_arm("lab_base", mode="computed"),
        "label gone entirely": label_arm("lab_off", mode="gone"),
        "judge states the realization": label_arm("lab_judged", mode="judged"),
    }

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
        print(f"  {label:32s} approved per run: {[len(s) for s in sets[label]]}")
    return against_first(sets, "conceptlabel")


PILOTS = {"conceptlabel": pilot_conceptlabel}


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
