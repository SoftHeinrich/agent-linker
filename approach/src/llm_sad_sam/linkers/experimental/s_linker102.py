"""S-Linker102 — the proposer stops authoring a set and judges membership instead.

The reading round measured that a single pass returns about one answer per item
however it is batched, obliged, windowed or instructed, and that the deficit lands
entirely on items with more than one correct answer. The 2026 literature names
this failure and prices it far outside our own measurement:

* **Silent omissions** (arXiv 2608.01000). Models judge set *membership* far
  better than they *author* the set: F1 0.60-0.77 judging against 0.26-0.48
  authoring, a +0.25 to +0.34 gap that **does not close over a 24x parameter
  range**. Planted over-inclusions are detected 6-7x more often than planted
  omissions -- so an omission is invisible to any later review, which is exactly
  why our three judges cannot repair a proposer's miss.
* **Extraction under-generation** (arXiv 2606.25656). Of answer entities already
  present in the prompt, only 57.4% are extracted, and one named cause is that
  singular query phrasing yields single-entity output even where several are
  valid.

Our own rung H tried to fix authoring with phrasing -- "report every component,
do not stop at the first" -- and moved the deficit metric 14.0 to 14.3 of 26.
The literature's own prompt-phrasing interventions likewise moved F1 by -0.06 to
+0.10 while the gap persisted. Phrasing is not the lever; the output *shape* is.

So this variant removes authoring from the proposal stage. The candidate set is
enumerated in code -- it always was, the components are given -- and the model is
asked, for each item and each candidate, a membership question it must answer.
Every component must appear in one of two lists per case -- referenced or
not-referenced -- so an omission is not expressible: there is no shorter answer
that is still well formed. The not-referenced list carries bare names, which keeps
the reply short enough to finish. An earlier version demanded a full decision
object per component; on a 12-component document the reply overflowed and came
back empty, which is why the cheap half of the checkbox is names only.

This is the general principle the round has been circling, in its strongest form:

    never let recall depend on an LLM enumerating a set.
    enumerate externally, and let the model judge membership.

Routing, evidence, all three judges, the alias module and the deterministic scan
are inherited untouched; only the shape of the proposal reply changes.
"""

from __future__ import annotations

import json

from llm_sad_sam.linkers.experimental.s_linker90 import (
    COREF_RULES,
    ENTITY_EXTRACTION_RULES,
    QUALIFIED_CLAUSE,
)
from llm_sad_sam.linkers.experimental.s_linker97 import SLinker97


class SLinker102(SLinker97):
    """The proposal reply is a decision per candidate, not a list of finds."""

    _VARIANT_NAME = "s_linker102"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"SLinker102 (membership decisions, batch {self.READING_BATCH})")

    def _prompt_reading(self, comp_names, mappings, batch, established, table=None) -> str:
        carried = (
            "ESTABLISHED EARLIER (component: the last sentence before this block "
            "that named it)\n" + json.dumps(established)
            if established else ""
        )
        blocks = [
            f"--- Case {i} ---\nTARGET S{s.number}: {s.text}"
            for i, s in enumerate(batch, 1)
        ]
        return f"""Decide, for every sentence below and every component listed, whether that sentence references that component.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings)}' if mappings else ''}
{carried}

A sentence references a component in one of two ways.

"names" — the sentence writes one of the component's names itself.

{ENTITY_EXTRACTION_RULES}

{QUALIFIED_CLAUSE}

"refers_back" — the sentence refers to the component without writing any of its
names, through a pronoun or a noun phrase. Give the earlier sentence that states
the name it refers back to.

{COREF_RULES}

Work through the cases in order. For each case, go through the COMPONENTS list in
order and account for every component: list the ones the sentence references under
"references", and list the names of all the others under "not_referenced". A case
is complete only when every component appears in exactly one of the two lists.

{chr(10).join(blocks)}

Return JSON, one entry per case, every component accounted for:
{{"cases": [{{"case": 1, "sentence": N_INTEGER, "references": [{{"component": "Name", "kind": "names_or_refers_back", "matched_text": "text found in sentence", "antecedent_sentence": M_INTEGER_OR_NULL}}], "not_referenced": ["OtherName", "AnotherName"]}}]}}
JSON only:"""

    def _ask(self, prompt, **kwargs):
        """The membership reply is flattened to the shape the reading loop reads.

        The loop above is inherited verbatim; only the wire format differs, so the
        translation happens here rather than by restating the loop -- a restated
        loop is where this round's one real bug hid.
        """
        if kwargs.get("require_present") != "references":
            return super()._ask(prompt, **kwargs)
        kwargs["require_present"] = "cases"
        data = super()._ask(prompt, **kwargs)
        refs = []
        for case in (data or {}).get("cases", []) or []:
            snum = case.get("sentence")
            for d in case.get("references", []) or []:
                refs.append({
                    "sentence": d.get("sentence", snum),
                    "component": d.get("component"),
                    "matched_text": d.get("matched_text") or "",
                    "antecedent_sentence": (
                        d.get("antecedent_sentence")
                        if d.get("kind") == "refers_back" else None),
                    "antecedent_text": d.get("antecedent_text") or "",
                })
        return {"references": refs}
