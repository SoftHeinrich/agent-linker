"""S-Linker103 — route a candidate by the evidence its sentence gives, not by the
stage that proposed it.

The head sends every candidate the coreference resolver proposes to the
coreference judge. That judge is asked whether the sentence *refers back* to the
component, and it answers that question faithfully -- including when the answer is
"no, because the sentence names the component outright":

    "objection": "the Logic component is named directly in the sentence rather
                  than being a referring expression to an unnamed component",
                  "approve": false

On teammates alone that objection rejects `Logic` 13 times and `Storage` 6 times
in one recorded run, while the full-name judge approves the same two components 33
and 24 times. The claims are true; they are refused for being the wrong *kind* of
claim. Whenever the extractor did not independently propose such a pair, it is
lost -- and the loss is invisible downstream, because a rejection leaves no trace.

That is where the head's remaining recall goes. Composed on terra, the head has 8
false negatives across five projects and **6 of them are on sentences that write
the component's name**, which is the population this objection covers.

The correction adds nothing and asks nothing new. Which sentences write a name is
a fact about the case, the deterministic layer already computes it, and this
branch's design law says facts stay in code. So a candidate whose sentence states
the component's name is judged by the named-mention judge, whichever proposer
found it; only candidates whose sentence names nothing reach the coreference
judge, which is the question that judge is actually asked.

Both proposers stay blind to each other -- the routing happens after they have
both spoken, so the mechanism the reading round measured is untouched. No prompt
changes, no new LLM calls, and the coreference judge's own rule text is left
exactly as measured.

Under an F2 budget this is the move with the best price: at the head's operating
point a point of recall is worth +0.76 F2 and a point of precision +0.24.
"""

from __future__ import annotations

from llm_sad_sam.core.data_types_v2 import CandidateLink
from llm_sad_sam.linkers.experimental.s_linker90 import SLinker90


class SLinker103(SLinker90):
    """Evidence decides the judge; the proposer only decides the candidate."""

    _VARIANT_NAME = "s_linker103"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._split: dict | None = None
        print("SLinker103 (candidates routed by evidence, not by proposing stage)")

    def _resolved_split(self, sentences, components, name_to_id, sent_map):
        """Run the head's resolver once, then split its output by the name relation."""
        if self._split is not None:
            return self._split
        links, metadata = SLinker90._resolve_references(
            self, sentences, components, name_to_id, sent_map)
        named_extra, kept, kept_meta = {}, [], {}
        for link in links:
            snum, cid = link.sentence_number, link.component_id
            sent = sent_map.get(snum)
            cname = link.component_name
            if sent is not None and self._states_a_name(sent.text, cname):
                matched = self._find_exact_form(sent.text, cname) or cname
                named_extra[(snum, cid)] = CandidateLink(
                    snum, sent.text, cname, cid, matched, source="full_name")
            else:
                kept.append(link)
                if (snum, cid) in metadata:
                    kept_meta[(snum, cid)] = metadata[(snum, cid)]
        print(f"    Routed {len(named_extra)} resolver candidates to the named judge; "
              f"{len(kept)} stay with the refer-back judge")
        self._split = {"named": named_extra, "coref": kept, "metadata": kept_meta}
        return self._split

    def _extract_named_mentions(self, sentences, components, name_to_id, sent_map) -> dict:
        base = SLinker90._extract_named_mentions(
            self, sentences, components, name_to_id, sent_map)
        split = self._resolved_split(sentences, components, name_to_id, sent_map)
        added = 0
        for key, cand in split["named"].items():
            if key not in base:
                base[key] = cand
                added += 1
        print(f"    Named stream gained {added} rerouted candidates")
        return base

    def _resolve_references(self, sentences, components, name_to_id, sent_map):
        split = self._resolved_split(sentences, components, name_to_id, sent_map)
        return split["coref"], split["metadata"]

    def link(self, text_path, model_path, **kwargs):
        self._split = None
        return super().link(text_path, model_path, **kwargs)
