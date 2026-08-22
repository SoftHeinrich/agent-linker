"""S-Linker94 — one reading pass proposes for all three linkers (ladder rung A).

The head asks the document two questions in two separate LLM stages: the
named-reference extractor ("which sentences write a component's name?") and the
coreference resolver ("which sentences refer back to one?"). They are the same
question at two reference forms, and the recorded runs say so: **54% of the
extractor's pairs and 46% of the resolver's are proposed by both stages**, and
the pairs both propose are the best candidates either produces (precision 0.947
against the extractor's 0.905 and the resolver's 0.614). The resolver then spends
8 of a project's ~17 calls re-deriving pairs the extractor already had, which the
union discards.

This variant asks once. One reading pass over the document, batched at
``EXTRACTION_BATCH``, reports every reference it finds and — for the ones the
sentence does not name outright — the earlier sentence that supplies the name.
Each claim is then routed to the judge its own evidence selects:

    antecedent reported   -> the coreference judge, unchanged
    no antecedent         -> the named-mention judge, unchanged
    deterministic scan    -> the denotation judge, unchanged

**Nothing about judging changes.** The three judges, their asymmetric defaults,
their batch sizes and their evidence formats are inherited from ``SLinker90``
untouched, and the alias module is untouched. This is deliberate: every merge the
s26-s35 line refuted folded *judging* or *alias discovery* into extraction, and
the standing finding — every consolidation of two LLM decisions into one call
raises recall and lowers precision — was earned on those. Merging the two
*proposal* stages with each other has never been tried on this branch.

Two measured facts license the merge:

* **Anchors are local.** Over 414 recorded resolutions the antecedent sits a
  median of 2 sentences back (mean 2.7, max 14); only **1.0%** fall outside a
  50-sentence block. The reading batch is wide enough to resolve what the
  dedicated resolver resolves.
* **The gap that remains is carried, not lost.** For the 1% that reach back
  further, and across every block boundary, the pass carries a per-component
  note of the last sentence that named it — the smallest form of the per-entity
  slot that let Recurrent Entity Networks (ICLR 2017) solve bAbI in one pass
  where a three-pass episodic reader needed three.

No authored rule text is added. The reading prompt composes
``ENTITY_EXTRACTION_RULES``, ``QUALIFIED_CLAUSE`` and ``COREF_RULES`` verbatim
from the head, so GATE-07's byte accounting is unchanged.
"""

from __future__ import annotations

import json

from llm_sad_sam.core.data_types_v2 import CandidateLink, SadSamLink
from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names, parse_snum
from llm_sad_sam.linkers.experimental.s_linker90 import (
    COREF_RULES,
    ENTITY_EXTRACTION_RULES,
    QUALIFIED_CLAUSE,
    SLinker90,
)


class SLinker94(SLinker90):
    """One reading pass; three judging routes."""

    _VARIANT_NAME = "s_linker94"

    #: Sentences per reading call. The merged question inherits the *naming*
    #: question's granularity by default; ``s_linker96`` asks it at the
    #: resolution question's granularity instead, which is the axis the stage
    #: pilot showed the merge actually turns on.
    READING_BATCH = SLinker90.EXTRACTION_BATCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reading: dict | None = None
        print("SLinker94 (one reading pass -> three judging routes)")

    # ── the single reading pass ──────────────────────────────────────────────

    def _prompt_reading(self, comp_names, mappings, batch, established, table=None) -> str:
        """One block of the document, one question, both reference forms.

        ``established`` is the carried per-component note: the last sentence of an
        earlier block that named each component. It is what makes a block-local
        read able to resolve a reference whose antecedent is behind it.
        """
        carried = (
            "ESTABLISHED EARLIER (component: the last sentence before this block "
            "that named it)\n" + json.dumps(established)
            if established else ""
        )
        return f"""Extract ALL references to components from this document.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings)}' if mappings else ''}
{carried}

{ENTITY_EXTRACTION_RULES}

{QUALIFIED_CLAUSE}

A sentence may also refer to a component without writing any of its names, through
a pronoun or a noun phrase that refers back. Report those too, and for each give
the earlier sentence that states the name it refers back to. A sentence that
writes the name needs no antecedent; leave it null.

{COREF_RULES}

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence", "antecedent_sentence": M_INTEGER_OR_NULL, "antecedent_text": "exact quote naming the component, or null"}}]}}
JSON only:"""

    def _read_document(self, sentences, components, name_to_id, sent_map) -> dict:
        """Read once. Cached: every linker below draws from this one pass."""
        if self._reading is not None:
            return self._reading

        comp_names = get_comp_names(components)
        mappings = (
            [f"{term}={component}"
             for term, component in self.doc_knowledge.aliases.items()]
            if self.doc_knowledge else []
        )
        named: dict = {}
        coref: list[SadSamLink] = []
        metadata: dict = {}
        established: dict[str, int] = {}

        self.llm.set_phase("phase_25_reading")
        for batch_num, batch in self._iter_batches(sentences, self.READING_BATCH):
            if len(sentences) > self.READING_BATCH:
                print(f"    batch {batch_num}: "
                      f"S{batch[0].number}-S{batch[-1].number} ({len(batch)} sents)")
            window_ids: set[int] = set()
            for sent in batch:
                window_ids.update(w.number for w in self._window(sent.number, sentences))
            table = [{"sentence": n, "text": sent_map[n].text}
                     for n in sorted(window_ids) if n in sent_map]
            data = self._ask(
                self._prompt_reading(comp_names, mappings, batch, established, table),
                timeout=600, label=f"Reading batch {batch_num}",
                require_present="references",
            )
            first = batch[0].number
            for ref in (data or {}).get("references", []) or []:
                cname = ref.get("component")
                snum = parse_snum(ref.get("sentence"))
                if snum is None or not cname or cname not in name_to_id:
                    continue
                sent = sent_map.get(snum)
                if sent is None:
                    continue
                cid = name_to_id[cname]
                key = (snum, cid)
                ant = parse_snum(ref.get("antecedent_sentence"))
                # An antecedent is usable only if it is a real earlier sentence.
                # A number the model invents cannot name one -- the head's rule,
                # applied here at the point the claim is made.
                if ant is not None and not (0 < ant < snum and ant in sent_map):
                    ant = None

                # Route on the sentence, not on the model's choice of field. A
                # sentence that states a name of the component belongs to the
                # named judge whether or not a refer-back was also reported;
                # that is the same relation, read the same way, the head uses to
                # decide which linker a sentence is for.
                if self._states_a_name(sent.text, cname):
                    ant = None

                if ant is None:
                    matched = ref.get("matched_text", "") or ""
                    if matched and matched.lower() not in sent.text.lower():
                        continue          # a fabricated span warrants nothing
                    if key not in named:
                        named[key] = CandidateLink(
                            snum, sent.text, cname, cid, matched, source="full_name",
                        )
                elif key not in metadata:
                    coref.append(SadSamLink(snum, cid, cname, source="coreference"))
                    metadata[key] = {
                        "reference": ref.get("matched_text", ""),
                        "antecedent_sentence": ant,
                        "antecedent_text": ref.get("antecedent_text", "") or "",
                        "raw_resolution": ref,
                    }
                # carry the naming sentence forward for the blocks that follow
                if ant is None and snum >= first:
                    established[cname] = max(established.get(cname, 0), snum)

        print(f"    Read: {len(named)} named, {len(coref)} refer-back")
        self._reading = {"named": named, "coref": coref, "metadata": metadata}
        return self._reading

    # ── the three routes: proposal replaced, judging inherited ───────────────

    def _extract_named_mentions(self, sentences, components, name_to_id, sent_map) -> dict:
        """The named route's candidates come from the reading pass."""
        return self._read_document(
            sentences, components, name_to_id, sent_map)["named"]

    def _resolve_references(self, sentences, components, name_to_id, sent_map):
        """The refer-back route's candidates come from the same reading pass."""
        reading = self._read_document(sentences, components, name_to_id, sent_map)
        return reading["coref"], reading["metadata"]

    def link(self, text_path, model_path, **kwargs):
        self._reading = None          # one reading pass per document, not per run
        return super().link(text_path, model_path, **kwargs)
