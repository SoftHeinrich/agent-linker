"""S-Linker126 — greedy name ownership on the exact antecedent arm.

The inherited coreference change remains: the resolver has no antecedent shortlist and
a resolution reaches its judge only when its cited antecedent writes the whole component
name or a document-established short form.

The name stream now gives a written whole catalog name ownership of the words it
contains. A contained word does not separately propose a different component. If one
remaining surface reaches several components, all proposals for that surface are
discarded because the written evidence does not identify one component.

The ownership predicate is the catalog-only form previously measured by s109/s121. A
fixed-input replay over six s122 runs removes 72 candidates, 0 gold, and one cached false
positive. Twelve residual groups remain, all BBB S27/S31 and all non-gold. Discarding
them is a semantic change, so the cached replay is screening evidence only; paired
end-to-end evaluation is required. See ``results/s127_greedy_merge``.
"""
from __future__ import annotations

from llm_sad_sam.linkers.experimental import s_linker123 as base
from llm_sad_sam.linkers.experimental.s_linker123 import NameForm
from llm_sad_sam.linkers.experimental.s_linker125 import SLinker125

_COMPETITORS_LINE = (
    "  competitors -- other components whose names carry the same word. They are "
    "what the expression could be reaching instead of this one.")
TRACE_LINK_RULE = base.TRACE_LINK_RULE.replace(_COMPETITORS_LINE + "\n", "", 1)
assert TRACE_LINK_RULE != base.TRACE_LINK_RULE

class SLinker126(SLinker125):
    """s125 with greedy unambiguous names and the exact-antecedent refusal."""

    _VARIANT_NAME = "s_linker126"
    ANTECEDENT_FORMS = ("whole name", "short form")

    def _covering_names(self, text, name, components):
        spans = []
        for component in components:
            if component.name != name:
                spans.extend(self._name_spans(
                    text, component.name, NameForm.ANY_CASE))
        return spans

    def _only_inside_another_name(self, text, name, components) -> bool:
        """Whether every matched word is owned by another written whole name."""
        mine = self._name_spans(text, name, NameForm.ANY_WORD)
        covering = self._covering_names(text, name, components)
        return bool(mine and covering) and all(
            any(start <= a and b <= end and end - start > b - a
                for start, end in covering)
            for a, b in mine)

    def _scan(self, sentences, components):
        return [candidate for candidate in super()._scan(sentences, components)
                if not self._only_inside_another_name(
                    candidate.sentence_text, candidate.component_name, components)]

    def _union_evidence(self, candidate, components, sent_map):
        evidence = super()._union_evidence(candidate, components, sent_map)
        evidence["competitors"] = []
        return evidence

    def _prompt_union(self, comp_names, sentence_table, cases) -> str:
        prompt = super()._prompt_union(comp_names, sentence_table, cases)
        replaced = prompt.replace(base.TRACE_LINK_RULE, TRACE_LINK_RULE, 1)
        assert replaced != prompt, "the inherited evidence rule was not found"
        return replaced

    @staticmethod
    def _group_key(candidate):
        return (candidate.sentence_number,
                (candidate.matched_text or candidate.component_name).casefold())

    def _judge_union(self, candidates, components, sentences, sent_map):
        grouped = {}
        for candidate in candidates:
            grouped.setdefault(self._group_key(candidate), []).append(candidate)
        singletons = [group[0] for group in grouped.values() if len(group) == 1]
        ambiguous = [group for group in grouped.values() if len(group) > 1]
        approved, decisions = super()._judge_union(
            singletons, components, sentences, sent_map)
        for group in ambiguous:
            for candidate in group:
                stage = self._stage_of(candidate)
                decisions[(candidate.sentence_number, candidate.component_id)] = {
                    "approved": False,
                    "claim": "",
                    "naming": "word only",
                    "path": f"{stage}_rejected",
                    "stage": "name_ambiguous_discard",
                }
        return approved, decisions

    def _antecedent_names_it(self, link, sent_map, metadata) -> bool:
        record = metadata.get((link.sentence_number, link.component_id), {})
        number = record.get("antecedent_sentence")
        sentence = sent_map.get(number) if number is not None else None
        if sentence is None:
            return True
        return self._written_as(
            sentence.text, link.component_name) in self.ANTECEDENT_FORMS

    def _validate_coref_links(self, coref_links, sent_map, components, metadata):
        return super()._validate_coref_links(
            [link for link in coref_links
             if self._antecedent_names_it(link, sent_map, metadata)],
            sent_map, components, metadata)
