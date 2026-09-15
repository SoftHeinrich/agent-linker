"""S-Linker126 — the shortlist's contract, enforced in code instead of asserted in English.

The coreference resolver's prompt prints a per-case shortlist and then makes a claim
about it:

    That list has already been checked against the document, so it is where the
    antecedent will be if there is one.

**That is a claim the prompt cannot enforce, and the model does not always honour it.**
Over six recorded runs, seven surviving false positives cite an antecedent sentence that
does not write the component's name at all — six of them the same sentence, offered for
`HTML5 Server` because it mentions `bbb-html5`. Ten more cite a sentence that writes the
name only inside a longer dotted identifier (`Package overview contains storage.api,
storage.entity, storage.search.` offered for `Storage`). **Seventeen false positives and
zero gold** (`pilot/coref_written_audit.py`, per model: qualified 0/5 terra and 0/5 luna,
off-list 0/7 luna).

This variant stops asserting the constraint and starts enforcing it:

* **the shortlist goes** — the `NAMED BEFORE THIS CASE` line, the paragraph about it, and
  `_named_before`, whose only caller was that paragraph's line (`s_linker125`);
* **the contract stays**, as a predicate: a resolution whose cited antecedent sentence
  does not write the component's name AS A NAME is not put to the judge.

**Simplicity, and reuse of what is already computed.** `_written_as` is derived for every
union-judging case already (`s_linker123._union_evidence`), so the refusal adds no new
computation and no new concept — it reads a fact the pipeline had and was discarding.
`_states_a_name` keeps its other caller (the partial-name scan's whole-name exclusion), so
nothing is orphaned. Net: one method deleted, one paragraph deleted, up to 198 shortlist
lines and 18 327 B off the resolver on teammates a run, against one predicate of four
lines. No authored rule constant changes, so GATE-07's accounting does not move.

**Why this is allowed to CLOSE a case when `s_linker124`'s mark was not.** `s_linker109`'s
rule is that a discovered fact may open a case and may not close one. s124 marked the
shortlist with the union judge's VERDICT — another stage's output, resampled every run —
and the doc-code gate refused it. `written` is GIVEN: catalog plus document, identical
every run, no sampling in it. s109's own nesting predicate is the precedent for a given
fact ending a case.

**What each half measured on its own** (stage pilots, three samples a model, alias table
and name-link set pinned, so the name stage is not resampled into the comparison):

    the refusal, with the shortlist still there
      terra  TP 183.7 -> 183.7   FP 13.0 -> 12.3   F1 95.76 -> 95.87
      luna   TP 180.0 -> 180.0   FP 40.3 -> 38.3   F1 90.50 -> 90.70   (FP -2 in 3 of 3)

    the shortlist removed, no refusal (`s_linker125`)
      terra  F1 95.92 -> 95.52      luna  F1 90.13 -> 90.53      (the models disagree)

The refusal costs **zero** true positives on both models — exactly what the audit
predicted, since the rows it empties hold no gold. Removing the shortlist alone is a sign
flip, i.e. inside noise, and its one measured cost is net spurious (terra 1.3 -> 2.7): the
resolver without a list proposes more freely and cites antecedents further afield. **That
is precisely the population the refusal deletes**, which is the argument for composing
them rather than choosing between them — and it is a hypothesis this file does not get to
assume. See the round report for what the composition actually reads.

**The finding that frames both.** The coreference stage contributes **~15 gold links a run
whatever the resolver is shown** — removing the shortlist raises gold PROPOSALS by 69%
(terra) and 53% (luna) and moves net gold by less than one link, because `link` merges by
pair and the name stage already holds them. The shortlist never bought recall. So the only
question it was ever answering is precision, and a deterministic predicate answers that
one better than a sentence of English can.

One added method and one predicate on top of `s_linker125`. Measurements:
`../results/coref_annot_round/README.md`.
"""
from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker125 import SLinker125


class SLinker126(SLinker125):
    """`s_linker125` with the antecedent contract the deleted prompt used to assert."""

    _VARIANT_NAME = "s_linker126"

    #: What it is to be an antecedent: the sentence writes the component's name, or a
    #: short form the document established for it. The other two values of `written` are
    #: `whole name (qualified)` -- every writing sits inside a longer joined or dotted
    #: identifier -- and `one word`, which for an antecedent means the name is not there
    #: at all. Both held 0 gold over six recorded runs on both models.
    ANTECEDENT_FORMS = ("whole name", "short form")

    def _antecedent_names_it(self, link, sent_map, metadata) -> bool:
        """Does the sentence this resolution cites write the component's name as a name?

        Reads `_written_as`, which `_union_evidence` already computes for every judging
        case, so the fact is reused rather than re-derived.
        """
        record = metadata.get((link.sentence_number, link.component_id), {})
        number = record.get("antecedent_sentence")
        sentence = sent_map.get(number) if number is not None else None
        if sentence is None:
            # `_resolve_references` already refuses a resolution that cites no antecedent
            # sentence; this predicate answers a different question and not that one.
            return True
        return self._written_as(
            sentence.text, link.component_name) in self.ANTECEDENT_FORMS

    def _validate_coref_links(self, coref_links, sent_map, components, metadata):
        """HEAD DELTA — the contract applied before the judge is asked.

        Before rather than after, for `s_linker109`'s reason: a case the code can already
        answer should not be spent on a call. Nothing downstream can be starved by the
        removal -- coreference is the last linker -- so the composition risk the
        measurement policy's level 3 tests for is structurally zero here.
        """
        return super()._validate_coref_links(
            [link for link in coref_links
             if self._antecedent_names_it(link, sent_map, metadata)],
            sent_map, components, metadata)
