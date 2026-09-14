"""S-Linker123 — the evidence fields merged into one readable field, on `s_linker122`.

`s_linker122` hands the union judge three pieces of evidence computed from the match
and names them `writes`, `alternatives` and `mention`. Replaying every recorded
judging call of the s122 round (184 calls, 3753 cases, `pilot/written_field_audit.py`)
shows two defects in that vocabulary, neither of them cosmetic:

* **One fact is printed twice.** `mention=via known alias` fires on exactly the 598
  cases where `writes` already says "a short form the document established for it" --
  598 of 598, in both directions. The judge is told the same thing in two wordings.
* **`mention` has two reachable values of the five `MentionType` declares**, and the
  other one, `lowercase, inside qualified name`, can only ever co-occur with the whole
  name: it is returned when `_find_exact_form` matches, which is the same `ANY_CASE`
  predicate `_writes_name` decides `naming` with. So the fields are not independent,
  and a reader -- or a judge -- who takes three field names for three facts is being
  misled by the format.

**What this variant does.** One field, `written`, whose four values are exactly the
partition the replay measured -- `whole name` (46.0% of cases), `one word` (29.1%),
`short form` (15.9%), `whole name (qualified)` (9.0%) -- computed by one function
instead of two predicates that have to agree. `alternatives` becomes `competitors`,
which answers "alternatives to what?". The evidence line is then two fields: one
always present, one when the match found it.

It is staged on `s_linker122` and keeps that variant's feature set whole: the anchor
block stays removed and `SURFACE_NOT_EVIDENCE` stays in the call. The rule is computed
off `s_linker122.TRACE_LINK_RULE`, so the clause and everything else the head carries
arrive here unchanged and a drift there is a drift here.

The identifier fact is folded into a *value* rather than kept as a field because the
containment is entailed, not merely observed: `qualified` implies the whole name is
written, since both go through `_find_exact_form`. That entailment has one
precondition -- `SKIP_QUALIFIED = False`, which makes `_writes_name` and
`_find_exact_form` the same relation -- and `_written_as` asserts it.

Parenthesised, not comma-separated: the evidence line separates fields with commas, so
`written=whole name, qualified` would be indistinguishable from two fields, to a
reader and to every tool that parses these lines.

**What it costs.** Judging bytes: the evidence line 189 325 -> 125 912 B over the
replayed cases (-33.5%), a mean -345 B a call; the rule's field block goes from three
lines to two but absorbs one sentence defining "qualified", +37 B a call. **Net -9434 B
a five-project run.** The readability is the point; the bytes are a side effect.

**What has to be measured and what does not.** Nothing about the candidate set moves --
the scan is untouched, which `pilot/written_field_audit.py --verify` checks by
rendering both arms over the recorded alias tables. What the judge does with the new
wording is a prompt change, so it is a stage arm. Read it on the `short form` row
first: that row keeps at 0.741 against a gold rate of 0.532, over-approving by ~0.20
on both models, and it is the row whose duplicate line this variant removes.

Measurements: `../results/s123_written_field/README.md`.
"""
from __future__ import annotations

import re

from llm_sad_sam.linkers.experimental import s_linker122 as base
from llm_sad_sam.linkers.experimental.s_linker122 import NameForm, SLinker122

#: The one field's line in the rule, replacing `writes`. It carries the sentence that
#: defines the fourth value, which is why the block loses a line and gains a sentence.
_WRITTEN_LINE = (
    "  written -- how much of the component's name this sentence writes: the whole "
    "name, a short form the document established for it, or one word of the name. "
    "Qualified means the whole name appears, but every writing of it sits inside a "
    "longer joined or dotted identifier. A shorter surface leaves more readings open; "
    "it does not make the reading in front of you wrong. Where the sentence does not "
    "write the name as such, ask what the expression itself denotes in its local "
    "context: a participant in the system, or something merely associated with "
    "software.")

#: What was `alternatives`, renamed and otherwise left alone. The `mention` line goes
#: with the field: a line about a field no case carries measures something else.
_COMPETITORS_LINE = (
    "  competitors -- other components whose names carry the same word. They are "
    "what the expression could be reaching instead of this one.")


def _rule() -> str:
    """The head's rule with the field block swapped. Both slices assert they hit."""
    rule = base.TRACE_LINK_RULE
    for old, new in ((base._WRITES_LINE, _WRITTEN_LINE),
                     (base._FIELD_LINES, _COMPETITORS_LINE)):
        assert old in rule, f"the head's rule no longer contains {old[:40]!r}"
        rule = rule.replace(old, new, 1)
    return rule


TRACE_LINK_RULE = _rule()

#: Which evidence fields a case may print, in the order it prints them.
UNION_FIELDS = ("written", "competitors")

#: The recorded decision-record projection. Six pilots and every recorded phase state
#: read `naming` with the head's three values, so the record keeps them -- but here
#: they are *derived from* `written` rather than computed beside it, which is the whole
#: point of the merge: the two cannot disagree.
NAMING_OF = {
    "whole name": "whole name",
    "whole name (qualified)": "whole name",
    "short form": "alias",
    "one word": "word only",
}


class SLinker123(SLinker122):
    """`s_linker122` with one merged evidence field in place of three named ones."""

    _VARIANT_NAME = "s_linker123"

    #: The four values, as printed. There is no second code vocabulary to map from.
    WRITTEN = ("whole name", "whole name (qualified)", "short form", "one word")

    def _only_in_identifier(self, text: str, name: str) -> bool:
        """Does every writing of the name in this sentence sit inside a longer path?

        `_mention_label`'s `CODE_TOKEN` arm, lifted out so the one fact that label
        still contributes is a predicate of its own rather than a value of an enum
        whose other four values are unreachable or redundant.
        """
        occurrences = list(re.finditer(rf"\b{re.escape(name.lower())}\b", text))
        return bool(occurrences) and all(
            self._in_dotted_path(text, m.start(), m.end()) for m in occurrences)

    def _written_as(self, text: str, name: str) -> str:
        """How much of the component's name this sentence writes: one of `WRITTEN`.

        One function where the head had `naming` and `_mention_label`, which each
        re-derived the same `ANY_CASE` match and agreed only because they happened to
        call the same helper.
        """
        assert not self.SKIP_QUALIFIED, (
            "`qualified` is a value of `written` because it entails the whole name; "
            "with SKIP_QUALIFIED set the two stop being the same relation")
        if self._writes_name(text, name):
            return ("whole name (qualified)" if self._only_in_identifier(text, name)
                    else "whole name")
        for term, owner in getattr(
                getattr(self, "doc_knowledge", None), "aliases", {}).items():
            if owner == name and self._find_exact_form(text, term):
                return "short form"
        return "one word"

    def _union_evidence(self, candidate, components, sent_map):
        """Every fact of the match this case carries. No weighing lives here.

        `written` is the whole of what the head split across `naming` and `mention`;
        `competitors` is the head's `alternatives` under a name that says what the
        alternatives are to. `naming` is not printed -- `UNION_FIELDS` names what is --
        and exists only so the decision record keeps the recorded runs' schema.
        """
        text = candidate.sentence_text
        name = candidate.component_name
        written = self._written_as(text, name)
        mine = {text[start:end].casefold() for start, end
                in self._name_spans(text, name, NameForm.ANY_WORD)}
        competitors = []
        for other in components:
            if other.name == name:
                continue
            spans = self._name_spans(text, other.name, NameForm.ANY_WORD)
            if spans and {text[s:e].casefold() for s, e in spans} & mine:
                competitors.append(other.name)
        return {
            "span": candidate.matched_text or name,
            "written": written,
            "competitors": competitors,
            "naming": NAMING_OF[written],
        }

    def _prompt_union(self, comp_names, sentence_table, cases) -> str:
        """The head's call, with the head's rule swapped for the merged-field one."""
        prompt = super()._prompt_union(comp_names, sentence_table, cases)
        swapped = prompt.replace(base.TRACE_LINK_RULE, TRACE_LINK_RULE, 1)
        assert swapped != prompt, "the head's rule was not found in its own prompt"
        return swapped

    def _format_union_case(self, index, candidate, evidence, sent_map):
        """One case: the span and its component, the sentence, and two fields.

        Every case is this shape. What a whole-name case and a one-word case differ in
        is the *value* of `written`, and whether the match found competitors at all.
        """
        previous = self._prev_prefix(candidate.sentence_number, sent_map)
        facts = [f"written={evidence['written']}"]
        if evidence["competitors"]:
            facts.append("competitors=" + ", ".join(evidence["competitors"]))
        return "\n".join([
            f'Case {index}: "{evidence["span"]}" -> {candidate.component_name}',
            f'  {previous}"{candidate.sentence_text}"',
            f"  Evidence: {', '.join(facts)}",
        ])
