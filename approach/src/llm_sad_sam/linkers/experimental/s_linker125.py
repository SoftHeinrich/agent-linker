"""S-Linker125 — the coreference resolver's antecedent shortlist, removed entirely.

`s_linker122`'s resolver prints a per-case list of the components the sentences above it
name, and a paragraph telling the model that list "has already been checked against the
document, so it is where the antecedent will be if there is one". This variant removes
both: the scan is not run, the list is not printed, and no sentence of the prompt speaks
about one. The resolver reads the sentences and decides.

**Why the list was suspected.** `s_linker124` marked the same list with the union judge's
verdicts and the doc-code gate refused it (`../results/coref_annot_round/README.md`). The
diagnosis was not mis-discrimination: the gold component was among the resolver's own
`candidates` in **0** of the surviving false positives, and 10 of the mark's 15 extra
false positives were in cases with a SINGLE candidate — no competition at all, only the
decision whether to attach. The mark was an attachment prior, and the paragraph above is
another one in authored text, so removing the list is the arm that tests whether the
mechanism helps at all.

**What the stage pilot measured, and it does NOT say this variant wins**
(`pilot/coref_shortlist_pilots.py`, three samples a model, alias table and name-link set
pinned, `../results/coref_shortlist_{terra_20260914,luna_20260915}`):

    terra   head F1 95.92 / F2 95.92   noshortlist F1 95.52 / F2 95.84
    luna    head F1 90.13 / F2 92.77   noshortlist F1 90.53 / F2 92.81

**The two models disagree on the sign** (terra F1 -0.40, luna +0.40), which by this
branch's standing rule is INSIDE NOISE and not an improvement. What the arm has is a
simplicity claim, and that is what it is to be read on:

* a whole mechanism gone — `_named_before`, `_states_a_name`'s live caller here, and the
  paragraph that endorses its output;
* the resolver prompt shrinks by **-8 680 B** on bigbluebutton and **-18 327 B** on
  teammates a run, the largest single cut left in that call;
* no authored rule constant changes, so GATE-07's accounting does not move.

**What it costs, stated because it is the uncomfortable half.** The resolver proposes far
more without the list — gold proposals **+69%** on terra (74.0 -> 125.0 a run) and
**+53%** on luna (72.7 -> 111.0) — and the judge is then asked about all of it, so calls
go **47.0 -> 49.7** (terra) and **49.0 -> 53.3** (luna). **The extra gold does not reach
the output**: net gold is 15.0 -> 14.7 and 15.7 -> 14.7, because `link` merges by pair
with the earlier linker winning and the name stage already holds those pairs. Net
spurious moves 1.3 -> 2.7 on terra and 4.7 -> 2.7 on luna — opposite directions, which is
the sign flip seen from the stage side.

**The finding that outlives whichever arm is chosen.** The coreference stage contributes
**~15 gold links a run whatever the resolver is shown**. The shortlist is not what limits
its recall; the merge and the judge are. That is why `s_linker124` had no headroom to win
and could only move false positives, and it is the reason to read this arm on simplicity
rather than on quality.

Two overrides, no signature changes. Measurements: `../results/coref_annot_round/README.md`.
"""
from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker123 import SLinker123

#: The per-case line the shortlist is rendered on, with the list emptied by the override
#: below so the ancestor renders its own "none" placeholder and this file removes it.
_CASE_LINE = "NAMED BEFORE THIS CASE:"

#: The paragraph that introduces the list and endorses it. Removed whole: a prompt that
#: no longer prints a list must not keep explaining one, and the second sentence is the
#: attachment prior the round is testing.
_SHORTLIST_PARA = (
    "Each case lists NAMED BEFORE THIS CASE: the components the sentences above it\n"
    "actually name, with the sentence that names each, nearest first. That list has\n"
    "already been checked against the document, so it is where the antecedent will be if\n"
    "there is one. Quote the referring expression first, then say which entries of that\n"
    "list could be what it points to, then name the one it does point to.")

#: What replaces it. The procedural half survives — quoting the referring expression
#: first is what makes the reply auditable and is not about the list — and nothing is
#: added that the ancestor does not also say.
_NO_LIST_PARA = (
    "Quote the referring expression first, then name the component it points to.")


class SLinker125(SLinker123):
    """`s_linker123` with no antecedent shortlist in the resolver's case, at all."""

    _VARIANT_NAME = "s_linker125"

    def _named_before(self, comp_names, sentence_table, target):
        """HEAD DELTA 1/2 — the scan is not run.

        Returning nothing rather than deleting the call: the ancestor renders its own
        `none` placeholder for an empty list, which `_prompt_coref` below then removes,
        so the two overrides cannot disagree about whether a list exists.
        """
        return []

    def _prompt_coref(self, comp_names, sentence_table, targets) -> str:
        """HEAD DELTA 2/2 — the line and the paragraph about it, both gone.

        Surgery on the ancestor's own prompt rather than a re-declared f-string, so a
        drift in the ancestor is a drift here; both slices assert they hit.
        """
        prompt = super()._prompt_coref(comp_names, sentence_table, targets)
        without = "\n".join(line for line in prompt.splitlines()
                            if not line.startswith(_CASE_LINE))
        assert without != prompt, "no shortlist line was found to remove"
        assert _SHORTLIST_PARA in without, "the ancestor's shortlist paragraph moved"
        return without.replace(_SHORTLIST_PARA, _NO_LIST_PARA, 1)
