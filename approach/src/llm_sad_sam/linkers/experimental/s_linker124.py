"""S-Linker124 — the two s123 mechanisms composed: one evidence vocabulary, one fact source.

Two rounds landed on `s_linker122` at the same time and touched different stages. Neither
is a superset of the other and they were priced on separate bases, so this file is the
composition and nothing else:

* **`s_linker123`** rewrites the union judge's evidence vocabulary — `writes` /
  `alternatives` / `mention` become `written` / `competitors` — because a replay of 184
  recorded judging calls showed one fact printed twice (598 of 598 alias cases) and a
  field with two reachable values of five. Read `s_linker123`'s own docstring for it;
  none of it is restated here.
* **This file's delta** marks the coreference resolver's antecedent shortlist with the
  verdict that same judge already reached about each mention:

      NAMED BEFORE THIS CASE: kurento (S68, linked), WebRTC-SFU (S68, named only)

  `_named_before` computes the list from `_states_a_name` — a purely **lexical** fact —
  and the module was discarding the verdicts and offering every entry as an equal.

**Why they compose without interacting.** The first changes the *judge's* evidence format;
the second changes the *resolver's* case. They share no method, no constant and no prompt:
`pilot/test_s124.py` asserts the judging prompts are `s_linker123`'s byte for byte on all
five projects and that the resolver prompts differ from `s_linker123`'s only in the
`NAMED BEFORE THIS CASE` lines. **They are priced separately and NOT yet as a
composition** — see the honesty note at the end.

**The design argument for the mark.** The shortlist was the one place in the module where
a stage is shown an UNREFINED version of a fact the pipeline has already refined. Every
other piece of evidence any judge reads is the best the system knows at that point. One
fact source, stated once, read everywhere. No rule speaks about the mark and no authored
rule text changes — it is rendered in the shortlist line, not written into any constant —
so the GATE-07 accounting does not move.

**What the mark is priced at, without rounding up.** Level 2 on the `s_linker122` base,
three samples a side, both arms in one invocation per model, alias table and name-link set
pinned (`pilot/coref_annot_pilots.py`, `../results/coref_annot_{terra,luna}_20260914`):

    terra   macro F2 +0.31 (p 0.50)   macro F1 +0.04 (p 1.00)   TP +0.67   FP +0.67
    luna    macro F2 +0.04 (p 1.00)   macro F1 -0.06 (p 1.00)   TP +0.67   FP +0.67

Link-level QUALITY-NEUTRAL on both models with the F2 point estimate favourable on both,
at the same call count. **It is not a precision result**: the exchange rate is one gold
per one spurious, and F2's 4:1 recall weighting is what turns that into a positive number.
Nothing reaches the n = 3 sign-flip floor of p = 0.25.

**Why the mark alone, with no clause about it.** An arm that adds one sentence — "a
`named only` entry is the weaker antecedent" — is the WORSE arm on both models (terra
F2 -0.19, luna -0.16). The two move the resolver 32 pairs apart in opposite directions and
only the unweighted one lands favourably. **A fact can be enough; the design law says
where a weighing goes when you want one, not that you want one.**

**What is refused, and it is `s_linker109`'s rule from the other side.** An arm that DROPS
the refused entries instead of marking them reads luna TP -1.0, **FP +4.0, macro F2 -0.92**,
worse in three samples of three: it proposes 36 fewer pairs a run and its net spurious
doubles, **+17 false positives added against 5 removed**, because taking an entry off the
list does not make the resolver abstain — it makes the resolver attach the same referring
expression to the next component down. A name verdict is a *discovered* fact, resampled
every run: **it may open a case and it may not close one. Marking is opening; withholding
is closing, and this file marks.**

**The blindness that is not spent.** `s_linker100` conditioned the second proposer on the
first's OUTPUT LIST and added zero pairs in two of three samples. This does not: the
resolver still reads every sentence, proposes independently, and is never told which pairs
to produce. What it receives is a property of each candidate ANTECEDENT — evidence about a
case, not a proposal to copy.

**HONESTY NOTE, and it governs how this file may be cited.** Every number above is
`pilot/score_runs.py`, which is **LINK-LEVEL**, and each mechanism was measured on its own
base rather than composed. The read this branch promotes an arm on is
`studies/compare_arms.py`, at the **doc-code, component-weighted** grain. `s_linker122` is
the standing warning: link-level QUALITY-NEUTRAL, then terra doc-code F1 -1.02 / F2 -0.71
with 3/3 runs agreeing on the sign — because the cut changed *which components* the links
landed on, not how many. The mark's entire effect is +0.67 gold and +0.67 spurious a run,
which is that same quantity. **This is a head candidate, not a reported arm.** The paper
arm is `s_linker120`.

**THE GATE HAS SINCE BEEN RUN, AND IT REFUSED THIS ARM (2026-09-14).** Six paired E2E runs
against an in-set `s_linker123` control read, on terra, **dc F1 -1.67 with 3/3 runs
agreeing and dc worst F1 -2.33 with 3/3 agreeing**; every doc-code metric on both models
has a negative mean, and luna's single BETTER is at the doc-model grain, which does not
promote an arm (`../evaluation/reports/ARM_COMPARE_s124.csv`). The warning in the paragraph
above came true against this file: the effect *is* which components a handful of links land
on, and the component-weighted metric is where it shows. **Do not cite this variant as an
improvement at any grain.** It stays the head only in the sense that later rounds fork from
it. What it would take to revisit the mark honestly -- six paired runs a model, and a
harness that pins the name stage across arms so its resampling noise cannot enter the
doc-code read -- is in the round README.

Round: `../results/coref_annot_round/README.md`; the evidence vocabulary's round is
`../results/s123_written_field/README.md`. None of the measurements live here.
"""
from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker123 import SLinker123


class SLinker124(SLinker123):
    """`s_linker123`, with the antecedent shortlist carrying the judge's own verdicts.

    Two overrides and no signature changes. `_run_linker` is where the name stage's
    result becomes visible, and `_named_before` is the only method whose result the
    shortlist line reads — so decorating there changes the case and nothing else.
    """

    _VARIANT_NAME = "s_linker124"

    #: What an entry's mark says: the verdict the union judge reached about that very
    #: mention. Named after the verdict and not after a quality ("strong"/"weak") so the
    #: entry stays a statement of fact — whether a `named only` antecedent is worth less
    #: is a weighing, and the round measured that adding one costs.
    MARK_LINKED = "linked"
    MARK_NAMED_ONLY = "named only"

    #: The name stage's kept mentions, as {(sentence, component name)}. Empty until the
    #: name linker has run, which `LINKERS` guarantees happens first; a resolver that
    #: somehow ran without it would mark every entry `named only` rather than break.
    _linked_mentions: frozenset = frozenset()

    def _run_linker(self, linker, sentences, components, name_to_id, sent_map):
        """Dispatch, and remember what the name stage kept.

        Captured here rather than threaded through four signatures: the resolver does
        not take a link set as input, and it must not — what it receives is a property
        of each shortlist entry, not a list of pairs to produce.
        """
        produced, feedback = super()._run_linker(
            linker, sentences, components, name_to_id, sent_map)
        if linker == "name":
            self._linked_mentions = frozenset(
                (link.sentence_number, link.component_name) for link in produced)
        return produced, feedback

    def _named_before(self, comp_names, sentence_table, target):
        """HEAD DELTA — the shortlist, each entry carrying its own mention's verdict.

        The ancestor returns (name, sentence) for a mention the document writes. The
        sentence number is what carries the mark, so the name stays the catalog's own
        string and the reply contract is untouched.
        """
        return [
            (name, f"{number}, "
             f"{self.MARK_LINKED if (number, name) in self._linked_mentions else self.MARK_NAMED_ONLY}")
            for name, number in super()._named_before(
                comp_names, sentence_table, target)
        ]
