"""S-Linker110-NoEvidence — the head's judges, with the context code computed removed.

The RQ4 arm for the paper (**experimental, NOT canonical**). The head's claim is that
each judge rules on a *case that code assembled*: the words that matched, how the
sentence writes them, the sentence before, the other sentences that name the component,
and — since `s_linker110` — the components the document names ahead of the target. This
arm removes exactly that, and nothing else, so what the case is worth can be read off
the difference.

**The rule, stated once.** *Remove every context code computed; keep the case
answerable.* The second half is what makes this an ablation rather than a different
system: a judge shown nothing it can rule on measures impossibility, not evidence. The
rule lands differently on each of the three judges because each holds a different case,
which is the head's own design claim:

    judge           removed                                    why the case survives
    full name       the whole `Evidence:` block -- span,        the case header still
                    mention label, [prev], anchor sentences     writes "<span>" -> <Name>
                                                                above the sentence
    partial name    the +/-5 sentence window shrinks to the     the expression and the
                    candidate's own sentence                    sentence it sits in are
                                                                what the step classifies
    coreference     `NAMED BEFORE THIS CASE` and the paragraph  the SENTENCES window
                    that explains it; the strict judge's        stays, so the antecedent
                    [prev] and evidence block                   is still in front of the
                                                                model to be found

The coreference window is the one thing this arm keeps, deliberately. `_prompt_coref`
without it cannot resolve a refer-back at all -- "It also writes them back on exit" has
no recoverable referent -- so removing it would price the task's impossibility rather
than the shortlist's worth. What the arm removes there is the *checked* shortlist: the
model must still find the antecedent, but out of the window instead of out of a list
`_named_before` already verified against the document.

**What is NOT removed**, and why. The strict coreference judge's `Claimed reference` and
`Claimed antecedent` lines survive: they are the resolver's own reply carried forward,
not a fact code computed, and the rule speaks only about the latter. The partial-name
step's withheld target survives too -- withholding it is a separate design decision the
head argues on its own evidence, and folding it into this arm would confound two changes
in one number.

**Held constant:** the linkers, so the candidate set entering each judge is the head's;
the rubrics (`LAYERED_ENTITY_RULES`, `LAYERED_COREF_RULES`, `QUALIFIED_CLAUSE`,
`STRICTER_CLAUSE`, `COREF_RULES`); every batch size; the reply shape and therefore the
parser. Only what the prompt shows the judge changes.

**Pairing.** `_VARIANT_NAME` is this variant's own, so its phase states nest under
`phase_states/s_linker110_noevidence/` and the arm can run beside `s_linker110` in one
invocation without clobbering it -- the branch's "never compare across invocation sets"
rule, and the reason the LANDMINE that applies to the `_noknow` variants does not apply
here.

**Measurement owed:** level 4, three runs a model on both backends, paired with
`s_linker110` in every invocation. Unmeasured at the time of writing.
"""

from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker92 import SLinker92
from llm_sad_sam.linkers.experimental.s_linker110 import SLinker110


class SLinker110NoEvidence(SLinker110):
    """The head, with every code-computed context withheld from the judges."""

    _VARIANT_NAME = "s_linker110_noevidence"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("SLinker110NoEvidence (judges see the case, not the computed context)")

    # ── the full-name judge, and the strict coreference judge ────────────────

    def _format_evidence(self, bundle, anchors=None, shown_in: int = 0) -> str:
        """No evidence lines at all.

        Drops the span restatement, the mention label, the preceding sentence and the
        anchor sentences in one place, because `_validate_with_evidence` renders the
        block through this method for both judges that have one.
        """
        return ""

    @staticmethod
    def _prev_prefix(snum, sent_map) -> str:
        """No `[prev: ...]` ahead of the sentence in any case.

        The head writes the preceding sentence twice -- once here, in the case header,
        and once inside the evidence block. Both are the same computed context, so both
        go; leaving this one would keep the arm from being the ablation it claims.
        """
        return ""

    # ── the partial-name denotation judge ────────────────────────────────────

    def _classify_denotations(self, candidates, sentences):
        """The step sees each candidate's own sentence instead of its +/-5 window.

        `CONTEXT_SENTENCES` is narrowed for the duration of this call only, so the
        coreference resolver -- which reads the same `_window` predicate and whose
        window this arm keeps -- is untouched. The constant is class-level and never
        set on an instance in this lineage, so deleting the shadow restores it exactly.
        """
        self.CONTEXT_SENTENCES = 0
        try:
            return super()._classify_denotations(candidates, sentences)
        finally:
            try:
                del self.CONTEXT_SENTENCES
            except AttributeError:
                pass

    # ── the coreference resolver ─────────────────────────────────────────────

    def _prompt_coref(self, comp_names, sentence_table, targets) -> str:
        """`s_linker92`'s resolver prompt: the window, and no shortlist.

        Delegated rather than copied so the two cannot drift. This renders the head's
        prompt minus `NAMED BEFORE THIS CASE`, minus the paragraph that tells the model
        the list has already been checked, and minus the `candidates` reply field --
        which is precisely `s_linker110`'s contribution and nothing else. `s_linker109`
        does not touch this prompt, so `SLinker92`'s version is the head's own base.
        """
        return SLinker92._prompt_coref(comp_names, sentence_table, targets)
