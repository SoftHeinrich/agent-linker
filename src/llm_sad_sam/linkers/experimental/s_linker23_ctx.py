"""s_linker23_ctx — s23 augmentation CONDITIONED on s21's own output, LLM-side.

Idea under test (validate/invalidate empirically): the augmentation currently
proposes candidates blind to what the base linker already found, so it re-proposes
things s21 handled (noise) and has no notion of *where* s21 fell short. Here the
base output is fed to the proposer as CONTEXT — each sentence is shown with the
components s21 already linked (`ALREADY LINKED: ...`), and the model is asked for
what the base MISSED (residual extraction). This is pure LLM-side conditioning:

  * NO hardcoded heuristics — no "skip sentences with >= N links", no thresholds,
    no regex. The model sees the base's decisions and decides where the gaps are.
  * Recovers the ~44% of missed gold that sits in sentences s21 already engaged
    (which under-linked-only targeting cannot reach), and lets the model stay quiet
    where the base already captured everything.

Everything else is inherited: SLinker23Verify's gate (s21's real evidence-bundle
validator) keeps precision, and the gate-floor invariant still holds. Only the
proposal step (`_propose`) changes. GATE-01: s21 untouched.
"""
from __future__ import annotations

from llm_sad_sam.linkers.experimental.proposer import GroundedTypedProposer
from llm_sad_sam.linkers.experimental.s_linker23_verify import SLinker23Verify


class SLinker23Ctx(SLinker23Verify):
    """s23 + s21-verify gate + proposer conditioned on s21's per-sentence links."""

    _VARIANT_NAME = "s_linker23_ctx"

    def _propose(self, sentences, names, prev_of, base_final):
        # s21's own decisions, per sentence, as LLM context (component NAMES)
        base_of: dict[int, list[str]] = {}
        for link in base_final:
            base_of.setdefault(link.sentence_number, []).append(link.component_name)
        proposer = GroundedTypedProposer(catalog_mode="name")
        return proposer.propose_batch(
            sentences, names, batch_size=20, strategy="residual",
            prev_of=prev_of, base_of=base_of, aliases=self._global_aliases())
