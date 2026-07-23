"""s_linker23_verify1p_all — extend the single-pass simplification UNIVERSALLY:
drop s21's Phase-4 second validation pass on the Framing-C floor too, not only on
the augmentation gate.

``SLinker23Verify1P`` (S1) single-passes only the augmentation candidates; the s21
floor still runs P1 AND P2 on its own Framing-C extraction. This variant asks the
question the offline pilot actually modelled (single-pass over the WHOLE union):
does the second pass earn its keep on the base FC candidates either?

  * augmentation gate  — single pass (inherited from SLinker23Verify1P)
  * s21 Phase-4 floor  — single pass (this file: override `_validate_with_evidence`)
  * Phase-5 coref      — already single pass in s21; unchanged

Everything else — the evidence bundles, the claim-before-verdict prompt, P1_FOCUS,
the batch size, coref, dedup/merge — is identical to s21. Only the P2 call is
removed. GATE-01: s21's file is byte-stable; this subclass overrides the hook.

Note this is a stronger claim than S1: it moves the s21 floor itself off two-pass,
so unlike S1 it is NOT a pure augmentation (the floor's own precision can change).
Measured e2e (checkpoint backend replays the shared P1/knowledge/extraction/coref
calls, so only the P2 omission differs).
"""
from __future__ import annotations

from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names
from llm_sad_sam.linkers.experimental.s_linker21 import P1_FOCUS
from llm_sad_sam.linkers.experimental.s_linker23_verify1p import SLinker23Verify1P


class SLinker23Verify1PAll(SLinker23Verify1P):
    """Single-pass everywhere: augmentation gate AND s21's Framing-C Phase-4."""

    _VARIANT_NAME = "s_linker23_verify1p_all"

    def _validate_with_evidence(self, candidates, bundles, components, sent_map,
                                p1_tag, p2_tag, stage_label):
        """s21's Phase-4 two-pass validator with the second pass removed: keep on P1
        alone. Same evidence-bundled cases and P1 prompt as s21 — only the P2 call is
        dropped (so P1 responses stay checkpoint-shared with the two-pass variants)."""
        if not candidates:
            return [], {}
        comp_names = get_comp_names(components)
        decisions: dict = {}
        approved = []
        for _, batch in self._iter_batches(candidates, 25):
            cases = []
            for i, c in enumerate(batch):
                p = self._prev_prefix(c.sentence_number, sent_map)
                bundle = bundles.get((c.sentence_number, c.component_id))
                evidence_block = self._format_evidence(bundle) if bundle else ""
                case_text = (
                    f'Case {i+1}: "{c.matched_text}" -> {c.component_name}\n'
                    f'  {p}"{c.sentence_text}"\n'
                    f'{evidence_block}'
                )
                cases.append((case_text, c))
            case_strings = [ct for ct, _ in cases]
            r1 = self._run_validation_pass(comp_names, case_strings, P1_FOCUS, p1_tag)
            for i, (case_text, c) in enumerate(cases):
                p1 = r1.get(i, False)
                key = (c.sentence_number, c.component_id)
                decisions[key] = {
                    "approved": p1, "p1": p1, "p2": None,
                    "path": f"{stage_label}_1pass" if p1 else f"{stage_label}_1pass_reject",
                    "stage": f"{stage_label}_1pass",
                }
                if p1:
                    approved.append(c)
        return approved, decisions
