"""s_linker23_verify1p — s23_verify with the augmentation gate reduced to a SINGLE
evidence pass (drop the redundant second pass), without losing precision.

Redundancy found (pilot/simplify_verify_pilot.py, offline over the 5-dataset tier
caches — captured router-less, so they isolate the GATE change):

  s23_verify  gate = p1 AND p2   F1 0.896   (reference)
  this        gate = p1          F1 0.920   (+0.024 pooled)

The two evidence passes agree ~90% of the time; where they differ, the second pass
removed ~2x more gold than non-gold on the augmentation population, so it costs more
recall than the precision it buys. Dropping it also halves the gate's API cost.

Only ``_evidence_validate`` changes: the evidence bundles, the claim-before-verdict
prompt, and the gate-floor invariant are identical to s23_verify's first pass — the
agentic router (VALIDATE/CODE/REJECT) and its CODE escape hatch are untouched.
GATE-01: s21 untouched; this only overrides a hook on SLinker23Verify.
"""
from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker21 import P1_FOCUS
from llm_sad_sam.linkers.experimental.s_linker23_verify import SLinker23Verify


class SLinker23Verify1P(SLinker23Verify):
    """s23_verify with a single-pass evidence gate (P1 only, no P2)."""

    _VARIANT_NAME = "s_linker23_verify1p"

    def _evidence_validate(self, links, bundles, components, comp_names, sent_map):
        if not links:
            return set()
        kept = set()
        for _, batch in self._iter_batches(links, 25):
            cases = []
            for i, c in enumerate(batch):
                p = self._prev_prefix(c.sentence_number, sent_map)
                bundle = bundles.get((c.sentence_number, c.component_id))
                ev = self._format_evidence(bundle) if bundle else ""
                cases.append(
                    f'Case {i+1}: "{c.matched_text}" -> {c.component_name}\n'
                    f'  {p}"{c.sentence_text}"\n{ev}'
                )
            r1 = self._run_validation_pass(comp_names, cases, P1_FOCUS,
                                           "phase_aug_evidence_1p")
            for i, c in enumerate(batch):
                if r1.get(i, False):
                    kept.add((c.sentence_number, c.component_id))
        return kept
