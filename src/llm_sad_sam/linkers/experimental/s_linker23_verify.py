"""s_linker23_verify — s23's LLM proposer+router COMBINED with s21's real
verification idea. The proposed (blocks) candidates the router sends to VALIDATE
are floored not by s23's lightweight case-text gate, but by s21's OWN Phase-4
evidence-bundle two-pass validator (claim-before-verdict) — the mechanism that
gives s21 its precision.

Motivation: plain s23 leaks false positives (its gate builds a minimal case text
without the evidence bundles s21 normally attaches, so it over-approves speculative
candidates). This variant routes the augmented candidates through the *identical*
verification s21 applies to its own candidates, making the bounded-autonomy floor a
real precision control, not just a recall floor.

    accept  ==  router-VALIDATE  AND  s21-evidence-bundle-validator approves

Only `_router_gate` changes; everything else (proposer, router, gate-floor, dedup,
try/except fallback) is inherited from SLinker23. GATE-01: s21 untouched.
"""
from __future__ import annotations

from llm_sad_sam.core.data_types_v2 import CandidateLink
from llm_sad_sam.linkers.experimental.s_linker23 import SLinker23


class SLinker23Verify(SLinker23):
    """s23 augmentation verified by s21's real evidence-bundle validator."""

    _VARIANT_NAME = "s_linker23_verify"

    def _router_gate(self, components, comp_names, sent_map):
        def gate(cands):
            cands = list(cands)
            if not cands:
                return {}
            # router Candidate -> s21 CandidateLink (id is "sentenceNumber|componentId")
            links, order = [], []
            for c in cands:
                snum_str, cid = c.id.split("|", 1)
                links.append(CandidateLink(
                    int(snum_str), c.sentence, c.component, cid,
                    c.quote or "", source="entity"))
                order.append(c.id)
            # s21's real Phase-4: build evidence bundles, then two-pass claim-before-verdict
            bundles = {
                (l.sentence_number, l.component_id): self._build_evidence_bundle(l, sent_map)
                for l in links
            }
            validated, _decisions = self._validate_with_evidence(
                links, bundles, components, sent_map,
                p1_tag="phase_aug_evidence_p1", p2_tag="phase_aug_evidence_p2",
                stage_label="augment")
            kept = {(v.sentence_number, v.component_id) for v in validated}
            out = {}
            for c in cands:
                snum_str, cid = c.id.split("|", 1)
                out[c.id] = (int(snum_str), cid) in kept
            return out
        return gate
