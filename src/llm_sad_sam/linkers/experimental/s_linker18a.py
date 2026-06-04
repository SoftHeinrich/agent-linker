"""s_linker18a — Cleanup F: drop generic-filter, route all candidates through twopass.

Feasibility verified empirically: the generic-filter pipeline is dead code at
gpt-5.4. Across all 5 ARDoCo benchmark projects, only 1 candidate (on teammates)
ever entered the generic-filter branch — and twopass agreed with its decision.

Removes the complex 3-condition gate in `_validate_with_evidence`:
    has_exact_case AND has_lowercase AND is_ambiguous → generic-filter prompt
    else → twopass

… plus the entire `CONTEXTUAL WORD USAGE` LLM prompt machinery (~80 LOC).

Behavior change: candidates that *would* have been routed through the
generic-filter prompt are now sent directly to twopass. The is_ambiguous
signal in the evidence bundle (set by `_build_evidence_bundle`) carries the
ambiguity info into twopass's referential-specificity pass (p2).

Inherits everything else from SLinker17f.

experimental=True, canonical=False.
"""
from __future__ import annotations

from llm_sad_sam.linkers.experimental.s_linker17f import SLinker17f
from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names


class SLinker18a(SLinker17f):
    """17f with cleanup F applied — single twopass validator, no generic-filter."""

    _VARIANT_NAME = "s_linker18a"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"SLinker18a (cleanup F: drop generic-filter, twopass-only validation)")

    def _validate_with_evidence(self, candidates, bundles, components, sent_map):
        """Unified twopass on every candidate — no generic-filter pre-pass.

        Returns (twopass_approved, decisions). The decisions dict has the same
        schema as 17f's twopass_decisions; the generic_filter sub-dict is
        simply never populated (downstream code already tolerates empty).
        """
        if not candidates:
            return [], {}

        comp_names = get_comp_names(components)
        decisions: dict = {}
        twopass_approved = []

        self.llm.set_phase("phase_4_twopass")
        print(f"    Unified twopass validation: {len(candidates)} candidates "
              "(no generic-filter pre-pass)")
        for batch_start in range(0, len(candidates), 25):
            batch = candidates[batch_start:batch_start + 25]
            cases = []
            for i, c in enumerate(batch):
                prev = sent_map.get(c.sentence_number - 1)
                p = f"[prev: {prev.text[:60]}] " if prev else ""
                bundle = bundles.get((c.sentence_number, c.component_id))
                evidence_block = self._format_evidence(bundle) if bundle else ""
                case_text = (
                    f'Case {i+1}: "{c.matched_text}" -> {c.component_name}\n'
                    f'  {p}"{c.sentence_text}"\n'
                    f'{evidence_block}'
                )
                cases.append((case_text, c))

            case_strings = [ct for ct, _ in cases]
            r1 = self._run_validation_pass(
                comp_names, case_strings,
                "Check architectural participation: does the sentence name this component as an architectural participant — performing operations, providing services, or taking part in the described system behavior?",
                phase_tag="phase_4_twopass_p1",
            )
            r2 = self._run_validation_pass(
                comp_names, case_strings,
                "Check referential specificity: is the component name used to identify this specific architectural element, or does it serve as a generic technical term in this sentence?",
                phase_tag="phase_4_twopass_p2",
            )

            for i, (case_text, c) in enumerate(cases):
                p1 = r1.get(i, False)
                p2 = r2.get(i, False)
                approved = p1 and p2
                key = (c.sentence_number, c.component_id)
                decisions[key] = {
                    "approved": approved,
                    "p1": p1,
                    "p2": p2,
                    "path": "twopass" if approved else "twopass_reject",
                    "stage": "twopass",
                }
                if approved:
                    twopass_approved.append(c)

        return twopass_approved, decisions
