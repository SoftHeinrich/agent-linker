"""s_linker18b — Cleanup E: unify coref validation with entity twopass.

Builds on s_linker18a (which already dropped generic-filter).

Before: coref-resolved candidates were validated through a separate single-pass
validator (`_validate_coref_links`) with a coref-specific focus prompt — an
asymmetric, parallel pipeline to the entity twopass.

After: coref candidates are validated through the same entity twopass
(p1 = architectural participation, p2 = referential specificity). The evidence
bundle for a coref candidate carries `mention_type='anaphoric reference'` so
twopass can read context appropriately.

Feasibility-test outcome on the 5 ARDoCo projects (gpt-5.4):
    twopass agrees with single-pass on 64 of 73 coref candidates
    net delta: +1 TP, −4 FP   (better on mediastore, teastore, teammates, jabref;
                               +1 FP on bigbluebutton — within tolerance)

Inherits everything from SLinker18a and overrides only `_validate_coref_links`.

experimental=True, canonical=False.
"""
from __future__ import annotations

from llm_sad_sam.core.data_types_v2 import CandidateLink
from llm_sad_sam.linkers.experimental.s_linker18a import SLinker18a
from llm_sad_sam.linkers.experimental.s_linker17f import EvidenceBundle
from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names


class SLinker18b(SLinker18a):
    """18a with cleanup E applied — entity twopass also validates coref candidates."""

    _VARIANT_NAME = "s_linker18b"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(f"SLinker18b (cleanup E: unified coref validation via entity twopass)")

    def _validate_coref_links(self, coref_links, sent_map, components):
        """Validate coref candidates through the same twopass used for entity links.

        Returns (validated, decisions) — schema-compatible with 17f, so the
        parent's `link()` method consumes our output without modification.

        Auto-keep coref links whose sentence text is missing (matches 17f's
        defensive behavior for sentences not in sent_map).
        """
        if not coref_links:
            return [], {}

        comp_names = get_comp_names(components)
        validated = []
        decisions: dict = {}

        self.llm.set_phase("phase_5_coref_validation")
        for batch_start in range(0, len(coref_links), 25):
            batch = coref_links[batch_start:batch_start + 25]
            # Build CandidateLinks with bundles that flag the anaphoric mention type.
            cases = []
            for i, lk in enumerate(batch):
                key = (lk.sentence_number, lk.component_id)
                sent = sent_map.get(lk.sentence_number)
                if not sent:
                    # Missing sentence — keep (matches 17f's behavior).
                    validated.append(lk)
                    decisions[key] = {
                        "approved": True,
                        "path": "coref_no_sentence_keep",
                    }
                    continue
                cand = CandidateLink(
                    lk.sentence_number, sent.text,
                    lk.component_name, lk.component_id,
                    matched_text=lk.component_name,  # the reference surface form
                    source="coreference",
                    mention_type="anaphoric",
                )
                bundle = self._build_evidence_bundle(
                    cand, sent_map, rationale="coref resolution to a prior antecedent")
                # Override mention_type to make the anaphoric nature explicit to twopass.
                bundle = EvidenceBundle(
                    source="coref",
                    matched_span=bundle.matched_span,
                    mention_type="anaphoric reference (pronoun or role-ref) — antecedent established in prior context",
                    preceding_text=bundle.preceding_text,
                    anchor_sentences=bundle.anchor_sentences,
                    is_ambiguous=bundle.is_ambiguous,
                    extraction_rationale="coref resolution to a prior antecedent",
                )
                prev = sent_map.get(lk.sentence_number - 1)
                p = f"[prev: {prev.text[:60]}] " if prev else ""
                cases.append((
                    i, lk,
                    f'Case {len(cases)+1}: pronoun/role-ref -> {lk.component_name}\n'
                    f'  {p}"{sent.text}"\n'
                    f'{self._format_evidence(bundle)}',
                ))

            if not cases:
                continue

            case_strings = [c for _, _, c in cases]
            r1 = self._run_validation_pass(
                comp_names, case_strings,
                "Check architectural participation: does the sentence name this component as an architectural participant — performing operations, providing services, or taking part in the described system behavior?",
                phase_tag="phase_5_coref_twopass_p1",
            )
            r2 = self._run_validation_pass(
                comp_names, case_strings,
                "Check referential specificity: is the component name used to identify this specific architectural element, or does it serve as a generic technical term in this sentence?",
                phase_tag="phase_5_coref_twopass_p2",
            )

            for idx, (i, lk, _) in enumerate(cases):
                key = (lk.sentence_number, lk.component_id)
                p1 = r1.get(idx, False)
                p2 = r2.get(idx, False)
                approved = p1 and p2
                decisions[key] = {
                    "approved": approved,
                    "p1": p1,
                    "p2": p2,
                    "path": "coref_twopass" if approved else "coref_twopass_reject",
                }
                if approved:
                    validated.append(lk)
                else:
                    print(f"    Coref twopass reject: S{lk.sentence_number} -> "
                          f"{lk.component_name} (p1={p1} p2={p2})")

        return validated, decisions
