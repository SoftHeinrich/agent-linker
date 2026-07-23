#!/usr/bin/env python3
"""Spike 004 — the layered no-reasoning validator (Mode 5 + Mode 1 [+ Mode 2 hints]).

Thin subclass of SLinker20Union that ONLY changes the validation-gate prompt:

  Mode 5 (justification scaffold): force a per-case `claim` field — the exact words
      that state the architectural claim — BEFORE the approve verdict. Relocates the
      deleted extended-thinking deliberation into output tokens.
  Mode 1 (architectural-claim rubric): approve on an ARCHITECTURAL CLAIM, explicitly
      NOT on name presence — so implicit (name_in_text=False) true links survive.
  Mode 2 as HINTS (not hard rules): the trap structures (code path, negation,
      listing/overview header) are named as reject cues inside the rubric. Stage 0b
      showed hard sentence-level rules net negative, so they live here as guidance the
      LLM applies per-(sentence,component), never as a blanket post-filter.

Everything upstream of validation is loaded from cache (see replay.py); this class
only re-runs the two gates. Set CLAUDE_DISABLE_THINKING=1 for the effort-0 condition.
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from llm_sad_sam.linkers.experimental.s_linker20_union import (
    SLinker20Union, COREF_VALIDATION_FOCUS)
from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names

# Mode 1 rubric + Mode 2 hints, fused. "Claim, not name presence" is the load-bearing
# clause that protects implicit links. Versioned so rubric variants can be A/B'd.
#
# v1 — initial. Recovered precision hard (teammates FP 16->7) but the
#      "reject listing/enumeration/overview header" hint nuked the architecture-
#      definition sentence ("the architecture contains X, Y, Z" = 7 true links).
# v2 — drop the listing/overview hint; keep the real FP discriminators (code path,
#      negation, generic, different-entity); state that naming a component as a part
#      of the system IS a claim, so enumeration/definition sentences approve.
RUBRICS = {
    "v1": (
        "Approve only when the sentence makes an ARCHITECTURAL CLAIM about this component: "
        "it performs an operation, provides or consumes a service, stores or routes data, "
        "contains or connects to another element, or is otherwise described as taking part "
        "in the system's behaviour or structure. What matters is the claim, not whether the "
        "exact name appears — an implicit or pronoun reference that clearly carries such a "
        "claim still approves. Reject when the matching word: is a generic technical term; "
        "names a different entity; appears only inside a code-level or package/member path "
        "(x.y.z); is stated in the negative (it is NOT a ...); or only appears in a listing, "
        "enumeration, or overview header with no claim attached to this component."
    ),
    "v2": (
        "Approve when the sentence makes an architectural claim about THIS specific "
        "component: it performs an operation, provides or consumes a service, stores or "
        "routes data, contains or connects to another element, is named as one of the "
        "system's parts, or otherwise takes part in the system's behaviour or structure. "
        "Naming the component as a part of the system or its architecture — e.g. 'the "
        "architecture contains X, Y, Z', or a heading that introduces the component — "
        "counts as such a claim, so approve those. What matters is that some claim attaches "
        "to this component, not whether its exact name appears: an implicit or pronoun "
        "reference that clearly carries a claim still approves. Reject only when the "
        "matching word: makes no architectural claim about this component in this sentence; "
        "is a generic technical term; names a different entity; appears only inside a "
        "code-level or package/member path (x.y.z); or is stated in the negative "
        "(it is NOT a ...)."
    ),
    # v3 = v2 + a hard code-path override: a path-only subject is rejected even when the
    # sentence states what the path does. Targets "storage.api provides ...",
    # "x.e2e contains system test cases", "Package overview contains logic.api".
    "v3": (
        "Approve when the sentence makes an architectural claim about THIS specific "
        "component: it performs an operation, provides or consumes a service, stores or "
        "routes data, contains or connects to another element, is named as one of the "
        "system's parts, or otherwise takes part in the system's behaviour or structure. "
        "Naming the component as a part of the system or its architecture — e.g. 'the "
        "architecture contains X, Y, Z', or a heading that introduces the component — "
        "counts as such a claim, so approve those. What matters is that some claim attaches "
        "to this component, not whether its exact name appears: an implicit or pronoun "
        "reference that clearly carries a claim still approves.\n"
        "Reject when the matching word: makes no architectural claim about this component "
        "in this sentence; is a generic technical term (e.g. the name used as a "
        "testing/technique word); names a different entity; or is stated in the negative "
        "(it is NOT a ...).\n"
        "Reject ALSO when the component is referred to ONLY through a code-level or "
        "package/member path of the form x.y or x.y.z — for example a sentence whose "
        "subject is such a dotted path — EVEN IF the sentence says what that package or "
        "path does, because the link must be to the named architectural component, not to "
        "one of its code packages. (If the component is ALSO named directly with a claim "
        "elsewhere in the sentence, approve.)"
    ),
}


# v4 — entity/coref ASYMMETRIC rubric. Diagnosis (Stage 1 full sweep): v3's
# "must make a claim" requirement recovers teammates precision but over-rejects
# bbb's bare-name headings ("FreeSWITCH.", "Kurento and WebRTC-SFU.") that bbb gold
# links. The discriminator: bbb true links are bare NAMED mentions (entity); the
# teammates FPs are code paths (logic.api) and anaphoric responsibility-bullets
# (coref). So make entity lenient (approve a named mention unless a hard reject
# signal) and keep coref strict (require a real referring expression + claim).
RUBRIC_V4_ENTITY = (
    "Approve the link by default: the component is named here and the document treats "
    "it as part of the system. A bare mention, a heading, or a list that includes the "
    "component name all count as valid links — approve them. Reject ONLY when one of "
    "these clearly holds: (1) the component is referred to only through a code-level or "
    "package/member path of the form x.y or x.y.z, even if that path is described as "
    "doing something; (2) the mention is negated (it is NOT a ...); (3) the matching "
    "word actually names a DIFFERENT entity; (4) the matching word is used as a generic "
    "technique or technology term, not as this system's component. When none of these "
    "reject-conditions clearly applies, approve."
)
RUBRIC_V4_COREF = (
    "These are coreference links: a pronoun or noun phrase in the sentence is claimed to "
    "refer back to the component, which is NOT named in the sentence itself. Approve only "
    "when the sentence contains a genuine referring expression (a pronoun or definite "
    "noun phrase) that unambiguously points to THIS component AND the sentence makes an "
    "architectural claim about it (it performs an operation, provides/consumes a service, "
    "stores or routes data, connects to another element). Reject when: the sentence is a "
    "bare continuation fragment, gerund phrase, or list item with no referring expression; "
    "the antecedent could equally be a different component; or the reference is only to a "
    "code/package path (x.y.z). When uncertain, reject."
)


def _active_rubric(focus: str = "") -> str:
    ver = os.environ.get("SPIKE_RUBRIC", "v2")
    if ver == "v4":
        return RUBRIC_V4_COREF if focus.startswith("Check coref resolution") else RUBRIC_V4_ENTITY
    return RUBRICS.get(ver, RUBRICS["v2"])


class LayeredValidator(SLinker20Union):
    """s_linker20_union with the Mode-5+1 validation prompt. Validator layer only."""

    _VARIANT_NAME = "s_linker20_union_layered"

    @staticmethod
    def _prompt_validation(comp_names, cases, focus) -> str:
        # Mode 5: require `claim` (the justification) before `approve`.
        return f"""Validate components in a document. {focus}

COMPONENTS: {', '.join(comp_names)}

{_active_rubric(focus)}

For each case, first quote the EXACT words from the sentence that state the
architectural claim about the component (or write "none" if the sentence makes no
such claim), then decide approve true/false based on that claim.

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "claim": "<exact quote or none>", "approve": true}}]}}
JSON only:"""

    # ── Mode 4: adversarial skeptic pass on coref survivors only ──────────────
    @staticmethod
    def _prompt_coref_skeptic(comp_names, cases) -> str:
        return f"""You are auditing candidate coreference links for FALSE POSITIVES. Each
case claims a pronoun or referring phrase in the sentence refers back to the named
component AND that the sentence makes an architectural claim about it. Your job is to
REFUTE weak links.

COMPONENTS: {', '.join(comp_names)}

Mark refute=true when ANY hold:
- the sentence has no actual referring expression (pronoun or noun phrase) that points to
  this component — e.g. it is a bare continuation fragment, gerund phrase, or list item;
- the reference could equally point to a different component (ambiguous antecedent);
- the sentence makes no architectural claim about THIS specific component;
- the component is referred to only through a code/package path (x.y.z).
Mark refute=false only when the referring expression is specific AND the architectural
claim clearly attaches to this component. When uncertain, refute=true.

CASES:
{chr(10).join(cases)}

Return JSON:
{{"verdicts": [{{"case": 1, "refute": true}}]}}
JSON only:"""

    def _validate_coref_links(self, coref_links, sent_map, components):
        validated, decisions = super()._validate_coref_links(coref_links, sent_map, components)
        if os.environ.get("SPIKE_CORE_SKEPTIC", "").strip().lower() not in ("1", "true", "on"):
            return validated, decisions

        comp_names = get_comp_names(components)
        kept = []
        self.llm.set_phase("phase_5_coref_skeptic")
        for _, batch in self._iter_batches(validated, 25):
            cases = []
            for lk in batch:
                sent = sent_map.get(lk.sentence_number)
                if not sent:
                    kept.append(lk)
                    continue
                p = self._prev_prefix(lk.sentence_number, sent_map)
                cases.append((lk, f'Case {len(cases)+1}: claimed reference -> '
                                  f'{lk.component_name}\n  {p}"{sent.text}"'))
            if not cases:
                continue
            data = self._ask(self._prompt_coref_skeptic(comp_names, [c for _, c in cases]),
                             timeout=120, label="Coref skeptic", require_present="verdicts")
            refuted = {}
            for v in (data.get("verdicts", []) if data else []):
                idx = v.get("case", 0) - 1
                if 0 <= idx < len(cases):
                    rv = v.get("refute", False)
                    refuted[idx] = rv is True or (isinstance(rv, str) and rv.lower() == "true")
            for idx, (lk, _) in enumerate(cases):
                key = (lk.sentence_number, lk.component_id)
                # default-KEEP on a missing verdict (parse gap != considered refute)
                if refuted.get(idx, False):
                    decisions[key] = {"approved": False, "path": "coref_skeptic_refuted"}
                else:
                    kept.append(lk)
                    decisions[key] = {"approved": True, "path": "coref_validated_skeptic"}
        return kept, decisions
