"""s_linker23_tiered — trace-linking as TIERED RANKING instead of binary keep/reject.

Instead of s21's binary Phase-4 gate (keep iff P1∧P2), each candidate gets an
EVIDENCE TIER from (name-match strength × gate votes × source), and the linker emits
links down to a fixed operating point. The tier scheme is the empirically-derived,
out-of-sample-validated v2 from pilot/tiered_ranking.py:

    FIRM     : votes==2 & match in {EXACT,TERMINAL}         (gold-rate ~1.00)
             | votes==2 & match==ROLE & source==FC          (~1.00)
    PROBABLE : votes==2 & match==ALIAS                      (~0.89)
             | votes==1 & match==ROLE & source==BLK         (~0.74, sibling-recovered)
    WEAK     : votes==2 & match==ROLE & source==BLK         (~0.50)
    REJECT   : everything else                              (<0.2)

`source` is the key signal: a blocks-proposer role reference ("the client" resolved
to HTML5 Client) is trustworthy; a Framing-C incidental role reference is not.

Two shipped operating points: FIRM+PROBABLE (precision/F1) and +WEAK (recall/F2).
Subclasses SLinker23Union (GATE-01 safe): reuses its alias+sibling union extraction,
retags blocks candidates as source="blocks", and re-selects Phase-4 by tier. Coref
(Phase 5) and the merge (Phase 6) are inherited unchanged, so it is a full linker
comparable head-to-head with s21.
"""
from __future__ import annotations

from dataclasses import replace

from llm_sad_sam.linkers.experimental.helper_v3 import has_standalone_mention
from llm_sad_sam.linkers.experimental.s_linker23_extract import SLinker23Union


def _match_type(name, quote, sentence, alias_terms):
    if has_standalone_mention(name, sentence):
        return "EXACT"
    words = name.split()
    if len(words) >= 2 and has_standalone_mention(words[-1], sentence):
        return "TERMINAL"
    if quote and quote.lower() in alias_terms:
        return "ALIAS"
    return "ROLE"


def _tier(match, votes, source):
    m, src = match, ("FC" if source != "blocks" else "BLK")
    if votes == 2 and m in ("EXACT", "TERMINAL"):
        return "FIRM"
    if votes == 2 and m == "ROLE" and src == "FC":
        return "FIRM"
    if votes == 2 and m == "ALIAS":
        return "PROBABLE"
    if votes == 1 and m == "ROLE" and src == "BLK":
        return "PROBABLE"
    if votes == 2 and m == "ROLE" and src == "BLK":
        return "WEAK"
    return "REJECT"


class SLinker23Tiered(SLinker23Union):
    """Tiered-ranking linker; select links down to ``_TIER_CUT`` (experimental)."""

    _TIER_CUT = ("FIRM", "PROBABLE")
    _VARIANT_NAME = "s_linker23_tier_f1"

    def _blocks_candidates(self, sentences, components, name_to_id, sent_map) -> dict:
        out = super()._blocks_candidates(sentences, components, name_to_id, sent_map)
        return {k: replace(c, source="blocks") for k, c in out.items()}

    def _alias_terms(self):
        dk = getattr(self, "doc_knowledge", None)
        if not dk or not getattr(dk, "aliases", None):
            return set()
        return {t.lower() for t in dk.aliases.keys()}

    def _validate_with_evidence(self, candidates, bundles, components, sent_map,
                                p1_tag, p2_tag, stage_label):
        """Run s21's two-pass validation for the votes, then re-select by TIER
        instead of the binary P1∧P2 approve."""
        _validated, decisions = super()._validate_with_evidence(
            candidates, bundles, components, sent_map, p1_tag, p2_tag, stage_label)
        alias_terms = self._alias_terms()
        cut = set(self._TIER_CUT)
        selected = []
        for c in candidates:
            key = (c.sentence_number, c.component_id)
            d = decisions.get(key, {})
            votes = int(bool(d.get("p1"))) + int(bool(d.get("p2")))
            match = _match_type(c.component_name, c.matched_text or "",
                                c.sentence_text, alias_terms)
            tier = _tier(match, votes, c.source)
            d["tier"] = tier
            keep = tier in cut
            d["approved"] = keep
            if keep:
                selected.append(c)
        return selected, decisions


class SLinker23TieredF2(SLinker23Tiered):
    """Recall/F2 operating point: FIRM+PROBABLE+WEAK."""
    _TIER_CUT = ("FIRM", "PROBABLE", "WEAK")
    _VARIANT_NAME = "s_linker23_tier_f2"
