"""s_linker23_extract — measure the batched `blocks` proposer AS the Phase-2
extractor, run through s21's REAL validation gate / coref / merge (not the agentic
router of s_linker23). Two variants answer the "replace vs integrate all" question:

  * SLinker23Replace  — Phase 2 = blocks proposer ONLY (blocks REPLACES Framing-C).
  * SLinker23Union    — Phase 2 = Framing-C UNION blocks (integrate ALL extractors).

Both SUBCLASS SLinker21 and only override `_run_framing_c` (GATE-01: s21 untouched);
everything after Phase 2 — the two-pass entity gate, coref, dedup merge — is s21's,
so the final P/R/F1 reflects what the gate keeps from each extraction set. The
extraction-ceiling comparison (recall of candidates before the gate) is in
`pilot/extraction_replace_compare.py`; these variants measure the DOWNSTREAM F1.
"""
from __future__ import annotations

from llm_sad_sam.core.data_types_v2 import CandidateLink
from llm_sad_sam.linkers.experimental.s_linker21 import SLinker21
from llm_sad_sam.linkers.experimental.proposer import (
    GroundedTypedProposer, filter_generic_aliases,
)


class _BlocksExtractBase(SLinker21):
    """s21 with Phase-2 extraction sourced from the batched `blocks` proposer."""

    _EXTRACT_MODE = "union"          # "replace" | "union"

    def _blocks_candidates(self, sentences, components, name_to_id, sent_map) -> dict:
        names = [c.name for c in components]
        prev_of = {s.number: (sent_map.get(s.number - 1).text
                              if sent_map.get(s.number - 1) else "")
                   for s in sentences}
        # s21's Phase-1 global aliases (populated before Phase 2) — same map s21
        # Framing-C uses; makes the blocks extractor alias-informed (recall superset).
        dk = getattr(self, "doc_knowledge", None)
        aliases = None
        if dk and getattr(dk, "aliases", None):
            pairs = [(t, getattr(e, "component", e)) for t, e in dk.aliases.items()
                     if getattr(e, "scope", "global") == "global"]
            aliases = filter_generic_aliases(pairs, sentences, 5) or None
        proposer = GroundedTypedProposer(catalog_mode="name")
        proposals = proposer.propose_batch(
            sentences, names, batch_size=20, strategy="blocks", prev_of=prev_of,
            aliases=aliases)
        out: dict = {}
        for r in proposals:
            cid = name_to_id.get(r["component"])
            sent = sent_map.get(r["sentence"])
            if cid is None or sent is None:
                continue
            matched = r.get("quote", "") or ""
            if matched and matched.lower() not in sent.text.lower():
                continue                      # same in-sentence guard as s21 Framing-C
            key = (r["sentence"], cid)
            if key not in out:
                out[key] = CandidateLink(
                    r["sentence"], sent.text, r["component"], cid, matched,
                    source="entity")
        return out

    def _run_framing_c(self, sentences, components, name_to_id, sent_map) -> dict:
        blocks = self._blocks_candidates(sentences, components, name_to_id, sent_map)
        if self._EXTRACT_MODE == "replace":
            print(f"    [blocks-extract] REPLACE: {len(blocks)} blocks candidates "
                  f"(Framing-C skipped)")
            return blocks
        base = super()._run_framing_c(sentences, components, name_to_id, sent_map)
        merged = {**blocks, **base}           # base (Framing-C) wins key collisions
        print(f"    [blocks-extract] UNION: Framing-C={len(base)} + blocks-only="
              f"{len(merged) - len(base)} -> {len(merged)}")
        return merged


class SLinker23Replace(_BlocksExtractBase):
    """Phase 2 = blocks proposer only (blocks REPLACES s21 Framing-C)."""
    _EXTRACT_MODE = "replace"
    _VARIANT_NAME = "s_linker23_replace"


class SLinker23Union(_BlocksExtractBase):
    """Phase 2 = s21 Framing-C UNION blocks proposer (integrate all extractors)."""
    _EXTRACT_MODE = "union"
    _VARIANT_NAME = "s_linker23_union"
