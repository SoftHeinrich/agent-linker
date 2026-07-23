"""s_linker23 — LLM-decision-driven augmentation of s_linker21.

EXPERIMENTAL (canonical=False). This is the clean, generalized typed router the
project converged on: routing and validation are decided by the LLM from GENERAL
GUIDELINES, not by hand-written structural rules. There is deliberately almost no
``if/else`` policy logic in this file — no regex evidence filters, no
mode->validator dispatch table, no exact/terminal/no-code predicates. Every
keep/route/reject decision is made by a model reading the sentence against a
generic English rubric; the Python here only wires steps together and enforces the
one bounded-autonomy invariant.

Pipeline
--------
1. Run canonical ``SLinker21.link()`` UNCHANGED as the floor (GATE-01: s21 is
   never edited). Its result is the precision/recall floor.
2. Surface floor-missed candidates with ``GroundedTypedProposer`` — a single
   reasoning-off LLM read per sentence over a generic prompt; a candidate survives
   only if it grounds to a real catalog component name (the sole non-LLM step, and
   it is data grounding, not a decision).
3. ``DocModelAgenticRouter`` lets the LLM choose ONE action per candidate —
   VALIDATE / CODE / REJECT — from general guidelines (``agentic_router``'s generic
   rubric). No structural pre-filter gates this decision.
4. Bounded autonomy: a VALIDATE candidate is accepted ONLY if s21's OWN two-pass
   entity validator (injected as the router's gate, so the *real* s21 rubric and
   passes decide, not a reimplementation) also approves it. The agent can only
   DIVERT (CODE/REJECT) or defer to the gate; it can never add a link the gate
   rejects. So the augmentation can never regress below the s21 floor.

The whole augmentation is wrapped in try/except and falls back to the untouched
s21 result on any failure. GATE-06: the proposer prompt, the router rubric, and
the gate rubric are all generic English; the only project-specific input is the
runtime component catalog, exactly as s21 already receives it.

Contrast with the earlier structural variant (s22) and the reasoning behind this
redesign are in ``pilot/COMPARISON_s22_vs_agentrouter.md``.
"""
from __future__ import annotations

from llm_sad_sam.core.data_types_v2 import SadSamLink
from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository

from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names
from llm_sad_sam.linkers.experimental.s_linker21 import SLinker21, P1_FOCUS, P2_FOCUS
from llm_sad_sam.linkers.experimental.agentic_router import (
    DocModelAgenticRouter, Candidate,
)
from llm_sad_sam.linkers.experimental.proposer import (
    GroundedTypedProposer, filter_generic_aliases,
)


class SLinker23(SLinker21):
    """s21 floor + an LLM-decided, gate-floored augmentation pass (experimental)."""

    _VARIANT_NAME = "s_linker23"
    _ALIAS_MAX_DF = 5          # single-word alias dropped if it occurs in > this many sentences
    _SIBLING_DISAMBIG = True   # resolve role/base refs to the right sibling (HTML5 Client vs Server)

    def link(self, text_path, model_path, **kwargs):
        base_final = super().link(text_path, model_path, **kwargs)
        self.router_decisions = []
        self.code_routed_candidates = []
        try:
            return self._augment(base_final, text_path, model_path, **kwargs)
        except Exception as exc:                       # never regress below the floor
            print(f"  [s23] LLM-router augmentation failed, base s21 result kept: {exc}")
            return base_final

    def _augment(self, base_final, text_path, model_path, **kwargs):
        """The LLM-driven augmentation pass over an existing s21 floor. Separated
        from ``link`` so it is testable without running the full s21 pipeline."""
        components = parse_pcm_repository(model_path)
        sentences = load_sentences(text_path)
        sent_map = build_sent_map(sentences)
        name_to_id = {c.name: c.id for c in components}
        names = [c.name for c in components]
        comp_names = get_comp_names(components)

        # (2) LLM proposer surfaces floor-missed candidates in a few BATCHED calls
        # (never one call per sentence — hard rule). The `blocks` strategy renders
        # each sentence as its own context-carrying item, so batching does NOT dilute
        # per-sentence recall; `blocks`@20 is the empirical optimum from
        # pilot/batch_strategy_compare.py (teammates: 10 calls, recall 1.000; bbb: 5
        # calls, 0.742 — vs a naive flat one-call read at 0.825/0.613). For a
        # 198-sentence doc this is ~10 calls, not 198.
        prev_of = {s.number: (sent_map.get(s.number - 1).text
                              if sent_map.get(s.number - 1) else "")
                   for s in sentences}
        proposals = self._propose(sentences, names, prev_of, base_final)
        candidates: list[Candidate] = []
        seen_ids = set()
        for r in proposals:
            cid = name_to_id.get(r["component"])
            sent = sent_map.get(r["sentence"])
            if cid is None or sent is None:
                continue
            cand_id = f"{r['sentence']}|{cid}"
            if cand_id in seen_ids:
                continue
            seen_ids.add(cand_id)
            candidates.append(Candidate(
                id=cand_id, sentence=sent.text, component=r["component"],
                prev=prev_of.get(r["sentence"], ""), quote=r.get("quote", ""),
            ))

        # (3)+(4) LLM router decides the action; a gate floors it. The gate is a
        # hook (`_router_gate`) so subclasses can plug in a stronger verifier.
        router = DocModelAgenticRouter(
            gate=self._router_gate(components, comp_names, sent_map))
        decisions = router.route(candidates)
        self.router_decisions = decisions
        self.code_routed_candidates = router.routed_to_code(decisions)

        existing = {(l.sentence_number, l.component_id) for l in base_final}
        id_to_comp = {c.id: c.component for c in candidates}
        augment: list[SadSamLink] = []
        for cand in router.accepted(decisions):
            snum_str, cid = cand.id.split("|", 1)
            key = (int(snum_str), cid)
            if key in existing:
                continue
            existing.add(key)
            augment.append(SadSamLink(key[0], cid, id_to_comp[cand.id],
                                      source="llmrouter"))
        print(f"  [s23] LLM router: {len(candidates)} proposed, "
              f"{len(augment)} gate-approved additions over the s21 floor.")
        return base_final + augment

    def _global_aliases(self, sentences=None):
        """s21's Phase-1 global doc aliases as ``[(term, component), ...]`` — the same
        alias map s21 Framing-C injects into its extraction prompt. Feeding it to the
        blocks proposer recovers alias-mediated mentions the blind read misses, making
        the proposal set a recall superset of Framing-C (pilot/KNOWLEDGE_PROPOSER_RESULTS.md).
        Generic single-word aliases are filtered by document frequency (see
        ``filter_generic_aliases``) — they caused the e2e teastore FP leak. ``sentences``
        supplies the frequency counts; ``self.doc_knowledge`` is set by the floor's Phase 1."""
        dk = getattr(self, "doc_knowledge", None)
        if not dk or not getattr(dk, "aliases", None):
            return None
        pairs = [(term, getattr(entry, "component", entry))
                 for term, entry in dk.aliases.items()
                 if getattr(entry, "scope", "global") == "global"]
        pairs = filter_generic_aliases(pairs, sentences, self._ALIAS_MAX_DF)
        return pairs or None

    def _propose(self, sentences, names, prev_of, base_final):
        """Hook: surface floor-missed candidates. Default = the batched `blocks`
        proposer, alias-informed with s21's Phase-1 global aliases (generic single-word
        aliases frequency-filtered). SLinker23Ctx overrides this to condition on s21's
        per-sentence links (LLM-side context)."""
        proposer = GroundedTypedProposer(catalog_mode="name")
        return proposer.propose_batch(
            sentences, names, batch_size=20, strategy="blocks", prev_of=prev_of,
            aliases=self._global_aliases(sentences),
            sibling_disambig=self._SIBLING_DISAMBIG)

    def _router_gate(self, components, comp_names, sent_map):
        """Hook: the verifier that floors the router's VALIDATE decisions. Default is
        the lightweight case-text two-pass gate; SLinker23Verify overrides it with
        s21's full evidence-bundle validator."""
        return self._s21_gate(comp_names)

    def _s21_gate(self, comp_names):
        """Return a gate callable(candidates)->{id: keep} that defers entirely to
        s21's OWN two-pass entity validator (P1 architectural-participation AND P2
        referential-specificity). No structural logic — the s21 LLM rubric decides.
        """
        def gate(cands):
            cands = list(cands)
            if not cands:
                return {}
            cases = []
            for i, c in enumerate(cands, 1):
                prev = f'[prev: "{c.prev}"] ' if c.prev else ""
                cases.append(
                    f'Case {i}: "{c.quote or c.component}" -> {c.component}\n'
                    f'  {prev}"{c.sentence}"'
                )
            r1 = self._run_validation_pass(comp_names, cases, P1_FOCUS, "phase_router_gate_p1")
            r2 = self._run_validation_pass(comp_names, cases, P2_FOCUS, "phase_router_gate_p2")
            return {c.id: bool(r1.get(i - 1) and r2.get(i - 1))
                    for i, c in enumerate(cands, 1)}
        return gate
