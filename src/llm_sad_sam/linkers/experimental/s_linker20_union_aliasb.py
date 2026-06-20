"""s_linker20_union_aliasb — v2.6.5 combined variant — standalone (no superclass dependency).

= s_linker20_union (Framing C 2-pass UNION consensus) + the aliasb "prompt swap": the
ANTECEDENT_ALIAS_RULES few-shot examples are rewritten to a non-SE (hardware-domain)
PowerSupplyUnit example (lio Candidate B — benchmark-distant, TM-stable). All prompt
constants inlined; no external prompt module import. experimental=True, canonical=False.

──────────────────────────────────────────────────────────────────────────────
Pipeline
──────────────────────────────────────────────────────────────────────────────

Phase 1 — Knowledge acquisition (model + document, parallel)
  • ambiguous_names (model_knowledge): ambiguity classification of names
  • aliases with scope (doc_knowledge): alias discovery + judge

Phase 2 — Entity extraction (Framing C only)
  • 2-pass alias-injected entity extraction, intersected for stability
  • (Framings A, B from s_linker17f are EMPIRICALLY redundant — 0 unique TPs
    across 5 gpt-5.4 benchmarks — and DROPPED here. ILinker4 is therefore
    unneeded; the entire dependency is gone.)

Phase 3 — Candidate pool
  • Just C's output (no union merge needed when only one framing runs)

Phase 4 — Unified evidence-bundle twopass validation (sole entity quality gate)
  • Build evidence bundles (typed mention_type, preceding text, anchors,
    is_ambiguous, extraction_rationale)
  • Two-pass twopass:
      p1 (P1_FOCUS): architectural participation — INCLUDES
         qualified-name-identifier exclusion (V3 + rename vs 17f, absorbing
         the role of 17f's Phase 4b code-path filter). The clause is the
         original V3 reframed in textbook SE vocabulary, with an explicit
         X.Y.Z schema as the structural anchor.
         Joint-backend probe (experiment_dotted_path_rename.py):
           - gpt-5.4 : 2/3 code-path FPs caught, 4/4 TPs preserved
           - sonnet  : 1/3 code-path FPs caught, 4/4 TPs preserved
         The original "dotted-path identifier" wording caught 2/3 on
         gpt-5.4 but 0/3 on Sonnet — strict joint improvement.
      p2 (P2_FOCUS): referential specificity
    Approve iff p1 ∧ p2.
  • No generic-filter pre-pass — empirically dead code at gpt-5.4 (1
    candidate across 5 projects ever triggered it).
  • No separate Phase 4b LLM phase — absorbed into p1 (V3 modification).

Phase 5 — Coreference
  • Discovery: LLM scans EVERY sentence in batches for anaphoric references.
    No hardcoded PRONOUN_PATTERN regex. No "the <terminal>" role-ref regex.
    No separate `_classify_specific_terminals` LLM step. The LLM identifies
    referential phrases in context.
  • Structural alias-aware antecedent gate (replaces 17f's
    antecedent_via_alias LLM-flag bypass). The antecedent sentence must
    contain either the canonical component name OR any discovered alias
    — purely structural, no LLM-flag coupling.
  • Single-pass coref-focused validator (COREF_VALIDATION_FOCUS).
    Kept ASYMMETRIC to entity twopass on principled grounds: anaphoric
    resolution asks a narrower epistemic question than name disambiguation.
    Cleanup E (unify with entity twopass) was empirically found too lenient
    on anaphora and is NOT applied here.

Phase 6 — Dedup-by-key merge of entity + coref links.

──────────────────────────────────────────────────────────────────────────────
What's gone from s_linker17f (paper-elegance)
──────────────────────────────────────────────────────────────────────────────
  • Framings A and B (0 unique TPs on gpt-5.4)
  • ILinker4 dependency (consequence of dropping A and B)
  • Generic-filter pre-pass (dead code at gpt-5.4)
  • Phase 4b code-path filter (absorbed into p1 — V3 +7 words)
  • PRONOUN_PATTERN hardcoded English regex
  • "the <terminal>" role-ref regex (role_ref_pat)
  • `_classify_specific_terminals` LLM call + cache
  • Anaphoric pre-filter (every sentence goes to coref LLM)
  • antecedent_via_alias LLM-flag bypass (replaced by structural check)
  • Ad-hoc string-typed mention classification (now MentionType enum)
  • Mixed-stage decisions dict (generic+twopass+codepath_filter paths)

experimental=True, canonical=False. s_linker13_min retains canonical=True.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from enum import Enum

from llm_sad_sam.core.data_types_v2 import (
    SadSamLink, CandidateLink,
    ModelKnowledge, DocumentKnowledge,
)
from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.linkers.experimental.helper_v3 import (
    has_standalone_mention, parse_snum, get_comp_names,
)
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient, LLMBackend, LLMResponse

# ─────────────────────────────────────────────────────────────────────────────
# Minimized prompt constants — inlined from Phase 46 minimization
# (replaces the external prompt-module import block used in the preceding variant)
# Phase 46 kept-cut inventory: AMB-01 (drop), AMB-02 (opener), DKJ-01 (drop),
# DKJ-07 (grouping), EXT-01 (opener), VAL-01 (matching entities), VAL-02 (opener),
# VAL-03 (noun phrase that refers back), COR-01 (noun phrase that refers back),
# COR-02 (topic of the surrounding section), COR-03 (opener), COR-04 (inline)
# ─────────────────────────────────────────────────────────────────────────────

AMBIGUITY_FEW_SHOT = ""

AMBIGUITY_RULES = """A name is ARCHITECTURAL when it identifies a specific role or mechanism. A name is AMBIGUOUS when ordinary technical writing about any system would use it generically without naming a specific component."""


DOC_KNOWLEDGE_EXTRACTION_RULES = """Find surface forms the document uses to refer to a single named component (introduced short forms, alternate names, or words of multi-word names when they alone clearly mean the full name). Reject terms whose ordinary English use dominates."""

DOC_KNOWLEDGE_JUDGE_EXAMPLES = ""

DOC_KNOWLEDGE_JUDGE_RULES = """An alias is valid when the document establishes an equivalence between a phrase and a single named component. An alias is invalid when the phrase is generic vocabulary, names the whole system, or names a different entity. An alias is also invalid when it names a grouping that encompasses multiple elements, because it identifies a grouping rather than a single named unit. When uncertain, prefer APPROVE."""

ALIAS_SCOPE_RULES = """For each alias, classify its SCOPE:
- "global": distinctive enough to unambiguously name the component anywhere in the document. Typical shapes: multi-word forms, hyphenated forms, CamelCase, all-caps abbreviations of length >= 2, or names beginning with an uppercase letter.
- "local": a single all-lowercase word overlapping with ordinary English vocabulary. Safe only where the surrounding context already establishes which component is being discussed.
Qualified-name fragments (package- or member-access paths of the form X.Y or X.Y.Z) are NOT aliases — do not include them."""


ENTITY_EXTRACTION_RULES = """Include a reference when the sentence refers to the component by name, alias, or as a participant in a described interaction. Exclude when the name appears only inside a code-level path — even if the compound identifier is semantically related to the component — or as ordinary English with no architectural intent. Favor inclusion."""


P1_FOCUS = (
    "Check architectural participation: does the sentence name this "
    "component as an architectural participant — performing operations, "
    "providing services, or taking part in the described system behavior, "
    "and not just as a qualified-name identifier (e.g. a package- or "
    "member-access path X.Y.Z)?"
)

P2_FOCUS = (
    "Check referential specificity: is the component name used to identify "
    "this specific architectural element, or does it serve as a generic "
    "technical term in this sentence?"
)

VALIDATION_RULES = """Approve when the sentence treats the component as an architectural participant, including matching entities. Reject when the matching word is generic, names a different entity, or describes a technique that merely shares the component's name."""


COREF_VALIDATION_FOCUS = (
    "Check coref resolution: does the pronoun, 'it', 'they', 'the service', "
    "or similar noun phrase that refers back in this sentence actually refer to "
    "the named component as an architectural participant — performing "
    "operations, providing services, or being the grammatical topic of the "
    "sentence?"
)

COREF_RULES = """For each case, decide whether a pronoun or noun phrase that refers back in the target sentence refers back to a component named or aliased earlier in the context. Resolve when: (a) the component's name or a known alias appears in the surrounding context sentences, or (b) only one component has been introduced in the immediately preceding sentences — treat it as the topic of the surrounding section and resolve role-referential phrases ("it", "the module", "the service", "the component", "the system") to that topic even without a direct name repetition. Avoid resolving when two or more equally plausible antecedents exist. Known aliases include the terminal word(s) of a multi-word name, documented abbreviations, and alternate forms used in the document. When the antecedent sentence uses a known alias rather than the full canonical name, set antecedent_via_alias=true."""

ANTECEDENT_ALIAS_RULES = """For each resolution, set antecedent_via_alias:
- true:  the antecedent quote refers to the component by an ALIAS — a terminal word of a multi-word name, an abbreviation, a hyphenated form, or any documented alternate name rather than the canonical name listed in COMPONENTS.
- false: the antecedent quote uses the canonical name verbatim as listed in COMPONENTS.

Examples:
- COMPONENTS contains "PowerSupplyUnit"; antecedent: "the unit regulates voltage" -> true (uses terminal "unit", not canonical "PowerSupplyUnit").
- COMPONENTS contains "PowerSupplyUnit"; antecedent: "PowerSupplyUnit regulates voltage" -> false (canonical name verbatim).

Default to true when the antecedent form clearly differs from the canonical name but unambiguously identifies the component."""

# ─────────────────────────────────────────────────────────────────────────────
# Tracing infrastructure — per-LLM-call audit trail for paper-time analysis
# ─────────────────────────────────────────────────────────────────────────────

_phase_local = threading.local()


def _current_phase() -> str:
    return getattr(_phase_local, "phase", "unknown")


class _TracingLLMClient:
    """Delegating wrapper that records every query() into a phase-tagged trace."""

    def __init__(self, inner: LLMClient, sink: list[dict]):
        self._inner = inner
        self._sink = sink
        self._sink_lock = threading.Lock()

    def set_phase(self, name: str) -> None:
        _phase_local.phase = name

    def query(self, prompt: str, timeout: int = 180, max_retries: int = 3) -> LLMResponse:
        phase = _current_phase()
        t0 = time.time()
        resp = self._inner.query(prompt, timeout=timeout, max_retries=max_retries)
        record = {
            "phase": phase, "ts": t0,
            "elapsed_s": round(time.time() - t0, 3),
            "timeout": timeout, "max_retries": max_retries,
            "prompt": prompt,
            "response_text": getattr(resp, "text", None),
            "success": getattr(resp, "success", None),
            "error": getattr(resp, "error", None),
            "latency_ms": getattr(resp, "latency_ms", None),
            "model": getattr(resp, "model", None),
        }
        usage = getattr(resp, "token_usage", None)
        if usage is not None:
            record["token_usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        with self._sink_lock:
            self._sink.append(record)
        return resp

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

class MentionType(Enum):
    """Classification of how a component name appears in a sentence."""
    PROPER_STANDALONE = "proper case, standalone"
    LOWERCASE_PROSE = "lowercase mention"
    CODE_TOKEN = "lowercase, inside qualified name"
    VIA_ALIAS = "via known alias"
    ANAPHORIC = "anaphoric reference"
    INDIRECT = "indirect/unclear match"


@dataclass
class EvidenceBundle:
    source: str
    matched_span: str
    mention_type: str          # MentionType.value (str for prompt embedding)
    preceding_text: str
    anchor_sentences: list[str]
    is_ambiguous: bool
    extraction_rationale: str


@dataclass(frozen=True)
class AliasEntry:
    component: str
    scope: str   # "global" | "local"


# ─────────────────────────────────────────────────────────────────────────────
# Main linker
# ─────────────────────────────────────────────────────────────────────────────

class SLinker20UnionAliasB:
    """v2.6.5 combined variant — s_linker20_union + aliasb prompt swap.

    Standalone (no superclass). = s_linker20_union (Framing C 2-pass UNION
    consensus) with the ANTECEDENT_ALIAS_RULES few-shot examples swapped to a
    non-SE (hardware-domain) PowerSupplyUnit example — the lio Candidate-B
    "prompt swap" (benchmark-distant, stable on TM). experimental=True,
    canonical=False."""

    _VARIANT_NAME = "s_linker20_union_aliasb"

    def __init__(
        self,
        backend: LLMBackend | None = None,
        model: str | None = None,
        checkpoint_fallback: LLMBackend | str | None = None,
        checkpoint_fallback_model: str | None = None,
    ):
        os.environ.setdefault("CLAUDE_MODEL", "sonnet")
        os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.4")
        real_llm = LLMClient(
            backend=backend or LLMBackend.CLAUDE,
            model=model,
            checkpoint_fallback=checkpoint_fallback,
            checkpoint_fallback_model=checkpoint_fallback_model,
        )
        self._llm_calls: list[dict] = []
        self.llm = _TracingLLMClient(real_llm, self._llm_calls)
        self.model_knowledge: ModelKnowledge | None = None
        self.doc_knowledge: DocumentKnowledge | None = None
        self._phase_log: list[dict] = []
        self._phase_metrics: dict[str, dict] = {}
        self._current_text_path: str | None = None
        print("SLinker20UnionAliasB (s20_union + aliasb hardware-domain prompt swap)")
        print(f"  Backend: {self.llm.describe_backend()}")

    # ── DAG infra ────────────────────────────────────────────────────────────

    @staticmethod
    def _run_parallel(tasks):
        if len(tasks) == 1:
            name, fn = next(iter(tasks.items()))
            return {name: fn()}
        results = {}
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(fn): name for name, fn in tasks.items()}
            try:
                for fut in as_completed(futures):
                    name = futures[fut]
                    results[name] = fut.result()
            except Exception:
                for other in futures:
                    other.cancel()
                raise
        return results

    # ── Small utility helpers ────────────────────────────────────────────────

    @staticmethod
    def _iter_batches(items, n):
        """Yield (batch_num, batch_slice) — batch_num is 1-indexed."""
        for i, start in enumerate(range(0, len(items), n), start=1):
            yield i, items[start:start + n]

    @staticmethod
    def _prev_prefix(snum, sent_map) -> str:
        prev = sent_map.get(snum - 1)
        return f"[prev: {prev.text}] " if prev else ""

    # ── Prompt builders (strings kept byte-identical to inline versions) ─────

    @staticmethod
    def _prompt_ambiguity(names) -> str:
        return f"""Classify these component names.

NAMES: {', '.join(names)}

{AMBIGUITY_FEW_SHOT}

NOW CLASSIFY THE NAMES ABOVE.

Return JSON:
{{
  "architectural": ["names that identify specific components"],
  "ambiguous": ["names that could easily be used as ordinary words in documentation"]
}}

{AMBIGUITY_RULES}

JSON only:"""

    @staticmethod
    def _prompt_doc_knowledge_extract(comp_names, doc_lines) -> str:
        return f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

{DOC_KNOWLEDGE_EXTRACTION_RULES}

{ALIAS_SCOPE_RULES}

DOCUMENT:
{chr(10).join(doc_lines)}

Return JSON:
{{
  "abbreviations": [{{"term": "short_form", "component": "FullComponent", "scope": "global"}}],
  "synonyms":      [{{"term": "specific_alternative_name", "component": "FullComponent", "scope": "local"}}]
}}
JSON only:"""

    @staticmethod
    def _prompt_doc_knowledge_judge(comp_names, mapping_list) -> str:
        return f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}

{DOC_KNOWLEDGE_JUDGE_EXAMPLES}

{DOC_KNOWLEDGE_JUDGE_RULES}

Return JSON:
{{"approved": ["term1", "term2"]}}
JSON only:"""

    @staticmethod
    def _prompt_extraction(comp_names, mappings, batch) -> str:
        return f"""Extract ALL references to components from this document.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings)}' if mappings else ''}

{ENTITY_EXTRACTION_RULES}

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence"}}]}}
JSON only:"""

    @staticmethod
    def _prompt_validation(comp_names, cases, focus) -> str:
        return f"""Validate components in a document. {focus}

COMPONENTS: {', '.join(comp_names)}

{VALIDATION_RULES}

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true}}]}}
JSON only:"""

    @staticmethod
    def _prompt_coref(comp_names, cases) -> str:
        prompt = f"""Resolve references (pronouns and noun phrases that refer back) to components.

COMPONENTS: {', '.join(comp_names)}

For each TARGET sentence below, identify any pronoun or noun phrase that
refers back to a component listed above. If a target sentence has no such
reference to a listed component, return no resolution for it. Be conservative — only include resolutions you are CERTAIN about.

"""
        for i, case in enumerate(cases):
            prompt += f"--- Case {i+1}: S{case['sent'].number} ---\n"
            prompt += "CONTEXT:\n" + "\n".join(case["context"]) + "\n"
            prompt += f"TARGET: S{case['sent'].number} (marked with >>>)\n\n"

        prompt += f"""{COREF_RULES}

{ANTECEDENT_ALIAS_RULES}

Return JSON:
{{"resolutions": [{{"case": 1, "sentence": N_INTEGER, "reference": "the server", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name", "antecedent_via_alias": false}}]}}

JSON only:"""
        return prompt

    # ── LLM call helper ──────────────────────────────────────────────────────

    def _ask(
        self,
        prompt: str,
        *,
        timeout: int = 120,
        label: str = "LLM call",
        phase: str | None = None,
        require: str | None = None,
        require_present: str | None = None,
    ) -> dict:
        """Query the LLM, parse JSON, retry once on empty/incomplete response.

        Success rule (in priority order):
          - require_present=KEY  → KEY must appear in the parsed dict (empty list OK)
          - require=KEY          → data[KEY] must be truthy (non-empty)
          - neither              → any non-empty parsed dict succeeds

        Returns the last parsed dict (possibly empty {}) — callers can still
        do `if not data: continue`.
        """
        if phase is not None:
            self.llm.set_phase(phase)

        def _ok(d: dict | None) -> bool:
            if not d:
                return False
            if require_present is not None:
                return require_present in d
            if require is not None:
                return bool(d.get(require))
            return True

        data: dict = {}
        for attempt in range(2):
            parsed = self.llm.extract_json(self.llm.query(prompt, timeout=timeout))
            if parsed is not None:
                data = parsed
            if _ok(data):
                return data
            if attempt == 0:
                print(f"    {label}: empty response, retrying...")
        return data

    # ── Main entry ───────────────────────────────────────────────────────────

    def link(self, text_path, model_path, **_kwargs):
        self._phase_log = []
        self._llm_calls.clear()
        self._phase_metrics = {}
        self._current_text_path = text_path
        t0 = time.time()

        components = parse_pcm_repository(model_path)
        sentences = load_sentences(text_path)
        name_to_id = {c.name: c.id for c in components}
        sent_map = build_sent_map(sentences)
        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        # ── Phase 1 ─────────────────────────────────────────────────────────
        t_p1 = time.time()
        print("\n[Phase 1] Knowledge acquisition (parallel)")
        knowledge = self._run_parallel({
            "model": lambda: self._analyze_model(components),
            "doc": lambda: self._learn_document_knowledge(sentences, components),
        })
        self.model_knowledge = knowledge["model"]
        self.doc_knowledge = knowledge["doc"]
        print(f"  Model: {len(self.model_knowledge.ambiguous_names)} ambiguous "
              f"of {len(components)} components")
        print(f"  Doc knowledge: {len(self.doc_knowledge.aliases)} aliases")
        self._save_phase(text_path, "layer1", {
            "model_knowledge": self.model_knowledge,
            "doc_knowledge": self.doc_knowledge,
            "elapsed_s": round(time.time() - t_p1, 2),
            "n_sentences": len(sentences), "n_components": len(components),
        })

        # ── Phase 2 ─────────────────────────────────────────────────────────
        t_p2 = time.time()
        print("\n[Phase 2] Entity extraction (Framing C, 2-pass UNION)")
        framing_c = self._run_framing_c(sentences, components, name_to_id, sent_map)
        print(f"  Framing C: {len(framing_c)} candidates")
        self._save_phase(text_path, "layer2", {
            "framing_c": framing_c,
            "framing_c_pass1": getattr(self, "_framing_c_pass1", None),
            "framing_c_pass2": getattr(self, "_framing_c_pass2", None),
            "elapsed_s": round(time.time() - t_p2, 2),
        })

        # ── Phase 3 ─────────────────────────────────────────────────────────
        # No multi-framing union needed — just C's output.
        candidates = list(framing_c.values())

        # ── Phase 4 ─────────────────────────────────────────────────────────
        t_p4 = time.time()
        print("\n[Phase 4] Twopass validation (V3 p1 + p2)")
        bundles = {
            (c.sentence_number, c.component_id): self._build_evidence_bundle(c, sent_map)
            for c in candidates
        }
        validated, entity_decisions = self._validate_with_evidence(
            candidates, bundles, components, sent_map,
            p1_tag="phase_4_twopass_p1", p2_tag="phase_4_twopass_p2",
            stage_label="entity")
        print(f"  Entity validated: {len(validated)} / {len(candidates)}")
        self._save_phase(text_path, "layer3", {
            "candidates": candidates, "validated": validated,
            "decisions": entity_decisions,
            "evidence_bundles": {k: asdict(v) for k, v in bundles.items()},
            "elapsed_s": round(time.time() - t_p4, 2),
        })

        # ── Phase 5 ─────────────────────────────────────────────────────────
        t_p5 = time.time()
        print("\n[Phase 5] Coreference ("
              ")")
        coref_raw, coref_metadata = self._run_coreference(
            sentences, components, name_to_id, sent_map)
        print(f"  Coref raw: {len(coref_raw)}")
        coref_validated, coref_decisions = self._validate_coref_links(
            coref_raw, sent_map, components)
        print(f"  Coref validated: {len(coref_validated)} / {len(coref_raw)}")
        self._save_phase(text_path, "layer4", {
            "coref_raw": coref_raw,
            "coref_validated": coref_validated,
            "coref_metadata": coref_metadata,
            "coref_decisions": coref_decisions,
            "elapsed_s": round(time.time() - t_p5, 2),
        })

        # ── Phase 6 ─────────────────────────────────────────────────────────
        print("\n[Phase 6] Dedup merge")
        entity_links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name,
                       source="entity")
            for c in validated
        ]
        all_links = entity_links + coref_validated
        seen: set[tuple] = set()
        final = []
        for lk in all_links:
            key = (lk.sentence_number, lk.component_id)
            if key not in seen:
                seen.add(key)
                final.append(lk)
        print(f"  Final: {len(final)} (from {len(all_links)} raw)")

        # ── Provenance + metrics ─────────────────────────────────────────────
        coref_keys = {(lk.sentence_number, lk.component_id) for lk in coref_validated}
        final_provenance: dict = {}
        for lk in final:
            key = (lk.sentence_number, lk.component_id)
            final_provenance[key] = {
                "from_coref": key in coref_keys,
                "source": lk.source,
                "entity_decision": entity_decisions.get(key),
                "coref_decision": coref_decisions.get(key),
                "coref_meta": coref_metadata.get(key),
            }

        total_elapsed = round(time.time() - t0, 1)
        self._phase_metrics = self._compute_phase_metrics()
        self._phase_metrics["_total"] = {
            "elapsed_s": total_elapsed, "llm_calls": len(self._llm_calls),
        }
        self._log("summary", {"total_time_s": total_elapsed},
                  {"final": len(final), "llm_calls": len(self._llm_calls)}, final)
        self._save_log(text_path)
        self._save_phase(text_path, "final", {
            "final": final, "final_provenance": final_provenance,
            "phase_metrics": self._phase_metrics,
            "backend": self._backend_tag(), "elapsed_s": total_elapsed,
        })
        print(f"\nFinal: {len(final)} links ({total_elapsed}s, "
              f"{len(self._llm_calls)} LLM calls)")
        return final

    # ── Phase 1 — Knowledge acquisition ─────────────────────────────────────

    def _analyze_model(self, components):
        self.llm.set_phase("phase_1_model")
        names = [c.name for c in components]
        knowledge = ModelKnowledge()
        prompt = self._prompt_ambiguity(names)
        data = self._ask(prompt, timeout=100, label="Ambiguity classification")
        if data:
            valid = set(names)
            raw_ambiguous = set(data.get("ambiguous", [])) & valid
            knowledge.ambiguous_names = {n for n in raw_ambiguous if len(n.split()) == 1}
        return knowledge

    def _learn_document_knowledge(self, sentences, components):
        self.llm.set_phase("phase_1_doc_extract")
        comp_names = [c.name for c in components]
        doc_lines = [s.text for s in sentences]

        prompt1 = self._prompt_doc_knowledge_extract(comp_names, doc_lines)

        data1 = self._ask(prompt1, timeout=300, label="Doc knowledge")

        all_mappings: dict[str, str] = {}
        all_scopes: dict[str, str] = {}
        if data1:
            abbr_recs = data1.get("abbreviations", [])
            syn_recs = data1.get("synonyms", [])
            if isinstance(abbr_recs, dict):
                abbr_recs = [{"term": k, "component": v, "scope": "local"} for k, v in abbr_recs.items()]
            if isinstance(syn_recs, dict):
                syn_recs = [{"term": k, "component": v, "scope": "local"} for k, v in syn_recs.items()]
            for rec in abbr_recs + syn_recs:
                if not isinstance(rec, dict):
                    continue
                term = rec.get("term")
                full = rec.get("component")
                scope = rec.get("scope", "local")
                if term and full in comp_names:
                    all_mappings[term] = full
                    all_scopes[term] = scope

        if all_mappings:
            mapping_list = [f"'{k}' -> {v}" for k, v in all_mappings.items()]
            prompt2 = self._prompt_doc_knowledge_judge(comp_names, mapping_list)
            data2 = self._ask(prompt2, timeout=120, label="Doc knowledge judge",
                              phase="phase_1_doc_judge", require="approved")
            approved = set(data2.get("approved", [])) if data2 else set(all_mappings.keys())
        else:
            approved = set()

        knowledge = DocumentKnowledge()
        for term, comp in all_mappings.items():
            if term in approved:
                scope = all_scopes.get(term, "local")
                if scope not in ("global", "local"):
                    scope = "local"
                knowledge.aliases[term] = AliasEntry(component=comp, scope=scope)
                print(f"    Alias: {term} -> {comp} [{scope}]")
        return knowledge

    # ── Phase 2 — Framing C only ────────────────────────────────────────────

    def _run_framing_c(self, sentences, components, name_to_id, sent_map) -> dict:
        comp_names = get_comp_names(components)
        mappings = (
            [f"{term}={entry.component}" for term, entry in self.doc_knowledge.aliases.items()
             if entry.scope == "global"]
            if self.doc_knowledge else []
        )
        results = self._run_parallel({
            "pass1": lambda: self._run_extraction_pass(
                sentences, comp_names, mappings, name_to_id, sent_map,
                pass_label="[C1] ", phase_tag="phase_2_framing_c_pass1"),
            "pass2": lambda: self._run_extraction_pass(
                sentences, comp_names, mappings, name_to_id, sent_map,
                pass_label="[C2] ", phase_tag="phase_2_framing_c_pass2"),
        })
        pass1, pass2 = results["pass1"], results["pass2"]
        # s19U / v2.6.5: UNION instead of intersection (mirrors s17g). The L3 intersection
        # gate killed ~5 BBB TPs for 0 FP saved; union keeps every candidate from either pass.
        unioned = {**pass2, **pass1}
        print(f"    Consensus: P1={len(pass1)} P2={len(pass2)} ∪={len(unioned)}")
        self._framing_c_pass1 = pass1
        self._framing_c_pass2 = pass2
        return unioned

    def _run_extraction_pass(self, sentences, comp_names, mappings,
                              name_to_id, sent_map, pass_label="", phase_tag=None):
        if phase_tag:
            self.llm.set_phase(phase_tag)
        batch_size = 50
        candidates: dict = {}
        for batch_num, batch in self._iter_batches(sentences, batch_size):
            if len(sentences) > batch_size:
                print(f"    {pass_label}batch {batch_num}: "
                      f"S{batch[0].number}-S{batch[-1].number} ({len(batch)} sents)")
            prompt = self._prompt_extraction(comp_names, mappings, batch)
            data = self._ask(prompt, timeout=240,
                             label=f"{pass_label}batch", require="references")
            if not data:
                continue
            for ref in data.get("references", []):
                cname = ref.get("component")
                snum = parse_snum(ref.get("sentence"))
                if snum is None or not cname or cname not in name_to_id:
                    continue
                sent = sent_map.get(snum)
                if not sent:
                    continue
                matched = ref.get("matched_text", "")
                if matched and matched.lower() not in sent.text.lower():
                    continue
                key = (snum, name_to_id[cname])
                if key not in candidates:
                    candidates[key] = CandidateLink(
                        snum, sent.text, cname, name_to_id[cname],
                        matched, source="entity",
                    )
        return candidates

    # ── Phase 4 — Mention classification + evidence bundle ──────────────────

    def _classify_mention_typed(self, comp_name: str, text: str) -> MentionType:
        if has_standalone_mention(comp_name, text):
            return MentionType.PROPER_STANDALONE
        comp_lower = comp_name.lower()
        if re.search(rf'\b{re.escape(comp_lower)}\b', text):
            if self._all_occurrences_in_qualified_path(comp_lower, text):
                return MentionType.CODE_TOKEN
            return MentionType.LOWERCASE_PROSE
        if self.doc_knowledge:
            for alias, entry in self.doc_knowledge.aliases.items():
                if entry.component == comp_name and re.search(
                    rf'\b{re.escape(alias)}\b', text, re.IGNORECASE
                ):
                    return MentionType.VIA_ALIAS
        return MentionType.INDIRECT

    @staticmethod
    def _all_occurrences_in_qualified_path(comp_lower: str, text: str) -> bool:
        any_match = False
        for m in re.finditer(rf'\b{re.escape(comp_lower)}\b', text):
            any_match = True
            s, e = m.start(), m.end()
            in_qualified_path = (
                (s > 0 and text[s - 1] == ".") or
                (e < len(text) and text[e] == "." and e + 1 < len(text)
                 and text[e + 1].isalpha())
            )
            if not in_qualified_path:
                return False
        return any_match

    def _build_evidence_bundle(self, candidate, sent_map, rationale="Framing C extraction"):
        comp_name = candidate.component_name
        snum = candidate.sentence_number
        mention_type = self._classify_mention_typed(comp_name, candidate.sentence_text).value
        prev_sent = sent_map.get(snum - 1)
        preceding_text = prev_sent.text if prev_sent else ""
        anchors = []
        for s in sorted(sent_map.values(), key=lambda x: x.number):
            if s.number == snum:
                continue
            if has_standalone_mention(comp_name, s.text):
                anchors.append(f"S{s.number}: {s.text}")
                if len(anchors) >= 5:
                    break
        is_ambig = bool(
            self.model_knowledge
            and self.model_knowledge.ambiguous_names
            and comp_name in self.model_knowledge.ambiguous_names
        )
        return EvidenceBundle(
            source=candidate.source,
            matched_span=candidate.matched_text or comp_name,
            mention_type=mention_type,
            preceding_text=preceding_text,
            anchor_sentences=anchors,
            is_ambiguous=is_ambig,
            extraction_rationale=rationale,
        )

    def _format_evidence(self, bundle: EvidenceBundle) -> str:
        lines = [
            f"  Evidence: source={bundle.source}, span=\"{bundle.matched_span}\", "
            f"mention={bundle.mention_type}, ambiguous={bundle.is_ambiguous}",
            f"  Rationale: {bundle.extraction_rationale}",
        ]
        if bundle.preceding_text:
            lines.append(f"  [prev: \"{bundle.preceding_text}\"]")
        if bundle.anchor_sentences:
            lines.append("  Anchors (confirmed refs):")
            for a in bundle.anchor_sentences:
                lines.append(f"    {a}")
        return "\n".join(lines)

    # ── Phase 4 — Twopass validation ────────────────────────────────────────

    def _validate_with_evidence(self, candidates, bundles, components, sent_map,
                                p1_tag, p2_tag, stage_label):
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
            r2 = self._run_validation_pass(comp_names, case_strings, P2_FOCUS, p2_tag)
            for i, (case_text, c) in enumerate(cases):
                p1 = r1.get(i, False)
                p2 = r2.get(i, False)
                ok = p1 and p2
                key = (c.sentence_number, c.component_id)
                decisions[key] = {
                    "approved": ok, "p1": p1, "p2": p2,
                    "path": f"{stage_label}_twopass" if ok else f"{stage_label}_twopass_reject",
                    "stage": f"{stage_label}_twopass",
                }
                if ok:
                    approved.append(c)
        return approved, decisions

    def _run_validation_pass(self, comp_names, cases, focus, phase_tag=None):
        if phase_tag:
            self.llm.set_phase(phase_tag)
        prompt = self._prompt_validation(comp_names, cases, focus)
        data = self._ask(prompt, timeout=120, label="Validation pass",
                         require="validations")
        results: dict[int, bool] = {}
        if data:
            for v in data.get("validations", []):
                idx = v.get("case", 0) - 1
                if 0 <= idx < len(cases):
                    val = v.get("approve", False)
                    results[idx] = val is True or (isinstance(val, str) and val.lower() == "true")
        return results

    # ── Phase 5 — Coreference discovery (no pre-filter regex) ───────────────

    def _antecedent_supports_resolution(self, comp_name: str, ant_text: str) -> bool:
        """Structural alias-aware antecedent gate (replaces 17f's LLM-flag bypass).

        True iff the antecedent sentence contains either the canonical component
        name (standalone match) or any known alias of that component.
        """
        if has_standalone_mention(comp_name, ant_text):
            return True
        if not self.doc_knowledge:
            return False
        for alias, entry in self.doc_knowledge.aliases.items():
            if entry.component != comp_name:
                continue
            if has_standalone_mention(alias, ant_text):
                return True
            if re.search(rf'\b{re.escape(alias)}\b', ant_text, re.IGNORECASE):
                return True
        return False

    def _run_coreference(self, sentences, components, name_to_id, sent_map):
        """Send EVERY sentence to LLM in batches; LLM identifies anaphoric
        references in context. No PRONOUN regex, no role-ref regex, no
        terminal classifier. Structural alias-aware antecedent gate.
        """
        comp_names = get_comp_names(components)
        all_coref = []
        coref_metadata: dict = {}
        self.llm.set_phase("phase_5_coref")

        # Batch all sentences in groups of 10 (LLM context window).
        for batch_num, batch in self._iter_batches(sentences, 10):
            cases = []
            for sent in batch:
                context = []
                for i in range(max(1, sent.number - 5), sent.number + 6):
                    s = sent_map.get(i)
                    if s:
                        marker = ">>>" if s.number == sent.number else "   "
                        context.append(f"{marker} S{s.number}: {s.text}")
                cases.append({"sent": sent, "context": context})

            prompt = self._prompt_coref(comp_names, cases)

            data = self._ask(prompt, timeout=300,
                             label=f"Coref batch {batch_num}",
                             require_present="resolutions")
            if not data:
                continue

            for res in data.get("resolutions", []):
                comp = res.get("component")
                snum = parse_snum(res.get("sentence"))
                if snum is None or not comp or comp not in name_to_id:
                    continue
                ant_snum = parse_snum(res.get("antecedent_sentence"))
                if ant_snum is None:
                    print(f"    Coref skip (no antecedent): S{snum} -> {comp}")
                    continue
                ant_sent = sent_map.get(ant_snum)
                if not ant_sent:
                    continue
                # Structural alias-aware gate.
                if not self._antecedent_supports_resolution(comp, ant_sent.text):
                    continue
                cid = name_to_id[comp]
                all_coref.append(SadSamLink(snum, cid, comp, source="coreference"))
                coref_metadata[(snum, cid)] = {
                    "reference": res.get("reference", ""),
                    "antecedent_sentence": ant_snum,
                    "antecedent_text": res.get("antecedent_text", ""),
                    # LLM flag retained for trace only — not used as a gate.
                    "antecedent_via_alias": bool(res.get("antecedent_via_alias", False)),
                    "raw_resolution": res,
                }
        return all_coref, coref_metadata

    # ── Phase 5 — Single-pass coref-focused validator ───────────────────────

    def _validate_coref_links(self, coref_links, sent_map, components):
        """Single-pass coref-focused validation. Asymmetric to entity twopass
        on principled grounds — anaphoric resolution asks a narrower question."""
        if not coref_links:
            return [], {}
        comp_names = get_comp_names(components)
        validated = []
        decisions: dict = {}
        self.llm.set_phase("phase_5_coref_validation")
        for _, batch in self._iter_batches(coref_links, 25):
            cases = []
            for i, lk in enumerate(batch):
                key = (lk.sentence_number, lk.component_id)
                sent = sent_map.get(lk.sentence_number)
                if not sent:
                    # Invariant violation: a coref link points at a sentence
                    # we don't have. Phase 4 built links from sent_map, so this
                    # means the map was mutated or the link was forged. Surface
                    # loudly — silently keeping it hides a real bug.
                    msg = (
                        f"!!! COREF VALIDATION INVARIANT VIOLATED !!! "
                        f"sentence_number=S{lk.sentence_number} missing from "
                        f"sent_map (component={lk.component_name}, "
                        f"component_id={lk.component_id}). "
                        f"sent_map keys present: {sorted(sent_map.keys())[:10]}"
                        f"{'…' if len(sent_map) > 10 else ''}. "
                        f"Keeping the link to preserve recall, but this should "
                        f"never happen — investigate Phase 4 extraction / "
                        f"sent_map construction."
                    )
                    print(f"\n{'!' * 80}\n{msg}\n{'!' * 80}\n", flush=True)
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)
                    validated.append(lk)
                    decisions[key] = {
                        "approved": True,
                        "path": "coref_no_sentence_keep",
                        "invariant_violation": True,
                    }
                    continue
                p = self._prev_prefix(lk.sentence_number, sent_map)
                cases.append((
                    i, lk,
                    f'Case {len(cases)+1}: pronoun/role-ref -> {lk.component_name}\n'
                    f'  {p}"{sent.text}"',
                ))
            if not cases:
                continue
            case_strings = [c for _, _, c in cases]
            results = self._run_validation_pass(
                comp_names, case_strings, COREF_VALIDATION_FOCUS,
                phase_tag="phase_5_coref_validation",
            )
            for idx, (i, lk, _) in enumerate(cases):
                key = (lk.sentence_number, lk.component_id)
                approved = bool(results.get(idx, False))
                decisions[key] = {
                    "approved": approved,
                    "path": "coref_validated" if approved else "coref_rejected",
                }
                if approved:
                    validated.append(lk)
                else:
                    print(f"    Coref reject: S{lk.sentence_number} -> {lk.component_name}")
        return validated, decisions

    # ── Logging / checkpointing ──────────────────────────────────────────────

    def _backend_tag(self) -> str:
        inner = getattr(self.llm, "_inner", self.llm)
        backend = getattr(inner, "backend", None)
        if backend is None:
            return "unknown"
        return getattr(backend, "value", str(backend))

    def _checkpoint_dir(self, text_path):
        cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
        ds = os.path.splitext(os.path.basename(text_path))[0]
        d = os.path.join(cache_dir, self._VARIANT_NAME, self._backend_tag(), ds)
        os.makedirs(d, exist_ok=True)
        return d

    def _save_phase(self, text_path, phase_name, state):
        d = self._checkpoint_dir(text_path)
        path = os.path.join(d, f"{phase_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"  Checkpoint: {phase_name} saved")

    def _log(self, phase, input_summary, output_summary, links=None):
        entry = {"phase": phase, "ts": time.time(),
                 "in": input_summary, "out": output_summary}
        if links is not None:
            entry["links"] = [
                {"s": l.sentence_number, "c": l.component_name, "src": l.source}
                for l in links
            ]
        self._phase_log.append(entry)

    def _save_log(self, text_path):
        log_dir = os.environ.get("LLM_LOG_DIR", "./results/llm_logs")
        os.makedirs(log_dir, exist_ok=True)
        ds = os.path.splitext(os.path.basename(text_path))[0]
        ts = time.strftime("%Y%m%d_%H%M%S")
        backend = self._backend_tag()
        summary_path = os.path.join(log_dir,
            f"{self._VARIANT_NAME}_{backend}_{ds}_{ts}.json")
        with open(summary_path, "w") as f:
            json.dump(self._phase_log, f, indent=2, default=str)
        print(f"  Phase log saved: {summary_path}")
        calls_path = os.path.join(log_dir,
            f"{self._VARIANT_NAME}_{backend}_{ds}_{ts}_calls.json")
        trunc_env = os.environ.get("CALLS_TRUNCATE_CHARS", "").strip()
        trunc = int(trunc_env) if trunc_env.isdigit() else 0
        if trunc > 0:
            calls = []
            for c in self._llm_calls:
                cc = dict(c)
                if cc.get("prompt") and len(cc["prompt"]) > trunc:
                    cc["prompt"] = cc["prompt"][:trunc] + f"... [truncated]"
                if cc.get("response_text") and len(cc["response_text"]) > trunc:
                    cc["response_text"] = cc["response_text"][:trunc] + f"... [truncated]"
                calls.append(cc)
        else:
            calls = self._llm_calls
        with open(calls_path, "w") as f:
            json.dump(calls, f, indent=2, default=str)
        print(f"  LLM call trace saved: {calls_path} ({len(self._llm_calls)} calls)")

    def _compute_phase_metrics(self) -> dict:
        metrics: dict[str, dict] = {}
        for call in self._llm_calls:
            ph = call.get("phase", "unknown")
            m = metrics.setdefault(ph, {"calls": 0, "elapsed_s": 0.0, "tokens": 0, "errors": 0})
            m["calls"] += 1
            m["elapsed_s"] = round(m["elapsed_s"] + call.get("elapsed_s", 0.0), 3)
            if call.get("success") is False:
                m["errors"] += 1
            usage = call.get("token_usage")
            if usage:
                m["tokens"] += usage.get("total_tokens", 0) or 0
        return metrics
