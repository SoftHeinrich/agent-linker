"""S-Linker17b — Multi-Framing Extraction (unified variant).

Three linguistically-motivated framings with sequential alias discovery and
k-voting merge. Each framing targets a distinct reference mechanism:
  Framing A (explicit-mention): "What components does this sentence name?"
  Framing B (actor-role): "What component is the agent of this sentence?"
  Framing C (alias-aware): "What components appear via known aliases?"

Phase 1: Alias discovery (doc_knowledge + model analysis) runs first.
Phase 2: All three framings run in parallel, all with alias knowledge.
Phase 3: k-voting merge — keep link if found by ≥2 framings.
Phase 4: Unified evidence-bundle validation on all k≥2 candidates.

This eliminates the execution-order artifact (seed runs before aliases,
entity runs after) and unifies the two validation approaches.

experimental=True, canonical=False.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from llm_sad_sam.core.data_types_v2 import (
    SadSamLink, CandidateLink,
    ModelKnowledge, DocumentKnowledge,
)
from llm_sad_sam.core.document_loader_v2 import (
    load_sentences, build_sent_map,
)
from llm_sad_sam.linkers.experimental.ilinker4 import ILinker4
from llm_sad_sam.linkers.experimental.helper_v3 import (
    has_standalone_mention,
    parse_snum,
    get_comp_names,
)
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient, LLMBackend


# ─────────────────────────────────────────────────────────────────────────────
# Inlined axiom prompts (v2.6.1 — standalone, no training/bank)
#
# Copied from prompts_v4_axiom (B-variant) with the three v2.6.1 FP root-cause
# fixes baked in directly. Inlined intentionally (standalone-file preference) so
# s_linker14_voyager and prompts_v4_axiom stay byte-for-byte untouched while
# s_linker17b carries the FP-fixed axioms. GATE-06: textbook SE domain terms only.
#
# FP fixes vs prompts_v4_axiom B-variant:
#   Cause A — DOC_KNOWLEDGE_JUDGE_RULES: + tier/technology-platform alias is invalid.
#   Cause B — ENTITY_EXTRACTION_RULES:   + code-path exclusion holds even when the
#             compound identifier is semantically related to the component.
#   Cause C — SEED_DISAMBIGUATION_RULES: not used in 17b (replaced by unified validation).
# ─────────────────────────────────────────────────────────────────────────────

AMBIGUITY_FEW_SHOT = """Example 1: Name = "Scheduler"
Sentence: "The Scheduler queues jobs and dispatches them to worker threads."
Classification: ARCHITECTURAL — "Scheduler" is the grammatical subject with a named role (queuing, dispatching). It identifies a specific mechanism, not a generic scheduling concept.

Example 2: Name = "Scheduler"
Sentence: "The system uses a scheduler-based approach to balance load across nodes."
Classification: AMBIGUOUS — "scheduler-based approach" describes a technique. Ordinary technical writing about any system would use "scheduler" here without naming a specific component."""

AMBIGUITY_RULES = """A name is ARCHITECTURAL when it identifies a specific role or mechanism. A name is AMBIGUOUS when ordinary technical writing about any system would use it generically without naming a specific component."""

DOC_KNOWLEDGE_EXTRACTION_RULES = """Find surface forms the document uses to refer to a single named component (introduced short forms, alternate names, or words of multi-word names when they alone clearly mean the full name). Reject terms whose ordinary English use dominates."""

DOC_KNOWLEDGE_JUDGE_EXAMPLES = """Example 1: Candidate = "Handler", Component = "RequestHandler"
Evidence: "The RequestHandler (hereafter Handler) processes incoming requests from clients."
Judgment: VALID — The document explicitly establishes "Handler" as an alternate name for RequestHandler via parenthetical definition. The alias is distinctive and scoped to one component.

Example 2: Candidate = "the system", Component = "CacheLayer"
Evidence: "The system stores frequently accessed records in the CacheLayer."
Judgment: INVALID — "the system" refers to the overall application, not to CacheLayer specifically. It names a different entity (the whole system) rather than establishing CacheLayer as an alias."""

# Cause A fix: tier/technology-platform alias rejection.
DOC_KNOWLEDGE_JUDGE_RULES = """An alias is valid when the document establishes an equivalence between a phrase and a single named component. An alias is invalid when the phrase is generic vocabulary, names the whole system, or names a different entity. An alias is also invalid when it names an architectural tier or technology platform that encompasses multiple elements, because it identifies a grouping rather than a single named unit. When uncertain, prefer APPROVE."""

# Cause B fix: code-path exclusion holds even for semantically-related compounds.
ENTITY_EXTRACTION_RULES = """Include a reference when the sentence refers to the component by name, alias, or as a participant in a described interaction. Exclude when the name appears only inside a code-level path — even if the compound identifier is semantically related to the component — or as ordinary English with no architectural intent. Favor inclusion."""

VALIDATION_RULES = """Approve when the sentence treats the component as an architectural participant, including counterparts. Reject when the matching word is generic, names a different entity, or describes a technique that merely shares the component's name."""

COREF_RULES = """For each case, decide whether a pronoun or role-referential noun phrase in the target sentence refers back to a component named or aliased earlier in the context. Resolve when: (a) the component's name or a known alias appears in the surrounding context sentences, or (b) only one component has been introduced in the immediately preceding sentences — treat it as the section-established topic and resolve role-referential phrases ("it", "the module", "the service", "the component", "the system") to that topic even without a direct name repetition. Avoid resolving when two or more equally plausible antecedents exist. Known aliases include the terminal word(s) of a multi-word name, documented abbreviations, and alternate forms used in the document. When the antecedent sentence uses a known alias rather than the full canonical name, set antecedent_via_alias=true."""

ALIAS_SCOPE_RULES = """For each alias, classify its SCOPE:
- "global": distinctive enough to unambiguously name the component anywhere in the document. Typical shapes: multi-word forms, hyphenated forms, CamelCase, all-caps abbreviations of length >= 2, or names beginning with an uppercase letter.
- "local": a single all-lowercase word overlapping with ordinary English vocabulary. Safe only where the surrounding context already establishes which component is being discussed.
Dotted-path fragments (tokens of the form X.Y or X.Y.Z) are NOT aliases — do not include them."""

ANTECEDENT_ALIAS_RULES = """For each resolution, set antecedent_via_alias:
- true:  the antecedent quote refers to the component by an ALIAS — a terminal word of a multi-word name, an abbreviation, a hyphenated form, or any documented alternate name rather than the canonical name listed in COMPONENTS.
- false: the antecedent quote uses the canonical name verbatim as listed in COMPONENTS.

Examples:
- COMPONENTS contains "TaskScheduler"; antecedent: "The scheduler queues jobs" -> true (uses terminal "scheduler", not canonical "TaskScheduler").
- COMPONENTS contains "TaskScheduler"; antecedent: "TaskScheduler queues jobs" -> false (canonical name verbatim).

Default to true when the antecedent form clearly differs from the canonical name but unambiguously identifies the component."""


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvidenceBundle:
    source: str
    matched_span: str
    mention_type: str
    preceding_text: str
    anchor_sentences: list[str]
    is_ambiguous: bool
    extraction_rationale: str


@dataclass(frozen=True)
class AliasEntry:
    component: str
    scope: str   # "global" | "local"


# ─────────────────────────────────────────────────────────────────────────────
# Main linker class
# ─────────────────────────────────────────────────────────────────────────────

class SLinker17b:
    """v2.6.1 unified multi-framing linker with k=2 voting.

    Sequential alias discovery followed by parallel extraction across three
    framings. k-voting merge (k=2) followed by unified evidence-bundle
    validation eliminates the execution-order artifact in s_linker15/17a
    where Framings A+B lack alias knowledge.

    experimental=True — research-grade; not canonical.
    canonical=False   — s_linker13_min remains canonical=True.
    """

    _VARIANT_NAME = "s_linker17b"

    PRONOUN_PATTERN = re.compile(
        r'\b(it|they|this|these|that|those|its|their)\b',
        re.IGNORECASE
    )

    def __init__(
        self,
        backend: LLMBackend | None = None,
        model: str | None = None,
        checkpoint_fallback: LLMBackend | str | None = None,
        checkpoint_fallback_model: str | None = None,
    ):
        os.environ.setdefault("CLAUDE_MODEL", "sonnet")
        os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.4")
        self.llm = LLMClient(
            backend=backend or LLMBackend.CLAUDE,
            model=model,
            checkpoint_fallback=checkpoint_fallback,
            checkpoint_fallback_model=checkpoint_fallback_model,
        )
        self.model_knowledge: ModelKnowledge | None = None
        self.doc_knowledge: DocumentKnowledge | None = None
        self._phase_log: list[dict] = []
        self._current_text_path: str | None = None

        # No bank, no training: axiom prompts (inlined above, with v2.6.1 FP fixes)
        # are used directly. No _wrap, no reload, no slot injection.
        self._AMBIGUITY_FEW_SHOT = AMBIGUITY_FEW_SHOT
        self._AMBIGUITY_RULES = AMBIGUITY_RULES
        self._DOC_KNOWLEDGE_EXTRACTION_RULES = DOC_KNOWLEDGE_EXTRACTION_RULES
        self._DOC_KNOWLEDGE_JUDGE_EXAMPLES = DOC_KNOWLEDGE_JUDGE_EXAMPLES
        self._DOC_KNOWLEDGE_JUDGE_RULES = DOC_KNOWLEDGE_JUDGE_RULES
        self._ENTITY_EXTRACTION_RULES = ENTITY_EXTRACTION_RULES
        self._VALIDATION_RULES = VALIDATION_RULES
        self._COREF_RULES = COREF_RULES
        self._ALIAS_SCOPE_RULES = ALIAS_SCOPE_RULES
        self._ANTECEDENT_ALIAS_RULES = ANTECEDENT_ALIAS_RULES

        # Note: no self._ilinker4 — 17b creates ILinker4 instances on-demand
        # in _run_framing_a and _run_framing_b with alias injection.

        print(f"SLinker17b (v2.6.1 unified multi-framing k=2, experimental=True)")
        print(f"  Backend: {self.llm.describe_backend()}")

    # ═══════════════════════════════════════════════════════════════════════
    # DAG Infrastructure
    # ═══════════════════════════════════════════════════════════════════════

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

    # ═══════════════════════════════════════════════════════════════════════
    # Main Entry Point
    # ═══════════════════════════════════════════════════════════════════════

    def link(self, text_path, model_path, **_kwargs):
        self._phase_log = []
        self._current_text_path = text_path
        t0 = time.time()

        components = parse_pcm_repository(model_path)
        sentences = load_sentences(text_path)
        name_to_id = {c.name: c.id for c in components}
        sent_map = build_sent_map(sentences)

        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        print("\n[Phase 1] Knowledge Acquisition (sequential)")
        knowledge = self._run_parallel({
            "model": lambda: self._analyze_model(components),
            "doc": lambda: self._learn_document_knowledge_enriched(sentences, components),
        })
        self.model_knowledge = knowledge["model"]
        self.doc_knowledge = knowledge["doc"]
        ambig = self.model_knowledge.ambiguous_names
        print(f"  Model: {len(ambig)} ambiguous (of {len(components)} components)")
        print(f"  Doc knowledge: {len(self.doc_knowledge.aliases)} aliases")

        self._log("layer1", {"sents": len(sentences), "comps": len(components)},
                  {"ambig": len(ambig), "aliases": len(self.doc_knowledge.aliases)})
        self._save_phase(text_path, "layer1", {
            "model_knowledge": self.model_knowledge,
            "doc_knowledge": self.doc_knowledge,
        })

        print("\n[Phase 2] Multi-Framing Extraction (parallel, all alias-aware)")
        framing_results = self._run_parallel({
            "framing_a": lambda: self._run_framing_a(sentences, components, name_to_id),
            "framing_b": lambda: self._run_framing_b(sentences, components, name_to_id),
            "framing_c": lambda: self._run_framing_c_raw(sentences, components, name_to_id, sent_map),
        })
        fa = framing_results["framing_a"]   # dict: (snum, cid) -> SadSamLink
        fb = framing_results["framing_b"]   # dict: (snum, cid) -> SadSamLink
        fc = framing_results["framing_c"]   # dict: (snum, cid) -> CandidateLink
        print(f"  Framing A: {len(fa)} links")
        print(f"  Framing B: {len(fb)} links")
        print(f"  Framing C: {len(fc)} links")

        self._save_phase(text_path, "layer2", {
            "framing_a": fa, "framing_b": fb, "framing_c": fc,
        })

        print("\n[Phase 3] k-Voting Merge (k=2)")
        candidates = self._kvoting_merge(fa, fb, fc, sent_map, k=2)
        print(f"  Candidates after k≥2 voting: {len(candidates)}")

        print("\n[Phase 4] Unified Validation")
        bundles = {
            (c.sentence_number, c.component_id): self._build_evidence_bundle(c, sent_map)
            for c in candidates
        }
        validated, decisions = self._validate_with_evidence(candidates, bundles, components, sent_map)
        print(f"  Validated: {len(validated)} / {len(candidates)}")

        self._save_phase(text_path, "layer3", {
            "candidates": candidates, "validated": validated, "decisions": decisions,
        })

        print("\n[Phase 5] Coreference")
        coref_links = self._run_coreference(sentences, components, name_to_id, sent_map)
        print(f"  Coreference: {len(coref_links)} links")

        print("\n[Phase 6] Final Merge")
        entity_links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name, source="multi_framing")
            for c in validated
        ]
        all_links = entity_links + coref_links
        seen: set[tuple] = set()
        final = []
        for lk in all_links:
            key = (lk.sentence_number, lk.component_id)
            if key not in seen:
                seen.add(key)
                final.append(lk)
        print(f"  After dedup: {len(final)} (from {len(all_links)} raw)")

        self._log("summary", {"total_time_s": round(time.time() - t0, 1)},
                  {"final": len(final)}, final)
        self._save_log(text_path)
        self._save_phase(text_path, "final", {"final": final})
        print(f"\nFinal: {len(final)} links ({time.time() - t0:.0f}s)")
        return final

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 1 — Knowledge Acquisition
    # ═══════════════════════════════════════════════════════════════════════

    def _analyze_model(self, components):
        names = [c.name for c in components]
        knowledge = ModelKnowledge()
        self._classify_components(names, knowledge)
        return knowledge

    def _classify_components(self, names, knowledge):
        prompt = f"""Classify these software architecture component names.

NAMES: {', '.join(names)}

{self._AMBIGUITY_FEW_SHOT}

NOW CLASSIFY THE NAMES ABOVE.

Return JSON:
{{
  "architectural": ["names that identify specific components"],
  "ambiguous": ["names that could easily be used as ordinary words in documentation"]
}}

{self._AMBIGUITY_RULES}

JSON only:"""

        data = None
        for attempt in range(2):
            data = self.llm.extract_json(self.llm.query(prompt, timeout=100))
            if data:
                break
            if attempt == 0:
                print("    Ambiguity classification: empty response, retrying...")
        if data:
            valid = set(names)
            raw_ambiguous = set(data.get("ambiguous", [])) & valid
            knowledge.ambiguous_names = {n for n in raw_ambiguous if len(n.split()) == 1}

    def _learn_document_knowledge_enriched(self, sentences, components):
        comp_names = [c.name for c in components]
        doc_lines = [s.text for s in sentences]

        prompt1 = f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

{self._DOC_KNOWLEDGE_EXTRACTION_RULES}

{self._ALIAS_SCOPE_RULES}

DOCUMENT:
{chr(10).join(doc_lines)}

Return JSON:
{{
  "abbreviations": [{{"term": "short_form", "component": "FullComponent", "scope": "global"}}],
  "synonyms":      [{{"term": "specific_alternative_name", "component": "FullComponent", "scope": "local"}}]
}}
JSON only:"""

        data1 = None
        for attempt in range(2):
            data1 = self.llm.extract_json(self.llm.query(prompt1, timeout=300))
            if data1:
                break
            if attempt == 0:
                print("    Doc knowledge: empty response, retrying...")

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

        data2 = None
        if all_mappings:
            mapping_list = [f"'{k}' -> {v}" for k, v in list(all_mappings.items())[:25]]
            prompt2 = f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}

{self._DOC_KNOWLEDGE_JUDGE_EXAMPLES}

{self._DOC_KNOWLEDGE_JUDGE_RULES}

Return JSON:
{{
  "approved": ["term1", "term2"]
}}
JSON only:"""

            for attempt in range(2):
                data2 = self.llm.extract_json(self.llm.query(prompt2, timeout=120))
                if data2 and data2.get("approved"):
                    break
                if attempt == 0:
                    print("    Doc knowledge judge: empty response, retrying...")
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

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2 — Multi-Framing Extraction (all alias-aware)
    # ═══════════════════════════════════════════════════════════════════════

    def _build_alias_rules(self) -> str:
        """Build alias injection string for ILinker4 slot injection."""
        if not self.doc_knowledge or not self.doc_knowledge.aliases:
            return ""
        lines = [
            f"- '{alias}' refers to '{entry.component}'"
            for alias, entry in self.doc_knowledge.aliases.items()
            if entry.scope == "global"
        ]
        if not lines:
            return ""
        return "Component aliases (treat as explicit references to the named component):\n" + "\n".join(lines)

    def _run_framing_a(self, sentences, components, name_to_id) -> dict:
        """Framing A: explicit-mention. ILinker4 Pass A with alias injection."""
        alias_rules = self._build_alias_rules()
        ilinker = ILinker4(llm=self.llm, seed_extraction_rules=alias_rules, seed_actor_rules="")
        comp_block = ilinker._build_comp_block(components)
        batches = ilinker._make_batches(sentences)
        links = ilinker._run_pass_batched(batches, comp_block, name_to_id, ilinker._prompt_extract)
        return {(lk.sentence_number, lk.component_id): lk for lk in links}

    def _run_framing_b(self, sentences, components, name_to_id) -> dict:
        """Framing B: actor-role. ILinker4 Pass B with alias injection."""
        alias_rules = self._build_alias_rules()
        ilinker = ILinker4(llm=self.llm, seed_extraction_rules="", seed_actor_rules=alias_rules)
        comp_block = ilinker._build_comp_block(components)
        batches = ilinker._make_batches(sentences)
        links = ilinker._run_pass_batched(batches, comp_block, name_to_id, ilinker._prompt_actor)
        return {(lk.sentence_number, lk.component_id): lk for lk in links}

    def _run_framing_c_raw(self, sentences, components, name_to_id, sent_map) -> dict:
        """Framing C: alias-aware entity extraction. Returns raw CandidateLink dict (pre-validation)."""
        candidates = self._extract_framing_c_candidates(sentences, components, name_to_id, sent_map)
        return {(c.sentence_number, c.component_id): c for c in candidates}

    def _extract_framing_c_candidates(self, sentences, components, name_to_id, sent_map):
        """Framing C extraction: 2-pass entity-style with alias injection + intersection."""
        comp_names = get_comp_names(components)
        mappings = (
            [f"{term}={entry.component}" for term, entry in self.doc_knowledge.aliases.items()
             if entry.scope == "global"]
            if self.doc_knowledge else []
        )
        results = self._run_parallel({
            "pass1": lambda: self._run_extraction_pass(
                sentences, comp_names, mappings, name_to_id, sent_map, pass_label="[C1] "),
            "pass2": lambda: self._run_extraction_pass(
                sentences, comp_names, mappings, name_to_id, sent_map, pass_label="[C2] "),
        })
        pass1, pass2 = results["pass1"], results["pass2"]
        intersected = {key: pass1[key] for key in pass1 if key in pass2}
        print(f"    Framing C consensus: Pass1={len(pass1)}, Pass2={len(pass2)}, "
              f"Intersect={len(intersected)}")
        return list(intersected.values())

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 3 — k-Voting Merge
    # ═══════════════════════════════════════════════════════════════════════

    def _kvoting_merge(self, fa: dict, fb: dict, fc: dict, sent_map, k: int = 2):
        """k-voting merge: keep (snum, cid) pairs with votes >= k across framings A, B, C."""
        all_keys = set(fa) | set(fb) | set(fc)
        candidates = []
        for key in all_keys:
            votes = (key in fa) + (key in fb) + (key in fc)
            if votes >= k:
                # Prefer CandidateLink from fc (has sentence_text + matched_text for evidence)
                if key in fc:
                    candidates.append(fc[key])
                else:
                    # Build CandidateLink from ExtractedLink (fa or fb)
                    lk = fa.get(key) or fb.get(key)
                    sent = sent_map.get(lk.sentence_number)
                    if sent:
                        candidates.append(CandidateLink(
                            lk.sentence_number, sent.text, lk.component_name, lk.component_id,
                            lk.matched_text, source="multi_framing",
                        ))
                    else:
                        print(f"    [kvoting] skip S{lk.sentence_number} -> {lk.component_name}: "
                              f"sentence not in sent_map (hallucinated snum?)")
        return candidates

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 4 — Unified Validation
    # ═══════════════════════════════════════════════════════════════════════

    def _classify_mention(self, comp_name: str, text: str) -> str:
        if has_standalone_mention(comp_name, text):
            return "proper case, standalone"
        comp_lower = comp_name.lower()
        if re.search(rf'\b{re.escape(comp_lower)}\b', text):
            for m in re.finditer(rf'\b{re.escape(comp_lower)}\b', text):
                s, e = m.start(), m.end()
                in_dotted = (
                    (s > 0 and text[s - 1] == ".") or
                    (e < len(text) and text[e] == "." and e + 1 < len(text)
                     and text[e + 1].isalpha())
                )
                if in_dotted:
                    return "lowercase, inside dotted path"
            return "lowercase mention"
        if self.doc_knowledge:
            for alias, entry in self.doc_knowledge.aliases.items():
                if entry.component == comp_name and re.search(
                    rf'\b{re.escape(alias)}\b', text, re.IGNORECASE
                ):
                    return f'via known alias "{alias}"'
        return "indirect/unclear match"

    def _build_evidence_bundle(self, candidate, sent_map, rationale="multi-framing k≥2 consensus"):
        comp_name = candidate.component_name
        snum = candidate.sentence_number
        mention_type = self._classify_mention(comp_name, candidate.sentence_text)
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
            lines.append(f"  [prev: \"{bundle.preceding_text[:80]}\"]")
        if bundle.anchor_sentences:
            lines.append("  Anchors (confirmed refs):")
            for a in bundle.anchor_sentences[:3]:
                lines.append(f"    {a[:100]}")
        return "\n".join(lines)

    def _validate_with_evidence(self, candidates, bundles, components, sent_map):
        """Unified 2-pass evidence-bundle validation for all k≥2 candidates."""
        if not candidates:
            return [], {}

        comp_names = get_comp_names(components)
        decisions: dict = {}

        generic_candidates: dict[str, list] = {}
        non_generic = []
        for c in candidates:
            sent = sent_map.get(c.sentence_number)
            if not sent:
                non_generic.append(c)
                continue
            comp_lower = c.component_name.lower()
            has_exact_case = has_standalone_mention(c.component_name, sent.text)
            has_lowercase = (not has_exact_case and
                             re.search(rf'\b{re.escape(comp_lower)}\b', sent.text))
            if has_lowercase and self.model_knowledge \
                    and self.model_knowledge.ambiguous_names \
                    and c.component_name in self.model_knowledge.ambiguous_names:
                generic_candidates.setdefault(c.component_name, []).append(c)
            else:
                non_generic.append(c)

        remaining = list(non_generic)
        for comp_name, cands in generic_candidates.items():
            anchor_lines = []
            for s in sent_map.values():
                if has_standalone_mention(comp_name, s.text):
                    anchor_lines.append(f"  S{s.number}: {s.text}")
                    if len(anchor_lines) >= 5:
                        break
            case_lines = []
            for i, c in enumerate(cands):
                s = sent_map.get(c.sentence_number)
                prev = sent_map.get(c.sentence_number - 1)
                prev_text = f" [prev: {prev.text[:60]}]" if prev else ""
                case_lines.append(f"  Case {i+1} (S{c.sentence_number}): {s.text}{prev_text}")
            anchor_section = (
                f'FULL-NAME REFERENCES (these definitely refer to the {comp_name} component):\n'
                + '\n'.join(anchor_lines) + '\n\n'
                if anchor_lines else ""
            )
            prompt = f"""CONTEXTUAL WORD USAGE: Does the word refer to the architecture component "{comp_name}", or is it used as an ordinary English word?

{anchor_section}SENTENCES TO CHECK (the component name appears only in lowercase or as part of a compound phrase):
{chr(10).join(case_lines)}

For each case, determine:
- COMPONENT: The word refers to the specific "{comp_name}" component as a system entity
  (e.g., "the {comp_name.lower()} handles requests" = component reference)
- GENERIC: The word is used as ordinary English describing a general concept, activity, or modifier
  (e.g., "provides {comp_name.lower()} access" or "{comp_name.lower()} operations" = generic usage)

Key distinction: A component reference names a specific system entity as a participant.
A generic use describes a type of activity or quality that happens to share the word.

Return JSON:
{{"results": [{{"case": 1, "usage": "component" or "generic", "reason": "brief"}}]}}
JSON only:"""

            data = None
            for attempt in range(2):
                data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
                if data and data.get("results"):
                    break
                if attempt == 0:
                    print(f"    Generic filter [{comp_name}]: empty response, retrying...")
            if not data:
                remaining.extend(cands)
                continue
            results_map: dict[int, dict] = {}
            for r in data.get("results", []):
                idx = r.get("case", 0) - 1
                results_map[idx] = r
            for i, c in enumerate(cands):
                result = results_map.get(i, {})
                usage = (result.get("usage", "component") or "component").lower()
                key = (c.sentence_number, c.component_id)
                if usage == "generic":
                    reason = result.get("reason", "")
                    print(f"    LLM generic reject: S{c.sentence_number} -> {c.component_name} ({reason})")
                    decisions[key] = {"approved": False, "path": f"generic_filter: {reason}"}
                else:
                    remaining.append(c)

        print(f"    LLM 2-pass validation (+evidence bundles): {len(remaining)} candidates")
        twopass_approved = []
        for batch_start in range(0, len(remaining), 25):
            batch = remaining[batch_start:batch_start + 25]
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
            r1 = self._run_validation_pass(comp_names, case_strings,
                "Check architectural participation: does the sentence name this component as an architectural participant — performing operations, providing services, or taking part in the described system behavior?")
            r2 = self._run_validation_pass(comp_names, case_strings,
                "Check referential specificity: is the component name used to identify this specific architectural element, or does it serve as a generic technical term in this sentence?")

            for i, (case_text, c) in enumerate(cases):
                p1 = r1.get(i, False)
                p2 = r2.get(i, False)
                approved = p1 and p2
                key = (c.sentence_number, c.component_id)
                decisions[key] = {
                    "approved": approved, "p1": p1, "p2": p2,
                    "path": "twopass" if approved else "twopass_reject",
                }
                if approved:
                    twopass_approved.append(c)

        return twopass_approved, decisions

    def _run_validation_pass(self, comp_names, cases, focus):
        prompt = f"""Validate component references in a software architecture document. {focus}

COMPONENTS: {', '.join(comp_names)}

{self._VALIDATION_RULES}

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true}}]}}
JSON only:"""

        data = None
        for attempt in range(2):
            data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
            if data and data.get("validations"):
                break
            if attempt == 0:
                print(f"    Validation pass: empty response, retrying...")
        results: dict[int, bool] = {}
        if data:
            for v in data.get("validations", []):
                idx = v.get("case", 0) - 1
                if 0 <= idx < len(cases):
                    val = v.get("approve", False)
                    results[idx] = val is True or (isinstance(val, str) and val.lower() == "true")
        return results

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 5 — Coreference
    # ═══════════════════════════════════════════════════════════════════════

    def _run_coreference(self, sentences, components, name_to_id, sent_map):
        anaphoric_count = sum(1 for s in sentences if self.PRONOUN_PATTERN.search(s.text))
        print(f"    Coreference: cases-in-context ({anaphoric_count} anaphoric sents / {len(sentences)} total)")
        return self._coref_cases_in_context(sentences, components, name_to_id, sent_map)

    def _classify_specific_terminals(self, components) -> set[str]:
        """LLM-driven: which multi-word component terminal words are specific enough for role-ref coref."""
        multi_word = [c for c in components if len(c.name.split()) > 1]
        if not multi_word:
            return set()
        terminal_words = sorted({c.name.split()[-1].lower() for c in multi_word})
        component_names = [c.name for c in multi_word]

        cache_key = tuple(terminal_words)
        if not hasattr(self, "_terminal_cache"):
            self._terminal_cache: dict = {}
        if cache_key in self._terminal_cache:
            return self._terminal_cache[cache_key]

        prompt = f"""Architecture components have multi-word names. Identify which terminal words (last word of each name) are SPECIFIC enough to serve as unambiguous role references in technical documentation.

Component names: {component_names}
Terminal words to evaluate: {terminal_words}

A terminal word is GENERIC if it could refer to any component in any system on its own — such as words that broadly describe a role or architectural tier.
A terminal word is SPECIFIC if it is distinctive or unusual enough that "the <word>" in a sentence about this system most likely refers to one specific component.
Also mark a terminal as GENERIC if multiple components in this list share the same terminal word (ambiguous).

Return JSON:
{{"specific": ["word1", "word2"], "generic": ["word3", "word4"]}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=60))
        if not data:
            print("  [coref] WARNING: terminal classification failed, role refs disabled")
            self._terminal_cache[cache_key] = set()
            return set()
        valid = set(terminal_words)
        result = {w.lower() for w in data.get("specific", []) if w.lower() in valid}
        self._terminal_cache[cache_key] = result
        return result

    def _coref_cases_in_context(self, sentences, components, name_to_id, sent_map):
        comp_names = get_comp_names(components)
        all_coref = []

        comp_terminals = self._classify_specific_terminals(components)
        role_ref_pat = re.compile(
            r'\bthe (' + '|'.join(re.escape(w) for w in sorted(comp_terminals)) + r')\b',
            re.IGNORECASE
        ) if comp_terminals else None

        anaphoric_sents = [
            s for s in sentences
            if self.PRONOUN_PATTERN.search(s.text)
            or (role_ref_pat and role_ref_pat.search(s.text))
        ]

        for batch_start in range(0, len(anaphoric_sents), 10):
            batch = anaphoric_sents[batch_start:batch_start + 10]
            cases = []
            for sent in batch:
                context = []
                for i in range(max(1, sent.number - 5), sent.number + 6):
                    s = sent_map.get(i)
                    if s:
                        marker = ">>>" if s.number == sent.number else "   "
                        context.append(f"{marker} S{s.number}: {s.text}")
                cases.append({"sent": sent, "context": context})

            prompt = f"""Resolve anaphoric references (pronouns and role-referential noun phrases) to architecture components.

COMPONENTS: {', '.join(comp_names)}

"""
            for i, case in enumerate(cases):
                prompt += f"--- Case {i+1}: S{case['sent'].number} ---\n"
                prompt += "CONTEXT:\n" + "\n".join(case["context"]) + "\n"
                prompt += f"TARGET: S{case['sent'].number} (marked with >>>)\n\n"

            prompt += f"""{self._COREF_RULES}

{self._ANTECEDENT_ALIAS_RULES}

Return JSON:
{{"resolutions": [{{"case": 1, "sentence": N_INTEGER, "reference": "the server", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name", "antecedent_via_alias": false}}]}}

Only include resolutions you are CERTAIN about. JSON only:"""

            data = None
            for attempt in range(2):
                data = self.llm.extract_json(self.llm.query(prompt, timeout=300))
                if data and data.get("resolutions"):
                    break
                if attempt == 0:
                    print(f"    Coref batch: empty response, retrying...")
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
                if not (has_standalone_mention(comp, ant_sent.text) or res.get("antecedent_via_alias", False)):
                    continue
                all_coref.append(SadSamLink(snum, name_to_id[comp], comp, source="coreference"))

        return all_coref

    # ═══════════════════════════════════════════════════════════════════════
    # Shared helpers (entity extraction pass)
    # ═══════════════════════════════════════════════════════════════════════

    def _run_extraction_pass(self, sentences, comp_names, mappings,
                             name_to_id, sent_map, pass_label=""):
        batch_size = 50
        candidates: dict = {}
        for batch_start in range(0, len(sentences), batch_size):
            batch = sentences[batch_start:batch_start + batch_size]
            if len(sentences) > batch_size:
                print(f"    {pass_label}Framing C batch {batch_start//batch_size + 1}: "
                      f"S{batch[0].number}-S{batch[-1].number} ({len(batch)} sents)")
            prompt = f"""Extract ALL references to software architecture components from this document.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings[:20])}' if mappings else ''}

{self._ENTITY_EXTRACTION_RULES}

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence"}}]}}
JSON only:"""

            data = None
            for attempt in range(2):
                data = self.llm.extract_json(self.llm.query(prompt, timeout=240))
                if data and data.get("references"):
                    break
                if attempt == 0:
                    print(f"    {pass_label}Empty response, retrying batch...")
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
                    candidates[key] = CandidateLink(snum, sent.text, cname, name_to_id[cname],
                                               matched, source="framing_c")
        return candidates

    # ═══════════════════════════════════════════════════════════════════════
    # Checkpoint & Logging
    # ═══════════════════════════════════════════════════════════════════════

    def _checkpoint_dir(self, text_path):
        cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
        ds = os.path.splitext(os.path.basename(text_path))[0]
        d = os.path.join(cache_dir, self._VARIANT_NAME, ds)
        assert self._VARIANT_NAME in d, (
            f"_checkpoint_dir must contain _VARIANT_NAME "
            f"('{self._VARIANT_NAME}' not in '{d}')"
        )
        os.makedirs(d, exist_ok=True)
        return d

    def _save_phase(self, text_path, phase_name, state):
        d = self._checkpoint_dir(text_path)
        path = os.path.join(d, f"{phase_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"  Checkpoint: {phase_name} saved")

    def _log(self, phase, input_summary, output_summary, links=None):
        entry = {"phase": phase, "ts": time.time(), "in": input_summary, "out": output_summary}
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
        path = os.path.join(
            log_dir,
            f"{self._VARIANT_NAME}_{ds}_{time.strftime('%Y%m%d_%H%M%S')}.json",
        )
        with open(path, "w") as f:
            json.dump(self._phase_log, f, indent=2, default=str)
        print(f"  Phase log saved: {path}")
