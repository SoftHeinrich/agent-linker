"""CNR + V26a Linker — Component Name Recovery + full V26a pipeline.

Discovers component names from the SAD text alone (no PCM model needed for discovery),
then runs the full V26a pipeline (Phases 0-9) using discovered components as if they
came from a PCM model.

Architecture:
  Phase DA (optional): Document Analysis — extract entities, synonyms, abbreviations
  Phase CNR: Component Name Discovery (from cnr_linker.py, DA-informed if enabled)
  Phase 4 override: ILinker2 extraction on discovered components (replaces TransArc CSV)
  Phases 0-9: Full V26a pipeline unchanged (Phase 3 gets DA warm-start if enabled)
  Eval Bridge: Map discovered names → PCM IDs (for evaluation only)

This uses composition over duplication: delegates V26a pipeline to AgentLinkerV26a,
overriding only the component source and Phase 4 seed.
"""

import json
import os
import re
import time
from typing import Optional

from ...core.data_types import SadSamLink, DocumentKnowledge
from ...core.document_loader import DocumentLoader, Sentence
from ...llm_client import LLMClient, LLMBackend
from ...pcm_parser import ArchitectureComponent, parse_pcm_repository
from .cnr_linker import CNRLinker, DocumentAnalysis
from .ilinker2 import ILinker2
from .agent_linker_v26a import AgentLinkerV26a


class CNRV26aLinker(AgentLinkerV26a):
    """Component Name Recovery + full V26a pipeline.

    Overrides V26a to:
    1. Discover components from text (CNR) instead of loading from PCM
    2. Use ILinker2 extraction as Phase 4 seed (on discovered components)
    3. Apply eval bridge to remap cnr IDs → PCM IDs for evaluation
    4. (If enable_da) Warm-start Phase 3 document knowledge from DA results
    """

    def __init__(self, backend: Optional[LLMBackend] = None, enable_da: bool = False):
        super().__init__(backend=backend)
        self._enable_da = enable_da
        self._cnr = CNRLinker(backend=backend or LLMBackend.CLAUDE, enable_da=enable_da)
        self._ilinker2 = ILinker2(backend=self.llm.backend)
        self._eval_bridge_map: dict[str, str] = {}
        self._discovered_components: list[ArchitectureComponent] = []
        da_str = " + DA warm-start" if enable_da else ""
        print(f"CNRV26aLinker: CNR discovery + full V26a pipeline{da_str}")

    def link(self, text_path: str, model_path: str = None,
             transarc_csv: str = None, resume_from_phase: int = None) -> list[SadSamLink]:
        """Run CNR discovery, then full V26a pipeline, then eval bridge.

        Args:
            text_path: Path to SAD text file.
            model_path: Optional PCM model path (used ONLY for eval bridge, never for discovery).
            transarc_csv: Accepted but ignored (ILinker2 replaces TransArc).
            resume_from_phase: Optional phase to resume from (for checkpointing).

        Returns:
            list[SadSamLink] with PCM IDs if model_path given, else synthetic cnr_N IDs.
        """
        self._phase_log = []
        t0 = time.time()

        sentences = DocumentLoader.load_sentences(text_path)
        print(f"Loaded {len(sentences)} sentences")

        # ── Phase CNR: Discover component names from text ──
        print(f"\n{'='*60}")
        print(f"PHASE CNR: Component Name Discovery")
        print(f"{'='*60}")
        self._discovered_components = self._cnr.discover_components_from_sentences(sentences)
        self._cached_text_path = text_path
        self._cached_model_path = model_path

        # Build synthetic model path info for V26a pipeline
        # V26a's link() calls parse_pcm_repository(model_path) — we override that behavior
        # by caching discovered components and injecting them
        self._cached_components = self._discovered_components
        name_to_id = {c.name: c.id for c in self._discovered_components}
        id_to_name = {c.id: c.name for c in self._discovered_components}
        sent_map = DocumentLoader.build_sent_map(sentences)
        self._cached_sent_map = sent_map

        components = self._discovered_components
        print(f"\nUsing {len(components)} discovered components for V26a pipeline")

        # ── Run V26a pipeline (Phases 0-9) with discovered components ──
        print(f"\n{'='*60}")
        print(f"V26A PIPELINE (Phases 0-9)")
        print(f"{'='*60}")

        # Phase 0: Document profile
        print("\n[Phase 0] Document Profile")
        self.doc_profile = self._learn_document_profile(sentences, components)
        self._is_complex = self._structural_complexity(sentences, components)
        spc = len(sentences) / max(1, len(components))
        print(f"  Stats: {spc:.1f} sents/comp, {self.doc_profile.pronoun_ratio:.0%} pronouns")
        print(f"  Complex: {self._is_complex}")
        from ...core.data_types import LearnedThresholds
        self.thresholds = LearnedThresholds(0, 0, 0, 0, "qualitative", 0)

        # Phase 1: Model Structure
        print("\n[Phase 1] Model Structure")
        self.model_knowledge = self._analyze_model(components)
        arch = self.model_knowledge.architectural_names
        ambig = self.model_knowledge.ambiguous_names
        self.GENERIC_COMPONENT_WORDS = set()
        for name in ambig:
            if ' ' not in name and not name.isupper():
                self.GENERIC_COMPONENT_WORDS.add(name.lower())
        print(f"  Architectural: {len(arch)}, Ambiguous: {sorted(ambig)}")
        print(f"  Discovered generic component words: {sorted(self.GENERIC_COMPONENT_WORDS)}")

        self.GENERIC_PARTIALS = set()
        for comp in components:
            parts = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)', comp.name)
            for part in parts:
                p_lower = part.lower()
                if part.isupper():
                    continue
                if len(p_lower) >= 3 and (p_lower in ambig or any(
                    p_lower == a.lower() for a in ambig
                )):
                    self.GENERIC_PARTIALS.add(p_lower)
        for name in ambig:
            if ' ' not in name and not name.isupper():
                self.GENERIC_PARTIALS.add(name.lower())
        print(f"  Discovered generic partials: {sorted(self.GENERIC_PARTIALS)}")

        # Phase 2: Pattern learning
        print("\n[Phase 2] Pattern Learning")
        self.learned_patterns = self._learn_patterns_with_debate(sentences, components)
        print(f"  Subprocess terms: {len(self.learned_patterns.subprocess_terms)}")

        # Phase 3: Document knowledge
        print("\n[Phase 3] Document Knowledge")
        self.doc_knowledge = self._learn_document_knowledge_enriched(sentences, components)
        print(f"  Abbrev: {len(self.doc_knowledge.abbreviations)}, "
              f"Syn: {len(self.doc_knowledge.synonyms)}, "
              f"Generic: {len(self.doc_knowledge.generic_terms)}")

        # Phase 3b: Multi-word partial enrichment
        self._enrich_multiword_partials(sentences, components)

        # Phase 4: ILinker2 extraction (replaces TransArc)
        print("\n[Phase 4] ILinker2 Extraction (replaces TransArc)")
        i2_links = self._run_ilinker2_extraction(sentences, components, name_to_id, sent_map)
        transarc_links = i2_links  # Relabeled as "transarc" for downstream compatibility
        transarc_set = {(l.sentence_number, l.component_id) for l in transarc_links}
        print(f"  Links: {len(transarc_links)}")

        # Phase 5: Entity extraction
        print("\n[Phase 5] Entity Extraction")
        candidates = self._extract_entities_enriched(sentences, components, name_to_id, sent_map)
        print(f"  Candidates: {len(candidates)}")

        before_guard = len(candidates)
        candidates = self._apply_abbreviation_guard_to_candidates(candidates, sent_map)
        if len(candidates) < before_guard:
            print(f"  After abbrev guard: {len(candidates)} (-{before_guard - len(candidates)})")

        # Phase 5b: Targeted recovery
        entity_comps = {c.component_name for c in candidates}
        transarc_comps = {l.component_name for l in transarc_links}
        covered_comps = entity_comps | transarc_comps
        unlinked = [c for c in components if c.name not in covered_comps]

        if unlinked:
            print(f"\n[Phase 5b] Targeted Recovery ({len(unlinked)} unlinked components)")
            extra = self._targeted_extraction(unlinked, sentences, name_to_id, sent_map,
                                              components=components, transarc_links=transarc_links,
                                              entity_candidates=candidates)
            if extra:
                print(f"  Found: {len(extra)} additional candidates")
                candidates.extend(extra)
            else:
                print(f"  Found: 0 additional candidates")

        # Phase 6: Validation
        print("\n[Phase 6] Validation")
        validated = self._validate_intersect(candidates, components, sent_map)
        print(f"  Validated: {len(validated)} (of {len(candidates)})")

        # Phase 7: Coreference
        print("\n[Phase 7] Coreference")
        if self._is_complex:
            print(f"  Mode: debate (complex, {len(sentences)} sents)")
            coref_links = self._coref_debate(sentences, components, name_to_id, sent_map)
        else:
            discourse_model = self._build_discourse_model(sentences, components, name_to_id)
            print(f"  Mode: discourse ({len(sentences)} sents)")
            coref_links = self._coref_discourse(sentences, components, name_to_id, sent_map, discourse_model)

        before_coref = len(coref_links)
        coref_links = self._filter_generic_coref(coref_links, sent_map)
        if len(coref_links) < before_coref:
            print(f"  After generic filter: {len(coref_links)} (-{before_coref - len(coref_links)})")

        coref_set = {(l.sentence_number, l.component_id) for l in coref_links}
        pronoun_links = self._deterministic_pronoun_coref(
            sentences, components, name_to_id, sent_map,
            transarc_set | {(c.sentence_number, c.component_id) for c in validated} | coref_set)
        if pronoun_links:
            coref_links.extend(pronoun_links)
            print(f"  Deterministic pronoun coref: +{len(pronoun_links)}")
        print(f"  Coref links: {len(coref_links)}")

        # Phase 8: Implicit References — SKIPPED
        reason = "complex doc" if self._is_complex else "dead weight"
        print(f"\n[Phase 8] Implicit References — SKIPPED ({reason})")

        # Phase 8b: Partial-reference injection
        existing = (transarc_set
                    | {(c.sentence_number, c.component_id) for c in validated}
                    | {(l.sentence_number, l.component_id) for l in coref_links})
        partial_links = self._inject_partial_references(
            sentences, components, name_to_id, transarc_set,
            {(c.sentence_number, c.component_id) for c in validated},
            {(l.sentence_number, l.component_id) for l in coref_links},
            set(),
        )
        if partial_links:
            print(f"\n[Phase 8b] Partial Injection")
            print(f"  Injected: {len(partial_links)} candidates")

        # Combine + deduplicate
        from collections import defaultdict
        entity_links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
            for c in validated
        ]
        all_links = transarc_links + entity_links + coref_links + partial_links
        link_map = {}
        for lk in all_links:
            key = (lk.sentence_number, lk.component_id)
            if key not in link_map:
                link_map[key] = lk
            else:
                old_p = self.SOURCE_PRIORITY.get(link_map[key].source, 0)
                new_p = self.SOURCE_PRIORITY.get(lk.source, 0)
                if new_p > old_p:
                    link_map[key] = lk
        preliminary = list(link_map.values())

        # Parent-overlap guard
        if self.model_knowledge and self.model_knowledge.impl_to_abstract:
            child_to_parent = self.model_knowledge.impl_to_abstract
            sent_comps = defaultdict(set)
            for lk in preliminary:
                sent_comps[lk.sentence_number].add(lk.component_name)
            before_po = len(preliminary)
            filtered_po = []
            for lk in preliminary:
                parent = child_to_parent.get(lk.component_name)
                if parent and parent in sent_comps[lk.sentence_number]:
                    print(f"    Parent-overlap drop: S{lk.sentence_number} -> {lk.component_name}")
                else:
                    filtered_po.append(lk)
            if len(filtered_po) < before_po:
                print(f"  Parent-overlap guard: dropped {before_po - len(filtered_po)}")
            preliminary = filtered_po

        # Phase 8c: Boundary filters
        print("\n[Phase 8c] Boundary Filters (non-TransArc only)")
        preliminary, boundary_rejected = self._apply_boundary_filters(
            preliminary, sent_map, transarc_set
        )
        if boundary_rejected:
            print(f"  Rejected: {len(boundary_rejected)}")

        # Phase 9: Judge Review
        print("\n[Phase 9] Judge Review (TransArc immune)")
        reviewed = self._judge_review(preliminary, sentences, components, sent_map, transarc_set)
        rejected = [l for l in preliminary if (l.sentence_number, l.component_id)
                    not in {(r.sentence_number, r.component_id) for r in reviewed}]
        print(f"  Approved: {len(reviewed)} (rejected {len(rejected)})")

        # Phase 10: FN Recovery — SKIPPED
        print("\n[Phase 10] FN Recovery — SKIPPED (dead weight)")
        final = reviewed

        # ── Eval bridge (if model_path provided) ──
        if model_path:
            print(f"\n{'='*60}")
            print(f"EVAL BRIDGE: Mapping to PCM IDs")
            print(f"{'='*60}")
            pcm_components = parse_pcm_repository(model_path)
            self._eval_bridge_map = self._cnr._build_eval_bridge(
                self._discovered_components, pcm_components)
            final = self._remap_links(final)
            print(f"  After eval bridge remap: {len(final)} links")

        # Save log
        self._log("summary", {"total_time_s": round(time.time() - t0, 1)},
                  {"final": len(final)}, final)
        self._save_log(text_path)

        print(f"\nFinal: {len(final)} links")
        return final

    # ── ILinker2 extraction (Phase 4 replacement) ────────────────────────

    def _run_ilinker2_extraction(self, sentences, components, name_to_id, sent_map):
        """Run ILinker2-style two-pass extraction on discovered components.

        Returns links with source="transarc" so downstream phases treat them
        identically to real TransArc/ILinker2 seed links.
        """
        from .ilinker2 import ILinker2, BATCH_SIZE, BATCH_OVERLAP, ExtractedLink

        comp_block = "\n".join(f"  {i+1}. {c.name}" for i, c in enumerate(components))

        # Include CNR discovered aliases
        alias_info = []
        for canonical, aliases in self._cnr._discovered_aliases.items():
            for a in aliases:
                alias_info.append(f"{a} = {canonical}")
        alias_block = f"\nKNOWN ALIASES: {', '.join(alias_info)}" if alias_info else ""

        # Build batches
        if len(sentences) <= BATCH_SIZE:
            batches = [sentences]
        else:
            batches, start = [], 0
            while start < len(sentences):
                end = min(start + BATCH_SIZE, len(sentences))
                batches.append(sentences[start:end])
                if end >= len(sentences):
                    break
                start = end - BATCH_OVERLAP

        # Pass A: Extraction
        pass_a = self._i2_pass_batched(batches, comp_block, alias_block, name_to_id,
                                        self._i2_prompt_extract)
        print(f"    Pass A (extract): {len(pass_a)} links")

        # Pass B: Actor
        pass_b = self._i2_pass_batched(batches, comp_block, alias_block, name_to_id,
                                        self._i2_prompt_actor)
        print(f"    Pass B (actor):   {len(pass_b)} links")

        # Merge: exact from either, non-exact intersection
        result_map: dict[tuple[int, str], tuple] = {}
        for link in pass_a + pass_b:
            key = (link[0], link[2])  # (snum, cid)
            if link[4] == "exact":
                result_map[key] = link

        a_keys = {(l[0], l[2]) for l in pass_a if l[4] != "exact"}
        b_keys = {(l[0], l[2]) for l in pass_b if l[4] != "exact"}
        agreed = a_keys & b_keys
        lookup = {}
        for link in pass_b + pass_a:
            key = (link[0], link[2])
            if link[4] != "exact":
                lookup[key] = link
        for key in agreed:
            if key not in result_map:
                result_map[key] = lookup[key]

        merged = list(result_map.values())
        print(f"    Merged: {len(merged)} links")

        # Convert to SadSamLink with source="transarc"
        links = []
        for snum, cname, cid, matched, mtype in merged:
            sent = sent_map.get(snum)
            if sent and self._in_dotted_path(sent.text, cname):
                continue
            links.append(SadSamLink(
                sentence_number=snum,
                component_id=cid,
                component_name=cname,
                confidence=0.92,
                source="transarc",
            ))
        return links

    def _i2_pass_batched(self, batches, comp_block, alias_block, name_to_id,
                          prompt_fn) -> list[tuple]:
        """Run one I2 extraction pass. Returns list of (snum, cname, cid, matched, type)."""
        seen: dict[tuple[int, str], tuple] = {}
        for i, batch in enumerate(batches):
            doc_block = "\n".join(f"S{s.number}: {s.text}" for s in batch)
            prompt = prompt_fn(doc_block, comp_block, alias_block)
            response = self.llm.query(prompt, timeout=300)
            if not response.success:
                continue
            data = self.llm.extract_json(response)
            if not data or "links" not in data:
                continue
            for item in data["links"]:
                snum = item.get("s")
                cname = item.get("c", "")
                if snum is None or not cname:
                    continue
                cid = name_to_id.get(cname)
                if not cid:
                    for name, nid in name_to_id.items():
                        if name.lower() == cname.lower():
                            cid, cname = nid, name
                            break
                if not cid:
                    for canonical, aliases in self._cnr._discovered_aliases.items():
                        if cname in aliases or cname.lower() in [a.lower() for a in aliases]:
                            cid = name_to_id.get(canonical)
                            cname = canonical
                            break
                if not cid:
                    continue
                key = (int(snum), cid)
                if key not in seen:
                    seen[key] = (int(snum), cname, cid,
                                 item.get("text", ""), item.get("type", "unknown"))
            if len(batches) > 1:
                print(f"      batch {i+1}/{len(batches)}: total {len(seen)}")
        return list(seen.values())

    def _i2_prompt_extract(self, doc_block, comp_block, alias_block):
        return f"""You are a software architecture traceability expert.

ARCHITECTURE COMPONENTS:
{comp_block}
{alias_block}

DOCUMENT:
{doc_block}

TASK: For each sentence, identify which architecture components are EXPLICITLY mentioned or referenced.

A valid reference is:
- Exact name: the component name appears verbatim in the sentence
- Synonym: a well-known alternative name for the component
- Abbreviation: a shortened form
- Partial name: a distinctive sub-phrase that unambiguously identifies the component

NOT a valid reference:
- A component name inside a dotted path (e.g., "renderer.utils.config")
- A generic English word used in its ordinary sense
- A sentence that merely describes related functionality without naming the component

Return ONLY valid JSON:
{{"links": [{{"s": <sentence_number>, "c": "<ComponentName>", "text": "<matched text>", "type": "exact|synonym|partial"}}]}}

Precision is critical — only include links with clear textual evidence.
JSON only:"""

    def _i2_prompt_actor(self, doc_block, comp_block, alias_block):
        return f"""You are a software architecture traceability expert performing an independent review.

ARCHITECTURE COMPONENTS:
{comp_block}
{alias_block}

DOCUMENT:
{doc_block}

TASK: For each sentence, determine which architecture component is the SUBJECT or primary ACTOR.

Ask yourself:
- Which component is this sentence ABOUT?
- Which component PERFORMS or RECEIVES the described action?
- Is the component named, abbreviated, or referred to by a recognizable alias?

Rules:
- Only report components that are explicitly named, abbreviated, or identified by a clear synonym/partial name IN THE SENTENCE TEXT.
- Do NOT report contextual or pronoun-based references.
- Do NOT match component names inside dotted package paths.
- Do NOT match generic English words used in their ordinary sense.

Return ONLY valid JSON:
{{"links": [{{"s": <sentence_number>, "c": "<ComponentName>", "text": "<evidence>", "type": "exact|synonym|partial"}}]}}

Be conservative — omit uncertain links.
JSON only:"""

    # ── Phase 3 DA warm-start ─────────────────────────────────────────

    def _learn_document_knowledge_enriched(self, sentences, components):
        """Override Phase 3 to warm-start from DA results if available."""
        da = self._cnr._doc_analysis
        if not da or not self._enable_da:
            return super()._learn_document_knowledge_enriched(sentences, components)

        comp_names = self._get_comp_names(components)
        comp_lower = {n.lower() for n in comp_names}

        # Build known mappings from DA
        known_abbrevs = {}
        known_synonyms = {}
        for abbr, full in da.abbreviations.items():
            # Match DA abbreviation full-forms to actual component names
            if full in comp_names:
                known_abbrevs[abbr] = full
            elif full.lower() in comp_lower:
                # Case-insensitive match
                for cn in comp_names:
                    if cn.lower() == full.lower():
                        known_abbrevs[abbr] = cn
                        break

        # Match DA synonym groups to component names
        for group in da.synonym_groups:
            # Find which group members match components vs are synonyms
            comp_members = []
            non_comp_members = []
            for member in group:
                if not member:
                    continue
                if member in comp_names:
                    comp_members.append(member)
                else:
                    # Case-insensitive check
                    matched = False
                    for cn in comp_names:
                        if cn.lower() == member.lower():
                            comp_members.append(cn)
                            matched = True
                            break
                    if not matched:
                        non_comp_members.append(member)
            # Only create synonym mappings if exactly one component in the group
            if len(comp_members) == 1 and non_comp_members:
                for syn in non_comp_members:
                    known_synonyms[syn] = comp_members[0]

        # Build warm-start block for the Phase 3 prompt
        warm_start_lines = []
        if known_abbrevs:
            warm_start_lines.append("KNOWN ABBREVIATIONS (from document analysis):")
            for abbr, comp in known_abbrevs.items():
                warm_start_lines.append(f"  {abbr} = {comp}")
        if known_synonyms:
            warm_start_lines.append("KNOWN SYNONYMS (from document analysis):")
            for syn, comp in known_synonyms.items():
                warm_start_lines.append(f"  {syn} = {comp}")
        warm_start_block = "\n".join(warm_start_lines) if warm_start_lines else ""

        if warm_start_block:
            print(f"    DA warm-start: {len(known_abbrevs)} abbrevs, {len(known_synonyms)} synonyms")

        # Run V26a's Phase 3 with warm-start injected into prompt
        doc_lines = [f"S{s.number}: {s.text}" for s in sentences[:100]]

        warm_inject = ""
        if warm_start_block:
            warm_inject = f"""
{warm_start_block}

These were discovered by document analysis. Include them if they are valid, and find any
additional mappings not listed above.
"""

        prompt1 = f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}
{warm_inject}
WHAT TO FIND:
1. ABBREVIATIONS: Short forms explicitly introduced in the document.
   Rule: The abbreviation must be defined in the text, e.g., "Full Name (FN)" introduces FN.
   Like "Abstract Syntax Tree (AST)" defines AST — look for the same parenthetical pattern.

2. SYNONYMS: Alternative names that SPECIFICALLY refer to one component.
   Rule: The alternative name must unambiguously identify exactly ONE component.
   APPROVE: A proper name, role title, or technical alias used interchangeably with the component
   REJECT: A generic description that could apply to anything (like "the system" or "the process")

3. PARTIAL REFERENCES: A shorter form of a multi-word component name used alone.
   Rule: A trailing word from a multi-word name that, in this document, consistently means the full name.
   APPROVE: Only if the short form is unambiguous — no other component shares this word
   REJECT: Common words that have ordinary English meanings beyond the component

DOCUMENT:
{chr(10).join(doc_lines)}

Return JSON:
{{
  "abbreviations": {{"short_form": "FullComponent"}},
  "synonyms": {{"specific_alternative_name": "FullComponent"}},
  "partial_references": {{"partial_name": "FullComponent"}}
}}
JSON only:"""

        data1 = self.llm.extract_json(self.llm.query(prompt1, timeout=150))

        all_mappings = {}
        # Pre-seed from DA warm-start
        for abbr, comp in known_abbrevs.items():
            all_mappings[abbr] = ("abbrev", comp)
        for syn, comp in known_synonyms.items():
            all_mappings[syn] = ("synonym", comp)

        if data1:
            for short, full in data1.get("abbreviations", {}).items():
                if full in comp_names and short not in all_mappings:
                    all_mappings[short] = ("abbrev", full)
            for syn, full in data1.get("synonyms", {}).items():
                if full in comp_names and syn not in all_mappings:
                    all_mappings[syn] = ("synonym", full)
            for partial, full in data1.get("partial_references", {}).items():
                if full in comp_names and partial not in all_mappings:
                    all_mappings[partial] = ("partial", full)

        if all_mappings:
            mapping_list = [f"'{k}' -> {v[1]} ({v[0]})" for k, v in list(all_mappings.items())[:25]]

            prompt2 = f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}

Apply these rules:

REJECT if ANY of these are true:
- The term is used in its ordinary English sense, NOT as a name for the component
  (e.g., "the scheduler runs every minute" uses "scheduler" as a generic concept, not as a named component)
- The term refers to a different component or to the system as a whole
- The mapping cannot be verified from the actual document text

APPROVE if ALL of these are true:
- The term is used AS A NAME for the component in context (even if the word exists in a dictionary)
  (e.g., "The Dispatcher routes incoming requests" uses "Dispatcher" as a proper name for a specific component)
- The term appears in the document in a context that makes the reference clear
- The term unambiguously identifies exactly one component

Return JSON:
{{
  "approved": ["term1", "term2"],
  "generic_rejected": ["generic_term1"]
}}
JSON only:"""

            data2 = self.llm.extract_json(self.llm.query(prompt2, timeout=120))
            approved = set(data2.get("approved", [])) if data2 else set(all_mappings.keys())
            generic_terms = set(data2.get("generic_rejected", [])) if data2 else set()

            # Apply the same V26a deterministic overrides
            for term in list(generic_terms):
                if not term:
                    generic_terms.discard(term)
                    continue
                if re.search(r'[a-z][A-Z]', term):
                    generic_terms.discard(term)
                    approved.add(term)
                    print(f"    CamelCase override (rescued): {term}")
            for term in list(generic_terms):
                if not term:
                    generic_terms.discard(term)
                    continue
                if term.isupper() and len(term) <= 4 and term in all_mappings:
                    generic_terms.discard(term)
                    approved.add(term)
                    print(f"    Uppercase override (rescued): {term}")
            for term in list(generic_terms):
                if not term or term not in all_mappings:
                    continue
                if any(term.lower() == cn.lower() for cn in comp_names):
                    generic_terms.discard(term)
                    approved.add(term)
                    print(f"    Exact-component override (rescued): {term}")
                elif term[0].isupper() and ' ' in term:
                    generic_terms.discard(term)
                    approved.add(term)
                    print(f"    Multi-word proper-name override (rescued): {term}")
            for term in list(generic_terms):
                if not term or term not in all_mappings:
                    continue
                _, target_comp = all_mappings[term]
                if ' ' in term or not term[0].isupper() or target_comp not in comp_names:
                    continue
                for s in sentences[:100]:
                    for m in re.finditer(rf'\b{re.escape(term)}\b', s.text):
                        if m.start() > 0:
                            generic_terms.discard(term)
                            approved.add(term)
                            print(f"    Capitalized-synonym override (rescued): {term}")
                            break
                    if term in approved:
                        break
        else:
            approved = set()
            generic_terms = set()

        knowledge = DocumentKnowledge()
        knowledge.generic_terms = generic_terms

        for term, (typ, comp) in all_mappings.items():
            if term in approved:
                if typ == "abbrev":
                    knowledge.abbreviations[term] = comp
                    print(f"    Abbrev: {term} -> {comp}")
                elif typ == "synonym":
                    knowledge.synonyms[term] = comp
                    print(f"    Syn: {term} -> {comp}")
                else:
                    knowledge.partial_references[term] = comp
                    print(f"    Partial: {term} -> {comp}")

        if generic_terms:
            print(f"    Generic (rejected): {', '.join(list(generic_terms)[:5])}")

        # Deterministic CamelCase-split synonym injection
        for comp in [c.name for c in components]:
            split = re.sub(r'([a-z])([A-Z])', r'\1 \2', comp)
            split = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', split)
            if split != comp and split not in knowledge.synonyms:
                knowledge.synonyms[split] = comp
                print(f"    CamelCase syn: {split} -> {comp}")

        return knowledge

    # ── Eval bridge remap ───────────────────────────────────────────────

    def _remap_links(self, links: list[SadSamLink]) -> list[SadSamLink]:
        """Remap links from cnr IDs to PCM IDs using eval bridge. Drop unmatched, dedup."""
        remapped: dict[tuple[int, str], SadSamLink] = {}
        for lk in links:
            pcm_id = self._eval_bridge_map.get(lk.component_id)
            if pcm_id:
                key = (lk.sentence_number, pcm_id)
                if key not in remapped:
                    remapped[key] = SadSamLink(
                        sentence_number=lk.sentence_number,
                        component_id=pcm_id,
                        component_name=lk.component_name,
                        confidence=lk.confidence,
                        source=lk.source,
                    )
        return list(remapped.values())
