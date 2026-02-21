"""AgentLinker V17: V16 per-mention + ablation flags for missing V6 features.

V16 base: per-mention generic check, no global ambiguity labels, enriched prompts.
Togglable features (all default OFF = V16 behavior):
  A: intersect_validation — Phase 6 requires BOTH passes (like V6), not EITHER
  B: enable_implicit — Phase 8 implicit refs for non-complex docs (like V6)
  C: enable_fn_recovery — Phase 10 FN recovery with judge (like V6)
  D: nuanced_transarc — Phase 9 sends generic-appearing TransArc to judge (like V6)
"""

import json
import os
import re
import time
from collections import defaultdict
from typing import Optional

from ...core import (
    SadSamLink,
    CandidateLink,
    DocumentProfile,
    LearnedThresholds,
    ModelKnowledge,
    DocumentKnowledge,
    LearnedPatterns,
    EntityMention,
    DiscourseContext,
    DocumentLoader,
    Sentence,
)
from ...pcm_parser import parse_pcm_repository
from ...llm_client import LLMClient, LLMBackend
from .agent_linker import AgentLinker


class AgentLinkerV17(AgentLinker):
    """V17: V16 + togglable V6 features for ablation."""

    SOURCE_PRIORITY = {
        "transarc": 5, "validated": 4, "entity": 3,
        "coreference": 2, "implicit": 1, "partial_inject": 1, "recovered": 0,
    }

    GENERIC_PARTIALS = {
        "conversion", "data", "process", "system",
        "core", "base", "app", "application",
    }

    NON_MODIFIERS = {
        "the", "a", "an", "this", "that", "these", "those", "its", "their",
        "our", "your", "my", "his", "her", "some", "any", "all", "each",
        "every", "no", "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "do", "does", "did", "will", "would", "can",
        "could", "shall", "should", "may", "might", "must",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "and", "or", "but", "not", "if", "then", "than", "as",
        "about", "into", "through", "during", "before", "after",
        "above", "below", "between", "under", "over",
    }

    def __init__(self, backend: Optional[LLMBackend] = None,
                 intersect_validation=False,
                 enable_implicit=False,
                 enable_fn_recovery=False,
                 nuanced_transarc=False,
                 judge_doc_knowledge=False,
                 judge_full_context=False,
                 judge_validated_bias=False,
                 judge_strict_def=False,
                 judge_adaptive_ctx=False,
                 judge_show_match=False):
        os.environ.setdefault("CLAUDE_MODEL", "sonnet")
        super().__init__(backend=backend or LLMBackend.CLAUDE)
        self._phase_log = []
        self._is_complex = None
        # self.doc_knowledge is set by Phase 3 (inherited)
        # Feature flags
        self.feat_intersect = intersect_validation
        self.feat_implicit = enable_implicit
        self.feat_fn_recovery = enable_fn_recovery
        self.feat_nuanced_ta = nuanced_transarc
        # Judge prompt flags
        self.judge_doc_knowledge = judge_doc_knowledge
        self.judge_full_context = judge_full_context
        self.judge_validated_bias = judge_validated_bias
        self.judge_strict_def = judge_strict_def
        self.judge_adaptive_ctx = judge_adaptive_ctx
        self.judge_show_match = judge_show_match
        flags = []
        if intersect_validation: flags.append("A:intersect")
        if enable_implicit: flags.append("B:implicit")
        if enable_fn_recovery: flags.append("C:fn_recovery")
        if nuanced_transarc: flags.append("D:nuanced_ta")
        if judge_doc_knowledge: flags.append("J1:doc_know")
        if judge_full_context: flags.append("J2:full_ctx")
        if judge_validated_bias: flags.append("J3:val_bias")
        if judge_strict_def: flags.append("J4:strict_def")
        if judge_adaptive_ctx: flags.append("J5:adapt_ctx")
        if judge_show_match: flags.append("J6:show_match")
        flag_str = ", ".join(flags) if flags else "none (= V16)"
        print(f"AgentLinkerV17: V16 + [{flag_str}]")
        print(f"  Backend: {self.llm.backend.value}, Model: {os.environ.get('CLAUDE_MODEL', 'default')}")

    # ── Logging ──────────────────────────────────────────────────────────

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
        path = os.path.join(log_dir, f"v17_{ds}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w") as f:
            json.dump(self._phase_log, f, indent=2, default=str)
        print(f"  Phase log saved: {path}")

    # ═════════════════════════════════════════════════════════════════════
    # Per-mention generic check (from V16)
    # ═════════════════════════════════════════════════════════════════════

    def _is_generic_mention(self, comp_name, sentence_text):
        if ' ' in comp_name or '-' in comp_name:
            return False
        if re.search(r'[a-z][A-Z]', comp_name):
            return False
        if comp_name.isupper():
            return False
        if comp_name[0].islower():
            return False
        if self._has_standalone_mention(comp_name, sentence_text):
            return False
        word_lower = comp_name.lower()
        if re.search(rf'\b{re.escape(word_lower)}\b', sentence_text):
            return True
        return False

    def _needs_antecedent_check(self, comp_name):
        if ' ' in comp_name or '-' in comp_name:
            return False
        if re.search(r'[a-z][A-Z]', comp_name):
            return False
        if comp_name.isupper():
            return False
        if comp_name[0].islower():
            return False
        return True

    # ═════════════════════════════════════════════════════════════════════
    # Main pipeline
    # ═════════════════════════════════════════════════════════════════════

    def link(self, text_path, model_path, transarc_csv=None):
        self._phase_log = []
        t0 = time.time()

        components = parse_pcm_repository(model_path)
        sentences = DocumentLoader.load_sentences(text_path)
        name_to_id = {c.name: c.id for c in components}
        id_to_name = {c.id: c.name for c in components}
        sent_map = DocumentLoader.build_sent_map(sentences)
        self._cached_sent_map = sent_map

        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        # ── Phase 0: Document profile ──────────────────────────────────
        print("\n[Phase 0] Document Profile")
        self.doc_profile = self._learn_document_profile(sentences, components)
        self._is_complex = self._structural_complexity(sentences, components)
        spc = len(sentences) / max(1, len(components))
        print(f"  Stats: {spc:.1f} sents/comp, {self.doc_profile.pronoun_ratio:.0%} pronouns")
        print(f"  Complex: {self._is_complex}")
        self.thresholds = LearnedThresholds(0, 0, 0, 0, "qualitative", 0)
        self._log("phase_0", {"sents": len(sentences), "comps": len(components)},
                  {"spc": spc, "complex": self._is_complex})

        # ── Phase 1: Model Structure ───────────────────────────────────
        print("\n[Phase 1] Model Structure")
        self.model_knowledge = self._analyze_model(components)
        arch = self.model_knowledge.architectural_names
        ambig = self.model_knowledge.ambiguous_names
        print(f"  Architectural: {len(arch)}, Ambiguous (logged only): {sorted(ambig)}")
        print(f"  NOTE: Ambiguity labels NOT used — per-mention check instead")
        self._log("phase_1", {"components": [c.name for c in components]},
                  {"architectural": sorted(arch), "ambiguous_logged_only": sorted(ambig)})

        # ── Phase 2: Pattern learning ─────────────────────────────────
        print("\n[Phase 2] Pattern Learning")
        self.learned_patterns = self._learn_patterns_with_debate(sentences, components)
        print(f"  Subprocess terms: {len(self.learned_patterns.subprocess_terms)}")
        self._log("phase_2", {}, {"subprocess": sorted(self.learned_patterns.subprocess_terms)})

        # ── Phase 3: Document knowledge (enriched prompt) ──────────────
        print("\n[Phase 3] Document Knowledge")
        self.doc_knowledge = self._learn_document_knowledge_enriched(sentences, components)
        print(f"  Abbrev: {len(self.doc_knowledge.abbreviations)}, "
              f"Syn: {len(self.doc_knowledge.synonyms)}, "
              f"Generic: {len(self.doc_knowledge.generic_terms)}")
        self._log("phase_3", {}, {
            "abbreviations": self.doc_knowledge.abbreviations,
            "synonyms": self.doc_knowledge.synonyms,
            "generic": list(self.doc_knowledge.generic_terms),
        })

        # ── Phase 3b: Multi-word partial enrichment
        self._enrich_multiword_partials(sentences, components)

        # ── Phase 4: TransArc baseline ────────────────────────────────
        print("\n[Phase 4] TransArc")
        transarc_links = self._process_transarc(transarc_csv, id_to_name, sent_map, name_to_id)
        transarc_set = {(l.sentence_number, l.component_id) for l in transarc_links}
        print(f"  Links: {len(transarc_links)}")
        self._log("phase_4", {"csv": transarc_csv}, {"count": len(transarc_links)}, transarc_links)

        # ── Phase 5: Entity extraction (enriched prompt) ──────────────
        print("\n[Phase 5] Entity Extraction")
        candidates = self._extract_entities_enriched(sentences, components, name_to_id, sent_map)
        print(f"  Candidates: {len(candidates)}")

        # Abbreviation guard
        before_guard = len(candidates)
        candidates = self._apply_abbreviation_guard_to_candidates(candidates, sent_map)
        if len(candidates) < before_guard:
            print(f"  After abbrev guard: {len(candidates)} (-{before_guard - len(candidates)})")
        self._log("phase_5", {}, {"count": len(candidates)})

        # ── Phase 5b: Targeted recovery for unlinked components ───────
        entity_comps = {c.component_name for c in candidates}
        transarc_comps = {l.component_name for l in transarc_links}
        covered_comps = entity_comps | transarc_comps
        unlinked = [c for c in components if c.name not in covered_comps]

        if unlinked:
            print(f"\n[Phase 5b] Targeted Recovery ({len(unlinked)} unlinked components)")
            extra = self._targeted_extraction(unlinked, sentences, name_to_id, sent_map)
            if extra:
                print(f"  Found: {len(extra)} additional candidates")
                candidates.extend(extra)
            else:
                print(f"  Found: 0 additional candidates")
            self._log("phase_5b", {"unlinked": [c.name for c in unlinked]},
                      {"found": len(extra) if extra else 0})

        # ── Phase 6: Validation ────────────────────────────────────────
        print("\n[Phase 6] Validation")
        validated = self._validate_split(candidates, components, sent_map)
        print(f"  Validated: {len(validated)} (of {len(candidates)})")
        self._log("phase_6", {"candidates": len(candidates)}, {"validated": len(validated)})

        # ── Phase 7: Coreference ──────────────────────────────────────
        print("\n[Phase 7] Coreference")
        discourse_model = None
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
        print(f"  Coref links: {len(coref_links)}")
        self._log("phase_7", {"method": "debate" if self._is_complex else "discourse"},
                  {"count": len(coref_links)}, coref_links)

        # ── Phase 8: Implicit References (Feature B) ──────────────────
        existing = (transarc_set
                    | {(c.sentence_number, c.component_id) for c in validated}
                    | {(l.sentence_number, l.component_id) for l in coref_links})

        if self.feat_implicit and not self._is_complex and discourse_model:
            print("\n[Phase 8] Implicit References")
            implicit_links = self._detect_implicit_references(
                sentences, components, name_to_id, sent_map, discourse_model, existing
            )
            print(f"  Detected: {len(implicit_links)}")
            if implicit_links:
                implicit_links = self._validate_implicit_links(
                    implicit_links, sentences, components, sent_map
                )
                print(f"  After validation: {len(implicit_links)}")
            self._log("phase_8", {"existing": len(existing)}, {"count": len(implicit_links)}, implicit_links)
        else:
            reason = "complex doc" if self._is_complex else ("disabled" if not self.feat_implicit else "no discourse model")
            print(f"\n[Phase 8] Implicit References — SKIPPED ({reason})")
            implicit_links = []

        # ── Phase 8b: Partial-reference injection ─────────────────────
        partial_links = self._inject_partial_references(
            sentences, components, name_to_id, transarc_set,
            {(c.sentence_number, c.component_id) for c in validated},
            {(l.sentence_number, l.component_id) for l in coref_links},
            {(l.sentence_number, l.component_id) for l in implicit_links},
        )
        if partial_links:
            print(f"\n[Phase 8b] Partial Injection")
            print(f"  Injected: {len(partial_links)} candidates")
            self._log("phase_8b", {}, {"count": len(partial_links)}, partial_links)

        # ── Combine + deduplicate ─────────────────────────────────────
        entity_links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
            for c in validated
        ]
        all_links = transarc_links + entity_links + coref_links + implicit_links + partial_links
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
        self._log("dedup", {"raw": len(all_links)}, {"deduped": len(preliminary)}, preliminary)

        # ── Phase 8c: Boundary filters (NON-TRANSARC ONLY) ───────────
        print("\n[Phase 8c] Boundary Filters (non-TransArc only)")
        preliminary, boundary_rejected = self._apply_boundary_filters(
            preliminary, sent_map, transarc_set
        )
        if boundary_rejected:
            print(f"  Rejected: {len(boundary_rejected)}")
            self._log("phase_8c", {},
                      {"rejected": len(boundary_rejected),
                       "details": [(lk.component_name, reason) for lk, reason in boundary_rejected]})

        # ── Phase 9: Judge Review ─────────────────────────────────────
        if self.feat_nuanced_ta:
            print("\n[Phase 9] Judge Review (nuanced TransArc)")
        else:
            print("\n[Phase 9] Judge Review (TransArc immune)")
        reviewed = self._judge_review(preliminary, sentences, components, sent_map, transarc_set)
        rejected = [l for l in preliminary if (l.sentence_number, l.component_id)
                    not in {(r.sentence_number, r.component_id) for r in reviewed}]
        print(f"  Approved: {len(reviewed)} (rejected {len(rejected)})")
        self._log("phase_9", {"input": len(preliminary)},
                  {"approved": len(reviewed), "rejected": len(rejected)}, reviewed)
        if rejected:
            self._log("phase_9_rejected", {}, {"count": len(rejected)}, rejected)

        # ── Phase 10: FN Recovery (Feature C) ─────────────────────────
        if self.feat_fn_recovery:
            print("\n[Phase 10] FN Recovery")
            final = self._fn_recovery(reviewed, sentences, components, name_to_id, sent_map)
            recovered = [l for l in final if l.source == "recovered"]
            print(f"  Final: {len(final)} (+{len(recovered)} recovered)")
            self._log("phase_10", {}, {"final": len(final), "recovered": len(recovered)}, recovered or None)
        else:
            print("\n[Phase 10] FN Recovery — SKIPPED (disabled)")
            final = reviewed

        # ── Save log ──────────────────────────────────────────────────
        self._log("summary", {"total_time_s": round(time.time() - t0, 1)},
                  {"final": len(final)}, final)
        self._save_log(text_path)

        print(f"\nFinal: {len(final)} links")
        return final

    # ═════════════════════════════════════════════════════════════════════
    # Phase 0: Document profile + structural complexity
    # ═════════════════════════════════════════════════════════════════════

    def _learn_document_profile(self, sentences, components):
        texts = [s.text for s in sentences]
        comp_names = [c.name for c in components]
        pron = r'\b(it|they|this|these|that|those|its|their)\b'
        pronoun_ratio = sum(1 for t in texts if re.search(pron, t.lower())) / max(1, len(sentences))
        mentions = sum(1 for t in texts for c in comp_names if c.lower() in t.lower())
        density = mentions / max(1, len(sentences))
        spc = len(sentences) / max(1, len(components))
        return DocumentProfile(
            sentence_count=len(sentences), component_count=len(components),
            pronoun_ratio=pronoun_ratio, technical_density=density,
            component_mention_density=density,
            complexity_score=min(1.0, spc / 20),
            recommended_strictness="balanced",
        )

    def _structural_complexity(self, sentences, components, spc_min=5):
        comp_names = [c.name for c in components]
        mention_count = sum(1 for sent in sentences
                          if any(cn.lower() in sent.text.lower() for cn in comp_names))
        uncovered_ratio = 1.0 - (mention_count / max(1, len(sentences)))
        spc = len(sentences) / max(1, len(components))
        result = uncovered_ratio > 0.5 and spc > spc_min
        print(f"  Structural complexity: uncovered={uncovered_ratio:.1%}, spc={spc:.1f}, min_spc={spc_min} -> {result}")
        return result

    # ═════════════════════════════════════════════════════════════════════
    # Phase 3b: Multi-word partial enrichment
    # ═════════════════════════════════════════════════════════════════════

    def _enrich_multiword_partials(self, sentences, components):
        if not self.doc_knowledge:
            return

        added = []
        for comp in components:
            parts = comp.name.split()
            if len(parts) < 2:
                continue
            last_word = parts[-1]
            if len(last_word) < 4:
                continue
            last_lower = last_word.lower()

            other_match = any(
                c.name != comp.name and c.name.lower().endswith(last_lower)
                for c in components
            )
            if other_match:
                continue
            if last_lower in {s.lower() for s in self.doc_knowledge.synonyms}:
                continue
            if last_lower in {p.lower() for p in self.doc_knowledge.partial_references}:
                continue

            is_generic_word = last_lower in self.GENERIC_PARTIALS
            full_lower = comp.name.lower()
            mention_count = 0
            for sent in sentences:
                sl = sent.text.lower()
                if last_lower in sl and full_lower not in sl:
                    if is_generic_word:
                        cap_word = last_word[0].upper() + last_word[1:]
                        if re.search(rf'\b{re.escape(cap_word)}\b', sent.text):
                            mention_count += 1
                    else:
                        if re.search(rf'\b{re.escape(last_word)}\b', sent.text, re.IGNORECASE):
                            mention_count += 1

            if mention_count >= 3:
                self.doc_knowledge.partial_references[last_word] = comp.name
                added.append(f"{last_word} -> {comp.name} ({mention_count} caps mentions)")

        if added:
            print(f"\n[Phase 3b] Multi-word Enrichment")
            for a in added:
                print(f"    Auto-partial: {a}")
            self._log("phase_3b", {}, {"added": added})

    # ═════════════════════════════════════════════════════════════════════
    # ENRICHED Phase 3: Document knowledge with examples
    # ═════════════════════════════════════════════════════════════════════

    def _learn_document_knowledge_enriched(self, sentences, components):
        comp_names = self._get_comp_names(components)
        doc_lines = [f"S{s.number}: {s.text}" for s in sentences[:100]]

        prompt1 = f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

WHAT TO FIND:
1. ABBREVIATIONS: Short forms explicitly introduced in the document.
   Example: "Application Server (AS)" → AS = Application Server
   Example: "the Message Broker Service (MBS)" → MBS = Message Broker Service

2. SYNONYMS: Alternative names that SPECIFICALLY refer to one component.
   GOOD: "auth provider" → AuthService (a specific role name for the component)
   GOOD: "message routing engine" → MessageBroker (describes THAT specific component)
   BAD: "stores data" → DataStore (a generic description, not a name)
   BAD: "the business logic" → Processor (generic English phrase, not a specific name)

3. PARTIAL REFERENCES: A shorter form of a multi-word component name used alone.
   GOOD: "Backend" alone → "ServiceBackend" (unique last-word reference)
   GOOD: "main interface" → "MainInterface" (case variant of full name)
   BAD: "server" alone → "AppServer" (too ambiguous — could be any server)
   BAD: "the client" → "WebClient" (generic usage with article)

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
        if data1:
            for short, full in data1.get("abbreviations", {}).items():
                if full in comp_names:
                    all_mappings[short] = ("abbrev", full)
            for syn, full in data1.get("synonyms", {}).items():
                if full in comp_names:
                    all_mappings[syn] = ("synonym", full)
            for partial, full in data1.get("partial_references", {}).items():
                if full in comp_names:
                    all_mappings[partial] = ("partial", full)

        if all_mappings:
            mapping_list = [f"'{k}' -> {v[1]} ({v[0]})" for k, v in list(all_mappings.items())[:25]]

            prompt2 = f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}

REJECT mappings that are:
- TOO GENERIC: common English phrases like "the server", "business logic", "data storage"
- WRONG TARGET: the term actually refers to a different component or to the system overall
- NOT IN DOCUMENT: the mapping was hallucinated and doesn't appear in the text

APPROVE mappings that are:
- SPECIFIC: the term is a recognizable name/alias for exactly that component
- DOCUMENT-SUPPORTED: the mapping can be verified from the document text
- UNAMBIGUOUS: the term clearly refers to one component, not multiple

Return JSON:
{{
  "approved": ["term1", "term2"],
  "generic_rejected": ["generic_term1"]
}}
JSON only:"""

            data2 = self.llm.extract_json(self.llm.query(prompt2, timeout=120))
            approved = set(data2.get("approved", [])) if data2 else set(all_mappings.keys())
            generic_terms = set(data2.get("generic_rejected", [])) if data2 else set()
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

        return knowledge

    # ═════════════════════════════════════════════════════════════════════
    # ENRICHED Phase 5: Entity extraction
    # ═════════════════════════════════════════════════════════════════════

    def _extract_entities_enriched(self, sentences, components, name_to_id, sent_map):
        comp_names = self._get_comp_names(components)
        comp_lower = {n.lower() for n in comp_names}

        mappings = []
        if self.doc_knowledge:
            mappings.extend([f"{a}={c}" for a, c in self.doc_knowledge.abbreviations.items()])
            mappings.extend([f"{s}={c}" for s, c in self.doc_knowledge.synonyms.items()])
            mappings.extend([f"{p}={c}" for p, c in self.doc_knowledge.partial_references.items()])

        batch_size = 100
        all_candidates = {}

        for batch_start in range(0, len(sentences), batch_size):
            batch = sentences[batch_start:batch_start + batch_size]

            if len(sentences) > batch_size:
                print(f"    Entity batch {batch_start//batch_size + 1}: "
                      f"S{batch[0].number}-S{batch[-1].number} ({len(batch)} sents)")

            prompt = f"""Extract ALL references to software architecture components from this document.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings[:20])}' if mappings else ''}

WHAT TO LOOK FOR:
- DIRECT MENTIONS: The component name appears in the sentence (exact or alias)
- FUNCTIONAL DESCRIPTIONS: The sentence describes what a specific component does
  Example: "responsible for converting media formats" → refers to a media conversion component
- ROLE REFERENCES: The sentence refers to a component by its architectural role
  Example: "the persistence layer handles all database operations" → refers to the persistence component
- CamelCase splits: "File Storage" in text may refer to component "FileStorage"

BE THOROUGH: Check every sentence against every component. Missing a reference is worse
than including a borderline one (validation will filter later).

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{{"references": [{{"sentence": N, "component": "Name", "matched_text": "text found in sentence", "match_type": "exact|synonym|partial|functional"}}]}}
JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=150))
            if not data:
                continue

            for ref in data.get("references", []):
                snum, cname = ref.get("sentence"), ref.get("component")
                if not (snum and cname and cname in name_to_id):
                    continue
                sent = sent_map.get(snum)
                if not sent or self._in_dotted_path(sent.text, cname):
                    continue

                matched = ref.get("matched_text", "").lower()
                is_exact = matched in comp_lower or cname.lower() in matched
                is_generic_here = self._is_generic_mention(cname, sent.text)
                needs_val = not is_exact or ref.get("match_type") != "exact" or is_generic_here

                key = (snum, name_to_id[cname])
                if key not in all_candidates:
                    all_candidates[key] = CandidateLink(snum, sent.text, cname, name_to_id[cname],
                                               ref.get("matched_text", ""), 0.85, "entity",
                                               ref.get("match_type", "exact"), needs_val)

        return list(all_candidates.values())

    # ═════════════════════════════════════════════════════════════════════
    # Phase 5b: Targeted extraction for unlinked components
    # ═════════════════════════════════════════════════════════════════════

    def _targeted_extraction(self, unlinked_components, sentences, name_to_id, sent_map):
        if not unlinked_components:
            return []

        all_extra = []
        for comp in unlinked_components:
            if len(sentences) <= 60:
                doc_sents = sentences
            else:
                doc_sents = sentences[:30] + sentences[-30:]

            doc_text = "\n".join([f"S{s.number}: {s.text}" for s in doc_sents])

            aliases = []
            if self.doc_knowledge:
                for a, c in self.doc_knowledge.abbreviations.items():
                    if c == comp.name: aliases.append(a)
                for s, c in self.doc_knowledge.synonyms.items():
                    if c == comp.name: aliases.append(s)
                for p, c in self.doc_knowledge.partial_references.items():
                    if c == comp.name: aliases.append(p)

            alias_str = f"\nKNOWN ALIASES: {', '.join(aliases)}" if aliases else ""

            prompt = f"""Find ALL sentences that discuss the software component "{comp.name}".
{alias_str}

Look for:
- Direct mentions of "{comp.name}" or any alias
- Descriptions of what {comp.name} does (functional descriptions)
- References to {comp.name}'s role in the architecture

DOCUMENT:
{doc_text}

Return JSON:
{{"references": [{{"sentence": N, "matched_text": "text found", "reason": "why this refers to {comp.name}"}}]}}

Be thorough — find ALL sentences that discuss this component.
JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
            if not data:
                continue

            for ref in data.get("references", []):
                snum = ref.get("sentence")
                if not snum:
                    continue
                try:
                    snum = int(snum)
                except (ValueError, TypeError):
                    continue
                sent = sent_map.get(snum)
                if not sent or self._in_dotted_path(sent.text, comp.name):
                    continue
                cid = name_to_id.get(comp.name)
                if not cid:
                    continue

                matched = ref.get("matched_text", comp.name)
                all_extra.append(CandidateLink(
                    snum, sent.text, comp.name, cid,
                    matched, 0.85, "entity", "targeted", True
                ))
                print(f"    Targeted: S{snum} -> {comp.name} ({matched})")

        return all_extra

    # ═════════════════════════════════════════════════════════════════════
    # Phase 6: Validation — Feature A controls intersect vs union
    # ═════════════════════════════════════════════════════════════════════

    def _validate_split(self, candidates, components, sent_map):
        if not candidates:
            return []

        comp_names = self._get_comp_names(components)
        needs = [c for c in candidates if c.needs_validation]
        direct = [c for c in candidates if not c.needs_validation]

        if not needs:
            return candidates

        # Per-mention pre-check: reject generic mentions
        pre_rejected = []
        remaining = []
        for c in needs:
            sent = sent_map.get(c.sentence_number)
            if sent and self._is_generic_mention(c.component_name, sent.text):
                pre_rejected.append(c)
                print(f"    Generic mention reject: S{c.sentence_number} -> {c.component_name}")
            else:
                remaining.append(c)

        if pre_rejected:
            print(f"    Generic mentions rejected: {len(pre_rejected)} candidates")

        needs = remaining

        ctx = []
        if self.learned_patterns:
            if self.learned_patterns.action_indicators:
                ctx.append(f"ACTION: {', '.join(self.learned_patterns.action_indicators[:4])}")
            if self.learned_patterns.effect_indicators:
                ctx.append(f"EFFECT (reject): {', '.join(self.learned_patterns.effect_indicators[:3])}")
            if self.learned_patterns.subprocess_terms:
                ctx.append(f"Subprocess (reject): {', '.join(list(self.learned_patterns.subprocess_terms)[:5])}")

        validated = []
        for batch_start in range(0, len(needs), 25):
            batch = needs[batch_start:batch_start + 25]
            cases = []
            for i, c in enumerate(batch):
                prev = sent_map.get(c.sentence_number - 1)
                p = f"[prev: {prev.text[:35]}...] " if prev else ""
                cases.append(f"Case {i+1}: \"{c.matched_text}\" -> {c.component_name}\n  {p}\"{c.sentence_text}\"")

            r1 = self._qual_validation_pass(comp_names, ctx, cases,
                "Focus on ACTOR role: is the component performing an action or being described?")
            r2 = self._qual_validation_pass(comp_names, ctx, cases,
                "Focus on DIRECT reference: does the text refer to the SPECIFIC architectural component, not a generic concept?")

            for i, c in enumerate(batch):
                p1 = r1.get(i, False)
                p2 = r2.get(i, False)
                # Feature A: intersect (both must approve) vs union (either approves)
                if self.feat_intersect:
                    approved = p1 and p2  # INTERSECT like V6
                else:
                    approved = p1 or p2   # UNION like V16

                if approved:
                    c.confidence = 1.0
                    c.source = "validated"
                    validated.append(c)

        return direct + validated

    def _qual_validation_pass(self, comp_names, ctx, cases, focus):
        prompt = f"""Validate component references in a software architecture document. {focus}

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(ctx)}

IMPORTANT DISTINCTIONS:
- "the routing logic" / "business logic" = generic English, NOT an architectural component → REJECT
- "client-side rendering" / "on the client" = generic usage, NOT a specific component → REJECT
- "data storage layer" / "in-memory cache" = generic concept, NOT a specific component → REJECT
- But "Router handles request processing" = the component IS the actor → APPROVE
- Section headings naming a component = introduces that component's section → APPROVE

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
        results = {}
        if data:
            for v in data.get("validations", []):
                idx = v.get("case", 0) - 1
                if 0 <= idx < len(cases):
                    results[idx] = v.get("approve", False)
        return results

    # ═════════════════════════════════════════════════════════════════════
    # Phase 7: Coreference
    # ═════════════════════════════════════════════════════════════════════

    def _coref_discourse(self, sentences, components, name_to_id, sent_map, discourse_model):
        comp_names = self._get_comp_names(components)
        all_coref = []
        pronoun_sents = [s for s in sentences if self.PRONOUN_PATTERN.search(s.text)]

        for batch_start in range(0, len(pronoun_sents), 12):
            batch = pronoun_sents[batch_start:batch_start + 12]
            cases = []
            for sent in batch:
                ctx = discourse_model.get(sent.number, DiscourseContext())
                prev = []
                for i in range(1, self.CONTEXT_WINDOW + 1):
                    p = sent_map.get(sent.number - i)
                    if p:
                        prev.append(f"S{p.number}: {p.text[:70]}")
                cases.append({"sent": sent, "ctx": ctx, "prev": prev,
                              "likely": ctx.get_likely_referent()})

            prompt = f"""Resolve pronoun references using discourse context.

COMPONENTS: {', '.join(comp_names)}

"""
            for i, case in enumerate(cases):
                ctx = case["ctx"]
                prompt += f"--- Case {i+1}: S{case['sent'].number} ---\n"
                prompt += f"DISCOURSE:\n"
                prompt += f"  Recent mentions: {ctx.get_context_summary(case['sent'].number)}\n"
                prompt += f"  Paragraph topic: {ctx.paragraph_topic or 'None'}\n"
                prompt += f"  Likely referent: {case['likely'] or 'Unknown'}\n"
                if case["prev"]:
                    prompt += f"PREVIOUS:\n  " + "\n  ".join(reversed(case["prev"])) + "\n"
                prompt += f">>> SENTENCE: {case['sent'].text}\n\n"

            prompt += """Return JSON:
{"resolutions": [{"case": 1, "sentence": N, "pronoun": "it", "component": "Name", "certainty": "high or low", "reasoning": "why"}]}

Guidelines:
- "high" certainty: component was the explicit grammatical subject of the previous 1-2 sentences AND the pronoun unambiguously refers to it
- "low" certainty: pronoun could refer to multiple things
- Only include resolutions where certainty is "high"

JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=150))
            if not data:
                continue

            for res in data.get("resolutions", []):
                comp = res.get("component")
                snum = res.get("sentence")
                if str(res.get("certainty", "low")).lower() != "high":
                    continue
                if not (comp and snum and comp in name_to_id):
                    continue
                try:
                    snum = int(snum)
                except (ValueError, TypeError):
                    continue
                sent = sent_map.get(snum)
                if sent and self.learned_patterns and self.learned_patterns.is_subprocess(sent.text):
                    continue
                all_coref.append(SadSamLink(snum, name_to_id[comp], comp, 1.0, "coreference"))

        return all_coref

    def _coref_debate(self, sentences, components, name_to_id, sent_map):
        comp_names = self._get_comp_names(components)
        all_coref = []

        ctx = []
        if self.learned_patterns and self.learned_patterns.action_indicators:
            ctx.append(f"ACTION: {', '.join(self.learned_patterns.action_indicators[:3])}")
        if self.learned_patterns and self.learned_patterns.subprocess_terms:
            ctx.append(f"Subprocesses (don't link): {', '.join(list(self.learned_patterns.subprocess_terms)[:5])}")

        for batch_start in range(0, len(sentences), 20):
            batch = sentences[batch_start:min(batch_start + 20, len(sentences))]
            ctx_start = max(0, batch_start - self.CONTEXT_WINDOW)
            ctx_sents = sentences[ctx_start:batch_start + 20]
            doc_lines = [
                f"{'*' if s.number >= batch[0].number else ' '}S{s.number}: {s.text}"
                for s in ctx_sents
            ]

            prompt1 = f"""Resolve pronoun references to components.

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(ctx)}

DOCUMENT (* = analyze):
{chr(10).join(doc_lines)}

Find pronouns (it, they, this) that refer to components.

Return JSON:
{{"resolutions": [{{"sentence": N, "pronoun": "it", "component": "Name", "certainty": "high or low", "reasoning": "why"}}]}}

Only include resolutions where certainty is "high".
JSON only:"""

            data1 = self.llm.extract_json(self.llm.query(prompt1, timeout=100))
            if not data1:
                continue
            proposed = data1.get("resolutions", [])
            if not proposed:
                continue

            judge_batch = proposed[:12]
            ptxt = [f"Case {i+1}: S{r.get('sentence')}: \"{r.get('pronoun', '?')}\" -> {r.get('component')} ({(r.get('reasoning') or '')[:40]})"
                    for i, r in enumerate(judge_batch)]

            prompt2 = f"""JUDGE: Validate these pronoun-to-component resolutions.

COMPONENTS: {', '.join(comp_names)}

PROPOSALS:
{chr(10).join(ptxt)}

CONTEXT:
{chr(10).join(doc_lines[:15])}

Reject if pronoun might refer to something else, or reference is to a subprocess.

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""

            data2 = self.llm.extract_json(self.llm.query(prompt2, timeout=100))
            rejected_indices = set()
            if data2:
                for j in data2.get("judgments", []):
                    idx = j.get("case", 0) - 1
                    if 0 <= idx < len(judge_batch) and not j.get("approve", False):
                        rejected_indices.add(idx)

            approved = set()
            for i, res in enumerate(judge_batch):
                if i not in rejected_indices:
                    snum = res.get("sentence")
                    if snum is not None:
                        approved.add(snum)

            for res in judge_batch:
                snum, comp = res.get("sentence"), res.get("component")
                if str(res.get("certainty", "low")).lower() != "high":
                    continue
                if not (snum and comp and comp in name_to_id):
                    continue
                try:
                    snum = int(snum)
                except (ValueError, TypeError):
                    continue
                if snum not in approved:
                    continue
                sent = sent_map.get(snum)
                if sent and self.learned_patterns and self.learned_patterns.is_subprocess(sent.text):
                    continue
                all_coref.append(SadSamLink(snum, name_to_id[comp], comp, 1.0, "coreference"))

        return all_coref

    # ═════════════════════════════════════════════════════════════════════
    # Generic coreference filter
    # ═════════════════════════════════════════════════════════════════════

    def _filter_generic_coref(self, coref_links, sent_map):
        kept = []
        for lk in coref_links:
            if not self._needs_antecedent_check(lk.component_name):
                kept.append(lk)
                continue
            found_recent = False
            for offset in range(1, 3):
                prev = sent_map.get(lk.sentence_number - offset)
                if prev and self._has_standalone_mention(lk.component_name, prev.text):
                    found_recent = True
                    break
            if found_recent:
                kept.append(lk)
            else:
                print(f"    Coref antecedent missing: S{lk.sentence_number} -> {lk.component_name}")
        return kept

    # ═════════════════════════════════════════════════════════════════════
    # Partial reference injection
    # ═════════════════════════════════════════════════════════════════════

    def _has_clean_mention(self, term, text):
        pattern = rf'\b{re.escape(term)}\b'
        for m in re.finditer(pattern, text, re.IGNORECASE):
            s, e = m.start(), m.end()
            if s > 0 and text[s-1] == '.':
                continue
            if e < len(text) and text[e] == '.' and e + 1 < len(text) and text[e+1].isalpha():
                continue
            if (s > 0 and text[s-1] == '-') or (e < len(text) and text[e] == '-'):
                continue
            return True
        return False

    def _inject_partial_references(self, sentences, components, name_to_id,
                                    transarc_set, validated_set, coref_set, implicit_set):
        if not self.doc_knowledge or not self.doc_knowledge.partial_references:
            return []

        existing = transarc_set | validated_set | coref_set | implicit_set
        injected = []

        for partial, comp_name in self.doc_knowledge.partial_references.items():
            if comp_name not in name_to_id:
                continue
            comp_id = name_to_id[comp_name]
            for sent in sentences:
                key = (sent.number, comp_id)
                if key in existing:
                    continue
                if self._has_clean_mention(partial, sent.text):
                    injected.append(SadSamLink(
                        sent.number, comp_id, comp_name, 0.8, "partial_inject"
                    ))
                    existing.add(key)

        return injected

    # ═════════════════════════════════════════════════════════════════════
    # Boundary filters (NON-TRANSARC ONLY)
    # ═════════════════════════════════════════════════════════════════════

    def _is_in_package_path(self, comp_name, sentence_text):
        words = comp_name.lower().split()
        text_lower = sentence_text.lower()
        for word in words:
            if len(word) < 3:
                continue
            dot_pattern = rf'[\w]+\.{re.escape(word)}|{re.escape(word)}\.[\w]+'
            if re.search(dot_pattern, text_lower):
                cleaned = re.sub(r'[\w]+\.[\w]+(?:\.[\w]+)*', '', text_lower)
                if re.search(rf'\b{re.escape(word)}\b', cleaned):
                    continue
                return True
        return False

    def _is_generic_word_usage(self, comp_name, sentence_text):
        words = comp_name.split()
        if len(words) != 1:
            return False
        word = words[0]
        if re.search(rf'\b{re.escape(word)}\b', sentence_text):
            return False
        pattern = rf'\b(\w+)\s+{re.escape(word.lower())}\b'
        for m in re.finditer(pattern, sentence_text.lower()):
            preceding = m.group(1).lower()
            if preceding not in self.NON_MODIFIERS:
                return True
        return False

    def _is_weak_partial_match(self, link, sent_map):
        if link.source != "partial_inject":
            return False
        sent = sent_map.get(link.sentence_number)
        if not sent:
            return False
        if re.search(rf'\b{re.escape(link.component_name)}\b', sent.text, re.IGNORECASE):
            return False
        if not self.doc_knowledge or not self.doc_knowledge.partial_references:
            return False
        for partial, comp in self.doc_knowledge.partial_references.items():
            if comp == link.component_name:
                if partial.lower() in self.GENERIC_PARTIALS:
                    if re.search(rf'\b{re.escape(partial)}\b', sent.text, re.IGNORECASE):
                        return True
        return False

    # ── Abbreviation guard ────────────────────────────────────────────

    def _abbreviation_match_is_valid(self, abbrev, comp_name, sentence_text):
        comp_parts = comp_name.split()
        if len(comp_parts) < 2:
            return True
        if not comp_name.upper().startswith(abbrev.upper()):
            return True
        pattern = rf'\b{re.escape(abbrev)}\b'
        full_rest = comp_name[len(abbrev):].strip()
        found_valid = False
        for m in re.finditer(pattern, sentence_text, re.IGNORECASE):
            end = m.end()
            rest = sentence_text[end:].lstrip()
            if not rest:
                found_valid = True
                break
            if rest.lower().startswith(full_rest.lower()):
                found_valid = True
                break
            next_word_m = re.match(r'(\w+)', rest)
            if next_word_m:
                next_word = next_word_m.group(1).lower()
                expected_next = full_rest.split()[0].lower() if full_rest else ""
                if next_word != expected_next and next_word.isalpha():
                    continue
            found_valid = True
            break
        return found_valid

    def _apply_abbreviation_guard_to_candidates(self, candidates, sent_map):
        if not self.doc_knowledge:
            return candidates
        abbrev_to_comp = {}
        comp_to_abbrevs = {}
        for abbr, comp in self.doc_knowledge.abbreviations.items():
            abbrev_to_comp[abbr.lower()] = comp
            comp_to_abbrevs.setdefault(comp, []).append(abbr)

        filtered = []
        for c in candidates:
            matched_lower = c.matched_text.lower() if c.matched_text else ""
            comp = c.component_name
            sent = sent_map.get(c.sentence_number)
            if matched_lower in abbrev_to_comp and abbrev_to_comp[matched_lower] == comp:
                if sent and not self._abbreviation_match_is_valid(c.matched_text, comp, sent.text):
                    print(f"    Abbrev guard: rejected S{c.sentence_number} {c.matched_text} -> {comp}")
                    continue
            if sent and comp in comp_to_abbrevs and ' ' in comp:
                full_in_text = re.search(rf'\b{re.escape(comp)}\b', sent.text, re.IGNORECASE)
                if not full_in_text:
                    rejected = False
                    for abbr in comp_to_abbrevs[comp]:
                        if re.search(rf'\b{re.escape(abbr)}\b', sent.text, re.IGNORECASE):
                            if not self._abbreviation_match_is_valid(abbr, comp, sent.text):
                                print(f"    Abbrev guard (inferred): rejected S{c.sentence_number} {abbr} -> {comp}")
                                rejected = True
                                break
                    if rejected:
                        continue
            filtered.append(c)
        return filtered

    def _abbreviation_guard_for_link(self, lk, sent_map):
        if not self.doc_knowledge:
            return True
        sent = sent_map.get(lk.sentence_number)
        if not sent:
            return True
        if re.search(rf'\b{re.escape(lk.component_name)}\b', sent.text, re.IGNORECASE):
            return True
        for abbr, comp in self.doc_knowledge.abbreviations.items():
            if comp == lk.component_name:
                if abbr.lower() in sent.text.lower():
                    if not self._abbreviation_match_is_valid(abbr, comp, sent.text):
                        return False
        return True

    def _apply_boundary_filters(self, links, sent_map, transarc_set):
        kept = []
        rejected = []
        for lk in links:
            sent = sent_map.get(lk.sentence_number)
            if not sent:
                kept.append(lk)
                continue
            is_ta = (lk.sentence_number, lk.component_id) in transarc_set
            if is_ta:
                kept.append(lk)
                continue
            reason = None
            if lk.source in ("validated", "entity"):
                if self._is_in_package_path(lk.component_name, sent.text):
                    reason = "package_path"
            if not reason and lk.source in ("validated", "entity", "coreference"):
                if self._is_generic_word_usage(lk.component_name, sent.text):
                    reason = "generic_word"
            if not reason:
                if self._is_weak_partial_match(lk, sent_map):
                    reason = "weak_partial"
            if not reason:
                if not self._abbreviation_guard_for_link(lk, sent_map):
                    reason = "abbrev_guard"
            if reason:
                rejected.append((lk, reason))
                print(f"    Boundary filter [{reason}]: S{lk.sentence_number} -> {lk.component_name} ({lk.source})")
            else:
                kept.append(lk)
        return kept, rejected

    # ═════════════════════════════════════════════════════════════════════
    # Phase 9: Judge — Feature D controls TransArc handling
    # ═════════════════════════════════════════════════════════════════════

    def _has_standalone_mention(self, comp_name, text):
        is_single = ' ' not in comp_name
        if is_single:
            cap_name = comp_name[0].upper() + comp_name[1:]
            pattern = rf'\b{re.escape(cap_name)}\b'
            flags = 0
        else:
            pattern = rf'\b{re.escape(comp_name)}\b'
            flags = re.IGNORECASE

        for m in re.finditer(pattern, text, flags):
            s, e = m.start(), m.end()
            if s > 0 and text[s-1] == '.':
                continue
            if e < len(text) and text[e] == '.' and e + 1 < len(text) and text[e+1].isalpha():
                continue
            if s > 0 and text[s-1] == '-':
                continue
            if e < len(text) and text[e] == '-' and '-' not in comp_name:
                continue
            return True
        return False

    def _judge_review(self, links, sentences, components, sent_map, transarc_set):
        if len(links) < 5:
            return links

        comp_names = self._get_comp_names(components)

        safe, review = [], []
        for l in links:
            is_ta = (l.sentence_number, l.component_id) in transarc_set
            sent = sent_map.get(l.sentence_number)

            if not self.feat_nuanced_ta:
                # V16 behavior: blanket TransArc immunity
                if is_ta:
                    safe.append(l)
                    continue
                if not sent:
                    review.append(l)
                    continue
                if self._has_standalone_mention(l.component_name, sent.text):
                    safe.append(l)
                else:
                    review.append(l)
            else:
                # V6 behavior: nuanced TransArc handling
                if not sent:
                    review.append(l)
                    continue
                has_standalone = self._has_standalone_mention(l.component_name, sent.text)
                if has_standalone:
                    safe.append(l)
                elif is_ta:
                    # TransArc but name appears generically — check if name present at all
                    has_any = re.search(rf'\b{re.escape(l.component_name)}\b',
                                        sent.text, re.IGNORECASE)
                    if has_any:
                        review.append(l)  # Generic mention → judge decides
                    else:
                        safe.append(l)  # Name absent → trust TransArc NLP
                else:
                    review.append(l)

        if not review:
            return safe

        cases = self._build_judge_cases(review, sent_map)
        prompt = self._build_judge_prompt(comp_names, cases)
        n = min(30, len(review))

        # Union voting: reject only if BOTH passes reject
        data1 = self.llm.extract_json(self.llm.query(prompt, timeout=180))
        data2 = self.llm.extract_json(self.llm.query(prompt, timeout=180))

        rej1 = self._parse_rejections(data1, n)
        rej2 = self._parse_rejections(data2, n)
        rejected = rej1 & rej2

        result = safe.copy()
        for i in range(n):
            if i not in rejected:
                result.append(review[i])
        result.extend(review[n:])
        return result

    def _parse_rejections(self, data, n):
        rejected = set()
        if data:
            for j in data.get("judgments", []):
                idx = j.get("case", 0) - 1
                if 0 <= idx < n and not j.get("approve", False):
                    rejected.add(idx)
        return rejected

    def _find_match_text(self, comp_name, sent_text):
        """Find what text in the sentence triggered the match to this component."""
        if not sent_text:
            return None
        text_lower = sent_text.lower()
        # Check exact component name
        if re.search(rf'\b{re.escape(comp_name)}\b', sent_text, re.IGNORECASE):
            return comp_name
        # Check known aliases from doc_knowledge
        if self.doc_knowledge:
            for alias, comp in self.doc_knowledge.synonyms.items():
                if comp == comp_name and re.search(rf'\b{re.escape(alias)}\b', sent_text, re.IGNORECASE):
                    return alias
            for alias, comp in self.doc_knowledge.abbreviations.items():
                if comp == comp_name and re.search(rf'\b{re.escape(alias)}\b', sent_text, re.IGNORECASE):
                    return alias
            for partial, comp in self.doc_knowledge.partial_references.items():
                if comp == comp_name and partial.lower() in text_lower:
                    return partial
        # No match found in text — likely pronoun/coreference
        return None

    def _build_judge_cases(self, review, sent_map):
        # Determine context mode
        use_full_ctx = self.judge_full_context
        if self.judge_adaptive_ctx:
            use_full_ctx = not self._is_complex  # Full for simple docs, truncated for complex

        cases = []
        for i, l in enumerate(review[:30]):
            sent = sent_map.get(l.sentence_number)
            ctx_lines = []
            if l.source in ("implicit", "coreference"):
                p2 = sent_map.get(l.sentence_number - 2)
                if p2:
                    txt = p2.text if use_full_ctx else f"{p2.text[:45]}..."
                    ctx_lines.append(f"    PREV2: {txt}")
            p1 = sent_map.get(l.sentence_number - 1)
            if p1:
                txt = p1.text if use_full_ctx else f"{p1.text[:45]}..."
                ctx_lines.append(f"    PREV: {txt}")
            ctx_lines.append(f"    >>> S{l.sentence_number}: {sent.text if sent else '?'}")

            # J6: Show match text
            src_info = f"src:{l.source}"
            if self.judge_show_match and sent:
                match = self._find_match_text(l.component_name, sent.text)
                if match and match.lower() != l.component_name.lower():
                    src_info += f', match:"{match}"'
                elif not match:
                    src_info += ', match:NONE(pronoun/context)'

            cases.append(f"Case {i+1}: S{l.sentence_number} -> {l.component_name} ({src_info})\n"
                         + chr(10).join(ctx_lines))
        return cases

    def _build_judge_prompt(self, comp_names, cases):
        # J4: Strict definition (no functional-only approval)
        if self.judge_strict_def:
            tl_def = """WHAT IS A TRACE LINK:
A trace link connects a documentation sentence to an architecture component when the sentence
DIRECTLY REFERENCES the component by name. Valid references (in order of frequency):
- EXPLICIT: The component name appears verbatim (e.g., "the MediaAccess component stores...")
- SYNONYM: A known synonym or abbreviation is used (e.g., "AudioAccess" for MediaAccess)
- PRONOUN: A pronoun (it, they, this) clearly resolves to a recently named component

CRITICAL: A sentence that only describes a FUNCTION or CONCEPT related to a component
(e.g., "data is cached" for a Cache component, "backends handle events" for HTML5 Server)
is NOT a valid trace link UNLESS the component name or a known alias actually appears.
Generic implementation terms (backends, frontends, caching, storage, server, client) are
NOT component references even if a component has a similar name.

"""
        else:
            tl_def = """WHAT IS A TRACE LINK:
A trace link connects a documentation sentence to an architecture component when the sentence
DIRECTLY REFERENCES the component. Valid references include:
- The component name appears verbatim in the sentence (~60% of links)
- A known synonym, abbreviation, or partial name is used (~20% of links)
- A pronoun (it, they, this) clearly resolves to a recently mentioned component (~10% of links)
A sentence that merely describes a generic concept (e.g., "caching", "storage") without
referencing a SPECIFIC component is NOT a valid trace link.

"""

        # J1: Document knowledge section
        doc_know_section = ""
        if self.judge_doc_knowledge and self.doc_knowledge:
            aliases = []
            for a, c in self.doc_knowledge.abbreviations.items():
                aliases.append(f"  {a} = {c} (abbreviation)")
            for s, c in self.doc_knowledge.synonyms.items():
                aliases.append(f"  {s} = {c} (synonym)")
            for p, c in self.doc_knowledge.partial_references.items():
                aliases.append(f"  {p} → {c} (partial)")
            if aliases:
                doc_know_section = "KNOWN ALIASES (discovered from this document):\n"
                doc_know_section += chr(10).join(aliases)
                doc_know_section += "\nIf a sentence uses a known alias, that counts as EXPLICIT MENTION.\n\n"

        # J3: Validated bias rule
        if self.judge_validated_bias:
            validated_rule = (
                '- "validated": Already confirmed by two independent LLM extraction passes. '
                'Default to APPROVE unless the link is CLEARLY wrong '
                '(e.g., the word is used in a completely generic sense unrelated to the component).'
            )
        else:
            validated_rule = (
                '- "validated": Entity extraction found this. '
                'Check: is name used as ARCHITECTURE REFERENCE or GENERIC WORD?'
            )

        # J4: Stricter criteria (require explicit reference, not functional-only)
        if self.judge_strict_def:
            criteria = """Evaluate each link:

1. EXPLICIT REFERENCE: Does the sentence contain the component name, a known synonym, or a resolvable pronoun?
2. UNIQUE REFERENCE: Could this sentence plausibly refer to a DIFFERENT component with equal or greater likelihood?

APPROVE if criterion 1 is YES and criterion 2 is NO.
REJECT if the sentence only describes a function/concept WITHOUT naming the component."""
        else:
            criteria = """Evaluate each link using these THREE criteria:

1. EXPLICIT MENTION: Does the sentence contain the component name (or a known synonym/abbreviation)?
2. FUNCTIONAL DESCRIPTION: Does the sentence describe what this SPECIFIC component does — not what the system does in general?
3. UNIQUE REFERENCE: Could this sentence plausibly refer to a DIFFERENT component with equal or greater likelihood?

APPROVE if criterion 1 OR 2 is YES, and criterion 3 is NO.
REJECT otherwise."""

        # J6: Match-aware rejection guidance
        match_guidance = ""
        if self.judge_show_match:
            match_guidance = """
MATCH TEXT GUIDANCE:
- If match text differs from component name (e.g., match:"backends" for HTML5 Server),
  ask: is this match text a DIRECT REFERENCE to the component, or a generic term?
- If match:NONE, the link was inferred from context/pronouns — verify the reference chain.
"""

        return f"""JUDGE: Review trace links in a software architecture document.

{tl_def}COMPONENTS: {', '.join(comp_names)}

{doc_know_section}{criteria}

CONCRETE EXAMPLES OF REJECTION PATTERNS:
- "the business logic handles..." → "logic" is generic English, NOT an architectural component → REJECT
- "client-side rendering" / "on the client" → generic usage, NOT a specific component → REJECT
- "data storage layer" / "in-memory cache" → generic concept, NOT a specific component → REJECT
- "pkg.subpkg.ComponentName" → package path, not an architectural reference → REJECT
- "backends handle events" → "backends" is generic, NOT a reference to "HTML5 Server" → REJECT

CONCRETE EXAMPLES OF APPROVAL PATTERNS:
- "Router handles incoming requests" → component is the ACTOR performing the action → APPROVE
- "The AuthService component validates tokens" → explicit component reference → APPROVE
- Section heading with a component name → introduces a section about that component → APPROVE
- "It delegates to Processor for computation" → component is the TARGET of delegation → APPROVE
{match_guidance}
SOURCE-SPECIFIC RULES:
- "coreference": Pronoun resolution. Verify pronoun CLEARLY refers to claimed component.
- "partial_inject": Partial name match. Verify the partial refers to THIS component, not generic usage.
{validated_rule}

LINKS:
{chr(10).join(cases)}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

    # ═════════════════════════════════════════════════════════════════════
    # Feature B: Phase 8 — Implicit reference detection (from V6)
    # ═════════════════════════════════════════════════════════════════════

    def _detect_implicit_references(self, sentences, components, name_to_id, sent_map,
                                     discourse_model, existing_links):
        comp_names = self._get_comp_names(components)
        candidates = []
        for sent in sentences:
            tl = sent.text.lower()
            if any(c.lower() in tl for c in comp_names):
                continue
            ctx = discourse_model.get(sent.number, DiscourseContext())
            if not ctx.active_entity or not ctx.paragraph_topic:
                continue
            if ctx.active_entity != ctx.paragraph_topic:
                continue
            likely = ctx.active_entity
            if (sent.number, name_to_id.get(likely, '')) in existing_links:
                continue
            dist = self._distance_to_last_mention(ctx.recent_mentions, likely, sent.number)
            if dist is None or dist > 2:
                continue
            candidates.append((sent, ctx, likely))

        if not candidates:
            return []
        print(f"  Candidates: {len(candidates)}")

        implicit = []
        for batch_start in range(0, len(candidates), 5):
            batch = candidates[batch_start:batch_start + 5]
            prompt = f"""Detect implicit component references. Be VERY conservative.

COMPONENTS: {', '.join(comp_names)}

"""
            for i, (sent, ctx, likely) in enumerate(batch):
                prev = sent_map.get(sent.number - 1)
                prompt += f"--- Case {i+1}: S{sent.number} ---\n"
                prompt += f"Previous: {prev.text[:70] if prev else 'N/A'}\n"
                prompt += f"Sentence: {sent.text}\n"
                prompt += f"Context: {ctx.get_context_summary(sent.number)}\n"
                prompt += f"Likely component: {likely}\n\n"

            prompt += """Return JSON:
{"detections": [{"case": 1, "sentence": N, "component": "Name", "is_implicit": true/false, "certainty": "high or low", "reasoning": "why"}]}

An implicit reference is ONLY when:
- The sentence continues describing the EXACT SAME component without re-naming it
- The action described can ONLY be performed by this specific component
- Subject is omitted but implied from IMMEDIATE context (previous 1-2 sentences)

NOT implicit if: result/effect, multiple possible actors, new topic, general process,
algorithm/data flow, configuration, concept, or system-wide behavior.

Only return is_implicit=true with certainty="high".
JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
            if not data:
                continue
            for det in data.get("detections", []):
                if not det.get("is_implicit"):
                    continue
                comp, snum = det.get("component"), det.get("sentence")
                if str(det.get("certainty", "low")).lower() != "high":
                    continue
                if not (comp and snum and comp in name_to_id):
                    continue
                try:
                    snum = int(snum)
                except (ValueError, TypeError):
                    continue
                implicit.append(SadSamLink(snum, name_to_id[comp], comp, 1.0, "implicit"))

        return implicit

    def _validate_implicit_links(self, implicit_links, sentences, components, sent_map):
        if not implicit_links:
            return []
        comp_names = self._get_comp_names(components)
        validated = []

        for batch_start in range(0, len(implicit_links), 10):
            batch = implicit_links[batch_start:batch_start + 10]
            cases = []
            for i, link in enumerate(batch):
                sent = sent_map.get(link.sentence_number)
                prev1 = sent_map.get(link.sentence_number - 1)
                prev2 = sent_map.get(link.sentence_number - 2)
                ctx_lines = []
                if prev2: ctx_lines.append(f"S{prev2.number}: {prev2.text[:60]}")
                if prev1: ctx_lines.append(f"S{prev1.number}: {prev1.text[:60]}")
                ctx_lines.append(f">>> S{link.sentence_number}: {sent.text if sent else '?'}")
                cases.append(f"Case {i+1}: S{link.sentence_number} -> {link.component_name}\n"
                             + "\n".join(f"    {l}" for l in ctx_lines))

            prompt = f"""VALIDATION: Verify these implicit component references.

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true/false, "reasoning": "brief"}}]}}

Reject if action could be any component, system-wide, or tenuous connection.
JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=150))
            if not data:
                continue
            for v in data.get("validations", []):
                idx = v.get("case", 0) - 1
                if 0 <= idx < len(batch) and v.get("approve", False):
                    validated.append(batch[idx])

        return validated

    # ═════════════════════════════════════════════════════════════════════
    # Feature C: Phase 10 — FN Recovery (from V6)
    # ═════════════════════════════════════════════════════════════════════

    def _fn_recovery(self, current_links, sentences, components, name_to_id, sent_map):
        comp_names = self._get_comp_names(components)

        covered = {l.sentence_number for l in current_links}
        synonyms_to_comp = {}
        if self.doc_knowledge:
            for syn, comp in self.doc_knowledge.synonyms.items():
                synonyms_to_comp[syn.lower()] = comp
            for abbr, comp in self.doc_knowledge.abbreviations.items():
                synonyms_to_comp[abbr.lower()] = comp
            for partial, comp in self.doc_knowledge.partial_references.items():
                synonyms_to_comp[partial.lower()] = comp

        potential = []
        for sent in sentences:
            if sent.number in covered:
                continue
            sl = sent.text.lower()
            found = False
            for cname in comp_names:
                if cname.lower() in sl and re.search(rf'\b{re.escape(cname)}\b', sent.text, re.IGNORECASE):
                    potential.append((sent.number, sent.text, cname))
                    found = True
            if not found:
                for syn, comp in synonyms_to_comp.items():
                    if syn in sl and comp in comp_names and re.search(
                        rf'\b{re.escape(syn)}\b', sent.text, re.IGNORECASE
                    ):
                        potential.append((sent.number, sent.text, comp))

        if not potential:
            print("    No candidates")
            return current_links

        print(f"    Candidates: {len(potential)}")

        recovered_candidates = []
        for batch_start in range(0, len(potential), 12):
            batch = potential[batch_start:batch_start + 12]
            cases = [f"Case {i+1}: S{sn}: \"{txt[:70]}...\" -> {cn}?"
                     for i, (sn, txt, cn) in enumerate(batch)]

            r1 = self._qual_recovery_pass(comp_names, cases, "Is component the ACTOR?")
            r2 = self._qual_recovery_pass(comp_names, cases, "Is this a valid missed link?")

            for i, (snum, _, cname) in enumerate(batch):
                if r1.get(i, False) and r2.get(i, False) and cname in name_to_id:
                    key = (snum, name_to_id[cname])
                    if not any((l.sentence_number, l.component_id) == key for l in current_links):
                        recovered_candidates.append(SadSamLink(snum, name_to_id[cname], cname, 1.0, "recovered"))

        # Judge confirmation
        if recovered_candidates:
            recovered_candidates = self._judge_recovered(recovered_candidates, comp_names, sent_map)

        result = current_links.copy()
        for lk in recovered_candidates:
            result.append(lk)
            print(f"    Recovered: S{lk.sentence_number} -> {lk.component_name}")
        return result

    def _judge_recovered(self, candidates, comp_names, sent_map):
        cases = []
        for i, lk in enumerate(candidates):
            sent = sent_map.get(lk.sentence_number)
            prev = sent_map.get(lk.sentence_number - 1)
            ctx_lines = []
            if prev:
                ctx_lines.append(f"    PREV: {prev.text[:50]}...")
            ctx_lines.append(f"    >>> S{lk.sentence_number}: {sent.text if sent else '?'}")
            cases.append(f"Case {i+1}: S{lk.sentence_number} -> {lk.component_name}\n" + chr(10).join(ctx_lines))

        prompt = f"""JUDGE: These are RECOVERED links — sentences that mention a component but were not linked by any earlier phase. Be STRICT: only approve if the component is clearly the ACTOR/SUBJECT performing an action.

COMPONENTS: {', '.join(comp_names)}

REJECT IF:
- Component name appears as a modifier, adjective, or in a compound word
- Component name is used as a generic English word, not an architecture reference
- Sentence describes package/directory/module structure using dot notation
- Sentence describes a general process/concept where the component is just mentioned
- Component is the OBJECT/TARGET of an action by another component
- Sentence is about configuration, deployment, or system-level behavior

LINKS:
{chr(10).join(cases)}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief"}}]}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
        if not data:
            return []

        approved = []
        for j in data.get("judgments", []):
            idx = j.get("case", 0) - 1
            if 0 <= idx < len(candidates) and j.get("approve", False):
                approved.append(candidates[idx])
        print(f"    Recovery judge: {len(approved)}/{len(candidates)} approved")
        return approved

    def _qual_recovery_pass(self, comp_names, cases, question):
        prompt = f"""Check potential missed links. {question}

COMPONENTS: {', '.join(comp_names)}

POTENTIAL:
{chr(10).join(cases)}

Return JSON:
{{"recoveries": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=100))
        results = {}
        if data:
            for r in data.get("recoveries", []):
                idx = r.get("case", 0) - 1
                if 0 <= idx < len(cases):
                    results[idx] = r.get("approve", False)
        return results
