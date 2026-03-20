"""S-Linker11: LLM-driven SAD-SAM traceability with source-adapted verification.

Three-phase pipeline with internal parallelism:

  Phase 1 — Knowledge Acquisition:
      Parallel: Model analysis | Document knowledge | LLM seed extraction
      Then: LLM word usage classification for multiword partial references

  Phase 2 — Link Recovery:
      Parallel: Seed validation | Entity extraction + validation | Coreference
      Then: Partial-reference injection + validation

  Phase 3 — Merge:
      Deduplication (first-seen priority: seed > entity > coref > partial)

Verification strategy (empirically motivated):
  Seed, entity, and partial candidates share the same validation
  infrastructure: generic-mention pre-filter, then dual-focus LLM voting.
  The voting threshold is adapted to evidence type: alias-backed matches
  use union (either pass), exact-name matches use intersection (both passes).
  Coreference uses its own antecedent-based verification (component mention
  in antecedent sentence within discourse window).
"""

import json
import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from llm_sad_sam.core.data_types import (
    SadSamLink, CandidateLink,
    ModelKnowledge, DocumentKnowledge,
)
from llm_sad_sam.core.document_loader import DocumentLoader
from llm_sad_sam.linkers.experimental.ilinker2 import ILinker2
from llm_sad_sam.linkers.experimental.prompts_v2 import (
    AMBIGUITY_FEW_SHOT, AMBIGUITY_RULES,
    DOC_KNOWLEDGE_JUDGE_EXAMPLES, DOC_KNOWLEDGE_JUDGE_RULES,
    DOC_KNOWLEDGE_EXTRACTION_RULES,
    ENTITY_EXTRACTION_RULES, VALIDATION_RULES, COREF_RULES,
    WORD_USAGE_PROMPT,
)
from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient, LLMBackend

class SLinker11:
    """LLM-driven SAD-SAM TLR with source-adapted verification."""

    PRONOUN_PATTERN = re.compile(
        r'\b(it|they|this|these|that|those|its|their)\b',
        re.IGNORECASE
    )
    _FEW_SHOT = AMBIGUITY_FEW_SHOT

    def __init__(self, backend: Optional[LLMBackend] = None):
        os.environ.setdefault("CLAUDE_MODEL", "sonnet")
        self.llm = LLMClient(backend=backend or LLMBackend.CLAUDE)
        self.model_knowledge: Optional[ModelKnowledge] = None
        self.doc_knowledge: Optional[DocumentKnowledge] = None
        self._phase_log = []
        self._ilinker2 = ILinker2(backend=self.llm.backend)
        self._generic_partials: set = set()
        print(f"SLinker11 (3-phase pipeline, source-adapted verification)")
        print(f"  Backend: {self.llm.backend.value}, Model: {os.environ.get('CLAUDE_MODEL', 'default')}")

    # ═══════════════════════════════════════════════════════════════════════
    # DAG Infrastructure
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _run_parallel(tasks):
        """Run named tasks concurrently, wait for all. Returns {name: result}.

        On first failure, cancels remaining futures and re-raises.
        """
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
    # Main Entry Point — DAG Orchestration
    # ═══════════════════════════════════════════════════════════════════════

    def link(self, text_path, model_path, **_kwargs):
        """Recover trace links between SAD and SAM via 5-layer pipeline.

        Args:
            text_path: Path to documentation text file (one sentence per line).
            model_path: Path to PCM .repository file.

        Returns:
            list[SadSamLink]: Recovered trace links.
        """
        self._phase_log = []
        t0 = time.time()

        # Load raw data
        components = parse_pcm_repository(model_path)
        sentences = DocumentLoader.load_sentences(text_path)
        name_to_id = {c.name: c.id for c in components}
        sent_map = DocumentLoader.build_sent_map(sentences)

        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        # ═══ LAYER 1: Knowledge Acquisition (all independent) ═══
        print("\n[Layer 1] Knowledge Acquisition (parallel)")
        l1 = self._run_parallel({
            "model": lambda: self._analyze_model(components),
            "doc_knowledge": lambda: self._learn_document_knowledge_enriched(sentences, components),
            "seed": lambda: self._run_seed(text_path, model_path),
        })

        self.model_knowledge = l1["model"]
        self.doc_knowledge = l1["doc_knowledge"]
        raw_seed_links = l1["seed"]

        # Derive generic partial set from model analysis
        self._compute_generic_partials(components)

        ambig = self.model_knowledge.ambiguous_names
        print(f"  Model: {len(ambig)} ambiguous (of {len(components)} components)")
        print(f"  Doc knowledge: {len(self.doc_knowledge.abbreviations)} abbrev, "
              f"{len(self.doc_knowledge.synonyms)} syn, "
              f"{len(self.doc_knowledge.partial_references)} partial")
        print(f"  Seed: {len(raw_seed_links)} raw links")
        print(f"  Generic partials: {sorted(self._generic_partials)}")

        self._log("layer1", {"sents": len(sentences), "comps": len(components)},
                  {"ambig": len(ambig), "seed": len(raw_seed_links),
                   "abbrev": len(self.doc_knowledge.abbreviations)})

        self._save_phase(text_path, "layer1", {
            "model_knowledge": self.model_knowledge,
            "doc_knowledge": self.doc_knowledge,
            "raw_seed_links": raw_seed_links,
            "generic_partials": self._generic_partials,
        })

        # ═══ LAYER 2: Knowledge Enrichment (needs Layer 1) ═══
        print("\n[Layer 2] Knowledge Enrichment")
        self._enrich_multiword_partials(sentences, components)

        self._save_phase(text_path, "layer2", {
            "doc_knowledge": self.doc_knowledge,
        })

        # ═══ LAYER 3: Link Recovery (all three parallel) ═══
        # Seed validation, entity extraction+validation, and coreference
        # all depend on Layer 1+2 knowledge but are independent of each other.
        print("\n[Layer 3] Link Recovery (parallel)")
        l3 = self._run_parallel({
            "seed_val": lambda: self._run_seed_validation(
                raw_seed_links, components, sent_map),
            "entity": lambda: self._run_entity_pipeline(
                sentences, components, name_to_id, sent_map),
            "coref": lambda: self._run_coreference(
                sentences, components, name_to_id, sent_map),
        })

        seed_links = l3["seed_val"]
        validated = l3["entity"]
        coref_links = l3["coref"]
        seed_set = {(l.sentence_number, l.component_id) for l in seed_links}
        print(f"  Seed validated: {len(seed_links)} / {len(raw_seed_links)}")
        print(f"  Entity pipeline: {len(validated)} validated")
        print(f"  Coreference: {len(coref_links)} links")

        # ═══ LAYER 4: Partial Recovery (needs Layer 3 outputs) ═══
        print("\n[Layer 4] Partial Recovery")
        partial_candidates = self._inject_partial_candidates(
            sentences, components, name_to_id, sent_map, seed_set,
            {(c.sentence_number, c.component_id) for c in validated},
            {(l.sentence_number, l.component_id) for l in coref_links},
        )
        if partial_candidates:
            print(f"  Partial candidates: {len(partial_candidates)}")
            partial_validated = self._validate_intersect(partial_candidates, components, sent_map)
            print(f"  Partial validated: {len(partial_validated)} / {len(partial_candidates)}")
        else:
            partial_validated = []

        self._save_phase(text_path, "layer3", {
            "seed_links": seed_links,
            "validated": validated,
            "coref_links": coref_links,
        })

        self._save_phase(text_path, "layer4", {
            "partial_validated": partial_validated,
        })

        # ═══ LAYER 5: Merge (dedup only) ═══
        print("\n[Layer 5] Merge")

        # Deduplication (first-seen wins — order: seed, entity, coref, partial)
        entity_links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name, source=c.source)
            for c in validated
        ]
        partial_links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name, source="partial")
            for c in partial_validated
        ]
        all_links = seed_links + entity_links + coref_links + partial_links
        seen = set()
        final = []
        for lk in all_links:
            key = (lk.sentence_number, lk.component_id)
            if key not in seen:
                seen.add(key)
                final.append(lk)
        print(f"  After dedup: {len(final)} (from {len(all_links)} raw)")

        # Save log + final checkpoint
        self._log("summary", {"total_time_s": round(time.time() - t0, 1)},
                  {"final": len(final)}, final)
        self._save_log(text_path)

        self._save_phase(text_path, "final", {
            "final": final,
        })

        print(f"\nFinal: {len(final)} links ({time.time() - t0:.0f}s)")
        return final

    # ═══════════════════════════════════════════════════════════════════════
    # Layer 1: Knowledge Acquisition
    # ═══════════════════════════════════════════════════════════════════════

    def _analyze_model(self, components):
        """Analyze model structure: classify component names as architectural/ambiguous."""
        names = [c.name for c in components]
        knowledge = ModelKnowledge()
        self._classify_components(names, knowledge)
        return knowledge

    @staticmethod
    def _is_structurally_unambiguous(name):
        """CamelCase, multi-word, or all-caps -> always architectural."""
        if ' ' in name or '-' in name:
            return True
        if re.search(r'[a-z][A-Z]', name):
            return True
        if name.isupper():
            return True
        return False

    def _classify_components(self, names, knowledge):
        """Classify components using few-shot prompt + structural code guard."""
        prompt = f"""Classify these software architecture component names.

NAMES: {', '.join(names)}

{self._FEW_SHOT}

NOW CLASSIFY THE NAMES ABOVE.

Return JSON:
{{
  "architectural": ["names that identify specific components"],
  "ambiguous": ["names that could easily be used as ordinary words in documentation"]
}}

{AMBIGUITY_RULES}

JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=100))
        if data:
            valid = set(names)
            raw_ambiguous = set(data.get("ambiguous", [])) & valid
            knowledge.ambiguous_names = {
                n for n in raw_ambiguous
                if len(n.split()) == 1 and not self._is_structurally_unambiguous(n)
            }

    def _compute_generic_partials(self, components):
        """Derive generic partial set from model analysis results.

        A partial is "generic" if it matches an ambiguous component name
        (e.g., "management" from "DataManagement" when "management" is ambiguous).
        Used to require capitalized mentions in multiword partial enrichment.
        """
        ambig = self.model_knowledge.ambiguous_names if self.model_knowledge else set()

        self._generic_partials = set()
        for comp in components:
            parts = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)', comp.name)
            for part in parts:
                p_lower = part.lower()
                if part.isupper():
                    continue
                if len(p_lower) >= 3 and (p_lower in ambig or any(
                    p_lower == a.lower() for a in ambig
                )):
                    self._generic_partials.add(p_lower)
        for name in ambig:
            if ' ' not in name and not name.isupper():
                self._generic_partials.add(name.lower())

    def _learn_document_knowledge_enriched(self, sentences, components):
        """Extract abbreviations, synonyms, partial references via few-shot calibrated judge."""
        comp_names = [c.name for c in components]
        doc_lines = [s.text for s in sentences]

        prompt1 = f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

{DOC_KNOWLEDGE_EXTRACTION_RULES}

DOCUMENT:
{chr(10).join(doc_lines)}

Return JSON:
{{
  "abbreviations": {{"short_form": "FullComponent"}},
  "synonyms": {{"specific_alternative_name": "FullComponent"}},
  "partial_references": {{"partial_name": "FullComponent"}}
}}
JSON only:"""

        data1 = self.llm.extract_json(self.llm.query(prompt1, timeout=300))

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

{DOC_KNOWLEDGE_JUDGE_EXAMPLES}

{DOC_KNOWLEDGE_JUDGE_RULES}

Return JSON:
{{
  "approved": ["term1", "term2"]
}}
JSON only:"""

            data2 = self.llm.extract_json(self.llm.query(prompt2, timeout=120))
            approved = set(data2.get("approved", [])) if data2 else set(all_mappings.keys())
        else:
            approved = set()

        knowledge = DocumentKnowledge()

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

        return knowledge

    def _run_seed(self, text_path, model_path):
        """LLM-based seed extraction (two-pass: broad recall + precision enrichment).

        Uses a lightweight LLM extractor as the seed strategy. The seed
        provides broad initial coverage; false positives are filtered by
        the same evidence-stratified validation applied to all strategies.
        """
        raw = self._ilinker2.link(text_path, model_path)
        return [SadSamLink(l.sentence_number, l.component_id, l.component_name,
                           source="seed") for l in raw]

    # ═══════════════════════════════════════════════════════════════════════
    # Layer 2: Knowledge Enrichment
    # ═══════════════════════════════════════════════════════════════════════

    def _enrich_multiword_partials(self, sentences, components):
        """Auto-discover multi-word partial references via LLM word usage classification.

        Instead of count>=3, asks the LLM whether the trailing word of a
        multi-word component name is used as a standalone entity reference.
        """
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

            is_generic_word = last_lower in self._generic_partials
            full_lower = comp.name.lower()

            # Find sentences where trailing word appears without full name
            relevant_sents = []
            for sent in sentences:
                sl = sent.text.lower()
                if last_lower in sl and full_lower not in sl:
                    if is_generic_word:
                        cap_word = last_word[0].upper() + last_word[1:]
                        if re.search(rf'\b{re.escape(cap_word)}\b', sent.text):
                            relevant_sents.append(sent)
                    else:
                        if re.search(rf'\b{re.escape(last_word)}\b', sent.text, re.IGNORECASE):
                            relevant_sents.append(sent)

            if not relevant_sents:
                continue

            # LLM word usage classification
            calibration = ""
            if is_generic_word:
                calibration = (f'NOTE: "{last_word}" is also an ordinary English word. '
                               f'Be careful to distinguish entity references from generic usage.\n\n')

            sent_block = "\n".join(f"  S{s.number}: {s.text}" for s in relevant_sents[:20])

            prompt = WORD_USAGE_PROMPT.format(
                partial=last_word,
                partial_lower=last_lower,
                comp_name=comp.name,
                calibration=calibration,
                sent_block=sent_block,
            )

            data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
            classification = data.get("classification", "ordinary") if data else "ordinary"
            reason = data.get("reason", "") if data else ""

            if classification == "name":
                self.doc_knowledge.partial_references[last_word] = comp.name
                added.append(f"{last_word} -> {comp.name} (LLM: {reason})")
            else:
                print(f"    LLM rejected: {last_word} -> {comp.name} ({reason})")

        if added:
            print(f"  [Enrichment] Multi-word partials (LLM):")
            for a in added:
                print(f"    Auto-partial: {a}")

    # ═══════════════════════════════════════════════════════════════════════
    # Layer 3: Link Recovery
    # ═══════════════════════════════════════════════════════════════════════

    def _run_seed_validation(self, raw_seed_links, components, sent_map):
        """Validate seed links through evidence-stratified 3-step validation.

        Killed TPs are recovered by entity/coref/partial via dedup
        (tested: zero net recall cost).
        """
        seed_candidates = []
        for sl in raw_seed_links:
            sent = sent_map.get(sl.sentence_number)
            if not sent:
                continue
            # Detect what actually matched in the sentence (component name or alias)
            # so evidence stratification can distinguish exact vs alias-backed.
            matched = self._find_matched_text(sl.component_name, sent.text)
            seed_candidates.append(CandidateLink(
                sl.sentence_number, sent.text, sl.component_name, sl.component_id,
                matched, source="seed",
            ))
        seed_validated = self._validate_intersect(seed_candidates, components, sent_map)
        return [SadSamLink(c.sentence_number, c.component_id, c.component_name, source="seed")
                for c in seed_validated]

    def _find_matched_text(self, comp_name, sentence_text):
        """Find what actually matched in the sentence: component name or known alias.

        Returns the matched text (alias string if alias-backed, component name
        if exact). This determines evidence stratification: alias-backed matches
        get union voting, exact-name matches get intersection.
        """
        # Check exact component name first
        if self._has_standalone_mention(comp_name, sentence_text):
            return comp_name
        # Check known aliases (abbreviations, synonyms, partial references)
        if self.doc_knowledge:
            for alias, target in self.doc_knowledge.abbreviations.items():
                if target == comp_name and re.search(rf'\b{re.escape(alias)}\b', sentence_text):
                    return alias
            for alias, target in self.doc_knowledge.synonyms.items():
                if target == comp_name and re.search(rf'\b{re.escape(alias)}\b', sentence_text, re.IGNORECASE):
                    return alias
            for partial, target in self.doc_knowledge.partial_references.items():
                if target == comp_name and re.search(rf'\b{re.escape(partial)}\b', sentence_text, re.IGNORECASE):
                    return partial
        # Fallback: component name (even if not found — validation will handle)
        return comp_name

    def _run_entity_pipeline(self, sentences, components, name_to_id, sent_map):
        """Dual-pass entity extraction with consensus, then 3-step validation."""
        candidates = self._extract_entities_enriched(sentences, components, name_to_id, sent_map)
        print(f"    Entity extraction: {len(candidates)} candidates")

        # Validation
        validated = self._validate_intersect(candidates, components, sent_map)
        print(f"    Validation: {len(validated)} / {len(candidates)}")
        return validated

    def _run_coreference(self, sentences, components, name_to_id, sent_map):
        """Unified coreference: cases-in-context (Variant E).

        Per-case presentation with +-5 bidirectional context window.
        """
        pronoun_count = sum(1 for s in sentences if self.PRONOUN_PATTERN.search(s.text))
        print(f"    Coreference: cases-in-context ({pronoun_count} pronoun sents / {len(sentences)} total)")
        return self._coref_cases_in_context(sentences, components, name_to_id, sent_map)

    def _run_single_extraction_pass(self, sentences, comp_names, comp_lower, mappings,
                                     name_to_id, sent_map, pass_label=""):
        """Run one pass of entity extraction over all batches. Returns dict of (snum, cid) -> CandidateLink."""
        batch_size = 50
        candidates = {}

        for batch_start in range(0, len(sentences), batch_size):
            batch = sentences[batch_start:batch_start + batch_size]

            if len(sentences) > batch_size:
                print(f"    {pass_label}Entity batch {batch_start//batch_size + 1}: "
                      f"S{batch[0].number}-S{batch[-1].number} ({len(batch)} sents)")

            prompt = f"""Extract ALL references to software architecture components from this document.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings[:20])}' if mappings else ''}

{ENTITY_EXTRACTION_RULES}

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence", "match_type": "exact|synonym|partial|functional"}}]}}
JSON only:"""

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
                snum = self._parse_snum(ref.get("sentence"))
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
                                               matched, source="entity",
                                               match_type=ref.get("match_type", "exact"))

        return candidates

    def _extract_entities_enriched(self, sentences, components, name_to_id, sent_map):
        """Dual-pass extraction consensus for variance reduction.

        Runs entity extraction twice independently, keeps only candidates
        found in BOTH passes (extraction consensus).
        """
        comp_names = self._get_comp_names(components)
        comp_lower = {n.lower() for n in comp_names}

        mappings = []
        if self.doc_knowledge:
            mappings.extend([f"{a}={c}" for a, c in self.doc_knowledge.abbreviations.items()])
            mappings.extend([f"{s}={c}" for s, c in self.doc_knowledge.synonyms.items()])
            mappings.extend([f"{p}={c}" for p, c in self.doc_knowledge.partial_references.items()])

        # Pass 1
        print("    Extraction pass A:")
        pass1 = self._run_single_extraction_pass(
            sentences, comp_names, comp_lower, mappings, name_to_id, sent_map, pass_label="[P1] ")

        # Pass 2
        print("    Extraction pass B:")
        pass2 = self._run_single_extraction_pass(
            sentences, comp_names, comp_lower, mappings, name_to_id, sent_map, pass_label="[P2] ")

        # Intersection: keep only candidates found in BOTH passes
        intersected = {key: pass1[key] for key in pass1 if key in pass2}

        print(f"    Extraction consensus: Pass1={len(pass1)}, Pass2={len(pass2)}, "
              f"Intersect={len(intersected)} (dropped {len(pass1) + len(pass2) - 2*len(intersected)} unique-to-one-pass)")

        return list(intersected.values())

    def _validate_intersect(self, candidates, components, sent_map):
        """3-step LLM validation with evidence-adapted voting threshold.

        Step 1 — Generic pre-filter: ambiguous component names appearing
            only in lowercase are classified by LLM as component reference
            vs generic English word. Generic uses are removed.
        Step 2 — Validation pass A (actor-role focus): LLM checks whether
            the component performs an action or is being described.
        Step 3 — Validation pass B (direct-reference focus): LLM checks
            whether the text refers to the specific architectural component.

        Voting threshold adapted to evidence type (empirically motivated):
        - Alias-backed matches: union (either pass approves)
        - Exact-name matches: intersection (both passes must approve)
        """
        if not candidates:
            return []

        comp_names = self._get_comp_names(components)

        # Pre-check: LLM-based contextual generic mention detection
        generic_candidates = {}  # comp_name -> [candidate]
        non_generic = []
        for c in candidates:
            sent = sent_map.get(c.sentence_number)
            if not sent:
                non_generic.append(c)
                continue
            comp_lower = c.component_name.lower()
            has_exact_case = self._has_standalone_mention(c.component_name, sent.text)
            has_lowercase = (not has_exact_case and
                             re.search(rf'\b{re.escape(comp_lower)}\b', sent.text))
            if not has_lowercase and self.doc_knowledge:
                for partial, target in self.doc_knowledge.partial_references.items():
                    if target == c.component_name:
                        partial_lower = partial.lower()
                        if (re.search(rf'\b{re.escape(partial_lower)}\b', sent.text.lower())
                                and not re.search(rf'\b{re.escape(partial)}\b', sent.text)):
                            has_lowercase = True
                            break
            if has_lowercase and self._is_ambiguous_name_component(c.component_name):
                generic_candidates.setdefault(c.component_name, []).append(c)
            else:
                non_generic.append(c)

        # For each ambiguous component with lowercase-only mentions, ask LLM
        remaining = list(non_generic)
        for comp_name, cands in generic_candidates.items():
            anchor_lines = []
            for s in sent_map.values():
                if self._has_standalone_mention(comp_name, s.text):
                    anchor_lines.append(f"  S{s.number}: {s.text}")
                    if len(anchor_lines) >= 5:
                        break

            case_lines = []
            for i, c in enumerate(cands):
                s = sent_map.get(c.sentence_number)
                prev = sent_map.get(c.sentence_number - 1)
                prev_text = f" [prev: {prev.text[:60]}]" if prev else ""
                case_lines.append(f"  Case {i+1} (S{c.sentence_number}): {s.text}{prev_text}")

            anchor_section = ""
            if anchor_lines:
                anchor_section = (
                    f'FULL-NAME REFERENCES (these definitely refer to the {comp_name} component):\n'
                    + '\n'.join(anchor_lines) + '\n\n'
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

            data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
            if not data:
                remaining.extend(cands)  # On failure, keep all (safe default)
                continue

            results_map = {}
            for r in data.get("results", []):
                idx = r.get("case", 0) - 1
                results_map[idx] = r

            for i, c in enumerate(cands):
                result = results_map.get(i, {})
                usage = result.get("usage", "component")
                if usage == "generic":
                    reason = result.get("reason", "")
                    print(f"    LLM generic reject: S{c.sentence_number} -> {c.component_name} ({reason})")
                else:
                    remaining.append(c)

        # Build alias lookup for context
        alias_map = {}
        for c in components:
            aliases = {}
            if self.doc_knowledge:
                for a, cn in self.doc_knowledge.abbreviations.items():
                    if cn == c.name:
                        aliases[a] = "abbreviation"
                for s, cn in self.doc_knowledge.synonyms.items():
                    if cn == c.name:
                        aliases[s] = "synonym"
                for p, cn in self.doc_knowledge.partial_references.items():
                    if cn == c.name:
                        aliases[p] = "partial reference"
            alias_map[c.name] = aliases

        # ALL candidates go through 2-pass validation with alias context.
        # Alias cases use UNION (either pass approves), others use INTERSECTION.
        print(f"    LLM 2-pass validation: {len(remaining)} candidates")
        twopass_approved = []
        for batch_start in range(0, len(remaining), 25):
            batch = remaining[batch_start:batch_start + 25]
            cases = []
            has_alias = []  # track which candidates have alias hints
            for i, c in enumerate(batch):
                prev = sent_map.get(c.sentence_number - 1)
                p = f"[prev: {prev.text[:60]}] " if prev else ""
                # Add alias context when matched text is not the exact component name
                alias_hint = ""
                matched_lower = c.matched_text.lower() if c.matched_text else ""
                if matched_lower and matched_lower != c.component_name.lower():
                    aliases = alias_map.get(c.component_name, {})
                    for alias, atype in aliases.items():
                        if alias.lower() in matched_lower or matched_lower in alias.lower():
                            alias_hint = f'\n  [KNOWN ALIAS: "{alias}" is a known {atype} for "{c.component_name}"]'
                            break
                has_alias.append(bool(alias_hint))
                cases.append(f'Case {i+1}: "{c.matched_text}" -> {c.component_name}{alias_hint}\n  {p}"{c.sentence_text}"')

            r1 = self._qual_validation_pass(comp_names, cases,
                "Focus on ACTOR role: is the component performing an action or being described?")
            r2 = self._qual_validation_pass(comp_names, cases,
                "Focus on DIRECT reference: does the text refer to the SPECIFIC architectural component, not a generic concept?")

            for i, c in enumerate(batch):
                p1 = r1.get(i, False)
                p2 = r2.get(i, False)
                # Union for alias cases (either pass), intersection for exact matches
                approved = (p1 or p2) if has_alias[i] else (p1 and p2)
                if approved:
                    twopass_approved.append(c)

        return twopass_approved

    def _qual_validation_pass(self, comp_names, cases, focus):
        """Single validation pass (Step 2 or Step 3 of 3-step validation)."""
        # Check if any cases have alias hints — if so, add alias-aware rule
        has_alias = any("[KNOWN ALIAS:" in c for c in cases)
        alias_rule = ""
        if has_alias:
            alias_rule = ("\n- When a KNOWN ALIAS is indicated, the word IS a reference to that component "
                          "unless the sentence clearly uses it in an unrelated sense")

        prompt = f"""Validate component references in a software architecture document. {focus}

COMPONENTS: {', '.join(comp_names)}

{VALIDATION_RULES}{alias_rule}

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

    def _coref_cases_in_context(self, sentences, components, name_to_id, sent_map):
        """Unified coreference: per-case presentation with +-5 bidirectional context.

        Cross-model Pareto winner (0 FP on both Claude and GPT-5.2).
        No complexity gate needed.
        """
        comp_names = self._get_comp_names(components)
        all_coref = []
        pronoun_sents = [s for s in sentences if self.PRONOUN_PATTERN.search(s.text)]

        for batch_start in range(0, len(pronoun_sents), 10):
            batch = pronoun_sents[batch_start:batch_start + 10]
            cases = []
            for sent in batch:
                context = []
                for i in range(max(1, sent.number - 5), sent.number + 6):
                    s = sent_map.get(i)
                    if s:
                        marker = ">>>" if s.number == sent.number else "   "
                        context.append(f"{marker} S{s.number}: {s.text}")
                cases.append({"sent": sent, "context": context})

            prompt = f"""Resolve pronoun references to architecture components.

COMPONENTS: {', '.join(comp_names)}

"""
            for i, case in enumerate(cases):
                prompt += f"--- Case {i+1}: S{case['sent'].number} ---\n"
                prompt += "CONTEXT:\n" + "\n".join(case["context"]) + "\n"
                prompt += f"TARGET: S{case['sent'].number} (marked with >>>)\n\n"

            prompt += f"""{COREF_RULES}

Return JSON:
{{"resolutions": [{{"case": 1, "sentence": N_INTEGER, "pronoun": "it", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

Only include resolutions you are CERTAIN about. JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=300))
            if not data:
                continue

            for res in data.get("resolutions", []):
                comp = res.get("component")
                snum = self._parse_snum(res.get("sentence"))
                if snum is None or not comp or comp not in name_to_id:
                    continue

                ant_snum = self._parse_snum(res.get("antecedent_sentence"))
                if ant_snum is None:
                    print(f"    Coref skip (no antecedent): S{snum} -> {comp}")
                    continue

                ant_sent = sent_map.get(ant_snum)
                if not ant_sent:
                    continue
                if not (self._has_standalone_mention(comp, ant_sent.text) or
                        self._has_alias_mention(comp, ant_sent.text)):
                    continue
                if abs(snum - ant_snum) > 3:
                    continue

                all_coref.append(SadSamLink(snum, name_to_id[comp], comp, source="coreference"))

        return all_coref

    def _inject_partial_candidates(self, sentences, components, name_to_id,
                                    sent_map, seed_set, validated_set, coref_set):
        """Find partial-reference matches and return as CandidateLinks for validation."""
        if not self.doc_knowledge or not self.doc_knowledge.partial_references:
            return []

        existing = seed_set | validated_set | coref_set
        candidates = []

        for partial, comp_name in self.doc_knowledge.partial_references.items():
            if comp_name not in name_to_id:
                continue
            comp_id = name_to_id[comp_name]
            for sent in sentences:
                key = (sent.number, comp_id)
                if key in existing:
                    continue
                if self._has_clean_mention(partial, sent.text):
                    candidates.append(CandidateLink(
                        sent.number, sent.text, comp_name, comp_id,
                        partial, source="partial_inject", match_type="partial",
                    ))
                    existing.add(key)

        return candidates

    # ═══════════════════════════════════════════════════════════════════════
    # Shared Helpers
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_snum(val) -> Optional[int]:
        """Parse sentence number from LLM output (handles 'S42', '42', 42)."""
        if val is None:
            return None
        if isinstance(val, str):
            val = val.lstrip("S")
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    def _has_clean_mention(self, term, text):
        """Check if term appears cleanly (not in dotted path or hyphenated compound)."""
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

    def _has_standalone_mention(self, comp_name, text):
        """Check for non-generic, clean standalone mention of component name."""
        if not comp_name:
            return False
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

    def _has_alias_mention(self, comp_name, sentence_text):
        """Check if any known synonym or partial reference appears in the text."""
        if not self.doc_knowledge:
            return False
        text_lower = sentence_text.lower()
        for syn, target in self.doc_knowledge.synonyms.items():
            if target == comp_name:
                if re.search(rf'\b{re.escape(syn.lower())}\b', text_lower):
                    return True
        for partial, target in self.doc_knowledge.partial_references.items():
            if target == comp_name:
                if re.search(rf'\b{re.escape(partial.lower())}\b', text_lower):
                    return True
        return False

    def _is_ambiguous_name_component(self, comp_name):
        """True if single-word, non-CamelCase, non-uppercase, classified ambiguous."""
        if self._is_structurally_unambiguous(comp_name):
            return False
        if not self.model_knowledge or not self.model_knowledge.ambiguous_names:
            return False
        return comp_name in self.model_knowledge.ambiguous_names

    def _get_comp_names(self, components) -> list[str]:
        """Get all component names."""
        return [c.name for c in components]


    # ═══════════════════════════════════════════════════════════════════════
    # Checkpoint & Logging
    # ═══════════════════════════════════════════════════════════════════════

    def _checkpoint_dir(self, text_path):
        cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
        ds = os.path.splitext(os.path.basename(text_path))[0]
        d = os.path.join(cache_dir, "s_linker11", ds)
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
        path = os.path.join(log_dir, f"s_linker11_{ds}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w") as f:
            json.dump(self._phase_log, f, indent=2, default=str)
        print(f"  Phase log saved: {path}")
