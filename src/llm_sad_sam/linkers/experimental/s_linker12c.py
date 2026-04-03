"""S-Linker12c: LLM-driven SAD-SAM traceability with structural guardrails.

3-tier DAG pipeline for recovering trace links between software architecture
documentation (SAD) and architecture models (SAM).

Tier 1 — Knowledge Acquisition (parallel):
  Model analysis     : classify component names as architectural vs ambiguous
  Document knowledge : discover aliases (abbreviations, synonyms) via LLM + judge
  Seed extraction    : baseline LLM extraction via ILinker3

Tier 2 — Link Recovery (parallel):
  Seed validation    : per-component disambiguation (single-pass LLM)
  Entity pipeline    : dual-pass extraction consensus + evidence-aware validation
  Coreference        : pronoun resolution with ±5-sentence context window

Tier 3 — Consolidation:
  Priority-ordered deduplication (seed > entity > coref)

Design: LLM-driven decisions with lightweight structural guardrails (CamelCase
detection, dotted-path exclusion, pronoun pattern, alias strength classification).
Alias stratification prevents weak single-word forms from polluting extraction
and coreference globally.  Intersection voting ensures conservative approval.
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
    Sentence, load_sentences, build_sent_map,
)
from llm_sad_sam.linkers.experimental.ilinker3 import ILinker3
from llm_sad_sam.linkers.experimental.prompts_v2 import (
    AMBIGUITY_FEW_SHOT, AMBIGUITY_RULES,
    DOC_KNOWLEDGE_JUDGE_EXAMPLES, DOC_KNOWLEDGE_JUDGE_RULES,
    DOC_KNOWLEDGE_EXTRACTION_RULES,
    ENTITY_EXTRACTION_RULES, VALIDATION_RULES, COREF_RULES,
)
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient, LLMBackend


@dataclass
class EvidenceBundle:
    """Proposer evidence for a single candidate link.

    Attached to each entity candidate before validation so the reviewer
    sees the full evidence trail rather than the bare sentence + matched_text.
    """
    source: str                     # "entity" or "seed"
    matched_span: str               # matched text in the sentence
    mention_type: str               # "proper case, standalone" / "lowercase mention" / "via known alias X" / ...
    preceding_text: str             # text of sentence N-1 (or "")
    anchor_sentences: list[str]     # "S{n}: {text}" for confirmed full-name mentions
    is_ambiguous: bool              # True if component is in ambiguous_names
    extraction_rationale: str       # e.g. "dual-pass extraction consensus"


class SLinker12c:
    """LLM-driven SAD-SAM traceability with structural guardrails."""

    PRONOUN_PATTERN = re.compile(
        r'\b(it|they|this|these|that|those|its|their)\b',
        re.IGNORECASE
    )

    SEED_DISAMBIGUATION_RULES = """REFERENCE DISAMBIGUATION — determine what the name means in each sentence.

COMPONENT (approve): The sentence discusses this architectural component —
it performs actions, provides services, is described, configured, listed,
or referenced by name in any grammatical role.

OTHER (reject): The name clearly carries a different meaning:
- Code-level notation: the name appears inside a package path, qualified
  identifier, or a sentence that enumerates code-level identifiers
- Technique or methodology: the sentence describes an algorithm, pattern,
  or approach that shares the component's name — not what the component
  does as an architectural participant
- Embedded sub-entity: the name appears only as part of a longer proper
  name that denotes a different, more specific entity
- Different entity: the sentence refers to a similarly-named but distinct
  thing (the name partially overlaps but the full reference is different)
- Generic English: the word is used with its ordinary dictionary meaning

When uncertain, choose COMPONENT — these candidates passed independent extraction."""

    def __init__(
        self,
        backend: LLMBackend | None = None,
        model: str | None = None,
        checkpoint_fallback: LLMBackend | str | None = None,
        checkpoint_fallback_model: str | None = None,
    ):
        os.environ.setdefault("CLAUDE_MODEL", "sonnet")
        os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-5.2")
        self.llm = LLMClient(
            backend=backend or LLMBackend.CLAUDE,
            model=model,
            checkpoint_fallback=checkpoint_fallback,
            checkpoint_fallback_model=checkpoint_fallback_model,
        )
        self.model_knowledge: ModelKnowledge | None = None
        self.doc_knowledge: DocumentKnowledge | None = None
        self._phase_log = []
        self._ilinker3 = ILinker3(llm=self.llm)
        self._current_text_path: str | None = None
        print("SLinker12c (12b - Tier 2, intersection voting)")
        print(f"  Backend: {self.llm.describe_backend()}")

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
        """Recover trace links between SAD and SAM via 3-layer pipeline.

        Args:
            text_path: Path to documentation text file (one sentence per line).
            model_path: Path to PCM .repository file.

        Returns:
            list[SadSamLink]: Recovered trace links.
        """
        self._phase_log = []
        self._current_text_path = text_path
        t0 = time.time()

        # Load raw data
        components = parse_pcm_repository(model_path)
        sentences = load_sentences(text_path)
        name_to_id = {c.name: c.id for c in components}
        sent_map = build_sent_map(sentences)

        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        # ═══ TIER 1: Knowledge Acquisition ═══
        print("\n[Tier 1] Knowledge Acquisition (parallel)")
        acq = self._run_parallel({
            "model": lambda: self._analyze_model(components),
            "doc_knowledge": lambda: self._learn_document_knowledge_enriched(sentences, components),
            "seed": lambda: self._run_seed(sentences, components),
        })

        self.model_knowledge = acq["model"]
        self.doc_knowledge = acq["doc_knowledge"]
        raw_seed_links = acq["seed"]

        ambig = self.model_knowledge.ambiguous_names
        print(f"  Model: {len(ambig)} ambiguous (of {len(components)} components)")
        print(f"  Doc knowledge: {len(self.doc_knowledge.aliases)} aliases")
        print(f"  Seed: {len(raw_seed_links)} raw links")

        self._log("layer1", {"sents": len(sentences), "comps": len(components)},
                  {"ambig": len(ambig), "seed": len(raw_seed_links),
                   "aliases": len(self.doc_knowledge.aliases)})

        self._save_phase(text_path, "layer1", {
            "model_knowledge": self.model_knowledge,
            "doc_knowledge": self.doc_knowledge,
            "raw_seed_links": raw_seed_links,
        })

        # ═══ TIER 2: Link Recovery (three strategies parallel) ═══
        # Seed validation, entity extraction+validation, and coreference
        # all depend on Tier 1 knowledge but are independent of each other.
        print("\n[Tier 2] Link Recovery (parallel)")
        rec = self._run_parallel({
            "seed_val": lambda: self._run_seed_validation(
                raw_seed_links, components, sent_map),
            "entity": lambda: self._run_entity_pipeline(
                sentences, components, name_to_id, sent_map),
            "coref": lambda: self._run_coreference(
                sentences, components, name_to_id, sent_map),
        })

        seed_links = rec["seed_val"]
        validated = rec["entity"]
        coref_links = rec["coref"]
        print(f"  Seed validated: {len(seed_links)} / {len(raw_seed_links)}")
        print(f"  Entity pipeline: {len(validated)} validated")
        print(f"  Coreference: {len(coref_links)} links")

        self._save_phase(text_path, "layer2", {
            "seed_links": seed_links,
            "validated": validated,
            "coref_links": coref_links,
        })

        # ═══ TIER 3: Link Consolidation (dedup) ═══
        print("\n[Tier 3] Link Consolidation")

        # Deduplication (first-seen wins — order: seed, entity, coref)
        entity_links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name, source=c.source)
            for c in validated
        ]
        all_links = seed_links + entity_links + coref_links
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
    # Tier 1 — Knowledge Acquisition
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

    @staticmethod
    def _is_strong_alias(term: str) -> bool:
        """True for aliases that are safe to use globally in extraction and coref.

        Strong: multi-word forms, CamelCase, all-caps, or starts with capital.
        Weak: single all-lowercase word — collides with ordinary English
        vocabulary and causes cross-sentence leakage when broadcast globally.
        """
        # Multi-word or hyphenated
        if ' ' in term or '-' in term:
            return True
        # CamelCase (internal capital)
        if re.search(r'[a-z][A-Z]', term):
            return True
        # All-caps (e.g. "RPC", "UI", "API")
        if term.isupper() and len(term) >= 2:
            return True
        # Starts with capital (distinctive proper name, e.g. "Auth", "Broker")
        if term and term[0].isupper():
            return True
        # Single lowercase word — weak alias
        return False

    def _get_strong_alias_mappings(self) -> list[str]:
        """Return alias mappings restricted to structurally strong aliases only."""
        if not self.doc_knowledge:
            return []
        return [f"{a}={c}" for a, c in self.doc_knowledge.aliases.items()
                if self._is_strong_alias(a)]

    def _classify_components(self, names, knowledge):
        """Classify components using few-shot prompt + structural code guard."""
        prompt = f"""Classify these software architecture component names.

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

        for attempt in range(2):
            data = self.llm.extract_json(self.llm.query(prompt, timeout=100))
            if data:
                break
            if attempt == 0:
                print("    Ambiguity classification: empty response, retrying...")
        if data:
            valid = set(names)
            raw_ambiguous = set(data.get("ambiguous", [])) & valid
            knowledge.ambiguous_names = {
                n for n in raw_ambiguous
                if len(n.split()) == 1 and not self._is_structurally_unambiguous(n)
            }

    def _learn_document_knowledge_enriched(self, sentences, components):
        """Discover aliases (abbreviations and synonyms) via LLM + judge."""
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
  "synonyms": {{"specific_alternative_name": "FullComponent"}}
}}
JSON only:"""

        for attempt in range(2):
            data1 = self.llm.extract_json(self.llm.query(prompt1, timeout=300))
            if data1:
                break
            if attempt == 0:
                print("    Doc knowledge: empty response, retrying...")

        all_mappings = {}
        if data1:
            for short, full in data1.get("abbreviations", {}).items():
                if full in comp_names:
                    all_mappings[short] = full
            for syn, full in data1.get("synonyms", {}).items():
                if full in comp_names:
                    all_mappings[syn] = full

        if all_mappings:
            mapping_list = [f"'{k}' -> {v}" for k, v in list(all_mappings.items())[:25]]

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
                knowledge.aliases[term] = comp
                print(f"    Alias: {term} -> {comp}")

        return knowledge

    def _run_seed(self, sentences, components):
        """LLM-based seed extraction via ILinker3 (v2 stack, no file I/O).

        Uses pre-loaded sentences and components — no redundant file reads
        or dual data-stack loading.
        """
        return self._ilinker3.extract(sentences, components)

    # ═══════════════════════════════════════════════════════════════════════
    # Tier 2 — Link Recovery
    # ═══════════════════════════════════════════════════════════════════════

    def _run_seed_validation(self, raw_seed_links, components, sent_map):
        """Knowledge-aware seed reference disambiguation.

        Per-component LLM pass with component dossier (ambiguity classification,
        known aliases, anchor sentences) and match context.  Single pass with
        approve-biased framing: seeds carry prior baseline evidence, so we ask
        "is there reason to doubt?" rather than "prove this is valid."
        """
        if not raw_seed_links:
            return []

        # Group seeds by component
        by_comp: dict[str, list[SadSamLink]] = {}
        for sl in raw_seed_links:
            by_comp.setdefault(sl.component_name, []).append(sl)

        comp_names = self._get_comp_names(components)
        verified = []

        for comp_name, seeds in sorted(by_comp.items()):
            seed_snums = {sl.sentence_number for sl in seeds}

            # ── Component profile ──
            profile = self._build_component_profile(comp_name)

            # ── Anchor sentences (proper-case mentions NOT in seed set) ──
            anchor_lines = []
            for s in sorted(sent_map.values(), key=lambda x: x.number):
                if s.number in seed_snums:
                    continue
                if self._has_standalone_mention(comp_name, s.text):
                    anchor_lines.append(f'  S{s.number}: "{s.text}"')
                    if len(anchor_lines) >= 5:
                        break

            if anchor_lines:
                anchor_section = (
                    f'KNOWN REFERENCES (these definitely refer to "{comp_name}"):\n'
                    + "\n".join(anchor_lines) + "\n\n"
                )
            else:
                anchor_section = (
                    f'NOTE: No standalone proper-case references to "{comp_name}" found '
                    f"elsewhere in the document. This component may not be discussed "
                    f"architecturally — be extra careful to verify each case.\n\n"
                )

            # ── Build cases with match context ──
            case_lines = []
            valid_seeds = []
            for sl in seeds:
                sent = sent_map.get(sl.sentence_number)
                if not sent:
                    continue
                valid_seeds.append(sl)
                prev = sent_map.get(sl.sentence_number - 1)
                prev_text = f' [prev: "{prev.text[:80]}"]' if prev else ""
                match_ctx = self._classify_mention(comp_name, sent.text)
                case_lines.append(
                    f'  Case {len(valid_seeds)} (S{sl.sentence_number}): '
                    f'"{sent.text}"{prev_text}\n    Mention: {match_ctx}'
                )

            if not valid_seeds:
                continue

            # ── LLM call ──
            prompt = f"""REFERENCE DISAMBIGUATION for component "{comp_name}"

COMPONENT PROFILE:
{profile}

{anchor_section}CASES TO VERIFY:
{chr(10).join(case_lines)}

{self.SEED_DISAMBIGUATION_RULES}

Return JSON:
{{"disambiguations": [{{"case": 1, "meaning": "component", "reason": "brief"}}]}}
JSON only:"""

            for attempt in range(2):
                data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
                if data and data.get("disambiguations"):
                    break
                if attempt == 0:
                    print(f"    [{comp_name}] Empty response, retrying...")
            if not data:
                verified.extend(valid_seeds)  # Keep all on failure (approve-biased)
                continue

            results = {}
            for d in data.get("disambiguations", []):
                idx = d.get("case", 0) - 1
                results[idx] = d

            approved = 0
            for i, sl in enumerate(valid_seeds):
                r = results.get(i, {})
                meaning = (r.get("meaning", "component") or "component").lower().strip()
                if meaning == "other":
                    reason = r.get("reason", "")
                    print(f"    Seed disambig reject: S{sl.sentence_number} -> "
                          f"{comp_name} ({reason})")
                else:
                    verified.append(sl)
                    approved += 1

            print(f"    [{comp_name}] {approved}/{len(valid_seeds)} seeds kept")

        return [SadSamLink(s.sentence_number, s.component_id,
                           s.component_name, source="seed")
                for s in verified]

    def _build_component_profile(self, comp_name: str) -> str:
        """Build textual component profile for disambiguation prompt."""
        lines = [f"- Name: {comp_name}"]

        is_ambig = (self.model_knowledge
                    and comp_name in self.model_knowledge.ambiguous_names)
        if is_ambig:
            lines.append(f'- Classification: AMBIGUOUS — "{comp_name}" is a common English word')
        else:
            lines.append("- Classification: DISTINCTIVE — architecturally specific name")

        aliases = []
        if self.doc_knowledge:
            for a, target in self.doc_knowledge.aliases.items():
                if target == comp_name:
                    aliases.append(f'"{a}"')

        if aliases:
            lines.append(f"- Known aliases: {', '.join(aliases)}")
        else:
            lines.append("- Known aliases: none")
        return "\n".join(lines)

    def _classify_mention(self, comp_name: str, text: str) -> str:
        """Classify how the component name appears in the sentence.

        Returns a human-readable match description for the LLM prompt.
        """
        # Check exact proper-case standalone mention
        if self._has_standalone_mention(comp_name, text):
            return "proper case, standalone"

        # Check lowercase mention
        comp_lower = comp_name.lower()
        if re.search(rf'\b{re.escape(comp_lower)}\b', text):
            # In dotted path?
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

        # Check alias match
        if self.doc_knowledge:
            for alias, target in self.doc_knowledge.aliases.items():
                if target == comp_name and re.search(
                    rf'\b{re.escape(alias)}\b', text, re.IGNORECASE
                ):
                    return f'via known alias "{alias}"'

        return "indirect/unclear match"

    def _build_evidence_bundle(
        self,
        candidate,
        sent_map,
        rationale: str = "dual-pass extraction consensus",
    ) -> EvidenceBundle:
        """Build an evidence bundle for a candidate link.

        Collects all contextual evidence the proposer can offer: matched span,
        mention type, preceding sentence, up to 5 anchor sentences (other sentences
        that mention the component by full proper name), and ambiguity flag.
        """
        comp_name = candidate.component_name
        snum = candidate.sentence_number

        mention_type = self._classify_mention(comp_name, candidate.sentence_text)

        prev_sent = sent_map.get(snum - 1)
        preceding_text = prev_sent.text if prev_sent else ""

        anchors = []
        for s in sorted(sent_map.values(), key=lambda x: x.number):
            if s.number == snum:
                continue
            if self._has_standalone_mention(comp_name, s.text):
                anchors.append(f"S{s.number}: {s.text}")
                if len(anchors) >= 5:
                    break

        is_ambig = self._is_ambiguous_name_component(comp_name)

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
        """Format an evidence bundle as a compact multi-line block for prompts."""
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

    def _run_entity_pipeline(self, sentences, components, name_to_id, sent_map):
        """Dual-pass entity extraction with consensus, then evidence-aware validation."""
        candidates = self._extract_entities_enriched(sentences, components, name_to_id, sent_map)
        print(f"    Entity extraction: {len(candidates)} candidates")

        # Build evidence bundles so validation prompts can use them
        bundles = {
            (c.sentence_number, c.component_id): self._build_evidence_bundle(c, sent_map)
            for c in candidates
        }

        if self._current_text_path:
            self._save_phase(self._current_text_path, "entity_candidates", {
                "entity_candidates": candidates,
                "bundles": bundles,
            })

        validated, decisions = self._validate_with_evidence(
            candidates, bundles, components, sent_map)
        print(f"    Validation: {len(validated)} / {len(candidates)}")

        if self._current_text_path:
            self._save_phase(self._current_text_path, "entity_decisions", {
                "decisions": decisions,
            })

        return validated

    def _run_coreference(self, sentences, components, name_to_id, sent_map):
        """Unified coreference: cases-in-context (Variant E).

        Per-case presentation with +-5 bidirectional context window.
        """
        pronoun_count = sum(1 for s in sentences if self.PRONOUN_PATTERN.search(s.text))
        print(f"    Coreference: cases-in-context ({pronoun_count} pronoun sents / {len(sentences)} total)")
        return self._coref_cases_in_context(sentences, components, name_to_id, sent_map)

    def _run_single_extraction_pass(self, sentences, comp_names, mappings,
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
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence"}}]}}
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
                                               matched, source="entity")

        return candidates

    def _extract_entities_enriched(self, sentences, components, name_to_id, sent_map):
        """Dual-pass extraction consensus for variance reduction.

        Runs entity extraction twice independently, keeps only candidates
        found in BOTH passes (extraction consensus).
        """
        comp_names = self._get_comp_names(components)

        # Use only strong aliases in extraction prompts to prevent global leakage
        # from weak single-word forms (e.g., "server"→HTML5Server matching everywhere)
        mappings = self._get_strong_alias_mappings()

        # Two independent extraction passes in parallel for variance reduction
        print("    Extraction pass A + B (parallel):")
        results = self._run_parallel({
            "pass1": lambda: self._run_single_extraction_pass(
                sentences, comp_names, mappings, name_to_id, sent_map, pass_label="[P1] "),
            "pass2": lambda: self._run_single_extraction_pass(
                sentences, comp_names, mappings, name_to_id, sent_map, pass_label="[P2] "),
        })
        pass1 = results["pass1"]
        pass2 = results["pass2"]

        # Intersection: keep only candidates found in BOTH passes
        intersected = {key: pass1[key] for key in pass1 if key in pass2}

        print(f"    Extraction consensus: Pass1={len(pass1)}, Pass2={len(pass2)}, "
              f"Intersect={len(intersected)} (dropped {len(pass1) + len(pass2) - 2*len(intersected)} unique-to-one-pass)")

        return list(intersected.values())

    def _validate_with_evidence(self, candidates, bundles, components, sent_map):
        """3-step LLM validation with evidence bundles, intersection voting.

        A candidate link (sentence S, component C) is valid when two conditions hold:
          (1) Architectural participation: S names C as a participant in the described
              system behavior — C performs an operation, provides a service, or is
              introduced in the context of the architecture.
          (2) Referential specificity: the name in S identifies the specific component C,
              not a homonymous generic term that happens to share C's name.

        These map to the two validation passes:
          Step 2 — Participation pass: checks condition (1).
          Step 3 — Specificity pass: checks condition (2).
          Each pass receives the evidence bundle so it can weigh the full
          proposer trail (matched span, mention type, anchor sentences).

        Pre-pass (Step 1): ambiguous-named components that appear only in lowercase
        are first sent through a word-usage classifier, which removes generic uses
        before the two-pass validation runs.

        Intersection voting: both conditions must hold for approval.

        Returns (validated_list, decisions_dict).
        """
        if not candidates:
            return [], {}

        comp_names = self._get_comp_names(components)
        decisions = {}  # (snum, cid) -> {"approved": bool, "path": str}

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

            for attempt in range(2):
                data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
                if data and data.get("results"):
                    break
                if attempt == 0:
                    print(f"    Generic filter [{comp_name}]: empty response, retrying...")
            if not data:
                remaining.extend(cands)  # On failure, keep all (safe default)
                continue

            results_map = {}
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

        # 2-pass validation with evidence bundles, intersection voting.
        print(f"    LLM 2-pass validation (+evidence bundles): {len(remaining)} candidates")
        twopass_approved = []
        for batch_start in range(0, len(remaining), 25):
            batch = remaining[batch_start:batch_start + 25]
            cases = []
            for i, c in enumerate(batch):
                prev = sent_map.get(c.sentence_number - 1)
                p = f"[prev: {prev.text[:60]}] " if prev else ""

                # Attach evidence bundle so the validator sees the full proposer trail
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
                    "approved": approved,
                    "p1": p1,
                    "p2": p2,
                    "path": "twopass" if approved else "twopass_reject",
                }
                if approved:
                    twopass_approved.append(c)

        return twopass_approved, decisions

    def _run_validation_pass(self, comp_names, cases, focus):
        """Single validation pass (Step 2 or Step 3 of 3-step validation)."""
        prompt = f"""Validate component references in a software architecture document. {focus}

COMPONENTS: {', '.join(comp_names)}

{VALIDATION_RULES}

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true}}]}}
JSON only:"""

        for attempt in range(2):
            data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
            if data and data.get("validations"):
                break
            if attempt == 0:
                print(f"    Validation pass: empty response, retrying...")
        results = {}
        if data:
            for v in data.get("validations", []):
                idx = v.get("case", 0) - 1
                if 0 <= idx < len(cases):
                    val = v.get("approve", False)
                    results[idx] = val is True or (isinstance(val, str) and val.lower() == "true")
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
                        self._has_strong_alias_mention(comp, ant_sent.text)):
                    continue

                all_coref.append(SadSamLink(snum, name_to_id[comp], comp, source="coreference"))

        return all_coref

    # ═══════════════════════════════════════════════════════════════════════
    # Shared Helpers
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_snum(val) -> int | None:
        """Parse sentence number from LLM output (handles 'S42', 's42', '42', 42)."""
        if val is None:
            return None
        if isinstance(val, str):
            val = val.lstrip("Ss")
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _has_standalone_mention(comp_name, text):
        """Check for non-generic, clean standalone mention of component name."""
        if not comp_name:
            return False
        is_single = ' ' not in comp_name
        if is_single:
            if comp_name[0].islower():
                pattern = rf'\b{re.escape(comp_name)}\b'
            else:
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

    def _has_strong_alias_mention(self, comp_name, sentence_text):
        """Check if any STRONG alias for comp_name appears in sentence_text.

        Used in coref antecedent verification to avoid accepting ordinary-word
        matches as antecedent evidence.
        """
        if not self.doc_knowledge:
            return False
        text_lower = sentence_text.lower()
        for alias, target in self.doc_knowledge.aliases.items():
            if target == comp_name and self._is_strong_alias(alias):
                if re.search(rf'\b{re.escape(alias.lower())}\b', text_lower):
                    return True
        return False

    def _is_ambiguous_name_component(self, comp_name):
        """True if single-word, non-CamelCase, non-uppercase, classified ambiguous."""
        if self._is_structurally_unambiguous(comp_name):
            return False
        if not self.model_knowledge or not self.model_knowledge.ambiguous_names:
            return False
        return comp_name in self.model_knowledge.ambiguous_names

    @staticmethod
    def _get_comp_names(components) -> list[str]:
        """Get all component names."""
        return [c.name for c in components]


    # ═══════════════════════════════════════════════════════════════════════
    # Checkpoint & Logging
    # ═══════════════════════════════════════════════════════════════════════

    def _checkpoint_dir(self, text_path):
        cache_dir = os.environ.get("PHASE_CACHE_DIR", "./results/phase_cache")
        ds = os.path.splitext(os.path.basename(text_path))[0]
        d = os.path.join(cache_dir, "s_linker12c", ds)
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
        path = os.path.join(log_dir, f"s_linker12c_{ds}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w") as f:
            json.dump(self._phase_log, f, indent=2, default=str)
        print(f"  Phase log saved: {path}")
