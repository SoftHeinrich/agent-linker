"""S-Linker15b — v2.6.1 alias-recovery variant of s_linker15.

Replaces the full entity pipeline (2-pass LLM scan × batches + 2-pass
validation ≈ 8–10 LLM calls) with a targeted alias-recovery pass that scans
only for sentences containing global-scope alias mentions and disambiguates
them through the existing _run_seed_validation logic. For datasets with no
aliases (MS, JAB) the pass is a no-op; for alias-rich datasets (BBB, TM) it
captures alias-based TPs at ~1–2 LLM calls instead of 8–10, eliminating the
entity pipeline FP surface (5 FP on TM at 50% precision) while preserving the
7 BBB alias TPs that motivate the pipeline's existence.

All other pipeline logic is identical to s_linker15 (standalone, no
inheritance, inlined axiom prompts, ILinker4 empty-seed extractor).
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
    SadSamLink,
    ModelKnowledge, DocumentKnowledge,
)
from llm_sad_sam.core.document_loader_v2 import (
    load_sentences, build_sent_map,
)
from llm_sad_sam.linkers.experimental.ilinker4 import ILinker4
from llm_sad_sam.linkers.experimental.helper_v3 import (
    has_standalone_mention,
    build_component_profile,
    parse_snum,
    get_comp_names,
)
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient, LLMBackend


# ─────────────────────────────────────────────────────────────────────────────
# Inlined axiom prompts (v2.6.1 — standalone, no training/bank)
#
# Copied verbatim from s_linker15 (B-variant + three v2.6.1 FP fixes).
# Inlined intentionally (standalone-file preference).
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

COREF_RULES = """For each case, decide whether a pronoun or role-referential noun phrase in the target sentence refers back to a component named or aliased earlier in the context. Resolve when: (a) the component's name or a known alias appears in the surrounding context sentences, or (b) only one component has been introduced in the immediately preceding sentences — treat it as the section-established topic and resolve role-referential phrases ("it", "the module", "the service", "the component", "the system") to that topic even without a direct name repetition. Avoid resolving when two or more equally plausible antecedents exist. Known aliases include the terminal word(s) of a multi-word name, documented abbreviations, and alternate forms used in the document. When the antecedent sentence uses a known alias rather than the full canonical name, set antecedent_via_alias=true."""

# Cause C fix: functional/process-alias removal check appended before the tie-breaker.
SEED_DISAMBIGUATION_RULES = """For each sentence, decide whether the matched name refers to the architectural component (COMPONENT) or carries a different meaning (OTHER: code identifier, technique sharing the name, sub-entity of a larger name, or ordinary English vocabulary). A sentence is COMPONENT when the matched name refers to the component — whether through behavior, interaction, role description, identity statement, or any architecturally meaningful mention. A sentence is OTHER only when the matched name is bare vocabulary, a code path fragment, or refers to a different entity that merely shares the name. When the matched alias is a functional or process description (a phrase describing what the component does), apply an additional check: if removing the alias from the sentence still leaves an accurate description of a process step or activity, classify as OTHER — the sentence describes the activity, not the component; classify as COMPONENT only when the sentence clearly treats the alias as the name of a specific architectural unit. When uncertain, choose COMPONENT."""

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


@dataclass(frozen=True)
class AliasEntry:
    component: str
    scope: str   # "global" | "local"


# ─────────────────────────────────────────────────────────────────────────────
# Main linker class
# ─────────────────────────────────────────────────────────────────────────────

class SLinker15b:
    """v2.6.1 alias-recovery variant — entity pipeline replaced by targeted alias scan.

    Identical to s_linker15 except Tier 2 replaces _run_entity_pipeline with
    _run_alias_recovery: regex-scan all sentences for global-scope alias
    mentions, then reuse _run_seed_validation for per-component LLM
    disambiguation. For components with no global aliases the pass is a no-op.

    experimental=True — research-grade; not canonical.
    canonical=False   — s_linker13_min remains canonical=True.
    """

    _VARIANT_NAME = "s_linker15b"

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
        self._COREF_RULES = COREF_RULES
        self._SEED_DISAMBIGUATION_RULES = SEED_DISAMBIGUATION_RULES
        self._ALIAS_SCOPE_RULES = ALIAS_SCOPE_RULES
        self._ANTECEDENT_ALIAS_RULES = ANTECEDENT_ALIAS_RULES

        # ILinker4 seed extractor with EMPTY seed rules: pure axiom seed, no bank.
        self._ilinker4 = ILinker4(
            llm=self.llm,
            seed_extraction_rules="",
            seed_actor_rules="",
        )

        print(f"SLinker15b (v2.6.1 alias-recovery, ILinker4, experimental=True)")
        print(f"  Backend: {self.llm.describe_backend()}")

    # ═══════════════════════════════════════════════════════════════════════
    # DAG Infrastructure (verbatim from s_linker13_clean_v3)
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

        print("\n[Tier 2] Link Recovery (parallel)")
        rec = self._run_parallel({
            "seed_val": lambda: self._run_seed_validation(raw_seed_links, components, sent_map),
            "alias_rec": lambda: self._run_alias_recovery(sentences, components, name_to_id, sent_map),
            "coref": lambda: self._run_coreference(sentences, components, name_to_id, sent_map),
        })

        seed_links = rec["seed_val"]
        alias_links = rec["alias_rec"]
        coref_links = rec["coref"]
        print(f"  Seed validated: {len(seed_links)} / {len(raw_seed_links)}")
        print(f"  Alias recovery: {len(alias_links)} links")
        print(f"  Coreference: {len(coref_links)} links")

        self._save_phase(text_path, "layer2", {
            "seed_links": seed_links,
            "alias_links": alias_links,
            "coref_links": coref_links,
        })

        print("\n[Tier 3] Link Consolidation")
        all_links = seed_links + alias_links + coref_links
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
    # Tier 1 — Knowledge Acquisition (uses self._SLOT prompts)
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

    def _run_seed(self, sentences, components):
        return self._ilinker4.extract(sentences, components)

    # ═══════════════════════════════════════════════════════════════════════
    # Tier 2 — Link Recovery
    # ═══════════════════════════════════════════════════════════════════════

    def _run_seed_validation(self, raw_seed_links, components, sent_map):
        if not raw_seed_links:
            return []

        by_comp: dict[str, list[SadSamLink]] = {}
        for sl in raw_seed_links:
            by_comp.setdefault(sl.component_name, []).append(sl)

        verified = []

        for comp_name, seeds in sorted(by_comp.items()):
            seed_snums = {sl.sentence_number for sl in seeds}
            profile = build_component_profile(comp_name, self.model_knowledge, self.doc_knowledge)

            anchor_lines = []
            for s in sorted(sent_map.values(), key=lambda x: x.number):
                if s.number in seed_snums:
                    continue
                if has_standalone_mention(comp_name, s.text):
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

            prompt = f"""REFERENCE DISAMBIGUATION for component "{comp_name}"

COMPONENT PROFILE:
{profile}

{anchor_section}CASES TO VERIFY:
{chr(10).join(case_lines)}

{self._SEED_DISAMBIGUATION_RULES}

Return JSON:
{{"disambiguations": [{{"case": 1, "meaning": "component", "reason": "brief"}}]}}
JSON only:"""

            data = None
            for attempt in range(2):
                data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
                if data and data.get("disambiguations"):
                    break
                if attempt == 0:
                    print(f"    [{comp_name}] Empty response, retrying...")
            if not data:
                verified.extend(valid_seeds)
                continue

            results: dict[int, dict] = {}
            for d in data.get("disambiguations", []):
                idx = d.get("case", 0) - 1
                results[idx] = d

            approved = 0
            for i, sl in enumerate(valid_seeds):
                r = results.get(i, {})
                meaning = (r.get("meaning", "component") or "component").lower().strip()
                if meaning == "other":
                    reason = r.get("reason", "")
                    print(f"    Seed disambig reject: S{sl.sentence_number} -> {comp_name} ({reason})")
                else:
                    verified.append(sl)
                    approved += 1

            print(f"    [{comp_name}] {approved}/{len(valid_seeds)} seeds kept")

        return [SadSamLink(s.sentence_number, s.component_id, s.component_name, source="seed")
                for s in verified]

    def _run_alias_recovery(self, sentences, components, name_to_id, sent_map):
        """Alias-targeted link recovery: scan for global-scope alias mentions only.

        Replaces full entity pipeline. Only runs for components with global aliases
        in doc_knowledge. Reuses _run_seed_validation for disambiguation.
        """
        if not self.doc_knowledge or not self.doc_knowledge.aliases:
            print("    Alias recovery: no aliases found, skipping")
            return []

        # Group global aliases by component
        comp_aliases: dict[str, list[str]] = {}
        for alias, entry in self.doc_knowledge.aliases.items():
            if entry.scope == "global":
                comp_aliases.setdefault(entry.component, []).append(alias)

        if not comp_aliases:
            print("    Alias recovery: no global aliases, skipping")
            return []

        print(f"    Alias recovery: {len(comp_aliases)} components with global aliases")
        candidates: list[SadSamLink] = []

        for comp_name, aliases in sorted(comp_aliases.items()):
            comp_id = name_to_id.get(comp_name)
            if not comp_id:
                continue
            alias_pats = [re.compile(rf'\b{re.escape(a)}\b', re.IGNORECASE) for a in aliases]
            for sent in sentences:
                for pat in alias_pats:
                    if pat.search(sent.text):
                        candidates.append(SadSamLink(
                            sent.number, comp_id, comp_name, source="alias_recovery"
                        ))
                        break  # one alias match per sentence per component

        if not candidates:
            print("    Alias recovery: 0 candidates found")
            return []

        print(f"    Alias recovery: {len(candidates)} candidates before disambiguation")
        # Reuse seed validation logic (per-component LLM disambiguation)
        validated = self._run_seed_validation(candidates, components, sent_map)
        # Re-tag source
        result = [SadSamLink(l.sentence_number, l.component_id, l.component_name, source="alias_recovery")
                  for l in validated]
        print(f"    Alias recovery: {len(result)} kept after disambiguation")
        return result

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

    def _run_coreference(self, sentences, components, name_to_id, sent_map):
        anaphoric_count = sum(1 for s in sentences if self.PRONOUN_PATTERN.search(s.text))
        print(f"    Coreference: cases-in-context ({anaphoric_count} anaphoric sents / {len(sentences)} total)")
        return self._coref_cases_in_context(sentences, components, name_to_id, sent_map)

    def _classify_specific_terminals(self, components) -> set[str]:
        """LLM-driven: which multi-word component terminal words are specific enough for role-ref coref.

        Avoids generic architectural nouns (service, manager, handler...) that match
        every sentence. Cached per component-list within a linker instance (called once per run).
        """
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

        # Build role_ref_pat: matches "the <terminal_word>" for multi-word component names.
        # Uses LLM-driven specificity check to exclude generic architectural nouns
        # (e.g. "service", "manager") that would match unrelated sentences.
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
    # Checkpoint & Logging (verbatim from s_linker13_clean_v3)
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
