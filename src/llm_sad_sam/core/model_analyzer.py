"""Architecture model analysis utilities."""

import re
from typing import Optional

from .data_types import ModelKnowledge
from ..pcm_parser import parse_pcm_repository, ArchitectureComponent
from ..llm_client import LLMClient


class ModelAnalyzer:
    """Analyze architecture model structure."""

    STRATEGY_TIGHT_PROMPT = "tight_prompt"
    STRATEGY_TWO_PASS = "two_pass"

    def __init__(self, llm_client: Optional[LLMClient] = None,
                 classify_strategy: str = "tight_prompt"):
        """Initialize analyzer.

        Args:
            llm_client: Optional LLM client for classification queries
            classify_strategy: Strategy for ambiguity classification.
                "tight_prompt" — single LLM call with restrictive prompt (default)
                "two_pass" — two LLM calls, intersect results
        """
        self.llm_client = llm_client
        self.classify_strategy = classify_strategy

    def load_components(self, model_path: str) -> list[ArchitectureComponent]:
        """Load components from PCM model file."""
        return parse_pcm_repository(model_path)

    def analyze(self, components: list[ArchitectureComponent]) -> ModelKnowledge:
        """Analyze model structure and build knowledge base."""
        names = [c.name for c in components]
        knowledge = ModelKnowledge()

        self._find_impl_patterns(names, knowledge)
        self._find_shared_vocabulary(names, knowledge)

        if self.llm_client:
            if self.classify_strategy == self.STRATEGY_TWO_PASS:
                self._classify_components_two_pass(names, knowledge)
            else:
                self._classify_components_tight_prompt(names, knowledge)

        return knowledge

    def _find_impl_patterns(self, names: list[str], knowledge: ModelKnowledge):
        """Find implementation/concrete variants of abstract components."""
        for name in names:
            for other in names:
                if other != name and len(other) >= 3 and other in name:
                    idx = name.find(other)
                    prefix = name[:idx]
                    suffix = name[idx + len(other):]

                    if prefix and len(prefix) >= 2:
                        knowledge.impl_indicators.append(prefix)
                        knowledge.impl_to_abstract[name] = other

                    if suffix and len(suffix) >= 2:
                        knowledge.impl_indicators.append(suffix)
                        knowledge.impl_to_abstract[name] = other

        knowledge.impl_indicators = list(set(knowledge.impl_indicators))

    def _find_shared_vocabulary(self, names: list[str], knowledge: ModelKnowledge):
        """Find words shared between multiple component names."""
        word_to_comps: dict[str, list[str]] = {}

        for name in names:
            words = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)', name)
            for word in words:
                w = word.lower()
                if len(w) >= 3:
                    if w not in word_to_comps:
                        word_to_comps[w] = []
                    word_to_comps[w].append(name)

        knowledge.shared_vocabulary = {
            word: list(set(comps))
            for word, comps in word_to_comps.items()
            if len(set(comps)) > 1
        }

    # ── Strategy A: Tight prompt (no code filter) ─────────────────────

    def _classify_components_tight_prompt(self, names: list[str], knowledge: ModelKnowledge):
        """Single LLM call with a restrictive prompt. No post-hoc code filter."""
        prompt = f"""Classify these software architecture component names.

NAMES: {', '.join(names)}

Return JSON:
{{
  "architectural": ["names that identify specific components"],
  "ambiguous": ["names that are too vague to identify a specific component"]
}}

CLASSIFICATION RULES:
1. A name is ARCHITECTURAL if it describes a specific responsibility or role in the system.
   Even single common English words are architectural if they name what the component DOES.
   Examples: "Scheduler" schedules, "Optimizer" optimizes, "Renderer" renders — these are architectural.
   Similarly, "Parser", "Dispatcher", "Validator", "Compiler" all name specific roles → architectural.

2. A name is AMBIGUOUS only if it is so vague that it could apply to ANY part of ANY system
   and tells you nothing about what the component does.
   Examples of truly ambiguous: "Util", "Misc", "Other", "Base", "Helper", "Core", "Main".
   These words describe no specific responsibility — they are organizational labels, not role names.

3. Multi-word names and CamelCase compounds are ALWAYS architectural — they are invented identifiers.

4. All-uppercase short names (DB, UI, API) are abbreviations → architectural.

5. When in doubt, classify as architectural. The bar for "ambiguous" is very high.

JSON only:"""

        response = self.llm_client.query(prompt, timeout=100)
        data = self.llm_client.extract_json(response)

        if data:
            valid_names = set(names)
            knowledge.architectural_names = set(data.get("architectural", [])) & valid_names
            raw_ambiguous = set(data.get("ambiguous", [])) & valid_names
            knowledge.ambiguous_names = {
                n for n in raw_ambiguous
                if len(n.split()) == 1
            }

    # ── Strategy B: Two-pass LLM intersect ────────────────────────────

    def _classify_components_two_pass(self, names: list[str], knowledge: ModelKnowledge):
        """Two LLM calls with different framings, intersect ambiguous results."""
        prompt_a = f"""Classify these component names from a software architecture model.

NAMES: {', '.join(names)}

Return JSON:
{{
  "architectural": ["names that refer to specific software components"],
  "ambiguous": ["names that are generic English words with no specific technical role"]
}}

Rules:
- Architectural: names that describe a specific responsibility (e.g., "Scheduler", "Parser", "Renderer")
- Ambiguous: names so generic they could mean anything (e.g., "Util", "Misc", "Helper", "Base")
- Multi-word names and CamelCase → always architectural
- When in doubt → architectural

JSON only:"""

        prompt_b = f"""You are reviewing component names from a software architecture.

NAMES: {', '.join(names)}

TASK: Which names could be confused with general English words when they appear in documentation?
A name is confusable ONLY if it is a single common word that people would use in ordinary
sentences without meaning the component (e.g., "other" in "other components", "base" in "base class").

Names that describe a specific function (scheduling, parsing, optimizing) are NOT confusable —
they clearly refer to the component when used in architecture documentation.

Return JSON:
{{
  "clear": ["names that clearly identify components"],
  "confusable": ["names that could be mistaken for ordinary English in documentation"]
}}

JSON only:"""

        data_a = self.llm_client.extract_json(self.llm_client.query(prompt_a, timeout=100))
        data_b = self.llm_client.extract_json(self.llm_client.query(prompt_b, timeout=100))

        valid_names = set(names)
        ambig_a = set()
        ambig_b = set()

        if data_a:
            knowledge.architectural_names = set(data_a.get("architectural", [])) & valid_names
            ambig_a = set(data_a.get("ambiguous", [])) & valid_names

        if data_b:
            ambig_b = set(data_b.get("confusable", [])) & valid_names

        knowledge.ambiguous_names = {
            n for n in (ambig_a & ambig_b)
            if len(n.split()) == 1
        }

    # ── Legacy entry point ────────────────────────────────────────────

    def _classify_components(self, names: list[str], knowledge: ModelKnowledge):
        """Use current strategy to classify components."""
        if self.classify_strategy == self.STRATEGY_TWO_PASS:
            self._classify_components_two_pass(names, knowledge)
        else:
            self._classify_components_tight_prompt(names, knowledge)

    def build_name_maps(
        self, components: list[ArchitectureComponent]
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Build name <-> ID mappings."""
        name_to_id = {c.name: c.id for c in components}
        id_to_name = {c.id: c.name for c in components}
        return name_to_id, id_to_name

    def get_abstract_names(
        self, components: list[ArchitectureComponent], knowledge: ModelKnowledge
    ) -> list[str]:
        """Get list of abstract (non-implementation) component names."""
        return [
            c.name for c in components
            if not knowledge.is_implementation(c.name)
        ]
