"""Architecture model analysis utilities."""

import re
from typing import Optional

from .data_types import ModelKnowledge
from ..pcm_parser import parse_pcm_repository, ArchitectureComponent
from ..llm_client import LLMClient


class ModelAnalyzer:
    """Analyze architecture model structure."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize analyzer.

        Args:
            llm_client: Optional LLM client for classification queries
        """
        self.llm_client = llm_client

    def load_components(self, model_path: str) -> list[ArchitectureComponent]:
        """Load components from PCM model file."""
        return parse_pcm_repository(model_path)

    def analyze(self, components: list[ArchitectureComponent]) -> ModelKnowledge:
        """Analyze model structure and build knowledge base.

        Args:
            components: List of architecture components

        Returns:
            ModelKnowledge with extracted information
        """
        names = [c.name for c in components]
        knowledge = ModelKnowledge()

        # Find implementation patterns (e.g., DefaultOrderProcessor -> OrderProcessor)
        self._find_impl_patterns(names, knowledge)

        # Find shared vocabulary between components
        self._find_shared_vocabulary(names, knowledge)

        # LLM-based classification if client available
        if self.llm_client:
            self._classify_components(names, knowledge)

        return knowledge

    def _find_impl_patterns(self, names: list[str], knowledge: ModelKnowledge):
        """Find implementation/concrete variants of abstract components."""
        for name in names:
            for other in names:
                if other != name and len(other) >= 3 and other in name:
                    idx = name.find(other)
                    prefix = name[:idx]
                    suffix = name[idx + len(other):]

                    # Record prefix indicators (e.g., "Basic", "Default", "Simple")
                    if prefix and len(prefix) >= 2:
                        knowledge.impl_indicators.append(prefix)
                        knowledge.impl_to_abstract[name] = other

                    # Record suffix indicators (e.g., "Impl", "Adapter")
                    if suffix and len(suffix) >= 2:
                        knowledge.impl_indicators.append(suffix)
                        knowledge.impl_to_abstract[name] = other

        # Deduplicate indicators
        knowledge.impl_indicators = list(set(knowledge.impl_indicators))

    def _find_shared_vocabulary(self, names: list[str], knowledge: ModelKnowledge):
        """Find words shared between multiple component names."""
        word_to_comps: dict[str, list[str]] = {}

        for name in names:
            # Split CamelCase and extract words
            words = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)', name)
            for word in words:
                w = word.lower()
                if len(w) >= 3:  # Ignore very short words
                    if w not in word_to_comps:
                        word_to_comps[w] = []
                    word_to_comps[w].append(name)

        # Keep only words shared by multiple components
        knowledge.shared_vocabulary = {
            word: list(set(comps))
            for word, comps in word_to_comps.items()
            if len(set(comps)) > 1
        }

    def _classify_components(self, names: list[str], knowledge: ModelKnowledge):
        """Use LLM to classify components as architectural vs ambiguous."""
        prompt = f"""Classify these software architecture component names from a Palladio Component Model (PCM).

NAMES: {', '.join(names)}

Return JSON:
{{
  "architectural": ["names that refer to specific software components"],
  "ambiguous": ["names that are extremely generic single English words with no technical meaning"]
}}

IMPORTANT GUIDELINES:
- Almost all names should be classified as "architectural" — these ARE component names from a real architecture model
- Architectural: ANY name that could identify a specific component in a software system. This includes short names like "Cache", "DB", "Auth", "UI", "Facade", "Logic", "Storage", "Server", "Client" — in context, these ARE specific components
- Ambiguous: ONLY classify a name as ambiguous if it is an extremely generic word that could never distinguish a component (e.g., "Common", "Util", "Misc", "Other", "Base")
- When in doubt, classify as architectural

JSON only:"""

        response = self.llm_client.query(prompt, timeout=100)
        data = self.llm_client.extract_json(response)

        if data:
            valid_names = set(names)
            knowledge.architectural_names = set(data.get("architectural", [])) & valid_names
            knowledge.ambiguous_names = set(data.get("ambiguous", [])) & valid_names

    def build_name_maps(
        self, components: list[ArchitectureComponent]
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Build name <-> ID mappings.

        Returns:
            Tuple of (name_to_id, id_to_name) dictionaries
        """
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
