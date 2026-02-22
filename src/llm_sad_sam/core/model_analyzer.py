"""Architecture model analysis utilities."""

import re
from typing import Optional

from .data_types import ModelKnowledge
from ..pcm_parser import parse_pcm_repository, ArchitectureComponent
from ..llm_client import LLMClient


# Few-shot examples for classification (safe textbook domains only, no benchmark names)
_FEW_SHOT = """
EXAMPLE 1:
NAMES: Lexer, Parser, CodeGenerator, Optimizer, Core, Util, AST, SymbolTable, Base
→ architectural: ["Lexer", "Parser", "CodeGenerator", "Optimizer", "AST", "SymbolTable"]
→ ambiguous: ["Core", "Util", "Base"]
Reasoning: Lexer/Parser/Optimizer name specific compilation roles. Core/Util/Base are
organizational labels that tell you nothing about what the component does.

EXAMPLE 2:
NAMES: Scheduler, Dispatcher, MemoryManager, Monitor, Pool, Helper, ProcessTable
→ architectural: ["Scheduler", "Dispatcher", "MemoryManager", "ProcessTable"]
→ ambiguous: ["Monitor", "Pool", "Helper"]
Reasoning: Scheduler/Dispatcher name specific OS roles. Monitor and Pool are common
English words regularly used generically ("monitor performance", "thread pool").
Helper is an organizational label.

EXAMPLE 3:
NAMES: RenderEngine, SceneGraph, Pipeline, Layer, Proxy, Socket, Router
→ architectural: ["RenderEngine", "SceneGraph", "Socket", "Router"]
→ ambiguous: ["Pipeline", "Layer", "Proxy"]
Reasoning: RenderEngine/SceneGraph are CamelCase compounds — always architectural.
Socket/Router name specific networking roles. Pipeline/Layer/Proxy are common words
used generically in documentation ("processing pipeline", "network layer", "behind a proxy").""".strip()


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
        """Analyze model structure and build knowledge base."""
        names = [c.name for c in components]
        knowledge = ModelKnowledge()

        self._find_impl_patterns(names, knowledge)
        self._find_shared_vocabulary(names, knowledge)

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

    @staticmethod
    def _is_structurally_unambiguous(name: str) -> bool:
        """Principled code guard: names that are structurally identifiers, not words."""
        if ' ' in name or '-' in name:
            return True
        if re.search(r'[a-z][A-Z]', name):
            return True
        if name.isupper():
            return True
        return False

    def _classify_components(self, names: list[str], knowledge: ModelKnowledge):
        """Classify components using few-shot prompt + structural code guard."""
        prompt = f"""Classify these software architecture component names.

NAMES: {', '.join(names)}

{_FEW_SHOT}

NOW CLASSIFY THE NAMES ABOVE.

Return JSON:
{{
  "architectural": ["names that identify specific components"],
  "ambiguous": ["names that could easily be used as ordinary words in documentation"]
}}

RULES:
1. ARCHITECTURAL: Names that refer to a specific role or responsibility. If the name tells you
   WHAT the component does (scheduling, parsing, rendering, storing data, managing users), it is
   architectural — even if the word also exists in a dictionary.
   Multi-word names, CamelCase compounds, and abbreviations (DB, API, UI) → always architectural.

2. AMBIGUOUS: Short single words that writers commonly use generically in software documentation.
   The test: "Could a technical writer naturally write this word in a sentence about ANY system
   without referring to a specific component?" If yes → ambiguous.

JSON only:"""

        response = self.llm_client.query(prompt, timeout=100)
        data = self.llm_client.extract_json(response)

        if data:
            valid_names = set(names)
            knowledge.architectural_names = set(data.get("architectural", [])) & valid_names
            raw_ambiguous = set(data.get("ambiguous", [])) & valid_names
            knowledge.ambiguous_names = {
                n for n in raw_ambiguous
                if len(n.split()) == 1 and not self._is_structurally_unambiguous(n)
            }

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
