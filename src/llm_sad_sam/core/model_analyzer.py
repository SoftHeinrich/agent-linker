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
Reasoning: Scheduler/Dispatcher name specific OS roles. Monitor and Pool are ordinary
English words regularly used generically ("monitor performance", "thread pool").
Helper is an organizational label.

EXAMPLE 3:
NAMES: RenderEngine, SceneGraph, Pipeline, Layer, Proxy, Socket, Router
→ architectural: ["RenderEngine", "SceneGraph", "Socket", "Router"]
→ ambiguous: ["Pipeline", "Layer", "Proxy"]
Reasoning: RenderEngine/SceneGraph are CamelCase compounds — always architectural.
Socket/Router name specific networking roles. Pipeline/Layer/Proxy are ordinary words
used generically in documentation ("processing pipeline", "network layer", "behind a proxy").

EXAMPLE 4:
NAMES: PaymentGateway, OrderProcessor, Connector, Controller, Adapter, Worker, Agent
→ architectural: ["PaymentGateway", "OrderProcessor", "Worker"]
→ ambiguous: ["Connector", "Controller", "Adapter", "Agent"]
Reasoning: PaymentGateway/OrderProcessor are CamelCase compounds naming specific roles.
Worker names a specific concurrency mechanism. But Connector/Controller/Adapter/Agent
seem functional yet are GENERIC categories writers use without referring to any specific
component: "a database connector", "the main controller", "a protocol adapter", "a
background agent". They describe WHAT KIND of thing it is, not WHICH specific mechanism
— so they are ambiguous.""".strip()


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
1. ARCHITECTURAL: Names that identify a specific component role or function.
   Always architectural:
   - Multi-word names, hyphenated names
   - CamelCase compounds
   - ALL-UPPERCASE abbreviations (API, TCP, RPC)
   - Well-known computing abbreviations in ANY letter case (vm, io, os, tcp, rpc)
   - Names that describe a specific technical function: Scheduler (=task scheduling),
     Dispatcher (=event routing), Router (=request routing), Renderer (=rendering output)

2. AMBIGUOUS: Single words that a technical writer would use as a GENERIC noun
   in everyday documentation, where the word has a strong plain-English meaning separate
   from any component. This includes TWO categories:

   Category A — Organizational labels (tell you nothing about function):
   - "the system core" → AMBIGUOUS
   - "a utility module" → AMBIGUOUS
   - "the base class" → AMBIGUOUS
   - "a helper function" → AMBIGUOUS

   Category B — Generic functional categories (describe WHAT KIND, not WHICH):
   - "a database connector" → "connector" describes a category → AMBIGUOUS
   - "the main controller" → "controller" describes a pattern → AMBIGUOUS
   - "a protocol adapter" → "adapter" describes a role type → AMBIGUOUS
   - "a background agent" → "agent" describes a pattern → AMBIGUOUS

   Counter-examples where the word IS a specific mechanism:
   - "the Scheduler assigns threads" → "Scheduler" names a SPECIFIC mechanism → ARCHITECTURAL
   - "the Router forwards packets" → "Router" names a SPECIFIC mechanism → ARCHITECTURAL
   Key difference: Scheduler/Router describe HOW something works (specific mechanism).
   Connector/Controller/Adapter describe WHAT KIND of thing it is (generic category).

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
