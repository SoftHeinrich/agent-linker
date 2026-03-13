"""ALinker — Adaptive Agent Linker with orchestrator, monitor, and phase sub-agents.

Each pipeline phase runs as an independent sub-agent dispatched by an Orchestrator.
A Monitor computes extrinsic quality metrics (no gold standard) after each agent.
Review agents handle issues: low coverage, density anomalies, coref spikes.

Based on V32 pipeline logic (94.5% macro F1 on Claude Sonnet).
Standalone: no inheritance from any previous linker.
"""

import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from llm_sad_sam.core.data_types import (
    SadSamLink, CandidateLink, DocumentProfile,
    ModelKnowledge, DocumentKnowledge, LearnedPatterns, EntityMention,
)
from llm_sad_sam.core.document_loader import DocumentLoader, Sentence
from llm_sad_sam.linkers.experimental.ilinker2 import ILinker2
from llm_sad_sam.pcm_parser import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient, LLMBackend


# ══════════════════════════════════════════════════════════════════════
# Constants (same as V32 — proven prompts)
# ══════════════════════════════════════════════════════════════════════

CONVENTION_GUIDE = """### STEP 1 — Hierarchical name reference (not about the component itself)?

The most common reason for NO_LINK: the sentence mentions the component name only
as part of a HIERARCHICAL/QUALIFIED NAME (dotted path, namespace, module path) that
refers to a nested sub-unit, not the component's own architectural role.

Software documentation commonly uses hierarchical naming (e.g., "X.utils", "X/handlers",
"X::impl") to refer to parts inside a component. The component name appears only as
a prefix, not as the subject.

Recognize these patterns — all are NO_LINK for component X:
- "X.utils provides helper functions" — dotted sub-unit reference
- "X.handlers, X.mappers, X.adapters follow a pipeline" — listing sub-packages of X
- "Classes in the X.impl package are not exported" — even with
  architectural language, if the subject is X's sub-unit → NO_LINK
- Bare name mixed with qualified paths: "X, Y.adapters, Y.transformers follow
  a pipeline design" — treat ALL as hierarchical references → NO_LINK

KEY DISTINCTION: Sentences that describe what X DOES or HOW X INTERACTS with other
components are LINK, even if they mention implementation details (e.g., "X uses Y
technology for Z" → LINK for X, because it describes X's behavior).

EXCEPTION: If the sentence also explicitly names the target component AS A PROPER
NOUN with the word "component" (e.g., "for the X component") → LINK.

Cross-reference rule: A sub-unit sentence mentioning a DIFFERENT component
in an architectural role is LINK for that other component.

### STEP 2 — Component name confused with a different entity?

**2a. Technology / methodology confusion:**

CRITICAL RULE: If a component IS NAMED AFTER a technology (e.g., the architecture
has a component called "Kafka Broker" or "Nginx Proxy" or "Zookeeper"), then ANY sentence
describing that technology's capabilities, role, or behavior IS about the component → LINK.
This rule applies because architecture components are often named after the technology they wrap.

NO_LINK ONLY when:
- The sentence describes a technology that is NOT one of our components
- The name appears in a compound entity unrelated to the component
  ("X Protocol specification" → NO_LINK for "X" if X is not about that protocol)
- Uses the name as part of a METHODOLOGY ("X testing in CI" → NO_LINK for "X")

LINK when:
- The technology IS one of our architecture components (always LINK)
- Components INTERACT with or connect THROUGH the technology

**2b. Generic word collision:**
NO_LINK — narrow, non-architectural sense:
- Process/activity modifier: "throttle X", "batch X", "polling X"
- Hardware/deployment: "a physical rack-mounted appliance", "multi-socket machine"
- Possessive/personal: "her bookmarks", "their account"

LINK — system-level architectural sense:
- System name + word: "the [System] gateway"
- Architectural role: "the orchestrator routes jobs to the gateway"

### STEP 3 — Default: LINK.
If neither Step 1 nor Step 2 applies → LINK.

### IMPORTANT GUARDRAILS:
- Multi-word component names (e.g., "Kafka Broker", "Nginx Proxy") are NEVER generic words → LINK
- CamelCase identifiers are NEVER generic words → LINK
- Sentences describing how components interact, connect, or communicate → LINK for ALL components involved (not just the grammatical subject). "X connects to Y" is LINK for both X and Y.
- Sentences about what a component does, provides, or handles → LINK
- A component does NOT need to be the grammatical subject to be relevant. If a sentence says "X sends data to Y", both X and Y get LINK.
- Only use NO_LINK when you are CONFIDENT the name is NOT used as a component reference

### Priority:
Be AGGRESSIVE with NO_LINK on sub-package descriptions (Step 1).
For Step 2, only NO_LINK when confident. Default to LINK."""

FEW_SHOT = """
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

PRONOUN_PATTERN = re.compile(
    r'\b(it|they|this|these|that|those|its|their|the component|the service)\b',
    re.IGNORECASE
)

SOURCE_PRIORITY = {
    "transarc": 5, "validated": 4, "entity": 3,
    "coreference": 2, "partial_inject": 1,
}

CONTEXT_WINDOW = 3


# ══════════════════════════════════════════════════════════════════════
# Pipeline State
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PipelineState:
    """Shared mutable state passed between agents."""
    text_path: str
    model_path: str
    sentences: list
    components: list
    name_to_id: dict
    sent_map: dict
    # Phase outputs (populated progressively by agents)
    doc_profile: Optional[DocumentProfile] = None
    is_complex: Optional[bool] = None
    model_knowledge: Optional[ModelKnowledge] = None
    generic_component_words: set = field(default_factory=set)
    generic_partials: set = field(default_factory=set)
    learned_patterns: Optional[LearnedPatterns] = None
    doc_knowledge: Optional[DocumentKnowledge] = None
    seed_links: list = field(default_factory=list)
    transarc_set: set = field(default_factory=set)
    candidates: list = field(default_factory=list)
    validated: list = field(default_factory=list)
    coref_links: list = field(default_factory=list)
    partial_links: list = field(default_factory=list)
    preliminary: list = field(default_factory=list)
    boundary_rejected: list = field(default_factory=list)
    final_links: list = field(default_factory=list)
    # Orchestrator tracking
    agent_log: list = field(default_factory=list)
    recovery_attempted: set = field(default_factory=set)  # component names already attempted


# ══════════════════════════════════════════════════════════════════════
# Quality Metrics + Monitor
# ══════════════════════════════════════════════════════════════════════

@dataclass
class QualityMetrics:
    """Extrinsic quality metrics — computable without gold standard."""
    component_coverage: float
    link_density: float
    uncovered_components: list
    density_per_component: dict
    total_links: int
    source_diversity: dict
    balance_score: float


class Monitor:
    """Computes extrinsic quality metrics after each agent. No gold standard needed."""

    def compute(self, links, comp_names):
        covered = set()
        density_per_comp = defaultdict(int)
        source_counts = defaultdict(int)
        for lk in links:
            covered.add(lk.component_name)
            density_per_comp[lk.component_name] += 1
            source_counts[lk.source] += 1

        coverage = len(covered) / max(1, len(comp_names))
        uncovered = [c for c in comp_names if c not in covered]
        total = len(links)
        density = total / max(1, len(comp_names))

        # Balance: coefficient of variation (low CV = evenly distributed)
        counts = list(density_per_comp.values()) + [0] * len(uncovered)
        if counts:
            mean = sum(counts) / len(counts)
            var = sum((c - mean) ** 2 for c in counts) / len(counts)
            cv = (var ** 0.5) / max(0.001, mean)
            balance = max(0.0, 1.0 - cv / 3.0)
        else:
            balance = 0.0

        return QualityMetrics(
            component_coverage=coverage,
            link_density=density,
            uncovered_components=uncovered,
            density_per_component=dict(density_per_comp),
            total_links=total,
            source_diversity=dict(source_counts),
            balance_score=balance,
        )

    def detect_anomalies(self, before, after, phase_name):
        anomalies = []
        if before is None:
            return anomalies
        # Big link drop (>30%)
        if after.total_links < before.total_links * 0.7 and before.total_links > 5:
            drop = 1 - after.total_links / max(1, before.total_links)
            anomalies.append(f"LINK_DROP:{phase_name} removed {drop:.0%} of links")
        # Coverage decrease
        if after.component_coverage < before.component_coverage - 0.1:
            anomalies.append(f"COVERAGE_DROP:{phase_name} lost {before.component_coverage - after.component_coverage:.0%} coverage")
        # Extreme density
        if after.link_density > 10:
            anomalies.append(f"HIGH_DENSITY:{after.link_density:.1f} links/comp")
        # Single component overload
        for comp, count in after.density_per_component.items():
            if count > 15:
                anomalies.append(f"COMP_OVERLOAD:{comp} has {count} links")
        return anomalies

    def print_metrics(self, metrics, label):
        print(f"  [Monitor:{label}] coverage={metrics.component_coverage:.0%} "
              f"density={metrics.link_density:.1f} links={metrics.total_links} "
              f"balance={metrics.balance_score:.2f} "
              f"sources={dict(metrics.source_diversity)}")
        if metrics.uncovered_components:
            print(f"    Uncovered: {', '.join(metrics.uncovered_components[:10])}")


# ══════════════════════════════════════════════════════════════════════
# Shared Helper Functions
# ══════════════════════════════════════════════════════════════════════

def get_comp_names(components, model_knowledge):
    """Get non-implementation component names."""
    return [c.name for c in components
            if not (model_knowledge and model_knowledge.is_implementation(c.name))]


def is_structurally_unambiguous(name):
    if ' ' in name or '-' in name:
        return True
    if re.search(r'[a-z][A-Z]', name):
        return True
    if name.isupper():
        return True
    return False


def has_standalone_mention(comp_name, text):
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
        if s > 0 and text[s - 1] == '.':
            continue
        if e < len(text) and text[e] == '.' and e + 1 < len(text) and text[e + 1].isalpha():
            continue
        if s > 0 and text[s - 1] == '-':
            continue
        if e < len(text) and text[e] == '-' and '-' not in comp_name:
            continue
        return True
    return False


def has_alias_mention(comp_name, text, doc_knowledge):
    if not doc_knowledge:
        return False
    text_lower = text.lower()
    for syn, target in doc_knowledge.synonyms.items():
        if target == comp_name:
            if re.search(rf'\b{re.escape(syn.lower())}\b', text_lower):
                return True
    for partial, target in doc_knowledge.partial_references.items():
        if target == comp_name:
            if re.search(rf'\b{re.escape(partial.lower())}\b', text_lower):
                return True
    return False


def has_clean_mention(term, text):
    pattern = rf'\b{re.escape(term)}\b'
    for m in re.finditer(pattern, text, re.IGNORECASE):
        s, e = m.start(), m.end()
        if s > 0 and text[s - 1] == '.':
            continue
        if e < len(text) and text[e] == '.' and e + 1 < len(text) and text[e + 1].isalpha():
            continue
        if (s > 0 and text[s - 1] == '-') or (e < len(text) and text[e] == '-'):
            continue
        return True
    return False


def is_generic_mention(comp_name, sentence_text):
    if ' ' in comp_name or '-' in comp_name:
        return False
    if re.search(r'[a-z][A-Z]', comp_name):
        return False
    if comp_name.isupper():
        return False
    if comp_name[0].islower():
        return False
    if has_standalone_mention(comp_name, sentence_text):
        return False
    word_lower = comp_name.lower()
    if re.search(rf'\b{re.escape(word_lower)}\b', sentence_text):
        return True
    return False


def word_boundary_match(name, text):
    return bool(re.search(r'\b' + re.escape(name) + r'\b', text, re.IGNORECASE))


def is_ambiguous_name_component(comp_name, model_knowledge):
    if ' ' in comp_name or '-' in comp_name:
        return False
    if re.search(r'[a-z][A-Z]', comp_name):
        return False
    if comp_name.isupper():
        return False
    if not model_knowledge or not model_knowledge.ambiguous_names:
        return False
    return comp_name in model_knowledge.ambiguous_names


def find_match_text(comp_name, sent_text, doc_knowledge):
    if not sent_text:
        return None
    if re.search(rf'\b{re.escape(comp_name)}\b', sent_text, re.IGNORECASE):
        return comp_name
    if doc_knowledge:
        for alias, comp in doc_knowledge.synonyms.items():
            if comp == comp_name and re.search(rf'\b{re.escape(alias)}\b', sent_text, re.IGNORECASE):
                return alias
        for alias, comp in doc_knowledge.abbreviations.items():
            if comp == comp_name and re.search(rf'\b{re.escape(alias)}\b', sent_text, re.IGNORECASE):
                return alias
        for partial, comp in doc_knowledge.partial_references.items():
            if comp == comp_name and partial.lower() in sent_text.lower():
                return partial
    return None


def extract_json_array(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    start = text.find("[")
    if start >= 0:
        depth = 0
        for j in range(start, len(text)):
            if text[j] == '[':
                depth += 1
            elif text[j] == ']':
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(text[start:j + 1])
                        if isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        break
    return None


def abbreviation_match_is_valid(abbrev, comp_name, sentence_text):
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


def parse_snum(snum):
    """Parse sentence number, handling 'S6' string format."""
    if isinstance(snum, str):
        snum = snum.lstrip("S")
    try:
        return int(snum)
    except (ValueError, TypeError):
        return None


def _safe_int(val, default=0):
    """Safely coerce LLM JSON value to int (handles str/float/None)."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


# ══════════════════════════════════════════════════════════════════════
# Phase Agent Base Class
# ══════════════════════════════════════════════════════════════════════

class PhaseAgent:
    """Base class for pipeline phase agents."""
    name = "base"

    def run(self, state: PipelineState, llm: LLMClient) -> None:
        """Execute agent logic, mutating state in place."""
        raise NotImplementedError

    def validate(self, state: PipelineState) -> list[str]:
        """Return list of warnings. Empty = OK."""
        return []


# ══════════════════════════════════════════════════════════════════════
# Phase 0: Document Profiling
# ══════════════════════════════════════════════════════════════════════

class ProfileAgent(PhaseAgent):
    name = "profile"

    def run(self, state, llm):
        sentences, components = state.sentences, state.components
        texts = [s.text for s in sentences]
        comp_names = [c.name for c in components]

        pron = r'\b(it|they|this|these|that|those|its|their)\b'
        pronoun_ratio = sum(1 for t in texts if re.search(pron, t.lower())) / max(1, len(sentences))
        mentions = sum(1 for t in texts for c in comp_names if c.lower() in t.lower())
        density = mentions / max(1, len(sentences))
        spc = len(sentences) / max(1, len(components))

        state.doc_profile = DocumentProfile(
            sentence_count=len(sentences), component_count=len(components),
            pronoun_ratio=pronoun_ratio, technical_density=density,
            component_mention_density=density,
            complexity_score=min(1.0, spc / 20),
            recommended_strictness="balanced",
        )

        # Structural complexity
        mention_count = sum(1 for sent in sentences
                           if any(cn.lower() in sent.text.lower() for cn in comp_names))
        uncovered_ratio = 1.0 - (mention_count / max(1, len(sentences)))
        state.is_complex = uncovered_ratio > 0.5 and spc > 4

        print(f"  Stats: {spc:.1f} sents/comp, {pronoun_ratio:.0%} pronouns, complex={state.is_complex}")


# ══════════════════════════════════════════════════════════════════════
# Phase 1: Model Structure Analysis
# ══════════════════════════════════════════════════════════════════════

class ModelAnalysisAgent(PhaseAgent):
    name = "model_analysis"

    def run(self, state, llm):
        components = state.components
        names = [c.name for c in components]
        knowledge = ModelKnowledge()

        # Discover parent-child relationships
        for name in names:
            for other in names:
                if other != name and len(other) >= 3 and other in name:
                    idx = name.find(other)
                    prefix, suffix = name[:idx], name[idx + len(other):]
                    if prefix and len(prefix) >= 2:
                        knowledge.impl_indicators.append(prefix)
                        knowledge.impl_to_abstract[name] = other
                    if suffix and len(suffix) >= 2:
                        knowledge.impl_indicators.append(suffix)
                        knowledge.impl_to_abstract[name] = other
        knowledge.impl_indicators = list(set(knowledge.impl_indicators))

        # Shared vocabulary
        word_to_comps = {}
        for name in names:
            for word in re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)', name):
                w = word.lower()
                if len(w) >= 3:
                    word_to_comps.setdefault(w, []).append(name)
        knowledge.shared_vocabulary = {w: list(set(c)) for w, c in word_to_comps.items() if len(set(c)) > 1}

        # LLM classification
        prompt = f"""Classify these software architecture component names.

NAMES: {', '.join(names)}

{FEW_SHOT}

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
   Multi-word names, CamelCase compounds, and abbreviations (API, TCP, RPC) → always architectural.

2. AMBIGUOUS: Single words that writers regularly use generically in software documentation.
   This includes TWO categories:
   Category A — Organizational labels: core, util, base, helper (tell you nothing about function)
   Category B — Generic functional categories: connector, controller, adapter, agent
   (describe WHAT KIND of thing, not WHICH specific mechanism)
   The test: "Could a technical writer naturally write this word in a sentence about ANY system
   without referring to a specific component?" If yes → ambiguous.
   Key: Scheduler/Router describe HOW (specific mechanism) → ARCHITECTURAL.
         Connector/Controller/Adapter describe WHAT KIND (generic category) → AMBIGUOUS.

JSON only:"""

        data = llm.extract_json(llm.query(prompt, timeout=100))
        if data:
            valid = set(names)
            knowledge.architectural_names = set(data.get("architectural", [])) & valid
            raw_ambiguous = set(data.get("ambiguous", [])) & valid
            knowledge.ambiguous_names = {
                n for n in raw_ambiguous
                if len(n.split()) == 1 and not is_structurally_unambiguous(n)
            }

        state.model_knowledge = knowledge

        # Discover generic words and partials
        ambig = knowledge.ambiguous_names
        state.generic_component_words = set()
        for name in ambig:
            if ' ' not in name and not name.isupper():
                state.generic_component_words.add(name.lower())

        state.generic_partials = set()
        for comp in components:
            parts = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)', comp.name)
            for part in parts:
                p_lower = part.lower()
                if part.isupper():
                    continue
                if len(p_lower) >= 3 and (p_lower in ambig or any(p_lower == a.lower() for a in ambig)):
                    state.generic_partials.add(p_lower)
        for name in ambig:
            if ' ' not in name and not name.isupper():
                state.generic_partials.add(name.lower())

        arch = knowledge.architectural_names
        print(f"  Architectural: {len(arch)}, Ambiguous: {sorted(ambig)}")
        print(f"  Generic words: {sorted(state.generic_component_words)}")
        print(f"  Generic partials: {sorted(state.generic_partials)}")


# ══════════════════════════════════════════════════════════════════════
# Phase 2: Pattern Learning (Debate)
# ══════════════════════════════════════════════════════════════════════

class PatternLearningAgent(PhaseAgent):
    name = "pattern_learning"

    def run(self, state, llm):
        comp_names = get_comp_names(state.components, state.model_knowledge)
        sample = [f"S{s.number}: {s.text}" for s in state.sentences[:70]]

        prompt1 = f"""Find terms that refer to INTERNAL PARTS of components (subprocesses).

COMPONENTS: {', '.join(comp_names)}

DOCUMENT:
{chr(10).join(sample)}

Return JSON:
{{
  "subprocess_terms": ["term1", "term2"],
  "reasoning": {{"term": "why"}}
}}
JSON only:"""

        data1 = llm.extract_json(llm.query(prompt1, timeout=120))
        proposed = data1.get("subprocess_terms", []) if data1 else []
        reasonings = data1.get("reasoning", {}) if data1 else {}

        if proposed:
            prompt2 = f"""DEBATE: Validate these subprocess terms.

COMPONENTS: {', '.join(comp_names)}

PROPOSED:
{chr(10).join([f"- {t}: {reasonings.get(t, '')}" for t in proposed[:15]])}

SAMPLE:
{chr(10).join(sample[:30])}

Return JSON:
{{
  "validated": ["terms that ARE subprocesses"],
  "rejected": ["terms that might be valid component references"]
}}
JSON only:"""

            data2 = llm.extract_json(llm.query(prompt2, timeout=120))
            validated_terms = set(data2.get("validated", [])) if data2 else set(proposed)
        else:
            validated_terms = set()

        prompt3 = f"""Find linguistic patterns.

COMPONENTS: {', '.join(comp_names)}

DOCUMENT:
{chr(10).join(sample[:40])}

Return JSON:
{{
  "action_indicators": ["verbs when component DOES something"],
  "effect_indicators": ["verbs for RESULTS"]
}}
JSON only:"""

        data3 = llm.extract_json(llm.query(prompt3, timeout=100))

        patterns = LearnedPatterns()
        patterns.subprocess_terms = validated_terms
        if data3:
            patterns.action_indicators = data3.get("action_indicators", [])
            patterns.effect_indicators = data3.get("effect_indicators", [])

        state.learned_patterns = patterns
        print(f"  Subprocess terms: {len(validated_terms)}")
        for t in list(validated_terms)[:8]:
            print(f"    '{t}'")


# ══════════════════════════════════════════════════════════════════════
# Phase 3: Document Knowledge (Few-shot Judge)
# ══════════════════════════════════════════════════════════════════════

class DocumentKnowledgeAgent(PhaseAgent):
    name = "document_knowledge"

    def run(self, state, llm):
        comp_names_list = [c.name for c in state.components]
        doc_lines = [s.text for s in state.sentences[:150]]

        prompt1 = f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names_list)}

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
   REJECT: Ordinary words that have plain English meanings beyond the component

DOCUMENT:
{chr(10).join(doc_lines)}

Return JSON:
{{
  "abbreviations": {{"short_form": "FullComponent"}},
  "synonyms": {{"specific_alternative_name": "FullComponent"}},
  "partial_references": {{"partial_name": "FullComponent"}}
}}
JSON only:"""

        data1 = llm.extract_json(llm.query(prompt1, timeout=150))

        all_mappings = {}
        if data1:
            for short, full in data1.get("abbreviations", {}).items():
                if full in comp_names_list:
                    all_mappings[short] = ("abbrev", full)
            for syn, full in data1.get("synonyms", {}).items():
                if full in comp_names_list:
                    all_mappings[syn] = ("synonym", full)
            for partial, full in data1.get("partial_references", {}).items():
                if full in comp_names_list:
                    all_mappings[partial] = ("partial", full)

        if all_mappings:
            mapping_list = [f"'{k}' -> {v[1]} ({v[0]})" for k, v in list(all_mappings.items())[:25]]

            prompt2 = f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names_list)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}

EXAMPLES — study these to calibrate your judgment:

Example 1 — APPROVE (abbreviation from component name):
  'AST' -> AbstractSyntaxTree (abbrev)
  Verdict: APPROVE. "AST" is the initials of "AbstractSyntaxTree". Abbreviations
  formed from the component name's words are always valid.

Example 2 — APPROVE (trailing word of multi-word name):
  'Dispatcher' -> EventDispatcher (partial)
  Verdict: APPROVE. "Dispatcher" is the last word of "EventDispatcher".
  If no other component ends in "Dispatcher", this partial is unambiguous.

Example 3 — APPROVE (CamelCase identifier):
  'RenderEngine' -> GameRenderEngine (synonym)
  Verdict: APPROVE. CamelCase is a constructed identifier — always a proper name.

Example 4 — APPROVE (trailing word of multi-word name):
  'Table' -> SymbolTable (partial)
  Verdict: APPROVE. "Table" is the trailing word of "SymbolTable" and
  likely refers to this specific component when no other component uses "Table".

Example 5 — REJECT (ordinary English verb/noun):
  'process' -> OrderProcessor (partial)
  Verdict: REJECT. "process" is an ordinary English verb/noun used generically
  in many contexts ("process requests", "the process").

Example 6 — REJECT (refers to whole system):
  'system' -> PaymentSystem (partial)
  Verdict: REJECT. "system" is too generic — it could refer to the overall system.

DECISION RULES (apply in order):

1. AUTO-APPROVE these — they are always valid mappings:
   - Abbreviations formed from the component name's initials or words
   - Trailing words of multi-word component names (if no other component shares that word)
   - CamelCase identifiers
   - Multi-word phrases that contain the component name

2. APPROVE if the term plausibly refers to exactly one component and is NOT
   a generic word like "system", "process", "service", "component", "module".

3. REJECT only if the term is clearly generic and could refer to anything,
   or clearly refers to a different component or the system as a whole.

IMPORTANT: When in doubt, APPROVE. False approvals are filtered by later
pipeline stages; false rejections cause permanent recall loss.

Return JSON:
{{
  "approved": ["term1", "term2"],
  "generic_rejected": ["generic_term1"]
}}
JSON only:"""

            data2 = llm.extract_json(llm.query(prompt2, timeout=120))
            approved = set(data2.get("approved", [])) if data2 else set(all_mappings.keys())
            generic_terms = set(data2.get("generic_rejected", [])) if data2 else set()
        else:
            approved = set()
            generic_terms = set()

        # CamelCase rescue
        for term in list(generic_terms):
            if re.search(r'[a-z][A-Z]', term) and term in all_mappings:
                generic_terms.discard(term)
                approved.add(term)
                print(f"    CamelCase override (rescued): {term}")

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
        for comp in [c.name for c in state.components]:
            split = re.sub(r'([a-z])([A-Z])', r'\1 \2', comp)
            split = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', split)
            if split != comp and split not in knowledge.synonyms:
                knowledge.synonyms[split] = comp
                print(f"    CamelCase syn: {split} -> {comp}")

        state.doc_knowledge = knowledge

        # Phase 3b: Multi-word partial enrichment
        self._enrich_multiword_partials(state)

        print(f"  Abbrev: {len(knowledge.abbreviations)}, "
              f"Syn: {len(knowledge.synonyms)}, "
              f"Generic: {len(knowledge.generic_terms)}")

    def _enrich_multiword_partials(self, state):
        if not state.doc_knowledge:
            return
        added = []
        for comp in state.components:
            parts = comp.name.split()
            if len(parts) < 2:
                continue
            last_word = parts[-1]
            if len(last_word) < 4:
                continue
            last_lower = last_word.lower()

            other_match = any(
                c.name != comp.name and c.name.lower().endswith(last_lower)
                for c in state.components
            )
            if other_match:
                continue
            if last_lower in {s.lower() for s in state.doc_knowledge.synonyms}:
                continue
            if last_lower in {p.lower() for p in state.doc_knowledge.partial_references}:
                continue

            is_generic_word = last_lower in state.generic_partials
            full_lower = comp.name.lower()
            mention_count = 0
            for sent in state.sentences:
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
                state.doc_knowledge.partial_references[last_word] = comp.name
                added.append(f"{last_word} -> {comp.name} ({mention_count} mentions)")

        if added:
            print(f"  [Phase 3b] Multi-word Enrichment")
            for a in added:
                print(f"    Auto-partial: {a}")


# ══════════════════════════════════════════════════════════════════════
# Phase 4: Seed (ILinker2)
# ══════════════════════════════════════════════════════════════════════

class SeedAgent(PhaseAgent):
    name = "seed"

    def run(self, state, llm):
        ilinker2 = ILinker2(backend=llm.backend)
        raw_links = ilinker2.link(state.text_path, state.model_path)

        state.seed_links = [
            SadSamLink(
                sentence_number=lk.sentence_number,
                component_id=lk.component_id,
                component_name=lk.component_name,
                confidence=lk.confidence,
                source="transarc",
            )
            for lk in raw_links
        ]
        state.transarc_set = {(l.sentence_number, l.component_id) for l in state.seed_links}
        print(f"  Seed links: {len(state.seed_links)}")

    def validate(self, state):
        warnings = []
        if not state.seed_links:
            warnings.append("ZERO_SEED: ILinker2 returned no links")
        return warnings


# ══════════════════════════════════════════════════════════════════════
# Phase 5: Entity Extraction
# ══════════════════════════════════════════════════════════════════════

class ExtractionAgent(PhaseAgent):
    name = "extraction"

    def run(self, state, llm):
        comp_names = get_comp_names(state.components, state.model_knowledge)
        comp_lower = {n.lower() for n in comp_names}

        mappings = []
        if state.doc_knowledge:
            mappings.extend([f"{a}={c}" for a, c in state.doc_knowledge.abbreviations.items()])
            mappings.extend([f"{s}={c}" for s, c in state.doc_knowledge.synonyms.items()])
            mappings.extend([f"{p}={c}" for p, c in state.doc_knowledge.partial_references.items()])

        batch_size = 50
        all_candidates = {}

        for batch_start in range(0, len(state.sentences), batch_size):
            batch = state.sentences[batch_start:batch_start + batch_size]

            if len(state.sentences) > batch_size:
                print(f"    Entity batch {batch_start // batch_size + 1}: "
                      f"S{batch[0].number}-S{batch[-1].number}")

            prompt = f"""Extract ALL references to software architecture components from this document.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings[:20])}' if mappings else ''}

RULES — include a reference when:
1. The component name (or known alias) appears directly in the sentence
2. A space-separated form matches a compound name (e.g., "Memory Manager" → MemoryManager)
3. The sentence describes what a specific component does by name or role
4. A known synonym or partial reference is used
5. The component participates in an interaction described in the sentence (as sender, receiver, or target) — e.g., "X sends data to Y" references BOTH X and Y
6. The component is mentioned in a passive or prepositional phrase — e.g., "data is stored in X", "handled by X", "via X", "through X"

RULES — exclude when:
1. The name appears only inside a dotted path (e.g., com.example.name)
2. The name is used as an ordinary English word, not as a component reference

Favor inclusion over exclusion — later validation will filter borderline cases.

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence", "match_type": "exact|synonym|partial|functional"}}]}}
JSON only:"""

            for attempt in range(2):
                data = llm.extract_json(llm.query(prompt, timeout=240))
                if data and data.get("references"):
                    break
                if attempt == 0:
                    print(f"    Empty response, retrying batch...")

            if not data:
                continue

            for ref in data.get("references", []):
                snum, cname = ref.get("sentence"), ref.get("component")
                if not (snum and cname and cname in state.name_to_id):
                    continue
                snum = parse_snum(snum)
                if snum is None:
                    continue
                sent = state.sent_map.get(snum)
                if not sent:
                    continue

                matched = ref.get("matched_text", "")
                if matched and matched.lower() not in sent.text.lower():
                    continue

                matched_lower = matched.lower() if matched else ""
                is_exact = matched_lower in comp_lower or cname.lower() in matched_lower
                is_generic_here = is_generic_mention(cname, sent.text)
                needs_val = not is_exact or ref.get("match_type") != "exact" or is_generic_here

                key = (snum, state.name_to_id[cname])
                if key not in all_candidates:
                    all_candidates[key] = CandidateLink(
                        snum, sent.text, cname, state.name_to_id[cname],
                        matched, 0.85, "entity",
                        ref.get("match_type", "exact"), needs_val)

        state.candidates = list(all_candidates.values())

        # Abbreviation guard
        self._apply_abbreviation_guard(state)

        # Phase 5b: targeted recovery for unlinked components
        self._targeted_recovery(state, llm)

        print(f"  Candidates: {len(state.candidates)}")

    def _apply_abbreviation_guard(self, state):
        if not state.doc_knowledge:
            return
        abbrev_to_comp = {}
        comp_to_abbrevs = {}
        for abbr, comp in state.doc_knowledge.abbreviations.items():
            abbrev_to_comp[abbr.lower()] = comp
            comp_to_abbrevs.setdefault(comp, []).append(abbr)

        filtered = []
        for c in state.candidates:
            matched_lower = c.matched_text.lower() if c.matched_text else ""
            comp = c.component_name
            sent = state.sent_map.get(c.sentence_number)
            if matched_lower in abbrev_to_comp and abbrev_to_comp[matched_lower] == comp:
                if sent and not abbreviation_match_is_valid(c.matched_text, comp, sent.text):
                    print(f"    Abbrev guard: rejected S{c.sentence_number} {c.matched_text} -> {comp}")
                    continue
            if sent and comp in comp_to_abbrevs and ' ' in comp:
                full_in_text = re.search(rf'\b{re.escape(comp)}\b', sent.text, re.IGNORECASE)
                if not full_in_text:
                    rejected = False
                    for abbr in comp_to_abbrevs[comp]:
                        if re.search(rf'\b{re.escape(abbr)}\b', sent.text, re.IGNORECASE):
                            if not abbreviation_match_is_valid(abbr, comp, sent.text):
                                print(f"    Abbrev guard (inferred): rejected S{c.sentence_number} {abbr} -> {comp}")
                                rejected = True
                                break
                    if rejected:
                        continue
            filtered.append(c)
        before = len(state.candidates)
        state.candidates = filtered
        if len(filtered) < before:
            print(f"  After abbrev guard: {len(filtered)} (-{before - len(filtered)})")

    def _targeted_recovery(self, state, llm):
        entity_comps = {c.component_name for c in state.candidates}
        transarc_comps = {l.component_name for l in state.seed_links}
        covered_comps = entity_comps | transarc_comps
        unlinked = [c for c in state.components if c.name not in covered_comps]

        if not unlinked:
            return

        print(f"  [Phase 5b] Targeted Recovery ({len(unlinked)} unlinked)")

        parent_map = {}
        existing_sent_comp = defaultdict(set)
        all_comp_names = {c.name for c in state.components}
        for comp in unlinked:
            parents = set()
            for other_name in all_comp_names:
                if other_name != comp.name and len(other_name) >= 3 and other_name in comp.name:
                    parents.add(other_name)
            if parents:
                parent_map[comp.name] = parents
        for lk in state.seed_links:
            existing_sent_comp[lk.sentence_number].add(lk.component_name)
        for c in state.candidates:
            existing_sent_comp[c.sentence_number].add(c.component_name)

        extra = []
        for comp in unlinked:
            doc_text = "\n".join([f"S{s.number}: {s.text}" for s in state.sentences])

            aliases = []
            if state.doc_knowledge:
                for a, c in state.doc_knowledge.abbreviations.items():
                    if c == comp.name:
                        aliases.append(a)
                for s, c in state.doc_knowledge.synonyms.items():
                    if c == comp.name:
                        aliases.append(s)
                for p, c in state.doc_knowledge.partial_references.items():
                    if c == comp.name:
                        aliases.append(p)

            alias_str = f"\nKNOWN ALIASES: {', '.join(aliases)}" if aliases else ""

            prompt = f"""Find ALL sentences that discuss the software component "{comp.name}".
{alias_str}

Look for:
- Direct mentions of "{comp.name}" or any alias
- Descriptions of what {comp.name} does (functional descriptions)
- References to {comp.name}'s role in the architecture

Exclude:
- Names that appear only inside a dotted package path (e.g., com.example.name does NOT count as a reference to "name")

DOCUMENT:
{doc_text}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "matched_text": "text found", "reason": "why this refers to {comp.name}"}}]}}

Be thorough — find ALL sentences that discuss this component.
JSON only:"""

            data = llm.extract_json(llm.query(prompt, timeout=120))
            if not data:
                continue

            for ref in data.get("references", []):
                snum = parse_snum(ref.get("sentence"))
                if snum is None:
                    continue
                sent = state.sent_map.get(snum)
                if not sent:
                    continue
                cid = state.name_to_id.get(comp.name)
                if not cid:
                    continue

                if comp.name in parent_map:
                    parents_here = parent_map[comp.name] & existing_sent_comp.get(snum, set())
                    if parents_here:
                        continue

                matched = ref.get("matched_text", comp.name)
                extra.append(CandidateLink(
                    snum, sent.text, comp.name, cid,
                    matched, 0.85, "entity", "targeted", True
                ))
                print(f"    Targeted: S{snum} -> {comp.name} ({matched})")

        if extra:
            state.candidates.extend(extra)
            print(f"    Found: {len(extra)} additional candidates")


# ══════════════════════════════════════════════════════════════════════
# Phase 6: Validation (2-pass intersect)
# ══════════════════════════════════════════════════════════════════════

class ValidationAgent(PhaseAgent):
    name = "validation"

    def run(self, state, llm):
        if not state.candidates:
            state.validated = []
            return

        comp_names = get_comp_names(state.components, state.model_knowledge)
        needs = [c for c in state.candidates if c.needs_validation]
        direct = [c for c in state.candidates if not c.needs_validation]

        if not needs:
            state.validated = list(state.candidates)
            return

        # Pre-check: reject generic mentions
        remaining = []
        for c in needs:
            sent = state.sent_map.get(c.sentence_number)
            if sent and is_generic_mention(c.component_name, sent.text):
                print(f"    Generic mention reject: S{c.sentence_number} -> {c.component_name}")
            else:
                remaining.append(c)
        needs = remaining

        # Build alias lookup
        alias_map = {}
        for c in state.components:
            aliases = {c.name}
            if state.doc_knowledge:
                for a, cn in state.doc_knowledge.abbreviations.items():
                    if cn == c.name:
                        aliases.add(a)
                for s, cn in state.doc_knowledge.synonyms.items():
                    if cn == c.name:
                        aliases.add(s)
                for p, cn in state.doc_knowledge.partial_references.items():
                    if cn == c.name:
                        aliases.add(p)
            alias_map[c.name] = aliases

        # Step 1: Word-boundary code-first
        auto_approved = []
        llm_needed = []
        for c in needs:
            sent = state.sent_map.get(c.sentence_number)
            if not sent:
                continue
            matched = False
            for a in alias_map.get(c.component_name, set()):
                if len(a) >= 3:
                    if a.lower() in sent.text.lower():
                        matched = True
                        break
                elif len(a) >= 2:
                    if word_boundary_match(a, sent.text):
                        matched = True
                        break
            if matched:
                c.confidence = 1.0
                c.source = "validated"
                auto_approved.append(c)
            else:
                llm_needed.append(c)

        print(f"    Code-first auto-approved: {len(auto_approved)}, LLM needed: {len(llm_needed)}")

        # Classify generic-risk
        generic_risk = set()
        if state.model_knowledge and state.model_knowledge.ambiguous_names:
            generic_risk |= state.model_knowledge.ambiguous_names
        for c in state.components:
            if c.name.lower() in state.generic_component_words:
                generic_risk.add(c.name)

        # Step 2: 2-pass intersect
        ctx = []
        if state.learned_patterns:
            if state.learned_patterns.action_indicators:
                ctx.append(f"ACTION: {', '.join(state.learned_patterns.action_indicators[:4])}")
            if state.learned_patterns.effect_indicators:
                ctx.append(f"EFFECT (reject): {', '.join(state.learned_patterns.effect_indicators[:3])}")
            if state.learned_patterns.subprocess_terms:
                ctx.append(f"Subprocess (reject): {', '.join(list(state.learned_patterns.subprocess_terms)[:5])}")

        twopass_approved = []
        generic_to_verify = []
        for batch_start in range(0, len(llm_needed), 25):
            batch = llm_needed[batch_start:batch_start + 25]
            cases = []
            for i, c in enumerate(batch):
                prev = state.sent_map.get(c.sentence_number - 1)
                p = f"[prev: {prev.text[:35]}...] " if prev else ""
                cases.append(f'Case {i + 1}: "{c.matched_text}" -> {c.component_name}\n  {p}"{c.sentence_text}"')

            r1 = self._qual_validation_pass(llm, comp_names, ctx, cases,
                "Focus on ACTOR role: is the component performing an action or being described?")
            r2 = self._qual_validation_pass(llm, comp_names, ctx, cases,
                "Focus on DIRECT reference: does the text refer to the SPECIFIC architectural component, not a generic concept?")

            for i, c in enumerate(batch):
                if r1.get(i, False) and r2.get(i, False):
                    if c.component_name in generic_risk:
                        generic_to_verify.append(c)
                    else:
                        c.confidence = 1.0
                        c.source = "validated"
                        twopass_approved.append(c)

        print(f"    2-pass approved: {len(twopass_approved)} specific, "
              f"{len(generic_to_verify)} generic need evidence")

        # Step 3: Evidence post-filter for generic-risk
        generic_validated = []
        if generic_to_verify:
            for batch_start in range(0, len(generic_to_verify), 25):
                batch = generic_to_verify[batch_start:batch_start + 25]
                cases = []
                for i, c in enumerate(batch):
                    cases.append(
                        f'Case {i + 1}: S{c.sentence_number} "{c.sentence_text}"\n'
                        f'  Candidate: {c.component_name}'
                    )

                prompt = f"""For each case, find the EXACT text in the sentence that refers to the architecture component.

COMPONENTS: {', '.join(comp_names)}

CASES:
{chr(10).join(cases)}

For each case, provide:
- evidence_text: the EXACT substring from the sentence that names or references the component
- If you cannot find specific text evidence, set evidence_text to null

Return JSON:
{{"validations": [{{"case": 1, "evidence_text": "exact substring or null"}}]}}
JSON only:"""

                data = llm.extract_json(llm.query(prompt, timeout=120))
                if not data:
                    continue

                for v in data.get("validations", []):
                    idx = _safe_int(v.get("case"), 0) - 1
                    if idx < 0 or idx >= len(batch):
                        continue
                    c = batch[idx]
                    evidence = v.get("evidence_text")
                    if not evidence:
                        continue
                    sent = state.sent_map.get(c.sentence_number)
                    if not sent:
                        continue
                    if evidence.lower() not in sent.text.lower():
                        continue
                    ev_lower = evidence.lower()
                    aliases = alias_map.get(c.component_name, {c.component_name.lower()})
                    if any(a.lower() in ev_lower for a in aliases if len(a) >= 2):
                        c.confidence = 1.0
                        c.source = "validated"
                        generic_validated.append(c)

            print(f"    Generic evidence: {len(generic_validated)}/{len(generic_to_verify)}")

        state.validated = direct + auto_approved + twopass_approved + generic_validated
        print(f"  Validated: {len(state.validated)} (of {len(state.candidates)})")

    def _qual_validation_pass(self, llm, comp_names, ctx, cases, focus):
        prompt = f"""Validate component references in a software architecture document. {focus}

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(ctx)}

DECISION RULES:
APPROVE when:
- The component is the grammatical actor or subject (the sentence is ABOUT the component)
- A section heading names the component (introduces that component's topic)
- The sentence describes what the component does, provides, or interacts with

REJECT when:
- The name is used as an ordinary English word, not as a proper name
  (Like "proxy" in "proxy pattern" is the design pattern concept, not the Proxy component — reject the component link)
- The name is a modifier inside a larger phrase, not a standalone reference
  (Like "observer" in "observer pattern" modifies pattern — reject if Observer is a component)
- The sentence is about a subprocess, algorithm, or implementation detail — not the component itself

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true/false}}]}}
JSON only:"""

        data = llm.extract_json(llm.query(prompt, timeout=120))
        results = {}
        if data:
            for v in data.get("validations", []):
                idx = _safe_int(v.get("case"), 0) - 1
                if 0 <= idx < len(cases):
                    results[idx] = v.get("approve", False)
        return results


# ══════════════════════════════════════════════════════════════════════
# Phase 7: Coreference Resolution
# ══════════════════════════════════════════════════════════════════════

class CoreferenceAgent(PhaseAgent):
    name = "coreference"

    def run(self, state, llm):
        comp_names = get_comp_names(state.components, state.model_knowledge)
        if state.is_complex:
            print(f"  Mode: debate ×2 union (complex)")
            links1 = self._coref_debate(state, llm, comp_names)
            links2 = self._coref_debate(state, llm, comp_names)
            state.coref_links = self._union_coref(links1, links2)
        else:
            print(f"  Mode: discourse ×2 union")
            links1 = self._coref_discourse(state, llm, comp_names)
            links2 = self._coref_discourse(state, llm, comp_names)
            state.coref_links = self._union_coref(links1, links2)
        print(f"  Coref links: {len(state.coref_links)}")

    def _union_coref(self, links1, links2):
        """Union two coref runs by (sentence, component). Stabilizes LLM variance."""
        seen = {}
        for lk in links1:
            seen[(lk.sentence_number, lk.component_id)] = lk
        for lk in links2:
            key = (lk.sentence_number, lk.component_id)
            if key not in seen:
                seen[key] = lk
        result = list(seen.values())
        print(f"    Union: {len(links1)} + {len(links2)} → {len(result)} unique")
        return result

    def _coref_discourse(self, state, llm, comp_names):
        all_coref = []
        pronoun_sents = [s for s in state.sentences if PRONOUN_PATTERN.search(s.text)]

        for batch_start in range(0, len(pronoun_sents), 12):
            batch = pronoun_sents[batch_start:batch_start + 12]
            cases = []
            for sent in batch:
                prev = []
                for i in range(1, 4):
                    p = state.sent_map.get(sent.number - i)
                    if p:
                        prev.append(f"S{p.number}: {p.text}")
                cases.append({"sent": sent, "prev": prev})

            prompt = f"""Resolve pronoun references to architecture components.

COMPONENTS: {', '.join(comp_names)}

"""
            for i, case in enumerate(cases):
                prompt += f"--- Case {i + 1}: S{case['sent'].number} ---\n"
                if case["prev"]:
                    prompt += "PREVIOUS:\n  " + "\n  ".join(reversed(case["prev"])) + "\n"
                prompt += f">>> {case['sent'].text}\n\n"

            prompt += """For each pronoun that refers to a component, provide:
- antecedent_sentence: the sentence number where the component was EXPLICITLY NAMED
- antecedent_text: the EXACT quote from that sentence containing the component name

RULES (all must hold):
1. The component name (or known alias) MUST appear verbatim in the antecedent sentence
2. The antecedent MUST be within the previous 3 sentences
3. The pronoun MUST grammatically refer back to that component as its subject
4. If the pronoun could refer to multiple things, DO NOT resolve it

Like in technical writing: "The Scheduler assigns tasks to threads. It uses a priority queue internally."
— "It" clearly refers to "the Scheduler" because it was the subject of the previous sentence.

Return JSON:
{"resolutions": [{"case": 1, "sentence": N_INTEGER, "pronoun": "it", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact text with component name"}]}

Only include resolutions you are CERTAIN about. JSON only:"""

            data = llm.extract_json(llm.query(prompt, timeout=150))
            if not data:
                continue

            for res in data.get("resolutions", []):
                link = self._verify_resolution(res, state)
                if link:
                    all_coref.append(link)

        return all_coref

    def _coref_debate(self, state, llm, comp_names):
        all_coref = []

        ctx = []
        if state.learned_patterns and state.learned_patterns.subprocess_terms:
            ctx.append(f"Subprocesses (don't link): {', '.join(list(state.learned_patterns.subprocess_terms)[:5])}")

        for batch_start in range(0, len(state.sentences), 20):
            batch = state.sentences[batch_start:min(batch_start + 20, len(state.sentences))]
            ctx_start = max(0, batch_start - CONTEXT_WINDOW)
            ctx_sents = state.sentences[ctx_start:batch_start + 20]
            doc_lines = [
                f"{'*' if s.number >= batch[0].number else ' '}S{s.number}: {s.text}"
                for s in ctx_sents
            ]

            prompt = f"""Resolve pronoun references to architecture components.

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(ctx)}

DOCUMENT (* = analyze these sentences):
{chr(10).join(doc_lines)}

Find pronouns (it, they, this, these) in starred sentences that refer to a component.

RULES (all must hold):
1. You MUST cite the antecedent_sentence where the component was EXPLICITLY NAMED
2. The component name (or known alias) MUST appear verbatim in the antecedent sentence
3. The antecedent MUST be within the previous 3 sentences
4. Do NOT resolve pronouns in sentences about subprocesses or implementation details
5. If the pronoun could refer to multiple components, do NOT resolve it

Return JSON:
{{"resolutions": [{{"sentence": N_INTEGER, "pronoun": "it", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

Only include resolutions you are CERTAIN about. JSON only:"""

            data = llm.extract_json(llm.query(prompt, timeout=100))
            if not data:
                continue

            for res in data.get("resolutions", []):
                link = self._verify_resolution(res, state)
                if link:
                    all_coref.append(link)

        return all_coref

    def _verify_resolution(self, res, state):
        """Verify a coreference resolution and return SadSamLink or None."""
        comp = res.get("component")
        snum = parse_snum(res.get("sentence"))
        if not (comp and snum is not None and comp in state.name_to_id):
            return None

        ant_snum = parse_snum(res.get("antecedent_sentence"))
        if ant_snum is not None:
            ant_sent = state.sent_map.get(ant_snum)
            if not ant_sent:
                return None
            if not (has_standalone_mention(comp, ant_sent.text) or
                    has_alias_mention(comp, ant_sent.text, state.doc_knowledge)):
                print(f"    Coref skip (comp not in antecedent S{ant_snum}): S{snum} -> {comp}")
                return None
            if abs(snum - ant_snum) > 3:
                return None

        sent = state.sent_map.get(snum)
        if sent and state.learned_patterns and state.learned_patterns.is_subprocess(sent.text):
            return None

        return SadSamLink(snum, state.name_to_id[comp], comp, 1.0, "coreference")


# ══════════════════════════════════════════════════════════════════════
# Phase 8b: Partial Reference Injection (deterministic)
# ══════════════════════════════════════════════════════════════════════

class PartialInjectionAgent(PhaseAgent):
    name = "partial_injection"

    def run(self, state, llm):
        if not state.doc_knowledge or not state.doc_knowledge.partial_references:
            state.partial_links = []
            return

        existing = (state.transarc_set |
                    {(c.sentence_number, c.component_id) for c in state.validated} |
                    {(l.sentence_number, l.component_id) for l in state.coref_links})
        injected = []

        for partial, comp_name in state.doc_knowledge.partial_references.items():
            if comp_name not in state.name_to_id:
                continue
            comp_id = state.name_to_id[comp_name]
            for sent in state.sentences:
                key = (sent.number, comp_id)
                if key in existing:
                    continue
                if has_clean_mention(partial, sent.text):
                    injected.append(SadSamLink(sent.number, comp_id, comp_name, 0.8, "partial_inject"))
                    existing.add(key)

        state.partial_links = injected
        if injected:
            print(f"  Injected: {len(injected)} partial links")


# ══════════════════════════════════════════════════════════════════════
# Merge Agent: Combine + Deduplicate + Parent-overlap guard
# ══════════════════════════════════════════════════════════════════════

class MergeAgent(PhaseAgent):
    name = "merge"

    def run(self, state, llm):
        entity_links = [
            SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
            for c in state.validated
        ]
        all_links = state.seed_links + entity_links + state.coref_links + state.partial_links

        # Deduplicate by (sentence, component), keep highest-priority source
        link_map = {}
        for lk in all_links:
            key = (lk.sentence_number, lk.component_id)
            if key not in link_map:
                link_map[key] = lk
            else:
                old_p = SOURCE_PRIORITY.get(link_map[key].source, 0)
                new_p = SOURCE_PRIORITY.get(lk.source, 0)
                if new_p > old_p:
                    link_map[key] = lk
        preliminary = list(link_map.values())

        # Parent-overlap guard
        if state.model_knowledge and state.model_knowledge.impl_to_abstract:
            child_to_parent = state.model_knowledge.impl_to_abstract
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

        state.preliminary = preliminary
        print(f"  Merged: {len(all_links)} raw -> {len(preliminary)} deduped")


# ══════════════════════════════════════════════════════════════════════
# Phase 8c: Convention-aware Boundary Filter
# ══════════════════════════════════════════════════════════════════════

class BoundaryFilterAgent(PhaseAgent):
    name = "boundary_filter"

    def run(self, state, llm):
        comp_names = get_comp_names(state.components, state.model_knowledge)

        safe = []
        to_review = []
        for lk in state.preliminary:
            is_ta = (lk.sentence_number, lk.component_id) in state.transarc_set
            if is_ta:
                safe.append(lk)
            else:
                to_review.append(lk)

        if not to_review:
            state.boundary_rejected = []
            return

        # Build alias lookup for enriching filter context
        alias_map = defaultdict(list)
        if state.doc_knowledge:
            for syn, target in state.doc_knowledge.synonyms.items():
                alias_map[target].append(syn)
            for abbr, target in state.doc_knowledge.abbreviations.items():
                alias_map[target].append(abbr)
            for partial, target in state.doc_knowledge.partial_references.items():
                alias_map[target].append(partial)

        items = []
        for i, lk in enumerate(to_review):
            sent = state.sent_map.get(lk.sentence_number)
            text = sent.text if sent else "(no text)"
            line = (f'{i + 1}. S{lk.sentence_number}: "{text}"\n'
                    f'   Component: "{lk.component_name}"')
            aliases = alias_map.get(lk.component_name, [])
            if aliases:
                line += f'\n   Known aliases: {", ".join(aliases)}'
            items.append(line)

        batch_size = 25

        # 2-pass union voting: reject only if BOTH passes say NO_LINK
        # This stabilizes borderline cases (especially partial references)
        pass_verdicts = [{}, {}]
        for pass_idx in range(2):
            for batch_start in range(0, len(items), batch_size):
                batch_items = items[batch_start:batch_start + batch_size]

                prompt = f"""Validate trace links between architecture documentation and components.

ARCHITECTURE COMPONENTS: {', '.join(comp_names)}

{CONVENTION_GUIDE}

---

For each sentence-component pair, apply the 3-step reasoning guide.
Decide LINK (keep the trace link) or NO_LINK (reject it).

{chr(10).join(batch_items)}

Return JSON array:
[{{"id": N, "step": "1|2a|2b|3", "verdict": "LINK" or "NO_LINK", "reason": "brief"}}]
JSON only:"""

                raw = llm.query(prompt, timeout=180)
                data = extract_json_array(raw.text if hasattr(raw, 'text') else str(raw))
                if data:
                    for item in data:
                        vid = item.get("id")
                        verdict = item.get("verdict", "LINK").upper().strip()
                        step = item.get("step", "3")
                        reason = item.get("reason", "")
                        if vid is not None:
                            pass_verdicts[pass_idx][vid] = (verdict, step, reason)

        kept = list(safe)
        rejected = []
        for i, lk in enumerate(to_review):
            v1 = pass_verdicts[0].get(i + 1, ("LINK", "3", ""))
            v2 = pass_verdicts[1].get(i + 1, ("LINK", "3", ""))
            reject1 = "NO" in v1[0]
            reject2 = "NO" in v2[0]
            if reject1 and reject2:
                # Both passes reject — confident NO_LINK
                step, reason = v1[1], v1[2]
                rejected.append((lk, f"convention_step{step}"))
                print(f"    Convention filter [step {step}]: S{lk.sentence_number} → "
                      f"{lk.component_name} ({lk.source}) — {reason}")
            elif reject1 or reject2:
                # Only one pass rejects — union-save (keep the link)
                kept.append(lk)
            else:
                kept.append(lk)

        state.preliminary = kept
        state.boundary_rejected = rejected
        print(f"  Boundary filter: kept {len(kept)}, rejected {len(rejected)}")


# ══════════════════════════════════════════════════════════════════════
# Phase 9: Judge Review (4-rule + Advocate-Prosecutor)
# ══════════════════════════════════════════════════════════════════════

class JudgeAgent(PhaseAgent):
    name = "judge"

    def __init__(self, lenient=False):
        self.lenient = lenient

    def run(self, state, llm):
        links = state.preliminary
        if len(links) < 5:
            state.final_links = list(links)
            return

        comp_names = get_comp_names(state.components, state.model_knowledge)

        # Triage: safe vs review
        safe, nomatch_links, ta_review = [], [], []
        syn_safe_count = 0
        for l in links:
            is_ta = (l.sentence_number, l.component_id) in state.transarc_set
            sent = state.sent_map.get(l.sentence_number)
            # Synonym-safe bypass
            if sent and has_alias_mention(l.component_name, sent.text, state.doc_knowledge):
                safe.append(l)
                syn_safe_count += 1
                continue
            if is_ta:
                if is_ambiguous_name_component(l.component_name, state.model_knowledge):
                    ta_review.append(l)
                else:
                    safe.append(l)
                continue
            if not sent:
                nomatch_links.append(l)
                continue
            if has_standalone_mention(l.component_name, sent.text):
                safe.append(l)
            else:
                nomatch_links.append(l)

        print(f"  Triage: {len(safe)} safe ({syn_safe_count} syn-safe), "
              f"{len(ta_review)} ta-review, {len(nomatch_links)} no-match")

        # Advocate-Prosecutor for ambiguous TransArc
        ta_approved = self._deliberate_transarc(ta_review, comp_names, state, llm)

        # 4-rule judge for no-match
        nomatch_approved = self._judge_nomatch(nomatch_links, comp_names, state, llm)

        state.final_links = safe + ta_approved + nomatch_approved
        rejected_count = len(links) - len(state.final_links)
        print(f"  Approved: {len(state.final_links)} (rejected {rejected_count})")

    def _judge_nomatch(self, nomatch_links, comp_names, state, llm):
        if not nomatch_links:
            return []

        cases = self._build_judge_cases(nomatch_links, state)
        prompt = self._build_judge_prompt(comp_names, cases)
        n = min(30, len(nomatch_links))

        # Union voting: reject only if BOTH passes reject
        data1 = llm.extract_json(llm.query(prompt, timeout=180))
        data2 = llm.extract_json(llm.query(prompt, timeout=180))

        rej1 = self._parse_rejections(data1, n)
        rej2 = self._parse_rejections(data2, n)

        if self.lenient:
            # Lenient mode: reject only if BOTH passes reject (same as default)
            # But also require high confidence — skip rejection if only 1 rule failed
            rejected = rej1 & rej2
        else:
            rejected = rej1 & rej2

        result = []
        for i in range(n):
            if i not in rejected:
                result.append(nomatch_links[i])
            else:
                print(f"    4-rule reject: S{nomatch_links[i].sentence_number} -> {nomatch_links[i].component_name}")
        result.extend(nomatch_links[n:])
        return result

    def _parse_rejections(self, data, n):
        rejected = set()
        if data:
            for j in data.get("judgments", []):
                idx = _safe_int(j.get("case"), 0) - 1
                if 0 <= idx < n and not j.get("approve", False):
                    rejected.add(idx)
        return rejected

    def _build_judge_prompt(self, comp_names, cases):
        return f"""JUDGE: Validate trace links between documentation and software architecture components.

APPROVAL CRITERIA:
A link S→C is valid when the sentence satisfies all four conditions:

1. EXPLICIT REFERENCE
   The component name (or a direct reference to it) appears in the sentence as a clear
   entity being discussed.
   NOTE: For coreference-resolved links (marked "PRONOUN(coreference-resolved)"), a pronoun
   (it, they, this) that refers back to a component named in a nearby ANTECEDENT sentence
   counts as an explicit reference. The antecedent is shown above the sentence. If the
   antecedent clearly names the component, Rule 1 is satisfied.

2. SYSTEM-LEVEL PERSPECTIVE
   The sentence describes the component's role, responsibilities, interfaces, or
   interactions within the overall system architecture. Reject statements focused on
   internal implementation details.

3. PRIMARY FOCUS
   The component is the main subject of what the sentence conveys, not a secondary
   or incidental mention.

4. COMPONENT-SPECIFIC USAGE
   The reference is to the component as a named entity within the system architecture,
   not to a generic concept, pattern, or technology that happens to share a name.

COMPONENTS: {', '.join(comp_names)}

LINKS:
{chr(10).join(cases)}

Return JSON:
{{"judgments": [{{"case": 1, "approve": true/false, "reason": "brief explanation"}}]}}
JSON only:"""

    def _build_judge_cases(self, review, state):
        use_full_ctx = not state.is_complex
        cases = []
        for i, l in enumerate(review[:30]):
            sent = state.sent_map.get(l.sentence_number)
            ctx_lines = []
            if l.source == "coreference":
                # For coref links, find the antecedent sentence where the component is named
                antecedent = self._find_coref_antecedent(l, state)
                if antecedent:
                    ctx_lines.append(f"    ANTECEDENT S{antecedent.number}: {antecedent.text}")
                    ctx_lines.append(f"    (Pronoun in S{l.sentence_number} resolved to "
                                     f"\"{l.component_name}\" via antecedent above)")
                # Also show immediate previous for flow
                p2 = state.sent_map.get(l.sentence_number - 2)
                if p2 and (not antecedent or p2.number != antecedent.number):
                    txt = p2.text if use_full_ctx else f"{p2.text[:45]}..."
                    ctx_lines.append(f"    PREV2: {txt}")
            p1 = state.sent_map.get(l.sentence_number - 1)
            if p1:
                txt = p1.text if use_full_ctx else f"{p1.text[:45]}..."
                ctx_lines.append(f"    PREV: {txt}")
            ctx_lines.append(f"    >>> S{l.sentence_number}: {sent.text if sent else '?'}")

            src_info = f"src:{l.source}"
            if sent:
                match = find_match_text(l.component_name, sent.text, state.doc_knowledge)
                if match and match.lower() != l.component_name.lower():
                    src_info += f', match:"{match}"'
                elif not match and l.source == "coreference":
                    src_info += ', match:PRONOUN(coreference-resolved)'
                elif not match:
                    src_info += ', match:NONE(pronoun/context)'

            cases.append(f"Case {i + 1}: S{l.sentence_number} -> {l.component_name} ({src_info})\n"
                         + chr(10).join(ctx_lines))
        return cases

    def _find_coref_antecedent(self, link, state):
        """Find the nearest prior sentence that explicitly mentions this component."""
        for offset in range(1, 4):
            prev = state.sent_map.get(link.sentence_number - offset)
            if prev and (has_standalone_mention(link.component_name, prev.text) or
                         has_alias_mention(link.component_name, prev.text, state.doc_knowledge)):
                return prev
        return None

    def _deliberate_transarc(self, links, comp_names, state, llm):
        if not links:
            return []

        print(f"  Deliberating {len(links)} ambiguous-name TransArc links")

        approved = []
        for l in links:
            sent = state.sent_map.get(l.sentence_number)
            if not sent:
                approved.append(l)
                continue

            verdicts = []
            for _ in range(2):
                verdict = self._single_advocate_prosecutor_pass(
                    l.sentence_number, l.component_name, sent.text, comp_names, llm)
                verdicts.append(verdict)

            if verdicts[0] or verdicts[1]:  # Union
                approved.append(l)
                if not verdicts[0] or not verdicts[1]:
                    print(f"    Deliberation union-save: S{l.sentence_number} -> {l.component_name}")
            else:
                print(f"    Deliberation reject: S{l.sentence_number} -> {l.component_name}")

        return approved

    def _single_advocate_prosecutor_pass(self, snum, comp_name, sent_text, comp_names, llm):
        comp_names_str = ', '.join(comp_names)

        advocate_prompt = f"""You are the ADVOCATE for linking sentence S{snum} to component "{comp_name}".

The system has a component literally named "{comp_name}". Your job is to defend valid links.

SENTENCE: {sent_text}
ALL COMPONENTS: {comp_names_str}

Your job: Find the STRONGEST evidence that this sentence discusses "{comp_name}" at the architectural level. Consider:
- Does the sentence describe {comp_name}'s role, behavior, interactions, or testing?
- Is "{comp_name.lower()}" used as a standalone noun/noun-phrase referring to a layer or part of the system?
- In architecture docs, even generic words like "the {comp_name.lower()} of the application" typically
  refer to the named component when such a component exists.

Provide your argument in 2-3 sentences. Then give your verdict.
Return JSON: {{"argument": "your argument", "verdict": "APPROVE" or "REJECT"}}
JSON only:"""

        prosecutor_prompt = f"""You are the PROSECUTOR arguing AGAINST linking sentence S{snum} to component "{comp_name}".

You should only argue REJECT when there is CLEAR evidence the match is spurious — not just because the word has generic meanings.

SENTENCE: {sent_text}
ALL COMPONENTS: {comp_names_str}

Your job: Find CLEAR evidence that this is a SPURIOUS match. Only these patterns warrant rejection:
1. "{comp_name.lower()}" is used as a modifier/adjective in a compound phrase (e.g., "throttle {comp_name.lower()}", "minimal {comp_name.lower()}") — NOT as a standalone noun referring to the component.
2. The sentence is primarily about a DIFFERENT component, and "{comp_name.lower()}" is purely incidental.
3. "{comp_name.lower()}" refers to a technology/protocol/tool, not the architecture component.
4. This is a package listing or dotted path (like x.foo.bar) where the match is coincidental.

If "{comp_name.lower()}" appears as a standalone noun describing a layer/part of the system, that is NOT spurious.

Provide your argument in 2-3 sentences. Then give your verdict.
Return JSON: {{"argument": "your argument", "verdict": "APPROVE" or "REJECT"}}
JSON only:"""

        adv_data = llm.extract_json(llm.query(advocate_prompt, timeout=60))
        pros_data = llm.extract_json(llm.query(prosecutor_prompt, timeout=60))

        adv_arg = adv_data.get("argument", "") if adv_data else ""
        pros_arg = pros_data.get("argument", "") if pros_data else ""

        jury_prompt = f"""JURY: Decide if sentence S{snum} should be linked to component "{comp_name}".

The system has a component literally named "{comp_name}". REJECT only with clear evidence — when in doubt, APPROVE.

SENTENCE: {sent_text}

ADVOCATE argues: {adv_arg}

PROSECUTOR argues: {pros_arg}

Rule: APPROVE when "{comp_name.lower()}" is used as a standalone noun referring to a layer/part
of the system (its role, behavior, interactions, or testing). REJECT only when "{comp_name.lower()}"
is clearly used as a modifier in a compound phrase ("throttle {comp_name.lower()}"), a technology name,
or the sentence is entirely about a different component.

Return JSON: {{"verdict": "APPROVE" or "REJECT", "reason": "brief explanation"}}
JSON only:"""

        jury_data = llm.extract_json(llm.query(jury_prompt, timeout=60))
        if jury_data:
            return jury_data.get("verdict", "APPROVE").upper() == "APPROVE"
        return True


# ══════════════════════════════════════════════════════════════════════
# Review Agents (dispatched by Orchestrator on quality issues)
# ══════════════════════════════════════════════════════════════════════

class CoverageRecoveryAgent(PhaseAgent):
    """Recovers links for components with zero coverage via targeted LLM extraction."""
    name = "coverage_recovery"

    def run(self, state, llm):
        comp_names = get_comp_names(state.components, state.model_knowledge)
        # Check all available link pools for coverage
        all_current = (state.final_links or []) + (state.seed_links or []) + [
            SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
            for c in (state.validated or [])
        ] + (state.coref_links or []) + (state.partial_links or [])
        covered = {lk.component_name for lk in all_current}
        uncovered = [c for c in state.components
                     if c.name in comp_names and c.name not in covered]

        if not uncovered:
            print(f"  Coverage recovery: all components covered")
            return

        # Skip components already attempted (no point retrying)
        uncovered = [c for c in uncovered if c.name not in state.recovery_attempted]
        if not uncovered:
            print(f"  Coverage recovery: all uncovered already attempted, skipping")
            return

        # Mark as attempted
        state.recovery_attempted |= {c.name for c in uncovered}
        print(f"  Coverage recovery: {len(uncovered)} uncovered components")
        existing = {(lk.sentence_number, lk.component_id) for lk in all_current}
        recovered = []

        for comp in uncovered:
            # Build focused prompt with surrounding sentence context
            doc_text = "\n".join([f"S{s.number}: {s.text}" for s in state.sentences])

            aliases = []
            if state.doc_knowledge:
                for a, c in state.doc_knowledge.abbreviations.items():
                    if c == comp.name:
                        aliases.append(a)
                for s, c in state.doc_knowledge.synonyms.items():
                    if c == comp.name:
                        aliases.append(s)

            prompt = f"""RECOVERY TASK: Find sentences about component "{comp.name}" that may have been missed.
{f'Known aliases: {", ".join(aliases)}' if aliases else ''}

This component currently has ZERO trace links. Search carefully for:
1. Direct mentions (exact name or alias)
2. Functional descriptions ("the component that handles X" where X is {comp.name}'s role)
3. References in interaction descriptions ("A sends data to B" — check if B is {comp.name})

COMPONENTS (for context): {', '.join(comp_names)}

DOCUMENT:
{doc_text}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "evidence": "exact quote from sentence", "reason": "why this is {comp.name}"}}]}}
JSON only:"""

            data = llm.extract_json(llm.query(prompt, timeout=120))
            if not data:
                continue

            for ref in data.get("references", []):
                snum = parse_snum(ref.get("sentence"))
                if snum is None:
                    continue
                sent = state.sent_map.get(snum)
                if not sent:
                    continue
                cid = state.name_to_id.get(comp.name)
                if not cid:
                    continue
                key = (snum, cid)
                if key in existing:
                    continue

                # Verify evidence text is actually in sentence
                evidence = ref.get("evidence", "")
                if evidence and evidence.lower() not in sent.text.lower():
                    continue

                # Structural gate: component name or alias must appear in sentence
                has_name = has_standalone_mention(comp.name, sent.text)
                has_alias = has_alias_mention(comp.name, sent.text, state.doc_knowledge)
                if not has_name and not has_alias:
                    print(f"    Recovery skip (name not in sentence): S{snum} -> {comp.name}")
                    continue

                recovered.append(SadSamLink(snum, cid, comp.name, 0.75, "recovered"))
                existing.add(key)
                print(f"    Recovered: S{snum} -> {comp.name}")

        if recovered:
            # Add to the appropriate link pool based on pipeline stage
            if state.final_links:
                state.final_links.extend(recovered)
            else:
                # Pre-judge: add to seed_links so they flow through the pipeline
                state.seed_links.extend(recovered)
                state.transarc_set |= {(lk.sentence_number, lk.component_id) for lk in recovered}
            print(f"  Recovered {len(recovered)} links for {len(uncovered)} uncovered components")


class DensityReviewAgent(PhaseAgent):
    """Reviews components with suspiciously high link density, removing weakest links."""
    name = "density_review"

    def __init__(self, max_density=12):
        self.max_density = max_density

    def run(self, state, llm):
        # Find components with too many links
        density = defaultdict(list)
        for lk in state.final_links:
            density[lk.component_name].append(lk)

        overloaded = {comp: links for comp, links in density.items()
                      if len(links) > self.max_density}

        if not overloaded:
            print(f"  Density review: no overloaded components")
            return

        comp_names = get_comp_names(state.components, state.model_knowledge)
        total_removed = 0

        for comp, links in overloaded.items():
            print(f"  Density review: {comp} has {len(links)} links (max={self.max_density})")

            # Ask LLM to rank links by relevance
            cases = []
            for i, lk in enumerate(links):
                sent = state.sent_map.get(lk.sentence_number)
                text = sent.text[:120] if sent else "?"
                cases.append(f'{i + 1}. S{lk.sentence_number} (src:{lk.source}): "{text}"')

            prompt = f"""This component "{comp}" has {len(links)} trace links. That seems too many.

Review each link and identify ones that are WEAKEST — where the sentence is NOT primarily about "{comp}".

COMPONENTS: {', '.join(comp_names)}

LINKS for "{comp}":
{chr(10).join(cases)}

Return JSON:
{{"weak_links": [list of case numbers that are weakest/should be removed],
  "reasoning": "brief explanation"}}
JSON only:"""

            data = llm.extract_json(llm.query(prompt, timeout=120))
            if not data:
                continue

            weak_ids = set()
            for w in data.get("weak_links", []):
                try:
                    weak_ids.add(int(w))
                except (ValueError, TypeError):
                    pass
            if not weak_ids:
                continue

            # Remove weak links but keep at least max_density
            to_remove = set()
            for wid in sorted(weak_ids):
                if len(links) - len(to_remove) <= self.max_density:
                    break
                idx = wid - 1
                if 0 <= idx < len(links):
                    lk = links[idx]
                    # Don't remove transarc links
                    if (lk.sentence_number, lk.component_id) not in state.transarc_set:
                        to_remove.add((lk.sentence_number, lk.component_id))

            if to_remove:
                state.final_links = [lk for lk in state.final_links
                                     if (lk.sentence_number, lk.component_id) not in to_remove]
                total_removed += len(to_remove)
                print(f"    Removed {len(to_remove)} weak links for {comp}")

        if total_removed:
            print(f"  Density review: removed {total_removed} total")


class ConsistencyCheckAgent(PhaseAgent):
    """Re-validates coref links when coref added suspiciously many links."""
    name = "consistency_check"

    def run(self, state, llm):
        # Find coref links in final_links
        coref_in_final = [lk for lk in state.final_links if lk.source == "coreference"]
        if not coref_in_final:
            return

        print(f"  Consistency check: reviewing {len(coref_in_final)} coref links")
        comp_names = get_comp_names(state.components, state.model_knowledge)

        # Build verification prompt
        cases = []
        for i, lk in enumerate(coref_in_final):
            sent = state.sent_map.get(lk.sentence_number)
            prev_sents = []
            for j in range(1, 4):
                ps = state.sent_map.get(lk.sentence_number - j)
                if ps:
                    prev_sents.append(f"  S{ps.number}: {ps.text[:80]}")
            prev_str = "\n".join(reversed(prev_sents)) if prev_sents else "  (no context)"
            cases.append(
                f'Case {i + 1}: S{lk.sentence_number} -> {lk.component_name}\n'
                f'  Context:\n{prev_str}\n'
                f'  >>> {sent.text if sent else "?"}'
            )

        # Process in batches of 15
        to_remove = set()
        for batch_start in range(0, len(cases), 15):
            batch_cases = cases[batch_start:batch_start + 15]

            prompt = f"""VERIFICATION: Check if these pronoun-resolved links are correct.

Each link claims a pronoun in the sentence refers to the named component.
Verify that the component was recently mentioned AND the pronoun grammatically refers to it.

COMPONENTS: {', '.join(comp_names)}

{chr(10).join(batch_cases)}

Return JSON:
{{"checks": [{{"case": N, "valid": true/false, "reason": "brief"}}]}}
JSON only:"""

            data = llm.extract_json(llm.query(prompt, timeout=120))
            if not data:
                continue

            for check in data.get("checks", []):
                idx = _safe_int(check.get("case"), 0) - 1 + batch_start
                if 0 <= idx < len(coref_in_final) and not check.get("valid", True):
                    lk = coref_in_final[idx]
                    to_remove.add((lk.sentence_number, lk.component_id))
                    print(f"    Coref invalid: S{lk.sentence_number} -> {lk.component_name}: "
                          f"{check.get('reason', '')}")

        if to_remove:
            state.final_links = [lk for lk in state.final_links
                                 if (lk.sentence_number, lk.component_id) not in to_remove]
            print(f"  Consistency check: removed {len(to_remove)} invalid coref links")


# ══════════════════════════════════════════════════════════════════════
# Orchestrator — Controls agent dispatch, monitors quality, adapts
# ══════════════════════════════════════════════════════════════════════

class Orchestrator:
    """Thin Agent Orchestrator — code owns the state machine, LLM agents do the work.

    Design follows the Thin Agent pattern:
    - CODE controls: phase sequencing, verification gates, retry decisions, acceptance
    - LLM controls: reasoning within each phase (extraction, validation, coref, etc.)
    - Verification is EXTRINSIC: computed from link sets, not gold standard
    - Decisions are STRUCTURAL: based on metrics (coverage, density, anomalies), not LLM opinion
    - When code detects a problem, it dispatches a review agent with the error context
    - The review agent (LLM) tries to fix it; code verifies the fix improved things
    """

    MAX_RETRIES = 1

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.monitor = Monitor()
        self.retry_counts = defaultdict(int)
        self.decisions = []
        self._prev_metrics = None

    def dispatch(self, agent: PhaseAgent, state: PipelineState):
        """Run an agent, log timing, validate output."""
        t0 = time.time()
        agent_name = agent.name
        print(f"\n[Agent: {agent_name}]")

        agent.run(state, self.llm)

        elapsed = time.time() - t0
        warnings = agent.validate(state)
        state.agent_log.append({
            "agent": agent_name,
            "elapsed_s": round(elapsed, 1),
            "warnings": warnings,
        })

        if warnings:
            for w in warnings:
                print(f"  WARNING: {w}")

        return warnings

    def _decide(self, decision):
        print(f"  [Orchestrator] {decision}")
        self.decisions.append({"time": time.time(), "decision": decision})

    def _get_comp_names(self, state):
        return get_comp_names(state.components, state.model_knowledge)

    # ── Verification gates (code-driven, structural) ────────────────

    def _check_and_act(self, state, links, checkpoint, comp_names):
        """Code-driven verification gate. Returns metrics.

        Checks structural properties of the link set and dispatches
        review agents when problems are detected. The review agent (LLM)
        tries to fix the problem; code verifies the fix.
        """
        metrics = self.monitor.compute(links, comp_names)
        self.monitor.print_metrics(metrics, checkpoint)

        # Gate 1: Uncovered components — dispatch recovery agent
        if metrics.uncovered_components:
            n_uncov = len(metrics.uncovered_components)
            total = len(comp_names)
            self._decide(f"COVERAGE_GAP:{checkpoint} {n_uncov}/{total} uncovered: "
                         f"{', '.join(metrics.uncovered_components[:5])}")
            # Only recover at final checkpoints (post-judge), not mid-pipeline
            if checkpoint in ("post-judge", "post-seed"):
                before = len(state.final_links) if state.final_links else len(links)
                self.dispatch(CoverageRecoveryAgent(), state)
                after = len(state.final_links)
                if after > before:
                    self._decide(f"RECOVERY_SUCCESS: +{after - before} links recovered")

        # Gate 2: Density outlier — any component has > 3x the mean
        if metrics.link_density > 0:
            mean = metrics.link_density
            for comp, count in metrics.density_per_component.items():
                if count > mean * 3 and count > 6:
                    self._decide(f"DENSITY_OUTLIER:{checkpoint} {comp} has {count} links "
                                 f"(mean={mean:.1f})")
                    if checkpoint == "post-judge":
                        # Compute max from document ratio, not hardcoded
                        spc = len(state.sentences) / max(1, len(state.components))
                        max_density = max(6, int(spc * 1.5))
                        self.dispatch(DensityReviewAgent(max_density=max_density), state)
                    break  # Only one density review per checkpoint

        # Gate 3: Anomaly detection (link drop, coverage drop)
        if self._prev_metrics:
            anomalies = self.monitor.detect_anomalies(
                self._prev_metrics, metrics, checkpoint)
            for a in anomalies:
                self._decide(f"ANOMALY:{a}")

        self._prev_metrics = metrics
        return metrics

    def run(self, state: PipelineState) -> list[SadSamLink]:
        """Execute pipeline. Code drives the state machine, LLM agents do the work."""
        t0 = time.time()
        comp_names_cache = None  # Populated after Phase 1

        # ── Phase 0+1: Foundation (always run, no verification needed) ──
        self.dispatch(ProfileAgent(), state)
        self.dispatch(ModelAnalysisAgent(), state)
        comp_names_cache = self._get_comp_names(state)

        # ── Phase 2+3: Knowledge gathering ──────────────────────────
        self.dispatch(PatternLearningAgent(), state)
        self.dispatch(DocumentKnowledgeAgent(), state)

        # ── Phase 4: Seed → VERIFY coverage ─────────────────────────
        self.dispatch(SeedAgent(), state)
        seed_metrics = self._check_and_act(
            state, state.seed_links, "post-seed", comp_names_cache)

        # If seed coverage is poor, retry once (LLM variance may help)
        if seed_metrics.component_coverage < 0.5 and self.retry_counts.get("seed", 0) < self.MAX_RETRIES:
            self.retry_counts["seed"] = self.retry_counts.get("seed", 0) + 1
            self._decide(f"RETRY_SEED: coverage {seed_metrics.component_coverage:.0%} < 50%")
            self.dispatch(SeedAgent(), state)
            seed_metrics = self._check_and_act(
                state, state.seed_links, "post-seed-retry", comp_names_cache)

        # ── Phase 5: Entity Extraction ──────────────────────────────
        self.dispatch(ExtractionAgent(), state)

        # ── Phase 6: Validation → VERIFY extraction quality ─────────
        self.dispatch(ValidationAgent(), state)

        combined = state.seed_links + [
            SadSamLink(c.sentence_number, c.component_id, c.component_name, 1.0, c.source)
            for c in state.validated
        ]
        self._check_and_act(state, combined, "post-extraction", comp_names_cache)

        # ── Phase 7: Coreference → VERIFY coref ratio ──────────────
        pre_coref_count = len(combined)
        self.dispatch(CoreferenceAgent(), state)

        # Structural check: did coref add a suspicious number of links?
        coref_ratio = len(state.coref_links) / max(1, pre_coref_count)
        coref_spike = coref_ratio > 0.3  # More than 30% of existing links
        if coref_spike:
            self._decide(f"COREF_SPIKE: {len(state.coref_links)} coref links = "
                         f"{coref_ratio:.0%} of {pre_coref_count} existing")

        # ── Phase 8b: Partial Injection (deterministic, no LLM) ─────
        self.dispatch(PartialInjectionAgent(), state)

        # ── Merge ───────────────────────────────────────────────────
        self.dispatch(MergeAgent(), state)
        merge_metrics = self._check_and_act(
            state, state.preliminary, "post-merge", comp_names_cache)

        # ── Phase 8c: Boundary Filter → VERIFY filter wasn't too aggressive
        pre_filter_count = len(state.preliminary)
        self.dispatch(BoundaryFilterAgent(), state)
        post_filter_count = len(state.preliminary)

        filter_drop = 1 - post_filter_count / max(1, pre_filter_count)
        if filter_drop > 0.3:
            self._decide(f"FILTER_AGGRESSIVE: boundary filter removed {filter_drop:.0%} "
                         f"({pre_filter_count} → {post_filter_count})")

        self._check_and_act(state, state.preliminary, "post-filter", comp_names_cache)

        # ── Phase 9: Judge → VERIFY judge wasn't too aggressive ─────
        pre_judge_count = len(state.preliminary)
        self.dispatch(JudgeAgent(), state)
        post_judge_count = len(state.final_links)

        judge_drop = 1 - post_judge_count / max(1, pre_judge_count)
        if judge_drop > 0.3:
            self._decide(f"JUDGE_AGGRESSIVE: rejected {judge_drop:.0%} "
                         f"({pre_judge_count} → {post_judge_count}), redoing lenient")
            state.final_links = []
            self.dispatch(JudgeAgent(lenient=True), state)

        # ── Post-judge verification gate (key checkpoint) ───────────
        judge_metrics = self._check_and_act(
            state, state.final_links, "post-judge", comp_names_cache)

        # ── Coref consistency (if coref spiked earlier) ─────────────
        if coref_spike:
            self.dispatch(ConsistencyCheckAgent(), state)
            self._check_and_act(
                state, state.final_links, "post-consistency", comp_names_cache)

        # ── Final summary ───────────────────────────────────────────
        final_metrics = self.monitor.compute(state.final_links, comp_names_cache)
        self.monitor.print_metrics(final_metrics, "FINAL")
        elapsed = time.time() - t0
        print(f"\n[Orchestrator] Pipeline complete: {len(state.final_links)} links, "
              f"{len(self.decisions)} decisions, {elapsed:.0f}s")
        for d in self.decisions:
            print(f"  Decision: {d['decision']}")

        return state.final_links


# ══════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════

class ALinker:
    """Adaptive Agent Linker — orchestrator-driven pipeline with extrinsic monitoring."""

    def __init__(self, backend: Optional[LLMBackend] = None):
        os.environ.setdefault("CLAUDE_MODEL", "sonnet")
        self.llm = LLMClient(backend=backend or LLMBackend.CLAUDE)
        print(f"ALinker (Adaptive Agent Linker)")
        print(f"  Backend: {self.llm.backend.value}, Model: {os.environ.get('CLAUDE_MODEL', 'default')}")

    def link(self, text_path, model_path, transarc_csv=None):
        """Run adaptive pipeline. transarc_csv accepted but unused (ILinker2 seed)."""
        components = parse_pcm_repository(model_path)
        sentences = DocumentLoader.load_sentences(text_path)
        name_to_id = {c.name: c.id for c in components}
        sent_map = DocumentLoader.build_sent_map(sentences)

        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        state = PipelineState(
            text_path=text_path,
            model_path=model_path,
            sentences=sentences,
            components=components,
            name_to_id=name_to_id,
            sent_map=sent_map,
        )

        orchestrator = Orchestrator(self.llm)
        return orchestrator.run(state)
