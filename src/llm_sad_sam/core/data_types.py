"""Shared data types for SAD-SAM trace link recovery."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class LinkSource(Enum):
    """Source of a trace link."""
    TRANSARC = "transarc"
    ENTITY = "entity"
    COREFERENCE = "coreference"
    VALIDATED = "validated"
    RECOVERED = "recovered"


@dataclass
class SadSamLink:
    """A trace link between a sentence and an architecture component."""
    sentence_number: int
    component_id: str
    component_name: str
    confidence: float
    source: str  # Keep as string for backward compatibility
    reasoning: Optional[str] = None

    def to_tuple(self) -> tuple:
        """Return (sentence_number, component_id) tuple for set operations."""
        return (self.sentence_number, self.component_id)


@dataclass
class CandidateLink:
    """A candidate trace link before validation."""
    sentence_number: int
    sentence_text: str
    component_name: str
    component_id: str
    matched_text: str
    confidence: float
    source: str
    match_type: str  # exact, synonym, partial, coreference
    needs_validation: bool
    context_sentences: list[int] = field(default_factory=list)


@dataclass
class DocumentProfile:
    """Learned document characteristics."""
    sentence_count: int
    component_count: int
    pronoun_ratio: float
    technical_density: float
    component_mention_density: float
    complexity_score: float
    recommended_strictness: str  # relaxed, balanced, strict


@dataclass
class LearnedThresholds:
    """Thresholds learned from document analysis."""
    coref_threshold: float
    validation_threshold: float
    fn_recovery_threshold: float
    disambiguation_threshold: float
    reasoning: str
    implicit_threshold: float = 0.85


@dataclass
class ModelKnowledge:
    """Knowledge extracted from architecture model."""
    impl_indicators: list[str] = field(default_factory=list)
    impl_to_abstract: dict[str, str] = field(default_factory=dict)
    architectural_names: set[str] = field(default_factory=set)
    ambiguous_names: set[str] = field(default_factory=set)
    shared_vocabulary: dict[str, list[str]] = field(default_factory=dict)

    def is_implementation(self, name: str) -> bool:
        """Check if a component name is an implementation variant."""
        return any(ind in name for ind in self.impl_indicators)

    def get_abstract(self, name: str) -> str:
        """Get abstract component name for an implementation."""
        return self.impl_to_abstract.get(name, name)


@dataclass
class DocumentKnowledge:
    """Knowledge extracted from document analysis."""
    abbreviations: dict[str, str] = field(default_factory=dict)
    synonyms: dict[str, str] = field(default_factory=dict)
    partial_references: dict[str, str] = field(default_factory=dict)
    generic_terms: set[str] = field(default_factory=set)

    def get_component(self, term: str) -> Optional[str]:
        """Get component name for a term (abbreviation, synonym, or partial)."""
        term_lower = term.lower()
        if term in self.abbreviations:
            return self.abbreviations[term]
        if term in self.synonyms:
            return self.synonyms[term]
        if term in self.partial_references:
            return self.partial_references[term]
        # Case-insensitive lookup
        for abbr, comp in self.abbreviations.items():
            if abbr.lower() == term_lower:
                return comp
        for syn, comp in self.synonyms.items():
            if syn.lower() == term_lower:
                return comp
        return None

    def is_generic(self, term: str) -> bool:
        """Check if a term is too generic to link reliably."""
        return term.lower() in {t.lower() for t in self.generic_terms}


@dataclass
class LearnedPatterns:
    """Patterns learned from document analysis."""
    action_indicators: list[str] = field(default_factory=list)
    effect_indicators: list[str] = field(default_factory=list)
    subprocess_terms: set[str] = field(default_factory=set)

    def is_subprocess(self, text: str) -> bool:
        """Check if text mentions a subprocess term."""
        text_lower = text.lower()
        return any(term.lower() in text_lower for term in self.subprocess_terms)


@dataclass
class EntityMention:
    """Track entity mentions for coreference resolution."""
    sentence_number: int
    component_name: str
    component_id: str
    mention_text: str
    is_subject: bool  # Is component the subject of sentence?
    position: int = 0  # Character position in sentence


@dataclass
class DiscourseContext:
    """Track discourse state for coreference resolution."""
    recent_mentions: list[EntityMention] = field(default_factory=list)
    paragraph_topic: Optional[str] = None
    paragraph_start: int = 0
    active_entity: Optional[str] = None  # Most likely referent for pronouns

    def add_mention(self, mention: EntityMention, max_history: int = 5):
        """Add a mention and update discourse state."""
        self.recent_mentions.append(mention)
        if len(self.recent_mentions) > max_history:
            self.recent_mentions.pop(0)

        # Update active entity (most recent subject mention)
        if mention.is_subject:
            self.active_entity = mention.component_name

    def get_likely_referent(self) -> Optional[str]:
        """Get most likely referent for a pronoun."""
        # Priority: active_entity > paragraph_topic > most recent mention
        if self.active_entity:
            return self.active_entity
        if self.paragraph_topic:
            return self.paragraph_topic
        if self.recent_mentions:
            return self.recent_mentions[-1].component_name
        return None

    def get_context_summary(self, sentence_num: int) -> str:
        """Get human-readable context summary for LLM prompts."""
        parts = []
        if self.active_entity:
            parts.append(f"Active subject: {self.active_entity}")
        if self.paragraph_topic and self.paragraph_topic != self.active_entity:
            parts.append(f"Paragraph topic: {self.paragraph_topic}")
        if self.recent_mentions:
            recent = [f"{m.component_name} (S{m.sentence_number})"
                     for m in self.recent_mentions[-3:]]
            parts.append(f"Recent mentions: {', '.join(recent)}")
        return "; ".join(parts) if parts else "No prior context"

    def start_new_paragraph(self, sentence_num: int):
        """Mark start of a new paragraph."""
        self.paragraph_start = sentence_num
        self.paragraph_topic = None
        # Don't clear recent_mentions entirely, but reset active_entity
        self.active_entity = None
