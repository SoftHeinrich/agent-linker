"""Core utilities for SAD-SAM trace link recovery.

This module contains stable, reusable utilities:
- Data types (dataclasses shared across linkers)
- Document loading and parsing
- Model analysis utilities

Agentic code (coreference, debate, validation) stays in linkers
since it evolves frequently between versions.
"""

from .data_types import (
    SadSamLink,
    CandidateLink,
    DocumentProfile,
    LearnedThresholds,
    ModelKnowledge,
    DocumentKnowledge,
    LearnedPatterns,
    EntityMention,
    DiscourseContext,
    LinkSource,
)
from .document_loader import DocumentLoader, Sentence
from .model_analyzer import ModelAnalyzer

__all__ = [
    # Data types
    "SadSamLink",
    "CandidateLink",
    "DocumentProfile",
    "LearnedThresholds",
    "ModelKnowledge",
    "DocumentKnowledge",
    "LearnedPatterns",
    "EntityMention",
    "DiscourseContext",
    "LinkSource",
    # Utilities
    "DocumentLoader",
    "Sentence",
    "ModelAnalyzer",
]
