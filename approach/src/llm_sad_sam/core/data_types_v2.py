"""Lean data types for the S-Linker family (v2).

Clean-slate replacement for core/data_types.py — no dead fields, no dead
methods.  Intended as the shared foundation for S-Linker11 and all future
linker generations.  Older linkers (pre-S11) still use data_types.py.
"""

from dataclasses import dataclass, field


@dataclass
class SadSamLink:
    """A trace link between a sentence and an architecture component."""
    sentence_number: int
    component_id: str
    component_name: str
    confidence: float = 1.0  # read by run_ablation.py
    source: str = ""


@dataclass
class CandidateLink:
    """A candidate trace link before validation."""
    sentence_number: int
    sentence_text: str
    component_name: str
    component_id: str
    matched_text: str
    source: str = ""
    # Added by S-Linker13d (VAR-04, Spike 003): LLM-emitted mention-type enum
    # and the alias used when mention_type == "via_alias". Defaults keep
    # pre-13d linkers byte-identical - only s_linker13d writes non-defaults.
    mention_type: str = "indirect"
    alias_used: str | None = None


@dataclass
class ModelKnowledge:
    """Knowledge extracted from architecture model.

    ambiguous_names: component names the LLM classifies as generic
    (used for generic-mention detection).
    """
    ambiguous_names: set[str] = field(default_factory=set)


@dataclass
class DocumentKnowledge:
    """Knowledge extracted from document analysis.

    aliases: unified map of alternative name -> component name.  Includes
        abbreviations (AST -> AbstractSyntaxTree), synonyms, and trailing-word
        forms (Dispatcher -> TaskDispatcher).  Used by 12c+.
    abbreviations/synonyms/partial_references: legacy fields, kept for
        backward compat with pre-12c linkers.
    """
    aliases: dict[str, str] = field(default_factory=dict)
    # Legacy (pre-12c) — do not use in new code
    abbreviations: dict[str, str] = field(default_factory=dict)
    synonyms: dict[str, str] = field(default_factory=dict)
    partial_references: dict[str, str] = field(default_factory=dict)
