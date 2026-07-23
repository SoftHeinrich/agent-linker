"""Vendored copy of agent-linker's core trace-link data types.

Verbatim copy of ``llm_sad_sam/core/data_types_v2.py`` from the agent-linker
repo (``../agent-linker/src/llm_sad_sam/core/data_types_v2.py``), plus the
``AliasEntry`` record that the s_linker20_union knowledge layer pickles.

Why vendored and not imported: the ``phase_cache/*.pkl`` run artifacts this
study reads are Python pickles of these dataclasses. Unpickling needs the class
definitions importable under their original module path
(``llm_sad_sam.core.data_types_v2``). Rather than depend on the agent-linker
package being pip-installed (this repo runs on a bare stdlib interpreter and
the mini-* studies copy definitions instead of importing them), ``rq34.py``
registers this module under that path before unpickling. Only ``SadSamLink``
and ``CandidateLink`` are exercised by RQ3/RQ4 (layer3/layer4/final); the
knowledge records are kept for fidelity so layer1 also loads if needed.

Depends only on the stdlib (``dataclasses``).
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
    mention_type: str = "indirect"
    alias_used: "str | None" = None


@dataclass
class ModelKnowledge:
    """Knowledge extracted from architecture model."""
    ambiguous_names: set = field(default_factory=set)


@dataclass(frozen=True)
class AliasEntry:
    """An alias -> component mapping with a resolution scope."""
    component: str
    scope: str  # "global" | "local"


@dataclass
class DocumentKnowledge:
    """Knowledge extracted from document analysis.

    ``aliases`` maps an alternative name to an :class:`AliasEntry` (newer runs)
    or to a plain component-name string (older runs); both unpickle fine here.
    """
    aliases: dict = field(default_factory=dict)
    # Legacy (pre-12c) — do not use in new code
    abbreviations: dict = field(default_factory=dict)
    synonyms: dict = field(default_factory=dict)
    partial_references: dict = field(default_factory=dict)
