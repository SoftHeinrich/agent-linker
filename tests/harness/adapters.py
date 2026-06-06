"""Builder → phase-tag adapter map for Phase 44 golden-replay harness.

Encodes the D-03 mapping verbatim (locked from code scout of s_linker19.py).

Exports:
- BUILDER_PHASE_TAGS : dict[str, tuple[str, ...]]
    Maps each builder name to the set of phase_tag values that appear in
    _calls.json when that builder is active.  Values are tuples (never lists
    or bare strings) to keep the schema explicit.

    D-03 GOTCHA: _prompt_validation handles THREE phase tags because
    phase_5_coref_validation reuses _prompt_validation with COREF_VALIDATION_FOCUS
    (s_linker19.py:893-916).  Its records go in test_s_linker20_prompt_validation.py,
    NOT ..._coref.py.

- BUILDERS : dict[str, Callable]
    Maps each builder name to the SLinker19._prompt_<name> @staticmethod.
    Values are callable; no SLinker19 instance is created here.
    Plan 02 test modules invoke the builders under test.

sys.path bootstrap: inherited from tests/conftest.py.
Do NOT modify sys.path in this module.
"""
from __future__ import annotations

from typing import Callable

from llm_sad_sam.linkers.experimental.s_linker19 import SLinker19


# ---------------------------------------------------------------------------
# D-03 builder → phase-tag mapping (verbatim from code scout)
# ---------------------------------------------------------------------------

BUILDER_PHASE_TAGS: dict[str, tuple[str, ...]] = {
    "_prompt_ambiguity": ("phase_1_model",),
    "_prompt_doc_knowledge_extract": ("phase_1_doc_extract",),
    "_prompt_doc_knowledge_judge": ("phase_1_doc_judge",),
    "_prompt_extraction": ("phase_2_framing_c_pass1", "phase_2_framing_c_pass2"),
    # NOTE: phase_5_coref_validation is handled by _prompt_validation (D-03 gotcha)
    "_prompt_validation": (
        "phase_4_twopass_p1",
        "phase_4_twopass_p2",
        "phase_5_coref_validation",
    ),
    "_prompt_coref": ("phase_5_coref",),
}

# ---------------------------------------------------------------------------
# Builder callable registry — @staticmethod references (no SLinker19 instance)
# ---------------------------------------------------------------------------

BUILDERS: dict[str, Callable] = {
    "_prompt_ambiguity": SLinker19._prompt_ambiguity,
    "_prompt_doc_knowledge_extract": SLinker19._prompt_doc_knowledge_extract,
    "_prompt_doc_knowledge_judge": SLinker19._prompt_doc_knowledge_judge,
    "_prompt_extraction": SLinker19._prompt_extraction,
    "_prompt_validation": SLinker19._prompt_validation,
    "_prompt_coref": SLinker19._prompt_coref,
}

# Sanity guard: every builder in BUILDERS must have a BUILDER_PHASE_TAGS entry.
assert set(BUILDERS.keys()) == set(BUILDER_PHASE_TAGS.keys()), (
    "BUILDERS and BUILDER_PHASE_TAGS keys are out of sync — update adapters.py"
)
