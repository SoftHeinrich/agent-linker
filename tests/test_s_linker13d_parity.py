"""Parity tests for s_linker13d's LLM-enum mention-type formatter.

D-21d (CONTEXT.md): byte-identical string parity is a hard acceptance
criterion. Every downstream prompt string consuming EvidenceBundle.mention_type
must read byte-identical text for at least one synthetic case per enum branch.

Mirrors .planning/spikes/003-llm-mention-classifier/spike.py tests 1, 2, 3
against the live SLinker13d class.
"""
from __future__ import annotations

import pytest

from llm_sad_sam.linkers.experimental.s_linker13d import SLinker13d


EXPECTED = {
    ("proper_case", None):           "proper case, standalone",
    ("lowercase", None):             "lowercase mention",
    ("dotted_path", None):           "lowercase, inside dotted path",
    ("via_alias", "Dispatcher"):     'via known alias "Dispatcher"',
    ("via_alias", None):             "via known alias",
    ("indirect", None):              "indirect/unclear match",
}


def test_mention_types_frozenset_matches_spike_003():
    """The enum set must exactly match Spike 003's MENTION_TYPES."""
    assert SLinker13d.MENTION_TYPES == frozenset({
        "proper_case", "lowercase", "dotted_path", "via_alias", "indirect"
    })


@pytest.mark.parametrize("inp,expected", list(EXPECTED.items()))
def test_format_mention_string_parity(inp, expected):
    """D-21d: byte-identical strings for all 6 case-branches (5 enum + via_alias-no-alias backstop)."""
    mt, alias = inp
    # Coerce first (real read site coerces) to mirror runtime behaviour
    coerced = SLinker13d._coerce_mention_type(mt)
    got = SLinker13d._format_mention_string(coerced, alias)
    assert got == expected, f"{inp!r} -> {got!r} expected {expected!r}"


def test_coerce_unknown_enum_raises():
    """D-21a STRICT coercion: an unknown value raises ValueError immediately,
    does NOT silently fall back to 'indirect' (D-21b rejects the Spike's lenient pattern).
    """
    with pytest.raises(ValueError, match="Unknown mention_type"):
        SLinker13d._coerce_mention_type("garbage_enum_value")


def test_classify_mention_method_removed():
    """VAR-04 acceptance: _classify_mention is fully removed from the class."""
    assert not hasattr(SLinker13d, "_classify_mention"), (
        "_classify_mention should be removed in 13d - the LLM emits mention_type "
        "via _extract_entities_enriched and the formatter consumes it."
    )


def test_format_mention_string_does_not_use_regex():
    """Mirrors Spike 003 test 4 - bytecode-level check that the formatter
    references zero regex (it must not - that was the whole point of VAR-04)."""
    names = SLinker13d._format_mention_string.__code__.co_names
    assert "re" not in names, f"_format_mention_string references re: {names}"
    names2 = SLinker13d._coerce_mention_type.__code__.co_names
    assert "re" not in names2, f"_coerce_mention_type references re: {names2}"
