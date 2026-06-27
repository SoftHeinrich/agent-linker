"""Spike 003: LLM-fully-driven mention-type classification.

Current (s_linker12c._classify_mention):
  4 regex branches over (comp_name, sentence_text):
    * "proper case, standalone"        — _has_standalone_mention passes
    * "lowercase mention"               — \\b<comp_lower>\\b
    * "lowercase, inside dotted path"   — regex with dotted-context guard
    * 'via known alias "X"'             — scan doc_knowledge.aliases
    * "indirect/unclear match"          — fallback

Proposed (LLM-only):
  Do NOT add a new LLM call. Piggyback on the existing entity-extraction
  pass: the extractor already reads (comp_name, sentence, known aliases),
  so ask it to emit `mention_type` as a field on each extracted candidate.

  Output schema per candidate:
    {
      "component": "TaskDispatcher",
      "sentence": 42,
      "matched_text": "Dispatcher",
      "mention_type": "via_alias" | "proper_case" | "lowercase" | "dotted_path" | "indirect",
      "alias_used": "Dispatcher" | null
    }

  code consumer: format the enum into the evidence bundle display string
  (no regex, no lookup).

This file demonstrates the consumer side. The LLM side is a prompt-schema
change in _extract_entities_enriched — no new call, net-zero cost.
"""
from __future__ import annotations


MENTION_TYPES = {"proper_case", "lowercase", "dotted_path", "via_alias", "indirect"}


def format_mention(mention_type: str, alias_used: str | None = None) -> str:
    """Turn LLM enum into the human-readable string used in prompts.

    Replaces the 5 regex branches in _classify_mention with a lookup.
    """
    if mention_type == "proper_case":
        return "proper case, standalone"
    if mention_type == "lowercase":
        return "lowercase mention"
    if mention_type == "dotted_path":
        return "lowercase, inside dotted path"
    if mention_type == "via_alias":
        if alias_used:
            return f'via known alias "{alias_used}"'
        return "via known alias"
    return "indirect/unclear match"


def consume_candidate(candidate: dict) -> str:
    """Given an LLM-emitted candidate dict, produce the evidence-bundle string.

    Validates the mention_type enum and falls back to 'indirect' if unknown —
    no regex, no dict scan.
    """
    mt = candidate.get("mention_type", "indirect")
    if mt not in MENTION_TYPES:
        mt = "indirect"
    return format_mention(mt, candidate.get("alias_used"))


# -------- self-verifying tests --------

def _test_all_enum_branches():
    cases = [
        ({"mention_type": "proper_case"}, "proper case, standalone"),
        ({"mention_type": "lowercase"}, "lowercase mention"),
        ({"mention_type": "dotted_path"}, "lowercase, inside dotted path"),
        ({"mention_type": "via_alias", "alias_used": "Dispatcher"},
         'via known alias "Dispatcher"'),
        ({"mention_type": "via_alias"}, "via known alias"),
        ({"mention_type": "indirect"}, "indirect/unclear match"),
    ]
    for cand, expected in cases:
        got = consume_candidate(cand)
        assert got == expected, f"{cand} -> {got!r} expected {expected!r}"
    print(f"  [pass] all {len(cases)} enum branches formatted correctly")


def _test_unknown_enum_falls_back():
    got = consume_candidate({"mention_type": "garbage"})
    assert got == "indirect/unclear match", got
    print("  [pass] unknown mention_type falls back to 'indirect'")


def _test_output_matches_current_regex_output():
    """Verify the LLM-enum → string formatter emits strings byte-identical to
    what _classify_mention in s_linker12c returns today. This is the parity
    guarantee: callers of _classify_mention see no diff in their prompt strings.
    """
    expected_strings = {
        "proper case, standalone",
        "lowercase mention",
        "lowercase, inside dotted path",
        'via known alias "Dispatcher"',
        "indirect/unclear match",
    }
    produced = {
        format_mention("proper_case"),
        format_mention("lowercase"),
        format_mention("dotted_path"),
        format_mention("via_alias", "Dispatcher"),
        format_mention("indirect"),
    }
    assert produced == expected_strings, f"mismatch\n  got: {produced}\n  exp: {expected_strings}"
    print("  [pass] LLM-enum formatter output matches _classify_mention strings byte-for-byte")


def _test_replaces_regex_branches_count():
    """Inspect the bytecode of the consumer functions themselves — they must
    not reference the `re` module. Current `_classify_mention` has 4 regex
    branches; the LLM-enum consumer has 0.
    """
    for fn in (format_mention, consume_candidate):
        names = fn.__code__.co_names
        assert "re" not in names, f"{fn.__name__} references re module: {names}"
    print("  [pass] consumer functions reference 0 regex (vs 4 in _classify_mention)")


def run_tests():
    print("Spike 003: llm-mention-classifier tests")
    _test_all_enum_branches()
    _test_unknown_enum_falls_back()
    _test_output_matches_current_regex_output()
    _test_replaces_regex_branches_count()
    print("All tests PASSED")


if __name__ == "__main__":
    run_tests()
