#!/usr/bin/env python3
"""Spike 004 — Mode 2 trap-pattern rejecter (rule-based, no LLM, BENCHMARK_TABOO-safe).

Each trap is a structural/linguistic predicate over a LinkCtx. True => the link's
sole architectural evidence matches a known false-positive trap => reject.

TABOO DISCIPLINE: every pattern below is generic English structure or a generic
dotted-identifier regex. No component name, alias, or project keyword from any of
the 5 benchmark projects appears here. (Feeds the open taboo-audit todo.)
"""
import re

_DEICTIC = {"this", "that", "these", "those", "it", "they", "them", "its", "their"}

# Generic listing/overview header cues (English structure, not benchmark terms).
_OVERVIEW = (
    "overview", "the following", "given above", "shown below", "as shown",
    "diagram below", "figure below", "table below", "listed below", "list of",
    "as follows", "consists of", "is composed of", "are listed", "shown in the",
    "depicted", "illustrat",
)

# Generic test-infrastructure prose (no benchmark component names).
_TEST = (
    "test case", "test cases", "unit test", "test suite", "testing framework",
    "test driver", "scaffold", "mock ", "stub ", "fixture", "test harness",
)

# Negation-of-identity cues — "X is not a Java package", "rather than".
_NEG = (
    "is not a", "is not an", "is not the", "are not a", "are not the",
    "isn't a", "isn't an", "aren't", "rather than", "instead of", "not actually",
)

# Lowercase dotted identifier path: pkg.member.access (member-access path X.Y.Z).
_QUALPATH = re.compile(r"\b[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+\b")


def _low(s):
    return (s or "").lower()


def trap_overview_header(c):
    """Sentence is an enumeration/overview header — names are listed, not asserted."""
    t = _low(c.sentence_text)
    return any(k in t for k in _OVERVIEW)


def trap_negation(c):
    """The architectural mention sits inside a negation-of-identity."""
    blob = _low(c.sentence_text) + " " + _low(c.antecedent_text)
    return any(k in blob for k in _NEG)


def trap_qualified_path(c):
    """The mention is via a lowercase dotted member/package path (X.Y.Z),
    not an architectural-participant reference."""
    span = c.matched_span or ""
    # entity: the matched span itself is a dotted path
    if _QUALPATH.search(span):
        return True
    # coref: antecedent rests on a dotted path
    if c.source == "coreference" and _QUALPATH.search(c.antecedent_text or ""):
        return True
    return False


def trap_deictic_pronoun(c):
    """Coref link resting on a bare deictic/pronoun whose antecedent does NOT
    standalone-name the component — weak pronoun resolution."""
    if c.source != "coreference":
        return False
    ref = _low(c.reference).strip()
    if ref not in _DEICTIC:
        return False
    # If the antecedent text actually names the component, keep it (strong link).
    name = _low(c.component_name)
    if name and re.search(rf"\b{re.escape(name)}\b", _low(c.antecedent_text)):
        return False
    return True


def trap_test_scaffolding(c):
    """Sentence describes generic test-infrastructure prose."""
    t = _low(c.sentence_text)
    return any(k in t for k in _TEST)


TRAPS = {
    "overview_header": trap_overview_header,
    "negation": trap_negation,
    "qualified_path": trap_qualified_path,
    "deictic_pronoun": trap_deictic_pronoun,
    "test_scaffolding": trap_test_scaffolding,
}


def trap_hits(c, names=None):
    """Return the list of trap names that fire for this LinkCtx."""
    names = names or list(TRAPS)
    return [n for n in names if TRAPS[n](c)]
