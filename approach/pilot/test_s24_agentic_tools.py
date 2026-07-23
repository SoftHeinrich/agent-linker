#!/usr/bin/env python3
"""Deterministic contract tests for SLinker24Agentic (no network)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_sad_sam.core.data_types_v2 import SadSamLink
from llm_sad_sam.core.document_loader_v2 import Sentence
from llm_sad_sam.linkers.experimental.s_linker24_agentic import (
    SLinker24Agentic,
    ToolInventory,
)


def bare_linker():
    return SLinker24Agentic.__new__(SLinker24Agentic)


def test_controller_is_bounded():
    linker = bare_linker()
    linker._ask = lambda *args, **kwargs: {
        "calls": ["alias_phase4", "alias_phase4", "anchored_reference"],
        "reason": "runtime evidence",
    }
    calls, reason = linker._plan_tools(
        ToolInventory(alias_phase4=2, anchored_reference=1), ["A"], []
    )
    assert calls == ["alias_phase4", "anchored_reference"]
    assert reason == "runtime evidence"

    linker._ask = lambda *args, **kwargs: {"calls": ["unknown"]}
    try:
        linker._plan_tools(ToolInventory(1, 1), ["A"], [])
    except RuntimeError as exc:
        assert "unknown tool" in str(exc)
    else:
        raise AssertionError("unknown tool was not rejected")


def test_weak_alias_uses_phase1_ambiguity():
    linker = bare_linker()
    linker.model_knowledge = SimpleNamespace(ambiguous_names={"DB"})
    linker.doc_knowledge = SimpleNamespace(
        aliases={
            "Database": SimpleNamespace(component="DB", scope="global"),
            "DataStorage": SimpleNamespace(component="FileStorage", scope="global"),
        }
    )
    components = [
        SimpleNamespace(name="DB", id="db"),
        SimpleNamespace(name="FileStorage", id="fs"),
    ]
    sentences = [Sentence(1, "Database and DataStorage participate.")]
    candidates = linker._alias_candidates(sentences, components, [])
    assert [(item.component_name, item.matched_text) for item in candidates] == [
        ("FileStorage", "DataStorage")
    ]


def test_longer_competing_alias_wins():
    linker = bare_linker()
    linker.doc_knowledge = SimpleNamespace(
        aliases={
            "Free Switch Event Layer": SimpleNamespace(
                component="EventAdapter", scope="global"
            )
        }
    )
    additions = [
        SadSamLink(1, "short", "Free Switch", source="s24_anchor"),
        SadSamLink(2, "short", "Free Switch", source="s24_anchor"),
    ]
    sentences = [
        Sentence(1, "The Free Switch Event Layer sends messages."),
        Sentence(2, "Free Switch accepts calls."),
    ]
    kept = linker._remove_competing_aliases(additions, sentences)
    assert [(item.sentence_number, item.component_name) for item in kept] == [
        (2, "Free Switch")
    ]


def main():
    test_controller_is_bounded()
    test_weak_alias_uses_phase1_ambiguity()
    test_longer_competing_alias_wins()
    print("PASS: SLinker24Agentic tool contracts")


if __name__ == "__main__":
    main()
