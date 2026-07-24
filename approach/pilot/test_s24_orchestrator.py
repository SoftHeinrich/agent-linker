#!/usr/bin/env python3
"""Deterministic contracts for the S24 replacement orchestrator."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_sad_sam.core.data_types_v2 import SadSamLink
from llm_sad_sam.core.document_loader_v2 import Sentence, build_sent_map
from llm_sad_sam.linkers.experimental.s_linker24_orchestrator import (
    SLinker24Orchestrator,
)


def bare():
    return SLinker24Orchestrator.__new__(SLinker24Orchestrator)


def test_controller_is_limited_by_available_capabilities():
    linker = bare()
    linker._ask = lambda *args, **kwargs: {
        "action": "coverage_audit",
        "document_reference_style": "aliases",
        "component_profile_effect": "unambiguous",
        "coverage_gap": "identity",
        "reason": "complementary evidence",
    }
    action, decision = linker._choose_tool(
        {}, ["coverage_audit"], [], [], {}
    )
    assert action == "coverage_audit"
    assert decision["reason"]

    linker._ask = lambda *args, **kwargs: {"action": "entity_pipeline"}
    try:
        linker._choose_tool({}, ["coverage_audit"], [], [], {})
    except RuntimeError as exc:
        assert "invalid replacement action" in str(exc)
    else:
        raise AssertionError("consumed/unavailable tool was accepted")


def test_audit_grounding_is_structural_only():
    linker = bare()
    linker.model_knowledge = SimpleNamespace(ambiguous_names=set())
    linker.doc_knowledge = SimpleNamespace(aliases={})
    linker._ask = lambda *args, **kwargs: {
        "omissions": [
            {
                "sentence": 1,
                "component": "Store",
                "quote": "storage service",
            },
            {
                "sentence": 1,
                "component": "Unknown",
                "quote": "storage service",
            },
            {
                "sentence": 1,
                "component": "Store",
                "quote": "words not present",
            },
        ]
    }
    sentences = [Sentence(1, "The storage service persists records.")]
    components = [SimpleNamespace(name="Store", id="store")]
    candidates = linker._coverage_candidates(
        sentences,
        components,
        [],
        build_sent_map(sentences),
    )
    assert [
        (
            candidate.sentence_number,
            candidate.component_name,
            candidate.matched_text,
        )
        for candidate in candidates
    ] == [(1, "Store", "storage service")]

    existing = [SadSamLink(1, "store", "Store")]
    assert (
        linker._coverage_candidates(
            sentences,
            components,
            existing,
            build_sent_map(sentences),
        )
        == []
    )


def test_link_is_replacement_not_s21_floor():
    source = inspect.getsource(SLinker24Orchestrator.link)
    assert "SLinker21.link" not in source
    assert "super().link" not in source


def test_controller_feedback_is_normalized_without_losing_outcomes():
    feedback = {
        "candidates": [
            {
                "sentence": 1,
                "text": "A long sentence already present in the profile.",
                "component": "Store",
                "source": "entity_candidate",
            },
            {
                "sentence": 2,
                "text": "Another repeated sentence.",
                "component": "UI",
                "source": "entity_candidate",
            },
        ],
        "accepted": [
            {
                "sentence": 1,
                "text": "A long sentence already present in the profile.",
                "component": "Store",
                "source": "entity",
            }
        ],
        "validator_decisions": {"large": "evidence stays in phase output"},
    }
    assert SLinker24Orchestrator._controller_feedback(feedback) == {
        "accepted": [{"sentence": 1, "component": "Store"}],
        "rejected": [{"sentence": 2, "component": "UI"}],
    }


if __name__ == "__main__":
    test_controller_is_limited_by_available_capabilities()
    test_audit_grounding_is_structural_only()
    test_link_is_replacement_not_s21_floor()
    test_controller_feedback_is_normalized_without_losing_outcomes()
    print("PASS: SLinker24Orchestrator contracts")
