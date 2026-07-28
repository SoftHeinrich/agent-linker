#!/usr/bin/env python3
"""Deterministic contracts for general S24 discovery grounding."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from llm_sad_sam.core.document_loader_v2 import Sentence

from pilot.s24_general_discovery_pilot import (
    SLinker24GeneralDiscoveryPilot,
)


def test_minimal_judge_retains_grounding():
    linker = SLinker24GeneralDiscoveryPilot.__new__(
        SLinker24GeneralDiscoveryPilot
    )
    linker.doc_knowledge = SimpleNamespace(aliases={})
    candidate = SimpleNamespace(
        sentence_number=2,
        sentence_text="The client sends requests.",
        component_name="HTML Client",
        component_id="client",
        matched_text="client",
    )
    sentences = [
        Sentence(1, "The HTML Client is the interface."),
        Sentence(2, "The client sends requests."),
    ]
    seen = {}

    def ask(prompt, **_kwargs):
        seen.setdefault("prompts", []).append(prompt)
        if "Classify what each expression" in prompt:
            return {
                "judgments": [{
                    "case": 1,
                    "denotation": "participant",
                    "claim": "client",
                }]
            }
        return {
            "judgments": [{
                "case": 1,
                "keep": True,
                "anchor_sentence": 1,
                "claim": '"The client sends requests."',
                "alternative": "none",
            }]
        }

    linker._ask = ask
    approved, decisions = linker.review_minimal(
        [candidate], sentences
    )
    assert approved == [candidate]
    assert decisions[(2, "client")]["evidence_valid"] is True
    combined = "\n".join(seen["prompts"])
    assert "hardware" not in combined
    assert "technology" not in combined
    assert "server" not in combined.casefold()


def test_minimal_judge_rejects_unlisted_anchor():
    linker = SLinker24GeneralDiscoveryPilot.__new__(
        SLinker24GeneralDiscoveryPilot
    )
    linker.doc_knowledge = SimpleNamespace(aliases={})
    candidate = SimpleNamespace(
        sentence_number=2,
        sentence_text="The client sends requests.",
        component_name="HTML Client",
        component_id="client",
        matched_text="client",
    )
    def ask(prompt, **_kwargs):
        if "Classify what each expression" in prompt:
            return {
                "judgments": [{
                    "case": 1,
                    "denotation": "participant",
                    "claim": "client",
                }]
            }
        return {
            "judgments": [{
                "case": 1,
                "keep": True,
                "anchor_sentence": 99,
                "claim": "The client sends requests.",
                "alternative": "none",
            }]
        }

    linker._ask = ask
    approved, decisions = linker.review_minimal(
        [candidate],
        [
            Sentence(1, "The HTML Client is the interface."),
            Sentence(2, "The client sends requests."),
        ],
    )
    assert approved == []
    assert decisions[(2, "client")]["evidence_valid"] is False


if __name__ == "__main__":
    test_minimal_judge_retains_grounding()
    test_minimal_judge_rejects_unlisted_anchor()
    print("PASS: S24 general-discovery contracts")
