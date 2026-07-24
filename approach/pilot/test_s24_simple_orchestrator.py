#!/usr/bin/env python3
"""Deterministic contracts for the minimal S24 orchestration pilot."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from llm_sad_sam.core.document_loader_v2 import Sentence

from pilot.s24_simple_orchestrator_pilot import (
    SLinker24SimplePilot,
)


def bare():
    linker = SLinker24SimplePilot.__new__(
        SLinker24SimplePilot
    )
    linker.doc_knowledge = SimpleNamespace(aliases={})
    return linker


def test_controller_prompt_is_compact_and_grounded():
    linker = bare()
    seen = {}

    def ask(prompt, **_kwargs):
        seen["prompt"] = prompt
        return {
            "action": "entity_pipeline",
            "evidence": [1],
            "reason": "named evidence remains",
        }

    linker._ask = ask
    profile = {
        "document": [
            {
                "sentence": 99,
                "text": "THIS FULL DOCUMENT MUST NOT BE SENT",
            }
        ],
        "tool_evidence": {
            "entity_pipeline": [
                {
                    "sentence": 1,
                    "quote": "Gateway",
                    "target": "Gateway",
                }
            ]
        },
    }
    action, decision = linker._choose_tool(
        profile,
        ["entity_pipeline"],
        [],
        [],
        {},
    )
    assert action == "entity_pipeline"
    assert decision["evidence"] == [1]
    assert "THIS FULL DOCUMENT MUST NOT BE SENT" not in seen["prompt"]
    assert len(seen["prompt"]) < 1200


def test_controller_uses_compact_prior_outcomes():
    linker = bare()
    seen = {}

    def ask(prompt, **_kwargs):
        seen["prompt"] = prompt
        return {
            "action": "coreference_pipeline",
            "evidence": [2],
            "reason": "reference evidence remains",
        }

    linker._ask = ask
    action, _decision = linker._choose_tool(
        {
            "tool_evidence": {
                "coreference_pipeline": [
                    {"sentence": 2, "quote": "it"}
                ]
            }
        },
        ["coreference_pipeline"],
        [
            {
                "action": "entity_pipeline",
                "feedback": {
                    "accepted": [{"sentence": 1, "component": "A"}],
                    "rejected": [],
                },
            }
        ],
        [],
        {},
    )
    assert action == "coreference_pipeline"
    assert '"accepted": 1' in seen["prompt"]
    assert "finalize" not in seen["prompt"]


def test_local_participant_prompt_excludes_unrelated_document():
    linker = bare()
    seen = {}
    candidate = SimpleNamespace(
        sentence_number=5,
        sentence_text="The client sends requests.",
        component_name="HTML Client",
        component_id="client",
        matched_text="client",
    )
    sentences = [
        Sentence(1, "The HTML Client is the user interface."),
        Sentence(4, "Requests are prepared."),
        Sentence(5, "The client sends requests."),
        Sentence(6, "Responses return."),
        Sentence(20, "UNRELATED DISTANT SENTENCE."),
    ]

    def ask(prompt, **_kwargs):
        seen["prompt"] = prompt
        return {
            "judgments": [
                {
                    "case": 1,
                    "keep": True,
                    "anchor_sentence": 1,
                    "claim": "The client sends requests.",
                    "alternative": "none",
                }
            ]
        }

    linker._ask = ask
    approved, decisions = linker._review_role_candidates(
        [candidate], sentences
    )
    assert approved == [candidate]
    assert decisions[(5, "client")]["evidence_valid"] is True
    assert "UNRELATED DISTANT SENTENCE" not in seen["prompt"]
    assert "section_anchor" not in seen["prompt"]
    assert "scope_bridge" not in seen["prompt"]


def test_local_participant_review_fails_closed():
    linker = bare()
    candidate = SimpleNamespace(
        sentence_number=2,
        sentence_text="The server stores state.",
        component_name="HTML Server",
        component_id="server",
        matched_text="server",
    )
    sentences = [
        Sentence(1, "The HTML Server is the backend."),
        Sentence(2, "The server stores state."),
    ]
    linker._ask = lambda *_args, **_kwargs: {
        "judgments": [
            {
                "case": 1,
                "keep": True,
                "anchor_sentence": 99,
                "claim": "The server stores state.",
                "alternative": "none",
            }
        ]
    }
    approved, decisions = linker._review_role_candidates(
        [candidate], sentences
    )
    assert approved == []
    assert decisions[(2, "server")]["requested_keep"] is True
    assert decisions[(2, "server")]["evidence_valid"] is False


if __name__ == "__main__":
    test_controller_prompt_is_compact_and_grounded()
    test_controller_uses_compact_prior_outcomes()
    test_local_participant_prompt_excludes_unrelated_document()
    test_local_participant_review_fails_closed()
    print("PASS: S24 simple-orchestrator contracts")
