#!/usr/bin/env python3
"""Deterministic contracts for discourse-scoped participant resolution."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from llm_sad_sam.core.data_types_v2 import SadSamLink
from llm_sad_sam.core.document_loader_v2 import Sentence
from llm_sad_sam.linkers.experimental.s_linker24_role_orchestrator import (
    SLinker24RoleOrchestrator,
)

from pilot.s24_discourse_scope_pilot import (
    discourse_candidates,
    find_prose_handle,
    review_discourse_candidates,
    terminal_role_handles,
)


def component(name, identifier):
    return SimpleNamespace(name=name, id=identifier)


def bare():
    linker = SLinker24RoleOrchestrator.__new__(
        SLinker24RoleOrchestrator
    )
    linker.doc_knowledge = SimpleNamespace(aliases={})
    return linker


def test_terminal_roles_include_regular_plural_only():
    components = [
        component("HTML5 Client", "client"),
        component("HTML5 Server", "server"),
        component("Presentation Conversion", "conversion"),
        component("Test Driver", "driver"),
        component("Other Driver", "other-driver"),
    ]
    assert terminal_role_handles(components) == [
        {
            "component": "HTML5 Client",
            "expressions": ["Client", "Clients"],
        },
        {
            "component": "HTML5 Server",
            "expressions": ["Server", "Servers"],
        },
    ]


def test_candidates_are_nonoverlapping_and_code_safe():
    linker = bare()
    components = [
        component("HTML5 Client", "client"),
        component("HTML5 Server", "server"),
    ]
    sentences = [
        Sentence(1, "The clients connect to the server."),
        Sentence(2, "The HTML5 client connects."),
        Sentence(3, "Use package.server.handler."),
        Sentence(4, "The client-side cache is local."),
    ]
    floor = [SadSamLink(1, "server", "HTML5 Server")]
    candidates = discourse_candidates(
        linker, sentences, components, floor
    )
    assert [
        (
            candidate.sentence_number,
            candidate.component_name,
            candidate.matched_text,
        )
        for candidate in candidates
    ] == [(1, "HTML5 Client", "clients")]


def test_terminal_period_is_prose_not_a_dotted_identifier():
    linker = bare()
    assert find_prose_handle(
        linker, "Messages are sent to clients.", "clients"
    ) == "clients"
    assert find_prose_handle(
        linker, "Use package.clients.handler.", "clients"
    ) == ""
    assert find_prose_handle(
        linker, "See clients. Then continue.", "clients"
    ) == "clients"


def test_review_fails_closed_on_unverified_evidence():
    linker = bare()
    sentences = [
        Sentence(1, "HTML5 Client is the browser component."),
        Sentence(2, "The client sends a request."),
    ]
    candidate = SimpleNamespace(
        sentence_number=2,
        sentence_text=sentences[1].text,
        component_name="HTML5 Client",
        component_id="client",
        matched_text="client",
    )
    linker._ask = lambda *args, **kwargs: {
        "judgments": [
            {
                "case": 1,
                "keep": True,
                "section_anchor": "The client sends a request.",
                "identity_anchor_sentence": 1,
                "identity_anchor": "not in the anchor",
                "scope_bridge_sentence": 1,
                "scope_bridge": "HTML5 Client",
                "claim": "The client sends a request.",
                "participant_role": "request sender",
                "competing_referent": "browser user",
            }
        ]
    }
    approved, decisions = review_discourse_candidates(
        linker, [candidate], sentences
    )
    assert approved == []
    assert decisions[(2, "client")]["requested_keep"] is True
    assert decisions[(2, "client")]["evidence_valid"] is False


def test_review_accepts_complete_grounded_evidence():
    linker = bare()
    sentences = [
        Sentence(1, "HTML5 Client is the browser component."),
        Sentence(2, "The client sends a request."),
    ]
    candidate = SimpleNamespace(
        sentence_number=2,
        sentence_text=sentences[1].text,
        component_name="HTML5 Client",
        component_id="client",
        matched_text="client",
    )
    linker._ask = lambda *args, **kwargs: {
        "judgments": [
            {
                "case": 1,
                "keep": True,
                "section_anchor": "HTML5 Client is the browser component.",
                "identity_anchor_sentence": 1,
                "identity_anchor": "HTML5 Client",
                "scope_bridge_sentence": 1,
                "scope_bridge": "HTML5 Client is the browser component.",
                "claim": "The client sends a request.",
                "participant_role": "request sender",
                "competing_referent": "browser user",
            }
        ]
    }
    approved, decisions = review_discourse_candidates(
        linker, [candidate], sentences
    )
    assert approved == [candidate]
    assert decisions[(2, "client")]["evidence_valid"] is True


if __name__ == "__main__":
    test_terminal_roles_include_regular_plural_only()
    test_candidates_are_nonoverlapping_and_code_safe()
    test_terminal_period_is_prose_not_a_dotted_identifier()
    test_review_fails_closed_on_unverified_evidence()
    test_review_accepts_complete_grounded_evidence()
    print("PASS: S24 discourse-scope contracts")
