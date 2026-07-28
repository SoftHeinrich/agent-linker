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
from llm_sad_sam.linkers.experimental._s_linker24_orchestrator_base import (
    _SLinker24OrchestratorBase,
)
from llm_sad_sam.linkers.experimental.s_linker24_role_orchestrator import (
    SLinker24RoleOrchestrator,
)


def bare():
    return _SLinker24OrchestratorBase.__new__(_SLinker24OrchestratorBase)


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
    source = inspect.getsource(_SLinker24OrchestratorBase.link)
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
    assert _SLinker24OrchestratorBase._controller_feedback(feedback) == {
        "accepted": [{"sentence": 1, "component": "Store"}],
        "rejected": [{"sentence": 2, "component": "UI"}],
    }


def test_role_controller_only_schedules_remaining_tools():
    linker = SLinker24RoleOrchestrator.__new__(
        SLinker24RoleOrchestrator
    )
    profile = {"private": "must not reach controller"}
    seen = {}

    def ask(prompt, **_kwargs):
        seen["prompt"] = prompt
        return {
            "action": "relation_role_resolution",
            "reason": "participant discovery remains",
        }

    linker._ask = ask
    action, decision = linker._choose_tool(
        profile,
        ["relation_role_resolution"],
        [],
        [],
        {},
    )
    assert action == "relation_role_resolution"
    assert decision["reason"] == "participant discovery remains"
    assert "private" not in seen["prompt"]
    assert '"evidence"' not in seen["prompt"]

    linker._ask = lambda *args, **kwargs: {
        "action": "finalize",
    }
    try:
        linker._choose_tool(
            profile,
            ["relation_role_resolution"],
            [],
            [],
            {},
        )
    except RuntimeError as exc:
        assert "invalid simple action" in str(exc)
    else:
        raise AssertionError("controller finalized with evidence-backed work")


def test_catalog_overlap_is_broad_unique_and_identifier_safe():
    linker = SLinker24RoleOrchestrator.__new__(
        SLinker24RoleOrchestrator
    )
    linker.doc_knowledge = SimpleNamespace(aliases={})
    sentences = [
        Sentence(1, "The clients call the server."),
        Sentence(2, "The serverless worker starts."),
        Sentence(3, "Use package.client.handler."),
        Sentence(4, "The client-side cache is local."),
        Sentence(5, "The conversion begins."),
        Sentence(6, "WebRTC is enabled."),
    ]
    components = [
        SimpleNamespace(name="HTML5 Client", id="client"),
        SimpleNamespace(name="HTML5 Server", id="server"),
        SimpleNamespace(
            name="Presentation Conversion", id="conversion"
        ),
        SimpleNamespace(name="Web", id="web"),
        SimpleNamespace(name="WebRTC-SFU", id="webrtc"),
    ]
    candidates = linker._catalog_overlap_candidates(
        sentences,
        components,
        [SadSamLink(1, "server", "HTML5 Server")],
    )
    assert [
        (
            candidate.sentence_number,
            candidate.component_name,
            candidate.matched_text,
        )
        for candidate in candidates
    ] == [
        (1, "HTML5 Client", "clients"),
        (2, "HTML5 Server", "serverless"),
        (5, "Presentation Conversion", "conversion"),
    ]


def test_all_tools_own_their_discovery():
    linker = SLinker24RoleOrchestrator.__new__(
        SLinker24RoleOrchestrator
    )
    assert linker._available_tools({}) == list(linker.PHASE_TOOLS)
    assert "coverage_audit" not in linker.PHASE_TOOLS


def test_role_variant_entity_ownership_is_name_or_approved_alias():
    linker = SLinker24RoleOrchestrator.__new__(
        SLinker24RoleOrchestrator
    )
    linker.doc_knowledge = SimpleNamespace(
        aliases={
            "data store": SimpleNamespace(
                component="Database", scope="global"
            )
        }
    )
    candidates = [
        SimpleNamespace(
            component_name="Database",
            sentence_text="The Database stores records.",
        ),
        SimpleNamespace(
            component_name="Database",
            sentence_text="The data store persists records.",
        ),
        SimpleNamespace(
            component_name="Database",
            sentence_text="Records are persisted.",
        ),
    ]
    assert linker._select_entity_candidates(candidates, {}) == candidates[:2]


def test_role_context_review_uses_project_anchors():
    linker = SLinker24RoleOrchestrator.__new__(
        SLinker24RoleOrchestrator
    )
    linker.doc_knowledge = SimpleNamespace(aliases={})
    candidates = [
        SimpleNamespace(
            sentence_number=2,
            component_id="backend",
            component_name="HTML Backend",
            matched_text="backend",
            sentence_text="The backend handles requests.",
        ),
        SimpleNamespace(
            sentence_number=3,
            component_id="backend",
            component_name="HTML Backend",
            matched_text="backend",
            sentence_text="The database backend is external.",
        ),
    ]
    seen = {}

    def ask(prompt, **_kwargs):
        seen.setdefault("prompts", []).append(prompt)
        if "Classify what each expression" in prompt:
            return {
                "judgments": [
                    {
                        "case": 1,
                        "denotation": "participant",
                        "claim": "backend",
                    },
                    {
                        "case": 2,
                        "denotation": "participant",
                        "claim": "backend",
                    },
                ]
            }
        return {
            "judgments": [
                {
                    "case": 1,
                    "keep": True,
                    "anchor_sentence": 1,
                    "claim": '"The backend handles requests."',
                    "alternative": "none",
                },
                {
                    "case": 2,
                    "keep": False,
                    "anchor_sentence": 1,
                    "claim": "The database backend is external.",
                    "alternative": "database engine",
                },
            ]
        }

    linker._ask = ask
    approved, decisions = linker._review_role_candidates(
        candidates,
        [
            Sentence(1, "The HTML Backend is the request service."),
            Sentence(2, "The backend handles requests."),
            Sentence(3, "The database backend is external."),
        ],
    )
    assert approved == candidates[:1]
    assert decisions[(2, "backend")]["approved"] is True
    assert decisions[(3, "backend")]["approved"] is False
    assert any(
        "The HTML Backend is the request service." in prompt
        for prompt in seen["prompts"]
    )


def test_lexical_entity_candidates_are_exact_and_nonoverlapping():
    linker = SLinker24RoleOrchestrator.__new__(
        SLinker24RoleOrchestrator
    )
    linker.doc_knowledge = SimpleNamespace(
        aliases={
            "DisplayGateway": SimpleNamespace(
                component="Display Gateway", scope="global"
            )
        }
    )
    components = [
        SimpleNamespace(name="Order Processor", id="order-processor"),
        SimpleNamespace(name="Display Gateway", id="display-gateway"),
        SimpleNamespace(name="Reencoding", id="reencoding"),
    ]
    sentences = [
        Sentence(1, "The order-processor component handles requests."),
        Sentence(2, "The DisplayGateway renders pages."),
        Sentence(3, "The re-encoding step runs."),
        Sentence(4, "See package.order-processor for details."),
        Sentence(5, "The order-processor."),
    ]
    candidates = linker._lexical_entity_candidates(
        sentences, components
    )
    assert [
        (
            candidate.sentence_number,
            candidate.matched_text,
            candidate.component_name,
        )
        for candidate in candidates
    ] == [
        (1, "order-processor", "Order Processor"),
        (2, "DisplayGateway", "Display Gateway"),
        (5, "order-processor", "Order Processor"),
    ]


if __name__ == "__main__":
    test_controller_is_limited_by_available_capabilities()
    test_audit_grounding_is_structural_only()
    test_link_is_replacement_not_s21_floor()
    test_controller_feedback_is_normalized_without_losing_outcomes()
    test_role_controller_only_schedules_remaining_tools()
    test_catalog_overlap_is_broad_unique_and_identifier_safe()
    test_all_tools_own_their_discovery()
    test_role_variant_entity_ownership_is_name_or_approved_alias()
    test_role_context_review_uses_project_anchors()
    test_lexical_entity_candidates_are_exact_and_nonoverlapping()
    print("PASS: SLinker24RoleOrchestrator contracts")
