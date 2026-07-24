#!/usr/bin/env python3
"""Deterministic contracts for S24 lexical entity normalization."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from llm_sad_sam.core.document_loader_v2 import Sentence

from llm_sad_sam.linkers.experimental.s_linker24_role_orchestrator import (
    SLinker24RoleOrchestrator,
)


def component(name, identifier):
    return SimpleNamespace(name=name, id=identifier)


def pairs(sentences, components):
    return {
        (
            candidate.sentence_number,
            candidate.component_name,
            candidate.matched_text,
        )
        for candidate in SLinker24RoleOrchestrator._lexical_entity_candidates(
            sentences, components
        )
    }


def test_signature_handles_separator_and_identifier_style():
    expected = ("bbb", "web")
    signature = SLinker24RoleOrchestrator._lexical_signature
    assert signature("BBB web") == expected
    assert signature("bbb-web") == expected
    assert signature("BBB_Web") == expected
    assert signature("BBBWeb") == expected


def test_exact_variants_are_catalog_driven():
    components = [
        component("BBB web", "bbb"),
        component("ImageProvider", "image"),
    ]
    sentences = [
        Sentence(1, "The bbb-web service handles the request."),
        Sentence(2, "The Image Provider returns an image."),
    ]
    assert pairs(sentences, components) == {
        (1, "BBB web", "bbb-web"),
        (2, "ImageProvider", "Image Provider"),
    }


def test_unsafe_extensions_are_rejected():
    components = [
        component("BBB web", "bbb"),
        component("HTML5 Client", "client"),
        component("WebRTC-SFU", "webrtc"),
        component("MediaAccess", "media"),
    ]
    sentences = [
        Sentence(1, "Use pkg.bbb-web.handler from code."),
        Sentence(2, "Several clients connect."),
        Sentence(3, "The WebRTC protocol is enabled."),
        Sentence(4, "AudioAccess stores the item."),
    ]
    assert pairs(sentences, components) == set()


def test_catalog_ambiguity_fails_closed():
    components = [
        component("Foo Bar", "one"),
        component("Foo-Bar", "two"),
    ]
    sentences = [Sentence(1, "FooBar performs the work.")]
    assert pairs(sentences, components) == set()


def test_identifier_is_not_a_controller_tool():
    assert "catalog_identifier_resolution" not in (
        SLinker24RoleOrchestrator.PHASE_TOOLS
    )
    assert "catalog-equivalent orthographic identity" in (
        SLinker24RoleOrchestrator.__new__(
            SLinker24RoleOrchestrator
        )._tool_catalog()
    )


if __name__ == "__main__":
    test_signature_handles_separator_and_identifier_style()
    test_exact_variants_are_catalog_driven()
    test_unsafe_extensions_are_rejected()
    test_catalog_ambiguity_fails_closed()
    test_identifier_is_not_a_controller_tool()
    print("PASS: S24 lexical entity contracts")
