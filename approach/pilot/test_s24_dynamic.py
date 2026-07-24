#!/usr/bin/env python3
"""Deterministic contracts for the profile-aware S24 controller."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_sad_sam.core.data_types_v2 import CandidateLink, SadSamLink
from llm_sad_sam.core.document_loader_v2 import Sentence
from llm_sad_sam.linkers.experimental.s_linker24_dynamic import SLinker24Dynamic


def bare():
    return SLinker24Dynamic.__new__(SLinker24Dynamic)


def test_profile_describes_project_and_floor():
    linker = bare()
    linker.model_knowledge = SimpleNamespace(ambiguous_names={"Cache"})
    linker.doc_knowledge = SimpleNamespace(
        aliases={"Store": SimpleNamespace(component="DB", scope="global")}
    )
    components = [
        SimpleNamespace(name="Cache", id="c"),
        SimpleNamespace(name="DB", id="d"),
    ]
    sentences = [Sentence(1, "Store writes."), Sentence(2, "It returns.")]
    floor = [SadSamLink(1, "d", "DB", source="entity")]
    aliases = [CandidateLink(1, "Store writes.", "DB", "d", "Store")]
    profile = linker._build_runtime_profile(
        components, sentences, floor, aliases, []
    )
    assert profile["document"]["sentences"] == 2
    assert profile["components"]["ambiguous_names"] == ["Cache"]
    assert profile["components"]["without_floor_links"] == ["Cache"]
    assert profile["recovery_evidence"]["alias_phase4"]["samples"][0][
        "matched_alias"
    ] == "Store"


def test_controller_can_only_select_available_action():
    linker = bare()
    linker._ask = lambda *args, **kwargs: {
        "action": "alias_phase4",
        "reason": "specific approved alias evidence",
        "assessment": {
            "document_regime": "architecture_prose",
            "catalog_risk": "low",
            "best_evidence": "alias_phase4",
            "expected_gain": "high",
            "false_positive_risk": "low",
        },
    }
    action, reason, assessment = linker._next_action(
        {}, ["alias_phase4"], []
    )
    assert action == "alias_phase4"
    assert reason
    assert assessment["expected_gain"] == "high"
    linker._ask = lambda *args, **kwargs: {
        "action": "anchored_reference",
        "assessment": {
            "document_regime": "mixed",
            "catalog_risk": "medium",
            "best_evidence": "anchored_reference",
            "expected_gain": "medium",
            "false_positive_risk": "medium",
        },
    }
    try:
        linker._next_action({}, ["alias_phase4"], [])
    except RuntimeError as exc:
        assert "invalid controller action" in str(exc)
    else:
        raise AssertionError("unavailable phase was accepted")


if __name__ == "__main__":
    test_profile_describes_project_and_floor()
    test_controller_can_only_select_available_action()
    print("PASS: SLinker24Dynamic contracts")
