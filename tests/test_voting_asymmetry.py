"""Unit tests for _validate_with_evidence voting modes (zero LLM calls).

The two-pass validation supports three voting modes:
  adaptive     — union for alias-backed candidates, intersection for exact-name
                 (current default; also enables alias_rule in prompts)
  intersection — both passes must approve, regardless of alias status
                 (symmetric; alias_rule disabled)
  union        — either pass may approve, regardless of alias status
                 (symmetric; alias_rule disabled)

Tests verify:
  1. All modes agree when both passes agree (universal behaviour).
  2. Modes diverge exactly on p1-XOR-p2 cases — and only there.
  3. p1, p2, and is_alias are recorded in the decisions dict.
  4. use_alias_rule is passed correctly to _run_validation_pass.
  5. Mixed-alias batches are split correctly in adaptive mode.
"""

import pytest
from unittest.mock import patch

from llm_sad_sam.linkers.experimental.s_linker12b import SLinker12b, EvidenceBundle
from llm_sad_sam.core.data_types_v2 import CandidateLink, DocumentKnowledge, ModelKnowledge
from llm_sad_sam.core.document_loader_v2 import Sentence


# ── Helpers ───────────────────────────────────────────────────────────────────

class _Comp:
    """Minimal stand-in for a PCM component."""
    def __init__(self, name, cid):
        self.name = name
        self.id = cid


def make_linker(synonyms=None, ambiguous=None):
    """SLinker12b instance with no LLM, minimal state."""
    linker = SLinker12b.__new__(SLinker12b)
    linker.doc_knowledge = DocumentKnowledge(
        abbreviations={},
        synonyms=synonyms or {},
        partial_references={},
    )
    linker.model_knowledge = ModelKnowledge(ambiguous_names=set(ambiguous or []))
    linker._phase_log = []
    linker._current_text_path = None
    return linker


def _cand(snum, comp, matched):
    return CandidateLink(snum, f"S{snum}.", comp.name, comp.id, matched, source="entity")


def _bundle(comp_name, matched, *, is_alias=False):
    return EvidenceBundle(
        source="entity",
        matched_span=matched,
        mention_type=f'via known synonym "{matched}"' if is_alias else "proper case, standalone",
        preceding_text="",
        anchor_sentences=[f"S0: The {comp_name} stores records."],
        is_ambiguous=False,
        extraction_rationale="dual-pass extraction consensus",
    )


def run_mocked(linker, candidates, bundles, components, sent_map,
               p1_map, p2_map, voting_mode="adaptive"):
    """Run _validate_with_evidence with _run_validation_pass mocked.

    p1_map / p2_map: {case_index: bool} — first call uses p1_map, second p2_map.
    """
    call_n = [0]
    captured_use_alias_rule = []

    def mock_pass(comp_names, cases, focus, use_alias_rule=True):
        captured_use_alias_rule.append(use_alias_rule)
        result_map = p1_map if call_n[0] == 0 else p2_map
        call_n[0] += 1
        return {i: result_map.get(i, False) for i in range(len(cases))}

    with patch.object(linker, "_run_validation_pass", side_effect=mock_pass):
        validated, decisions = linker._validate_with_evidence(
            candidates, bundles, components, sent_map, voting_mode=voting_mode
        )

    return validated, decisions, captured_use_alias_rule


# ── Voting matrix ─────────────────────────────────────────────────────────────

# (p1, p2, is_alias, exp_adaptive, exp_intersection, exp_union, label)
MATRIX = [
    # Both agree → all modes agree
    (True,  True,  False, True,  True,  True,  "both-approve_exact"),
    (True,  True,  True,  True,  True,  True,  "both-approve_alias"),
    (False, False, False, False, False, False, "both-reject_exact"),
    (False, False, True,  False, False, False, "both-reject_alias"),
    # One pass disagrees — this is where modes diverge
    (True,  False, False, False, False, True,  "p1only_exact"),
    (False, True,  False, False, False, True,  "p2only_exact"),
    (True,  False, True,  True,  False, True,  "p1only_alias"),
    (False, True,  True,  True,  False, True,  "p2only_alias"),
]


@pytest.mark.parametrize(
    "p1,p2,is_alias,exp_adap,exp_inter,exp_union,label",
    MATRIX,
    ids=[r[-1] for r in MATRIX],
)
def test_voting_modes(p1, p2, is_alias, exp_adap, exp_inter, exp_union, label):
    """Each voting mode produces the correct outcome for every p1/p2/alias combo."""
    comp = _Comp("Database", "db-001")
    matched = "DB" if is_alias else "Database"
    sent_text = "Queries go to DB." if is_alias else "The Database stores records."

    cand = _cand(1, comp, matched)
    bundle = _bundle(comp.name, matched, is_alias=is_alias)
    sent_map = {1: Sentence(1, sent_text)}
    linker = make_linker(synonyms={"DB": "Database"} if is_alias else {})

    for mode, expected in [("adaptive", exp_adap), ("intersection", exp_inter), ("union", exp_union)]:
        validated, decisions, _ = run_mocked(
            linker, [cand], {(1, comp.id): bundle}, [comp], sent_map,
            p1_map={0: p1}, p2_map={0: p2},
            voting_mode=mode,
        )
        actual = len(validated) == 1
        assert actual == expected, (
            f"[{mode}] {label}: expected={expected}, got={actual}"
        )


# ── decisions dict records p1, p2, is_alias ──────────────────────────────────

def test_decisions_record_pass_details():
    """decisions must expose p1, p2, is_alias for post-hoc analysis."""
    comp = _Comp("Database", "db-001")
    cand = _cand(1, comp, "Database")
    bundle = _bundle(comp.name, "Database")
    sent_map = {1: Sentence(1, "The Database stores records.")}
    linker = make_linker()

    _, decisions, _ = run_mocked(
        linker, [cand], {(1, comp.id): bundle}, [comp], sent_map,
        p1_map={0: True}, p2_map={0: False},
    )
    d = decisions[(1, comp.id)]
    assert d["p1"] is True
    assert d["p2"] is False
    assert d["is_alias"] is False
    assert d["approved"] is False   # adaptive + exact + p1-only → intersection → reject
    assert "path" in d


# ── alias_rule forwarding ─────────────────────────────────────────────────────

@pytest.mark.parametrize("mode,expect_alias_rule", [
    ("adaptive",     True),
    ("intersection", False),
    ("union",        False),
])
def test_alias_rule_forwarded_correctly(mode, expect_alias_rule):
    """use_alias_rule is True only in adaptive mode."""
    comp = _Comp("Database", "db-001")
    cand = _cand(1, comp, "DB")
    bundle = _bundle(comp.name, "DB", is_alias=True)
    sent_map = {1: Sentence(1, "Queries go to DB.")}
    linker = make_linker(synonyms={"DB": "Database"})

    _, _, captured = run_mocked(
        linker, [cand], {(1, comp.id): bundle}, [comp], sent_map,
        p1_map={0: True}, p2_map={0: True},
        voting_mode=mode,
    )
    # Both passes are called; check whether any saw use_alias_rule=True
    assert any(captured) == expect_alias_rule, (
        f"[{mode}] use_alias_rule={expect_alias_rule} not propagated correctly"
    )


# ── Mixed-alias batch (adaptive mode) ────────────────────────────────────────

def test_mixed_batch_adaptive():
    """Alias and exact candidates in one batch get different voting in adaptive mode."""
    comp = _Comp("Database", "db-001")
    cand_alias = _cand(1, comp, "DB")          # alias match
    cand_exact = _cand(2, comp, "Database")    # exact match
    bundles = {
        (1, comp.id): _bundle(comp.name, "DB", is_alias=True),
        (2, comp.id): _bundle(comp.name, "Database", is_alias=False),
    }
    sent_map = {
        1: Sentence(1, "Queries go to DB."),
        2: Sentence(2, "The Database stores records."),
    }
    linker = make_linker(synonyms={"DB": "Database"})

    # Arrange: alias gets p1=T/p2=F (pass 1 index 0), exact gets p1=F/p2=T (pass 1 index 1)
    _, decisions, _ = run_mocked(
        linker,
        [cand_alias, cand_exact],
        bundles, [comp], sent_map,
        p1_map={0: True,  1: False},
        p2_map={0: False, 1: True},
        voting_mode="adaptive",
    )
    # alias: is_alias=True, p1=T,p2=F → union → approved
    assert decisions[(1, comp.id)]["approved"] is True,  "alias p1-only should approve under adaptive"
    # exact: is_alias=False, p1=F,p2=T → intersection → rejected
    assert decisions[(2, comp.id)]["approved"] is False, "exact p2-only should reject under adaptive"


def test_mixed_batch_intersection():
    """Same mixed batch under intersection — both candidates rejected (p1 XOR p2)."""
    comp = _Comp("Database", "db-001")
    cand_alias = _cand(1, comp, "DB")
    cand_exact = _cand(2, comp, "Database")
    bundles = {
        (1, comp.id): _bundle(comp.name, "DB", is_alias=True),
        (2, comp.id): _bundle(comp.name, "Database", is_alias=False),
    }
    sent_map = {
        1: Sentence(1, "Queries go to DB."),
        2: Sentence(2, "The Database stores records."),
    }
    linker = make_linker(synonyms={"DB": "Database"})

    _, decisions, _ = run_mocked(
        linker,
        [cand_alias, cand_exact],
        bundles, [comp], sent_map,
        p1_map={0: True,  1: False},
        p2_map={0: False, 1: True},
        voting_mode="intersection",
    )
    assert decisions[(1, comp.id)]["approved"] is False, "alias p1-only should reject under intersection"
    assert decisions[(2, comp.id)]["approved"] is False, "exact p2-only should reject under intersection"


# ── Empty candidates ──────────────────────────────────────────────────────────

def test_empty_candidates():
    """Empty input returns empty output for all modes."""
    comp = _Comp("Database", "db-001")
    linker = make_linker()
    for mode in ("adaptive", "intersection", "union"):
        validated, decisions = linker._validate_with_evidence(
            [], {}, [comp], {}, voting_mode=mode
        )
        assert validated == []
        assert decisions == {}
