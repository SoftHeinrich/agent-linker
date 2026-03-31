"""Checkpoint-based tests for s_linker12b (zero LLM calls, zero mocks).

Loads real checkpoint data (p1, p2, is_alias per decision) and tests:
  1. Structural invariants of the saved decisions.
  2. Alias-backed vs exact-name TP/FP breakdown vs gold standard.
  3. Voting-asymmetry hypothesis: simulate all three modes by replaying
     the real p1/p2/is_alias values through each voting rule.

Key empirical finding (from real LLM output):
  The only p1-XOR-p2 alias cases in mediastore are FPs ('database' used
  generically). Intersection correctly rejects them; adaptive (union) does not.
  This is evidence that the asymmetry hurts precision without helping recall.

Checkpoint: results/phase_cache/s_linker12b/mediastore/
Gold standard: ardoco/.../goldstandard_sad_2016-sam_2016.csv
"""

import csv
import pickle
import pytest
from pathlib import Path

CHECKPOINT_DIR = Path("results/phase_cache/s_linker12b/mediastore")
GOLD_CSV = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
    "/mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv"
)

EXPECTED_GOLD_LINKS    = 31
EXPECTED_ALIAS_BACKED  = 14
EXPECTED_EXACT_NAME    = 20


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def checkpoints_exist():
    required = [
        CHECKPOINT_DIR / "entity_candidates.pkl",
        CHECKPOINT_DIR / "entity_decisions.pkl",
        CHECKPOINT_DIR / "layer3.pkl",
        CHECKPOINT_DIR / "final.pkl",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        pytest.skip(f"Checkpoints not found (run ablation first): {missing}")


@pytest.fixture(scope="module")
def gold(checkpoints_exist):
    g = set()
    with open(GOLD_CSV) as f:
        for row in csv.DictReader(f):
            g.add((int(row["sentence"]), row["modelElementID"]))
    assert len(g) == EXPECTED_GOLD_LINKS
    return g


@pytest.fixture(scope="module")
def decisions(checkpoints_exist):
    with open(CHECKPOINT_DIR / "entity_decisions.pkl", "rb") as f:
        data = pickle.load(f)["decisions"]
    # Require the new-format checkpoint (with p1/p2/is_alias).
    sample = next(iter(data.values()))
    if "p1" not in sample:
        pytest.skip("Old-format checkpoint (no p1/p2/is_alias). Re-run ablation to regenerate.")
    return data


@pytest.fixture(scope="module")
def bundles(checkpoints_exist):
    with open(CHECKPOINT_DIR / "entity_candidates.pkl", "rb") as f:
        return pickle.load(f)["bundles"]


@pytest.fixture(scope="module")
def layer3(checkpoints_exist):
    with open(CHECKPOINT_DIR / "layer3.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def final_links(checkpoints_exist):
    with open(CHECKPOINT_DIR / "final.pkl", "rb") as f:
        return pickle.load(f)["final"]


# ── Helper ────────────────────────────────────────────────────────────────────

def _simulate_mode(decisions, mode):
    """Replay real p1/p2/is_alias through a voting rule. No LLM, no mock."""
    result = {}
    for k, v in decisions.items():
        p1, p2, is_alias = v["p1"], v["p2"], v["is_alias"]
        if mode == "intersection":
            approved = p1 and p2
        elif mode == "union":
            approved = p1 or p2
        else:  # adaptive
            approved = (p1 or p2) if is_alias else (p1 and p2)
        result[k] = approved
    return result


# ── 1. Structural invariants ──────────────────────────────────────────────────

def test_decisions_have_required_fields(decisions):
    """Every decision has p1, p2, is_alias, approved, path."""
    for k, v in decisions.items():
        for field in ("p1", "p2", "is_alias", "approved", "path"):
            assert field in v, f"{k} missing field '{field}'"


def test_approved_count_consistent(decisions):
    """approved=True ↔ path='twopass'; approved=False ↔ path='twopass_reject'."""
    for k, v in decisions.items():
        if v["approved"]:
            assert v["path"] == "twopass",        f"{k}: approved but path={v['path']!r}"
        else:
            assert v["path"] == "twopass_reject", f"{k}: rejected but path={v['path']!r}"


def test_approved_matches_adaptive_rule(decisions):
    """saved 'approved' equals the adaptive voting rule applied to real p1/p2/is_alias."""
    for k, v in decisions.items():
        p1, p2, is_alias = v["p1"], v["p2"], v["is_alias"]
        expected = (p1 or p2) if is_alias else (p1 and p2)
        assert v["approved"] == expected, (
            f"{k}: approved={v['approved']} but adaptive({p1},{p2},alias={is_alias})={expected}"
        )


def test_total_candidates(decisions, bundles):
    """decisions and bundles have the same keys."""
    assert set(decisions.keys()) == set(bundles.keys())


def test_alias_backed_count(bundles):
    alias_keys = {k for k, b in bundles.items() if "via known synonym" in b.mention_type}
    assert len(alias_keys) == EXPECTED_ALIAS_BACKED


def test_exact_name_count(bundles):
    exact_keys = {k for k, b in bundles.items() if "via known synonym" not in b.mention_type}
    assert len(exact_keys) == EXPECTED_EXACT_NAME


# ── 2. TP/FP breakdown (real data vs gold) ───────────────────────────────────

def test_no_tp_rejections_in_saved_run(decisions, gold):
    """The real LLM run never rejected a gold link at the entity-validation stage."""
    tp_rejected = [k for k, v in decisions.items() if not v["approved"] and k in gold]
    assert tp_rejected == [], f"TP candidates rejected: {tp_rejected}"


def test_alias_backed_classification(decisions, bundles, gold):
    """Every alias-backed candidate is classified correctly in is_alias field."""
    for k, b in bundles.items():
        expected_alias = "via known synonym" in b.mention_type
        assert decisions[k]["is_alias"] == expected_alias, (
            f"{k}: bundle says alias={expected_alias} but decision says {decisions[k]['is_alias']}"
        )


# ── 3. Voting asymmetry hypothesis — replayed on real p1/p2/is_alias ─────────

def test_xor_alias_cases_are_fps(decisions, gold):
    """p1-XOR-p2 alias cases (the only ones affected by the asymmetry) are all FPs.

    These are the only cases where adaptive ≠ intersection. If they are FPs,
    the asymmetry is actively harmful: it admits FPs that intersection would reject.
    """
    xor_alias = {k: v for k, v in decisions.items()
                 if v["is_alias"] and v["p1"] != v["p2"]}
    assert len(xor_alias) >= 1, "Need at least one XOR-alias case to test the hypothesis"

    false_positives = [k for k in xor_alias if k not in gold]
    true_positives  = [k for k in xor_alias if k in gold]

    assert true_positives == [], (
        f"XOR-alias TPs exist — asymmetry IS protecting real links: {true_positives}\n"
        f"Do NOT switch to intersection without further analysis."
    )
    assert len(false_positives) == len(xor_alias), (
        f"Expected all XOR-alias to be FPs, but found TPs: {true_positives}"
    )


def test_intersection_rejects_xor_alias_fps(decisions, gold):
    """Under intersection, p1-XOR-p2 alias FPs are correctly rejected."""
    intersection = _simulate_mode(decisions, "intersection")
    xor_alias_fps = {k for k, v in decisions.items()
                     if v["is_alias"] and v["p1"] != v["p2"] and k not in gold}

    for k in xor_alias_fps:
        assert not intersection[k], (
            f"Intersection failed to reject alias FP {k} "
            f"(p1={decisions[k]['p1']}, p2={decisions[k]['p2']})"
        )


def test_adaptive_incorrectly_approves_xor_alias_fps(decisions, gold):
    """Adaptive (union for aliases) approves alias cases where p1=T, p2=F even when FP.
    This is the direct cost of the asymmetry on this dataset.
    """
    adaptive = _simulate_mode(decisions, "adaptive")
    xor_alias_fps = {k for k, v in decisions.items()
                     if v["is_alias"] and v["p1"] != v["p2"] and k not in gold}

    wrongly_approved = [k for k in xor_alias_fps if adaptive[k]]
    assert len(wrongly_approved) == len(xor_alias_fps), (
        f"Expected adaptive to approve all alias FPs where p1≠p2, "
        f"but some were rejected: {set(xor_alias_fps) - set(wrongly_approved)}"
    )


def test_intersection_recovers_no_tps_vs_adaptive(decisions, gold):
    """Intersection loses zero TPs vs adaptive on real data.
    Together with the FP-only XOR-alias result, this means intersection
    is strictly better (fewer FPs, same TPs) on this dataset.
    """
    adaptive     = _simulate_mode(decisions, "adaptive")
    intersection = _simulate_mode(decisions, "intersection")

    tps_lost_by_intersection = [
        k for k in gold
        if k in adaptive and adaptive[k] and k in intersection and not intersection[k]
    ]
    assert tps_lost_by_intersection == [], (
        f"Intersection loses TPs vs adaptive: {tps_lost_by_intersection}"
    )


def test_all_modes_agree_on_unanimous_cases(decisions, gold):
    """When both passes agree (p1==p2), all three modes produce identical outcomes."""
    unanimous = {k for k, v in decisions.items() if v["p1"] == v["p2"]}

    adaptive     = _simulate_mode(decisions, "adaptive")
    intersection = _simulate_mode(decisions, "intersection")
    union        = _simulate_mode(decisions, "union")

    for k in unanimous:
        assert adaptive[k] == intersection[k] == union[k], (
            f"{k}: modes disagree on unanimous case (p1=p2={decisions[k]['p1']})"
        )


def test_precision_improves_under_intersection(decisions, gold):
    """Intersection has strictly fewer FPs than adaptive (no worse TPs).
    Quantifies the benefit of removing the alias asymmetry.
    """
    adaptive     = _simulate_mode(decisions, "adaptive")
    intersection = _simulate_mode(decisions, "intersection")

    fps_adaptive     = {k for k, approved in adaptive.items()     if approved and k not in gold}
    fps_intersection = {k for k, approved in intersection.items() if approved and k not in gold}

    assert len(fps_intersection) <= len(fps_adaptive), (
        f"Intersection should have ≤ FPs: intersection={len(fps_intersection)}, "
        f"adaptive={len(fps_adaptive)}"
    )
    # The FPs eliminated by intersection are exactly the XOR-alias FPs
    eliminated = fps_adaptive - fps_intersection
    assert all(decisions[k]["is_alias"] and decisions[k]["p1"] != decisions[k]["p2"]
               for k in eliminated), (
        f"Non-XOR-alias FP eliminated by intersection: {eliminated}"
    )
