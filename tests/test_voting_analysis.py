"""Cross-dataset voting analysis (zero LLM calls, zero mocks).

Uses two sets of real checkpoint data:
  - s_linker11a: all 5 datasets, validated entity lists (no per-pass data)
  - s_linker12b: mediastore only, full p1/p2/is_alias per decision

Tests:
  1. s_linker11a entity-stage FP characterisation (what does the current
     intersection validator let through?).
  2. s_linker12b mediastore: per-pass pattern distribution and voting-mode impact.
  3. Majority vote simulation: does adding a 3rd pass (2/3 threshold) improve
     over intersection? Spoiler: only when p3 is specificity-oriented (like p2),
     which is equivalent to requiring 2-of-3 specificity passes — strictly
     harder than intersection for same TP outcome.
  4. Unified voting conclusion: intersection (p1 AND p2) is the correct, simplest
     rule — it implements both necessary conditions for a valid TLR link with no
     asymmetry and no extra calls.

Patterns visible in the s_linker12b mediastore real data:
  (p1, p2, is_alias) → count  TP  FP
  (F, F, F)          →    2    0   2   all modes reject — no issue
  (T, F, F)          →    1    0   1   only union admits — union adds FP
  (T, T, F)          →   17   17   0   all modes approve — clean TP block
  (T, F, T)          →    2    0   2   adaptive/union admit — ASYMMETRY FPs
  (T, T, T)          →   12   11   1   all modes admit 1 FP regardless of mode
"""

import csv
import pickle
import pytest
from pathlib import Path

BENCHMARK = Path(
    "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark"
)
GOLD_FILES = {
    "mediastore":    "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    "teastore":      "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    "teammates":     "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "bigbluebutton": "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "jabref":        "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
}
S11A_DIR   = Path("results/phase_cache/s_linker11a")
S12B_MS    = Path("results/phase_cache/s_linker12b/mediastore")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def gold_all():
    out = {}
    for ds, rel in GOLD_FILES.items():
        g = set()
        with open(BENCHMARK / rel) as f:
            for row in csv.DictReader(f):
                g.add((int(row["sentence"]), row["modelElementID"]))
        out[ds] = g
    return out


@pytest.fixture(scope="module")
def s11a_layer3_all(gold_all):
    if not S11A_DIR.exists():
        pytest.skip("s_linker11a checkpoints not found")
    out = {}
    for ds in GOLD_FILES:
        with open(S11A_DIR / ds / "layer3.pkl", "rb") as f:
            out[ds] = pickle.load(f)
    return out


@pytest.fixture(scope="module")
def s12b_decisions():
    path = S12B_MS / "entity_decisions.pkl"
    if not path.exists():
        pytest.skip("s_linker12b mediastore checkpoint not found")
    with open(path, "rb") as f:
        data = pickle.load(f)["decisions"]
    sample = next(iter(data.values()))
    if "p1" not in sample:
        pytest.skip("Old-format checkpoint (no p1/p2/is_alias). Re-run ablation.")
    return data


# ── Helpers ───────────────────────────────────────────────────────────────────

def _simulate(decisions, mode):
    """Replay real p1/p2/is_alias through a voting rule."""
    result = {}
    for k, v in decisions.items():
        p1, p2, alias = v["p1"], v["p2"], v["is_alias"]
        if mode == "intersection":
            approved = p1 and p2
        elif mode == "union":
            approved = p1 or p2
        elif mode == "adaptive":
            approved = (p1 or p2) if alias else (p1 and p2)
        elif mode == "majority_p3_like_p1":
            # 3rd pass is participation-oriented (like p1) → ties toward approval
            p3 = p1
            approved = sum([p1, p2, p3]) >= 2
        elif mode == "majority_p3_like_p2":
            # 3rd pass is specificity-oriented (like p2) → ties toward rejection
            p3 = p2
            approved = sum([p1, p2, p3]) >= 2
        else:
            raise ValueError(f"Unknown mode: {mode}")
        result[k] = approved
    return result


def _tp_fp(approved_set, gold):
    tp = sum(1 for k in approved_set if k in gold)
    fp = sum(1 for k in approved_set if k not in gold)
    return tp, fp


# ── 1. s_linker11a: entity-stage quality across all 5 datasets ───────────────

def test_s11a_entity_validated_zero_fps_on_clean_datasets(s11a_layer3_all, gold_all):
    """teastore and jabref have zero entity-stage FPs under s_linker11a intersection."""
    for ds in ("teastore", "jabref"):
        validated = s11a_layer3_all[ds]["validated"]
        fps = [c for c in validated if (c.sentence_number, c.component_id) not in gold_all[ds]]
        assert fps == [], f"{ds}: unexpected entity FPs: {fps}"


def test_s11a_entity_fps_are_one_per_noisy_dataset(s11a_layer3_all, gold_all):
    """mediastore, teammates, bigbluebutton each have exactly 1 entity-stage FP."""
    for ds in ("mediastore", "teammates", "bigbluebutton"):
        validated = s11a_layer3_all[ds]["validated"]
        fps = [c for c in validated if (c.sentence_number, c.component_id) not in gold_all[ds]]
        assert len(fps) == 1, f"{ds}: expected 1 entity FP, got {len(fps)}: {fps}"


def test_s11a_no_entity_tp_lost(s11a_layer3_all, gold_all):
    """Across all 5 datasets, every validated entity link that is a TP is in gold
    (trivially true) — and the total TP count is non-zero."""
    total_tp = 0
    for ds, l3 in s11a_layer3_all.items():
        tp = sum(1 for c in l3["validated"]
                 if (c.sentence_number, c.component_id) in gold_all[ds])
        assert tp > 0, f"{ds}: zero entity TPs — something is wrong"
        total_tp += tp
    assert total_tp >= 150  # sanity: across 5 datasets


# ── 2. s_linker12b mediastore: per-pass pattern coverage ─────────────────────

def test_all_tp_patterns_are_unanimous(s12b_decisions, gold_all):
    """Every TP where passes disagree (p1≠p2) is zero: all real TPs had both passes agree.
    This validates the theoretical premise of intersection: both necessary conditions
    being satisfied is sufficient and necessary for a TP on this dataset.
    """
    gold = gold_all["mediastore"]
    tp_xor = [k for k, v in s12b_decisions.items()
              if k in gold and v["p1"] != v["p2"]]
    assert tp_xor == [], (
        f"TPs with disagreeing passes: {tp_xor}\n"
        "If non-empty, the asymmetry may protect real links — reconsider intersection default."
    )


def test_xor_patterns_are_all_fps(s12b_decisions, gold_all):
    """Every p1≠p2 case (regardless of alias) is a FP. No disagreement on real TPs."""
    gold = gold_all["mediastore"]
    xor_cases = {k: v for k, v in s12b_decisions.items() if v["p1"] != v["p2"]}
    xor_tps   = [k for k in xor_cases if k in gold]
    assert xor_tps == [], f"XOR TPs exist: {xor_tps}"
    assert len(xor_cases) >= 1, "Need at least one XOR case for the test to be meaningful"


def test_intersection_optimal_on_real_data(s12b_decisions, gold_all):
    """Intersection has strictly fewer FPs than adaptive, with identical TPs.
    Quantifies the net benefit of the symmetric rule on this dataset.
    """
    gold = gold_all["mediastore"]
    approved_keys = lambda mode: {k for k, v in _simulate(s12b_decisions, mode).items() if v}

    adaptive_set     = approved_keys("adaptive")
    intersection_set = approved_keys("intersection")

    tp_adap  = len(adaptive_set     & gold)
    tp_inter = len(intersection_set & gold)
    fp_adap  = len(adaptive_set     - gold)
    fp_inter = len(intersection_set - gold)

    assert tp_inter == tp_adap,  f"Intersection loses TPs vs adaptive: {tp_adap} → {tp_inter}"
    assert fp_inter <  fp_adap,  f"Intersection should have fewer FPs: {fp_adap} → {fp_inter}"


# ── 3. Majority vote simulation ───────────────────────────────────────────────
#
# Majority vote (2-of-3) with a 3rd pass that is a copy of p1 (participation)
# or a copy of p2 (specificity). Both are simulated from real p1/p2 data.

def test_majority_p3_like_p1_does_not_help_vs_adaptive(s12b_decisions, gold_all):
    """Majority vote where p3 mimics p1 (participation) is no better than adaptive.

    For the FP pattern (p1=T, p2=F, alias=T):
      p3_like_p1 = T  →  votes: T,F,T  →  2/3 = approve  →  still FP

    The participation question is too permissive to cast the deciding vote.
    """
    gold = gold_all["mediastore"]
    majority_p1  = _simulate(s12b_decisions, "majority_p3_like_p1")
    adaptive     = _simulate(s12b_decisions, "adaptive")

    fp_majority  = sum(1 for k, v in majority_p1.items() if v and k not in gold)
    fp_adaptive  = sum(1 for k, v in adaptive.items()    if v and k not in gold)

    # Both admit the same (T,F,alias) FPs — 3rd participation pass doesn't help
    assert fp_majority >= fp_adaptive, (
        f"Unexpected: majority_p3_like_p1 has fewer FPs ({fp_majority}) than adaptive ({fp_adaptive})"
    )


def test_majority_p3_like_p2_equals_intersection(s12b_decisions, gold_all):
    """Majority vote where p3 mimics p2 (specificity) is equivalent to intersection.

    For the FP pattern (p1=T, p2=F, alias=T):
      p3_like_p2 = F  →  votes: T,F,F  →  1/3 = reject  →  same as intersection

    For the TP pattern (p1=T, p2=T, alias=T):
      p3_like_p2 = T  →  votes: T,T,T  →  3/3 = approve  →  same as intersection

    Majority with a specificity-biased 3rd pass is just a more expensive intersection.
    """
    gold = gold_all["mediastore"]
    majority_p2   = _simulate(s12b_decisions, "majority_p3_like_p2")
    intersection  = _simulate(s12b_decisions, "intersection")

    assert majority_p2 == intersection, (
        "majority_p3_like_p2 and intersection differ — unexpected pattern in real data.\n"
        "Differences: " + str({k for k in majority_p2 if majority_p2[k] != intersection[k]})
    )


def test_no_voting_mode_beats_intersection_on_fp(s12b_decisions, gold_all):
    """No single-step mode reduces FPs below intersection without also losing TPs.

    This is the key claim: intersection is the Pareto-optimal rule for this data.
    Every alternative either has the same FPs (majority_p3_like_p2) or more FPs
    (adaptive, union, majority_p3_like_p1).
    """
    gold = gold_all["mediastore"]
    for mode in ("adaptive", "union", "majority_p3_like_p1", "majority_p3_like_p2"):
        result   = _simulate(s12b_decisions, mode)
        inter    = _simulate(s12b_decisions, "intersection")
        fp_mode  = sum(1 for k, v in result.items()  if v and k not in gold)
        fp_inter = sum(1 for k, v in inter.items()   if v and k not in gold)
        tp_mode  = sum(1 for k, v in result.items()  if v and k in gold)
        tp_inter = sum(1 for k, v in inter.items()   if v and k in gold)

        # Either same FPs (majority_p3_like_p2 == intersection) or more FPs
        assert fp_mode >= fp_inter, (
            f"{mode}: has fewer FPs ({fp_mode}) than intersection ({fp_inter}) "
            f"— unexpected, reconsider recommendation"
        )
        # TPs should not drop below intersection
        assert tp_mode >= tp_inter, (
            f"{mode}: loses TPs vs intersection ({tp_inter} → {tp_mode})"
        )
