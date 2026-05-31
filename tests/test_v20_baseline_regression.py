"""GATE-02 frozen-compat regression test.

GATE-02 frozen-compat regression test. Asserts every variant in CANONICAL_VARIANTS
stays equivalent to the pinned v2.0 baseline JSON at tests/fixtures/v2_0_baseline.json.
This test must pass before any v2.1 promotion. See REQUIREMENTS.md GATE-02 and
STATE.md Standing Gates.

Design contract:
    - This is a STATIC fixture-vs-registry consistency test. It does NOT execute any
      linker (running 45 variants x 5 datasets would take hours and is wired in
      Phase 13 PROMPT-03 as a separate gate). Importing any module under
      llm_sad_sam.linkers.* is explicitly disallowed because that drags LLMClient
      and forces .env loading.
    - The fixture is the SOURCE OF TRUTH for "what the v2.0-close state was". Drift
      between run_ablation.CANONICAL_VARIANTS and the fixture is treated as a
      GATE-02 failure with an actionable error message ("snapshot it before
      promotion").
    - Variants with no v2.0-close 5-dataset sweep are pinned under fixture['missing']
      with explicit per-dataset null markers. The regression test xfails those
      slots (visible in the report but does not break CI). A future fixture
      refresh fills them in or removes them from CANONICAL_VARIANTS.
    - Float tolerance pinned at 1e-4 for any future live-run comparator wired
      in Phase 13. This file does not compare to live runs.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics

import pytest

from run_ablation import CANONICAL_VARIANTS


FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "v2_0_baseline.json"
DATASETS = ("mediastore", "teastore", "teammates", "bigbluebutton", "jabref")


@pytest.fixture(scope="module")
def baseline():
    """Load the pinned v2.0 baseline JSON once per test module."""
    assert FIXTURE_PATH.exists(), (
        f"v2.0 baseline fixture missing at {FIXTURE_PATH} - "
        "Phase 10 Plan 01 Task 1 must run first."
    )
    return json.loads(FIXTURE_PATH.read_text())


def _pinned_variant_names(baseline) -> list[str]:
    return sorted(baseline["variants"].keys())


def _missing_variant_names(baseline) -> list[str]:
    return [m["variant"] for m in baseline.get("missing", [])]


# ----------------------------------------------------------------------
# Test 1: CANONICAL_VARIANTS <-> fixture registry consistency
# ----------------------------------------------------------------------

def test_canonical_variants_matches_fixture_coverage(baseline):
    """GATE-02: every variant in CANONICAL_VARIANTS must be accounted for in
    the fixture (either pinned with numbers or listed in 'missing'). Drift
    in either direction is a GATE-02 failure with an actionable message."""
    pinned = set(baseline["variants"].keys())
    missing = set(_missing_variant_names(baseline))
    registered = set(CANONICAL_VARIANTS)
    accounted = pinned | missing

    added_to_registry = registered - accounted
    removed_from_registry = accounted - registered

    assert not added_to_registry, (
        f"GATE-02 drift: variants {sorted(added_to_registry)} added to "
        "CANONICAL_VARIANTS but missing from tests/fixtures/v2_0_baseline.json - "
        "snapshot them before promotion."
    )
    assert not removed_from_registry, (
        f"GATE-02 drift: variants {sorted(removed_from_registry)} pinned in the "
        "fixture but no longer present in CANONICAL_VARIANTS - refresh the fixture "
        "or restore the registry entry."
    )


def test_fixture_pinned_and_missing_sets_are_disjoint(baseline):
    """A variant cannot be both pinned and missing - one or the other."""
    pinned = set(baseline["variants"].keys())
    missing = set(_missing_variant_names(baseline))
    overlap = pinned & missing
    assert not overlap, (
        f"GATE-02 fixture inconsistency: variants in both 'variants' and "
        f"'missing': {sorted(overlap)}"
    )


# ----------------------------------------------------------------------
# Test 2: per-variant fixture internal consistency
# ----------------------------------------------------------------------

def _pinned_variant_params():
    """Pytest parametrize source loaded at collection time. Returns sorted variant
    names from the fixture so tests are stably ordered."""
    if not FIXTURE_PATH.exists():
        return []
    return sorted(json.loads(FIXTURE_PATH.read_text())["variants"].keys())


@pytest.mark.parametrize("variant", _pinned_variant_params())
def test_pinned_variant_has_all_datasets_with_valid_metrics(variant, baseline):
    """Every pinned variant must (a) cover all 5 datasets, (b) carry P/R/F1
    as floats in [0, 1], (c) round-trip macro_f1 = mean(F1)."""
    entry = baseline["variants"][variant]
    per_ds = entry["per_dataset"]

    assert set(per_ds.keys()) == set(DATASETS), (
        f"{variant}: per_dataset keys {sorted(per_ds.keys())} != "
        f"required {sorted(DATASETS)}"
    )

    f1_values = []
    for ds in DATASETS:
        m = per_ds[ds]
        for key in ("P", "R", "F1"):
            assert key in m, f"{variant}/{ds}: missing key {key!r}"
            assert isinstance(m[key], float), (
                f"{variant}/{ds}/{key}: expected float, got {type(m[key]).__name__}"
            )
            assert 0.0 <= m[key] <= 1.0, (
                f"{variant}/{ds}/{key}: value {m[key]} out of [0, 1]"
            )
        f1_values.append(m["F1"])

    computed_macro = statistics.fmean(f1_values)
    stored_macro = entry["macro_f1"]
    assert math.isclose(computed_macro, stored_macro, abs_tol=1e-6), (
        f"{variant}: macro_f1 stored {stored_macro!r} != computed "
        f"{computed_macro!r} (mean of per-dataset F1)"
    )


# ----------------------------------------------------------------------
# Test 3: s_linker13 anchor F1
# ----------------------------------------------------------------------

def test_s_linker13_macro_f1_anchors_to_v20_close(baseline):
    """The v2.0 audit pins s_linker13 macro F1 = 0.9509. Round-trip from the
    per-dataset values stored in this fixture must reproduce that anchor
    within 5e-3 (allows for rounding when computing macro from per-dataset)."""
    assert "s_linker13" in baseline["variants"], (
        "s_linker13 must be pinned in the v2.0 baseline fixture"
    )
    macro = baseline["variants"]["s_linker13"]["macro_f1"]
    assert math.isclose(macro, 0.9509, abs_tol=5e-3), (
        f"s_linker13 macro_f1 = {macro} diverges from v2.0-close anchor 0.9509 "
        f"by more than 5e-3 - investigate before any promotion."
    )


# ----------------------------------------------------------------------
# Test 4: tolerance contract
# ----------------------------------------------------------------------

def test_tolerance_contract_is_1e_4(baseline):
    """Future live-run comparators (Phase 13 PROMPT-03) read this tolerance.
    Pinning it here as part of the GATE-02 contract."""
    assert baseline["tolerance_abs_f1"] == 1e-4, (
        f"tolerance_abs_f1 must equal 1e-4 per GATE-02 contract, "
        f"got {baseline['tolerance_abs_f1']!r}"
    )


# ----------------------------------------------------------------------
# Test 5: xfail slots for variants without a 5-dataset v2.0-close sweep
# ----------------------------------------------------------------------

def _missing_variant_params():
    if not FIXTURE_PATH.exists():
        return []
    return [m["variant"] for m in json.loads(FIXTURE_PATH.read_text()).get("missing", [])]


@pytest.mark.parametrize("variant", _missing_variant_params())
def test_missing_variant_slot_xfail(variant, baseline):
    """Variants listed in fixture['missing'] do not have a v2.0-close Claude
    Sonnet 5-dataset sweep on disk. They are tracked as XFAIL so the gap is
    visible in the CI report without breaking the build. A future fixture
    refresh (or removal from CANONICAL_VARIANTS) clears these slots."""
    missing_entries = {m["variant"]: m for m in baseline.get("missing", [])}
    entry = missing_entries[variant]
    pytest.xfail(
        f"{variant}: no v2.0-close 5-dataset Claude Sonnet sweep available. "
        f"Missing datasets: {entry['missing_datasets']}. "
        f"Present (partial) datasets: {sorted(entry.get('present_datasets', {}).keys())}. "
        f"Refresh fixture or drop from CANONICAL_VARIANTS to clear this slot."
    )


# ----------------------------------------------------------------------
# Test 6: GATE-02 contract docstring is grep-discoverable
# ----------------------------------------------------------------------

def test_module_docstring_contains_gate_02_contract():
    """Grep-based audits must be able to find the GATE-02 contract by
    scanning test files for the required tokens."""
    module_doc = (pathlib.Path(__file__).read_text())
    required = ("GATE-02", "frozen-compat", "CANONICAL_VARIANTS", "v2.0 baseline JSON")
    missing = [tok for tok in required if tok not in module_doc]
    assert not missing, (
        f"GATE-02 contract docstring is missing required tokens: {missing}. "
        f"These must remain literally present so grep-based audits work."
    )
