"""Tests for s_linker13_clean_v3 import + registration (Phase 12 Step 0).

Verifies the thin sibling variant:
1. Imports cleanly and exposes SLinker13CleanV3.
2. Has _VARIANT_NAME == "s_linker13_clean_v3" (separate checkpoint subdir).
3. Is a standalone class (subclassing object, not the parent SLinker13Clean).
4. Is registered in run_ablation.CANONICAL_VARIANTS.
5. Has the expected VARIANT_SPECS shape (canonical=False, class_name set).
"""
from __future__ import annotations

import pathlib
import sys

# run_ablation.py sits at the project root, not on sys.path under pytest's default
# rootdir-based collection. Match the pattern used by test_v20_baseline_regression.py
# (which is invoked with PYTHONPATH=.) and inject the project root explicitly so this
# test file is invocable both ways.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def test_import_s_linker13_clean_v3():
    from llm_sad_sam.linkers.experimental.s_linker13_clean_v3 import SLinker13CleanV3
    assert SLinker13CleanV3 is not None


def test_variant_name_separate_from_parent():
    from llm_sad_sam.linkers.experimental.s_linker13_clean_v3 import SLinker13CleanV3
    assert SLinker13CleanV3._VARIANT_NAME == "s_linker13_clean_v3"


def test_standalone_class_not_subclass():
    """V3 sibling must be standalone (bases == (object,)) per Plan 12-01 rules."""
    from llm_sad_sam.linkers.experimental.s_linker13_clean_v3 import SLinker13CleanV3
    assert SLinker13CleanV3.__bases__ == (object,), (
        f"SLinker13CleanV3 must be standalone, got bases={SLinker13CleanV3.__bases__}"
    )


def test_registered_in_canonical_variants():
    from run_ablation import CANONICAL_VARIANTS
    assert "s_linker13_clean_v3" in CANONICAL_VARIANTS


def test_variant_specs_canonical_false():
    from run_ablation import VARIANT_SPECS
    spec = VARIANT_SPECS["s_linker13_clean_v3"]
    assert spec["canonical"] is False, f"expected canonical=False, got {spec.get('canonical')}"


def test_variant_specs_class_name():
    from run_ablation import VARIANT_SPECS
    spec = VARIANT_SPECS["s_linker13_clean_v3"]
    assert spec["class_name"] == "SLinker13CleanV3"
    assert spec["module"] == "llm_sad_sam.linkers.experimental.s_linker13_clean_v3"
