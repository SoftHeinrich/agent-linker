"""Registration + no-inheritance guard for s_linker20 (Phase 47 SHIP, REQ-V264-08).

Covers Phase 47 acceptance criteria:
  1. s_linker20 is registered in CANONICAL_VARIANTS.
  2. s_linker20 is registered in VARIANT_SPECS.
  3. VARIANT_SPECS entry has experimental=True.
  4. VARIANT_SPECS entry has canonical=False.
  5. VARIANT_SPECS entry has correct module and class_name.
  6. SLinker20._VARIANT_NAME == "s_linker20".
  7. SLinker20 does NOT inherit from SLinker19 (no-inheritance guard).
  8. s_linker20.py imports neither prompts_v5 nor s_linker19 (self-contained guard).
"""

from __future__ import annotations

import ast
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 1. Registration in run_ablation
# ─────────────────────────────────────────────────────────────────────────────

def test_registered_in_canonical_variants():
    import run_ablation
    assert "s_linker20" in run_ablation.CANONICAL_VARIANTS


def test_variant_spec_exists():
    import run_ablation
    assert "s_linker20" in run_ablation.VARIANT_SPECS


def test_variant_spec_experimental_true():
    import run_ablation
    spec = run_ablation.VARIANT_SPECS["s_linker20"]
    assert spec.get("experimental") is True


def test_variant_spec_canonical_false():
    import run_ablation
    spec = run_ablation.VARIANT_SPECS["s_linker20"]
    assert spec.get("canonical") is False


def test_variant_spec_module_and_class():
    import run_ablation
    spec = run_ablation.VARIANT_SPECS["s_linker20"]
    assert spec["module"] == "llm_sad_sam.linkers.experimental.s_linker20"
    assert spec["class_name"] == "SLinker20"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Structural guards
# ─────────────────────────────────────────────────────────────────────────────

def test_variant_name():
    from llm_sad_sam.linkers.experimental.s_linker20 import SLinker20
    assert SLinker20._VARIANT_NAME == "s_linker20"


def test_not_subclass_of_slinker19():
    """Arch spec: SLinker20 must NOT inherit from SLinker19."""
    from llm_sad_sam.linkers.experimental.s_linker20 import SLinker20
    from llm_sad_sam.linkers.experimental.s_linker19 import SLinker19
    assert not issubclass(SLinker20, SLinker19)


def test_does_not_import_prompts_v5():
    """Arch spec: s_linker20.py must NOT import prompts_v5 or s_linker19.

    All prompt constants must be inlined (self-contained module).
    """
    src = Path(
        "src/llm_sad_sam/linkers/experimental/s_linker20.py"
    ).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in getattr(node, "names", [])]
            module = getattr(node, "module", "") or ""
            assert "prompts_v5" not in module, (
                "s_linker20 must not import prompts_v5 — constants must be inlined"
            )
            assert "s_linker19" not in module, (
                "s_linker20 must not import s_linker19 — no superclass dependency allowed"
            )
            for name in names:
                assert "prompts_v5" not in name, (
                    "s_linker20 must not import prompts_v5"
                )
                assert "s_linker19" not in name, (
                    "s_linker20 must not import s_linker19"
                )
