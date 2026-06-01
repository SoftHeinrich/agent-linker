"""Registration + structural-guard tests for s_linker14_voyager (Phase 14).

Covers Phase 14 success criteria:
  1. s_linker14_voyager instantiates (backend=CHECKPOINT) and produces valid output structure.
  2. Registration in CANONICAL_VARIANTS + VARIANT_SPECS with experimental=True, canonical=False.
  3. GATE-06 helpers (gate06_ok, reviewer_critic_stub) are callable and return correct types.
  4. Cache reads/writes correctly for a synthetic LLM output; VOYAGER4B_CACHE_ROOT env override works.
  5. GATE-02: frozen-compat check — no frozen artifact modified.
  6. Bank loading: empty bank → axiom-only mode; non-empty bank → patterns injected.
  7. _wrap() wraps axiom with LEARNED PATTERNS section.
  8. _VARIANT_NAME is distinct from all parent classes.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 1. Module import + symbol exports
# ─────────────────────────────────────────────────────────────────────────────

def test_module_imports_required_symbols():
    from llm_sad_sam.linkers.experimental.s_linker14_voyager import (  # noqa: F401
        SLinker14Voyager,
        SLOT_NAMES,
        DEFAULT_BANK_PATH,
        _load_bank,
        _wrap,
    )


def test_slot_names_complete():
    from llm_sad_sam.linkers.experimental.s_linker14_voyager import SLOT_NAMES
    expected = {
        "AMBIGUITY_FEW_SHOT", "AMBIGUITY_RULES", "DOC_KNOWLEDGE_EXTRACTION_RULES",
        "DOC_KNOWLEDGE_JUDGE_EXAMPLES", "DOC_KNOWLEDGE_JUDGE_RULES",
        "ENTITY_EXTRACTION_RULES", "VALIDATION_RULES", "COREF_RULES",
        "SEED_DISAMBIGUATION_RULES",
    }
    assert set(SLOT_NAMES) == expected, f"Missing or extra slots: {set(SLOT_NAMES) ^ expected}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Registration in run_ablation
# ─────────────────────────────────────────────────────────────────────────────

def test_registered_in_canonical_variants():
    import run_ablation
    assert "s_linker14_voyager" in run_ablation.CANONICAL_VARIANTS


def test_variant_spec_exists():
    import run_ablation
    assert "s_linker14_voyager" in run_ablation.VARIANT_SPECS


def test_variant_spec_experimental_true():
    import run_ablation
    spec = run_ablation.VARIANT_SPECS["s_linker14_voyager"]
    assert spec.get("experimental") is True


def test_variant_spec_canonical_false():
    import run_ablation
    spec = run_ablation.VARIANT_SPECS["s_linker14_voyager"]
    assert spec.get("canonical") is False


def test_variant_spec_module_and_class():
    import run_ablation
    spec = run_ablation.VARIANT_SPECS["s_linker14_voyager"]
    assert spec["module"] == "llm_sad_sam.linkers.experimental.s_linker14_voyager"
    assert spec["class_name"] == "SLinker14Voyager"


def test_variant_spec_description_mentions_beta():
    import run_ablation
    desc = run_ablation.VARIANT_SPECS["s_linker14_voyager"]["description"].lower()
    assert "β" in desc or "beta" in desc


# ─────────────────────────────────────────────────────────────────────────────
# 3. Structural guards
# ─────────────────────────────────────────────────────────────────────────────

def test_variant_name_distinct():
    from llm_sad_sam.linkers.experimental.s_linker14_voyager import SLinker14Voyager
    assert SLinker14Voyager._VARIANT_NAME == "s_linker14_voyager"


def test_not_subclass_of_slinker13_clean():
    """Arch spec: must NOT inherit from s_linker13_clean or s_linker13_clean_v3."""
    from llm_sad_sam.linkers.experimental.s_linker14_voyager import SLinker14Voyager
    from llm_sad_sam.linkers.experimental.s_linker13_clean import SLinker13Clean
    from llm_sad_sam.linkers.experimental.s_linker13_clean_v3 import SLinker13CleanV3
    assert not issubclass(SLinker14Voyager, SLinker13Clean)
    assert not issubclass(SLinker14Voyager, SLinker13CleanV3)


def test_does_not_import_prompts_v2():
    """Arch spec: must NOT import prompts_v2 (imports prompts_v3_axiom directly)."""
    import ast
    src = Path(
        "src/llm_sad_sam/linkers/experimental/s_linker14_voyager.py"
    ).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in getattr(node, "names", [])]
            module = getattr(node, "module", "") or ""
            assert "prompts_v2" not in module, "s_linker14_voyager must not import prompts_v2"
            for name in names:
                assert "prompts_v2" not in name, "s_linker14_voyager must not import prompts_v2"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Bank loading
# ─────────────────────────────────────────────────────────────────────────────

def test_load_bank_missing_file_returns_empty():
    from llm_sad_sam.linkers.experimental.s_linker14_voyager import _load_bank
    result = _load_bank("/nonexistent/path/final_bank.json")
    assert result == {}


def test_load_bank_valid_file():
    from llm_sad_sam.linkers.experimental.s_linker14_voyager import _load_bank, SLOT_NAMES
    bank_data = {
        "version": "v4b",
        "slot_patterns": {
            "AMBIGUITY_RULES": [
                {"pattern_id": "p_001", "rule_text": "Test rule.", "example_block": "TP: x\nFP: y"}
            ]
        }
    }
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(bank_data, f)
        path = f.name
    try:
        result = _load_bank(path)
        assert result["AMBIGUITY_RULES"][0]["pattern_id"] == "p_001"
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_bank_invalid_json_returns_empty():
    from llm_sad_sam.linkers.experimental.s_linker14_voyager import _load_bank
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write("not json {")
        path = f.name
    try:
        result = _load_bank(path)
        assert result == {}
    finally:
        Path(path).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. _wrap() behavior
# ─────────────────────────────────────────────────────────────────────────────

def test_wrap_empty_patterns_returns_axiom_unchanged():
    from llm_sad_sam.linkers.experimental.s_linker14_voyager import _wrap
    axiom = "Axiom text here."
    result = _wrap(axiom, "AMBIGUITY_RULES", {})
    assert result == axiom


def test_wrap_with_patterns_adds_learned_section():
    from llm_sad_sam.linkers.experimental.s_linker14_voyager import _wrap
    axiom = "Axiom text here."
    slot_patterns = {
        "AMBIGUITY_RULES": [
            {"pattern_id": "p_001", "rule_text": "Use subject position as signal.", "example_block": ""}
        ]
    }
    result = _wrap(axiom, "AMBIGUITY_RULES", slot_patterns)
    assert "LEARNED PATTERNS" in result
    assert "Use subject position as signal." in result
    assert result.startswith(axiom)


def test_wrap_wrong_slot_leaves_axiom_unchanged():
    from llm_sad_sam.linkers.experimental.s_linker14_voyager import _wrap
    axiom = "Axiom text here."
    slot_patterns = {"COREF_RULES": [{"pattern_id": "p_001", "rule_text": "Some rule."}]}
    result = _wrap(axiom, "AMBIGUITY_RULES", slot_patterns)
    assert result == axiom


# ─────────────────────────────────────────────────────────────────────────────
# 6. Instantiation smoke test (CHECKPOINT backend — no LLM calls)
# ─────────────────────────────────────────────────────────────────────────────

def test_instantiation_checkpoint_empty_bank():
    from llm_sad_sam.linkers.experimental.s_linker14_voyager import SLinker14Voyager
    from llm_sad_sam.llm_client import LLMBackend
    linker = SLinker14Voyager(backend=LLMBackend.CHECKPOINT, bank_path="/nonexistent/bank.json")
    assert linker._VARIANT_NAME == "s_linker14_voyager"
    # Axiom-only mode: wrapped prompts should equal the axiom constants
    from llm_sad_sam.linkers.experimental import prompts_v3_axiom as _ax
    assert linker._AMBIGUITY_RULES == _ax.AMBIGUITY_RULES
    assert linker._COREF_RULES == _ax.COREF_RULES
    assert linker._SEED_DISAMBIGUATION_RULES == _ax.SEED_DISAMBIGUATION_RULES


def test_instantiation_checkpoint_with_bank():
    from llm_sad_sam.linkers.experimental.s_linker14_voyager import SLinker14Voyager, _wrap
    from llm_sad_sam.linkers.experimental import prompts_v3_axiom as _ax
    from llm_sad_sam.llm_client import LLMBackend
    bank_data = {
        "version": "v4b",
        "slot_patterns": {
            "AMBIGUITY_RULES": [
                {"pattern_id": "p_001", "rule_text": "A subject-position name is a stronger signal.", "example_block": "TP: x\nFP: y"}
            ]
        }
    }
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(bank_data, f)
        path = f.name
    try:
        linker = SLinker14Voyager(backend=LLMBackend.CHECKPOINT, bank_path=path)
        # AMBIGUITY_RULES should have LEARNED PATTERNS injected
        assert "LEARNED PATTERNS" in linker._AMBIGUITY_RULES
        assert "A subject-position name" in linker._AMBIGUITY_RULES
        # Other slots should be axiom-only
        assert linker._COREF_RULES == _ax.COREF_RULES
    finally:
        Path(path).unlink(missing_ok=True)


def test_reload_bank_updates_prompts():
    from llm_sad_sam.linkers.experimental.s_linker14_voyager import SLinker14Voyager
    from llm_sad_sam.linkers.experimental import prompts_v3_axiom as _ax
    from llm_sad_sam.llm_client import LLMBackend
    linker = SLinker14Voyager(backend=LLMBackend.CHECKPOINT, bank_path="/nonexistent/bank.json")
    assert linker._AMBIGUITY_RULES == _ax.AMBIGUITY_RULES  # axiom-only

    bank_data = {"version": "v4b", "slot_patterns": {
        "AMBIGUITY_RULES": [{"pattern_id": "p_001", "rule_text": "New rule.", "example_block": ""}]
    }}
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(bank_data, f)
        path = f.name
    try:
        count = linker.reload_bank(path)
        assert count == 1
        assert "New rule." in linker._AMBIGUITY_RULES
    finally:
        Path(path).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 7. GATE-06 helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_gate06_ok_clean_text():
    from scripts.voyager_train_tlr_v4_beta import gate06_ok
    ok, hits = gate06_ok("A subject-position reference is a stronger signal.")
    assert ok is True
    assert hits == []


def test_gate06_ok_taboo_hit():
    from scripts.voyager_train_tlr_v4_beta import gate06_ok
    ok, hits = gate06_ok("The FreeSWITCH component handles calls.")
    assert ok is False
    assert "FreeSWITCH" in hits


def test_gate06_ok_project_name_hit():
    from scripts.voyager_train_tlr_v4_beta import gate06_ok
    ok, hits = gate06_ok("In mediastore the component is called X.")
    assert ok is False


def test_reviewer_critic_stub_returns_dict():
    from scripts.voyager_train_tlr_v4_beta import reviewer_critic_stub
    result = reviewer_critic_stub("A subject-position name is a stronger signal.", "AMBIGUITY_RULES")
    assert isinstance(result, dict)
    assert "verdict" in result
    assert result["verdict"] in ("ACCEPT", "REJECT")


def test_reviewer_critic_stub_accepts_clean_pattern():
    from scripts.voyager_train_tlr_v4_beta import reviewer_critic_stub
    result = reviewer_critic_stub("A subject-position name is a stronger signal.", "AMBIGUITY_RULES")
    assert result["verdict"] == "ACCEPT"


def test_reviewer_critic_stub_rejects_empty():
    from scripts.voyager_train_tlr_v4_beta import reviewer_critic_stub
    result = reviewer_critic_stub("", "AMBIGUITY_RULES")
    assert result["verdict"] == "REJECT"


def test_reviewer_critic_stub_rejects_taboo():
    from scripts.voyager_train_tlr_v4_beta import reviewer_critic_stub
    result = reviewer_critic_stub("The FreeSWITCH rule applies here.", "AMBIGUITY_RULES")
    assert result["verdict"] == "REJECT"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Cache adapter — VOYAGER4B_CACHE_ROOT env override
# ─────────────────────────────────────────────────────────────────────────────

def test_cache_write_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("VOYAGER4B_CACHE_ROOT", str(tmp_path))
    from scripts.voyager_train_tlr_v4_beta import _cache_read, _cache_write
    key = "test_synthetic_llm_output_key_001"
    payload = {"iter": 1, "failure_modes": [{"id": "FM-1", "title": "test"}]}
    _cache_write(key, payload)
    result = _cache_read(key)
    assert result is not None
    assert result["iter"] == 1
    assert result["failure_modes"][0]["id"] == "FM-1"


def test_cache_read_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("VOYAGER4B_CACHE_ROOT", str(tmp_path))
    from scripts.voyager_train_tlr_v4_beta import _cache_read
    result = _cache_read("nonexistent_key_xyz_999")
    assert result is None


def test_cache_root_env_override(tmp_path, monkeypatch):
    custom_root = tmp_path / "custom_cache"
    monkeypatch.setenv("VOYAGER4B_CACHE_ROOT", str(custom_root))
    from scripts.voyager_train_tlr_v4_beta import _cache_write, _cache_path
    key = "override_test_key"
    _cache_write(key, {"test": True})
    p = _cache_path(key)
    assert str(custom_root) in str(p)
    assert p.exists()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Training script dry-run smoke test (success criterion 1)
# ─────────────────────────────────────────────────────────────────────────────

def test_dry_run_probe_single_project(tmp_path, monkeypatch):
    """Phase 14 SC1: voyager_train_tlr_v4_beta.py runs dry-run end-to-end."""
    monkeypatch.setenv("VOYAGER4B_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("VOYAGER4B_OUT_ROOT", str(tmp_path / "out"))
    from scripts.voyager_train_tlr_v4_beta import run_probe, LLMBackend
    result = run_probe(
        projects=["mediastore"],
        backend=LLMBackend.OPENAI,
        model="gpt-5.4",
        dry_run=True,
        split_name="test_dry_run",
    )
    assert result["tier"] == "probe"
    assert result["passes_run"] >= 1
    assert "verdict" in result
    assert result["verdict"] in ("CONTINUE", "KILL")


# ─────────────────────────────────────────────────────────────────────────────
# 10. GATE-02: frozen artifacts unchanged
# ─────────────────────────────────────────────────────────────────────────────

def test_gate02_frozen_artifacts_unchanged():
    """GATE-02: no frozen artifact has uncommitted edits."""
    import subprocess
    frozen = [
        "src/llm_sad_sam/linkers/experimental/prompts_v2.py",
        "src/llm_sad_sam/linkers/experimental/s_linker13.py",
        "src/llm_sad_sam/linkers/experimental/s_linker13_min.py",
        "src/llm_sad_sam/core/data_types_v2.py",
        "src/llm_sad_sam/core/document_loader_v2.py",
        "src/llm_sad_sam/pcm_parser_v2.py",
    ]
    result = subprocess.run(["git", "diff", "--quiet"] + frozen, capture_output=True)
    assert result.returncode == 0, (
        "GATE-02 violation: frozen artifact has uncommitted edits. "
        f"Run: git status {' '.join(frozen)}"
    )
