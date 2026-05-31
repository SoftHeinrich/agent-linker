"""Smoke tests for the Phase 12-02 single-step ablation harness.

Tests are grouped in two layers:

1. **Contract tests** (Task 1): pin PHASE_ORDER and DOWNSTREAM_DEPS, the
   canonical phase-to-downstream re-run table that 12-03/04/05 plans cite.
2. **Engine tests** (Task 2): smoke-test `run_single_step` against the
   cached `s_linker13_clean` checkpoints using the checkpoint backend so
   no live LLM is called.

All Task 2 tests are skipped automatically if the baseline checkpoint
fixtures under `results/phase_cache/s_linker13_clean/mediastore/` do not
exist; this keeps the contract layer green even in clean CI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASELINE_CACHE = ROOT / "results" / "phase_cache" / "s_linker13_clean" / "mediastore"
REQUIRED_PKLS = ("layer1.pkl", "layer2.pkl", "entity_candidates.pkl",
                 "entity_decisions.pkl", "final.pkl")

_baseline_present = all((BASELINE_CACHE / name).exists() for name in REQUIRED_PKLS)
requires_baseline = pytest.mark.skipif(
    not _baseline_present,
    reason=("baseline s_linker13_clean checkpoints missing under "
            f"{BASELINE_CACHE} — Task 2 smoke tests require Phase 10 fixtures"),
)


# ============================================================
# Task 1: Contract tests — pin PHASE_ORDER + DOWNSTREAM_DEPS
# ============================================================

def test_phase_order_constant():
    from llm_sad_sam.ablation.single_step import PHASE_ORDER
    assert PHASE_ORDER == (
        "layer1", "layer2", "entity_candidates", "entity_decisions", "final"
    )


def test_downstream_deps_layer1():
    from llm_sad_sam.ablation.single_step import DOWNSTREAM_DEPS
    assert DOWNSTREAM_DEPS["layer1"] == (
        "layer2", "entity_candidates", "entity_decisions", "final"
    )


def test_downstream_deps_layer2():
    from llm_sad_sam.ablation.single_step import DOWNSTREAM_DEPS
    assert DOWNSTREAM_DEPS["layer2"] == ("final",)


def test_downstream_deps_entity_candidates():
    from llm_sad_sam.ablation.single_step import DOWNSTREAM_DEPS
    assert DOWNSTREAM_DEPS["entity_candidates"] == ("entity_decisions", "final")


def test_downstream_deps_entity_decisions():
    from llm_sad_sam.ablation.single_step import DOWNSTREAM_DEPS
    assert DOWNSTREAM_DEPS["entity_decisions"] == ("final",)


def test_downstream_deps_final():
    from llm_sad_sam.ablation.single_step import DOWNSTREAM_DEPS
    assert DOWNSTREAM_DEPS["final"] == ()


# ============================================================
# Task 2: Engine tests — smoke-test run_single_step + CLI
# ============================================================

@requires_baseline
def test_run_single_step_baseline_equivalence(tmp_path):
    """phase=layer1 on the unchanged variant reproduces baseline F1.

    Uses backend="checkpoint" so every LLM query hits the disk cache; no
    live LLM call is made. The F1 must lie within 0.02 of the cached
    final.pkl-derived F1 — there is no semantic change, so this is the
    no-op contract.
    """
    from llm_sad_sam.ablation.single_step import run_single_step
    result = run_single_step(
        variant="s_linker13_clean",
        dataset="mediastore",
        phase="layer1",
        results_dir=tmp_path,
        backend="checkpoint",
    )
    assert result["variant"] == "s_linker13_clean"
    assert result["dataset"] == "mediastore"
    assert result["phase"] == "layer1"
    for key in ("F1", "P", "R", "fp", "fn", "baseline_F1", "delta_F1"):
        assert key in result, f"missing key {key} in {result!r}"
    # No-op contract: the harness produced the same F1 as the cached pipeline.
    assert abs(result["delta_F1"]) <= 0.02, (
        f"baseline-equivalence violated: F1={result['F1']}, "
        f"baseline_F1={result['baseline_F1']}, delta={result['delta_F1']}"
    )
    out = tmp_path / "s_linker13_clean" / "mediastore" / "layer1.json"
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["phase"] == "layer1"


def test_run_single_step_unknown_phase(tmp_path):
    from llm_sad_sam.ablation.single_step import run_single_step
    with pytest.raises(ValueError, match="not_a_real_phase"):
        run_single_step(
            variant="s_linker13_clean",
            dataset="mediastore",
            phase="not_a_real_phase",
            results_dir=tmp_path,
            backend="checkpoint",
        )


@requires_baseline
def test_run_single_step_missing_upstream(tmp_path):
    """phase=entity_decisions requires layer1 + entity_candidates pickles.

    Point phase_cache_dir at an empty tmp dir; the harness must raise
    FileNotFoundError naming the missing pickle, NOT silently regenerate.
    """
    from llm_sad_sam.ablation.single_step import run_single_step
    empty_cache = tmp_path / "empty_cache"
    empty_cache.mkdir()
    with pytest.raises(FileNotFoundError) as exc:
        run_single_step(
            variant="s_linker13_clean",
            dataset="mediastore",
            phase="entity_decisions",
            results_dir=tmp_path / "out",
            backend="checkpoint",
            phase_cache_dir=str(empty_cache),
        )
    msg = str(exc.value)
    assert "layer1.pkl" in msg or "entity_candidates.pkl" in msg


@requires_baseline
def test_cli_runs_layer1_smoke(tmp_path):
    """CLI invocation must exit 0 and write the expected JSON."""
    cmd = [
        sys.executable, "-m", "llm_sad_sam.ablation",
        "single_step",
        "--variant", "s_linker13_clean",
        "--dataset", "mediastore",
        "--phase", "layer1",
        "--results-dir", str(tmp_path),
        "--backend", "checkpoint",
    ]
    env = dict(os.environ)
    # Ensure src is on PYTHONPATH for subprocess
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=env)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    out = tmp_path / "s_linker13_clean" / "mediastore" / "layer1.json"
    assert out.exists(), f"expected {out}; stdout={proc.stdout}\nstderr={proc.stderr}"
    data = json.loads(out.read_text())
    assert data["variant"] == "s_linker13_clean"
    assert data["phase"] == "layer1"


def test_cli_rejects_unknown_variant(tmp_path):
    cmd = [
        sys.executable, "-m", "llm_sad_sam.ablation",
        "single_step",
        "--variant", "not_a_variant",
        "--dataset", "mediastore",
        "--phase", "layer1",
        "--results-dir", str(tmp_path),
        "--backend", "checkpoint",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=env)
    assert proc.returncode != 0
    assert "not_a_variant" in (proc.stdout + proc.stderr)


def test_cli_rejects_unknown_dataset(tmp_path):
    cmd = [
        sys.executable, "-m", "llm_sad_sam.ablation",
        "single_step",
        "--variant", "s_linker13_clean",
        "--dataset", "not_a_dataset",
        "--phase", "layer1",
        "--results-dir", str(tmp_path),
        "--backend", "checkpoint",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=env)
    assert proc.returncode != 0
    assert "not_a_dataset" in (proc.stdout + proc.stderr)
