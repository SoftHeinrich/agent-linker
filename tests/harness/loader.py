"""Fixture loader for Phase 44 golden-replay harness.

Provides:
- load_records(project, phase_tag)  — filter _calls.json by phase tag, return list[dict]
- load_pkl(project, layer)          — deserialise results/phase_cache/s_linker19/openai/<project>/<layer>.pkl
- fixture_missing_reason(project)   — None if all files present, otherwise a skip-reason string

sys.path bootstrap: inherited from tests/conftest.py (tests/harness/ is under tests/).
Do NOT modify sys.path in this module.
"""
from __future__ import annotations

import functools
import json
import pickle
from pathlib import Path
from typing import Optional

from harness.manifest import load_manifest, FixtureEntry

# Layer names that must be present for a project to be considered complete.
_REQUIRED_LAYERS: tuple[str, ...] = (
    "layer1",
    "layer2",
    "layer3",
    "layer4",
    "final",
)


# ---------------------------------------------------------------------------
# Internal helpers (lazy + cached)
# ---------------------------------------------------------------------------

def _get_entry(project: str) -> FixtureEntry:
    """Return the FixtureEntry for *project*, or raise KeyError if not found."""
    entries = {e.project: e for e in load_manifest()}
    if project not in entries:
        raise KeyError(
            f"Project {project!r} not found in MANIFEST.json. "
            f"Known projects: {sorted(entries)}"
        )
    return entries[project]


@functools.lru_cache(maxsize=32)
def _load_calls_json(project: str) -> list[dict]:
    """Load and cache the entire _calls.json for *project*."""
    entry = _get_entry(project)
    return json.loads(entry.calls_json.read_text())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_records(project: str, phase_tag: str) -> list[dict]:
    """Return all _calls.json records for *project* whose phase equals *phase_tag*.

    Records are returned in original order.  An empty list is returned (not an
    error) if no record matches *phase_tag*.

    Args:
        project:   one of the 5 project names in DATASETS
        phase_tag: e.g. "phase_1_model", "phase_4_twopass_p1"

    Returns:
        list of raw record dicts from the _calls.json file, filtered by phase.
    """
    records = _load_calls_json(project)
    return [r for r in records if r.get("phase") == phase_tag]


def load_pkl(project: str, layer: str) -> object:
    """Deserialise results/phase_cache/s_linker19/openai/<project>/<layer>.pkl.

    Args:
        project: one of the 5 project names in DATASETS
        layer:   one of "layer1", "layer2", "layer3", "layer4", "final"

    Returns:
        The deserialised Python object (a dataclass from data_types_v2.py).

    Raises:
        FileNotFoundError: if the .pkl file is absent.
    """
    entry = _get_entry(project)
    pkl_path: Path = entry.pkl_dir / f"{layer}.pkl"
    if not pkl_path.is_file():
        raise FileNotFoundError(
            f"PKL not found for project={project!r}, layer={layer!r}: {pkl_path}"
        )
    with open(pkl_path, "rb") as fh:
        return pickle.load(fh)


def fixture_missing_reason(project: str) -> Optional[str]:
    """Return None if all fixture files for *project* are present on disk.

    Returns a human-readable string suitable as a pytest.skip(reason=...) argument
    if any required file is absent.  Mirrors the skip-on-missing convention from
    tests/test_single_step_harness.py.

    Checks performed:
    - The manifest entry for *project* exists
    - entry.calls_json is a file
    - entry.pkl_dir / "{layer}.pkl" exists for each layer in ("layer1".."layer4", "final")
    """
    try:
        # load_manifest() raises FileNotFoundError if the manifest lists missing files
        entry = _get_entry(project)
    except (FileNotFoundError, KeyError) as exc:
        return str(exc)

    if not entry.calls_json.is_file():
        return (
            f"calls_json missing for project={project!r}: {entry.calls_json} — "
            "re-run s_linker19 with --backend openai to regenerate"
        )

    for layer in _REQUIRED_LAYERS:
        pkl_path = entry.pkl_dir / f"{layer}.pkl"
        if not pkl_path.is_file():
            return (
                f"PKL missing for project={project!r}, layer={layer!r}: {pkl_path} — "
                "re-run s_linker19 with --backend openai to regenerate"
            )

    return None
