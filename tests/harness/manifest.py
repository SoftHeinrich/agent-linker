"""Manifest reader for Phase 44 golden-replay fixture infrastructure.

Exposes:
- MANIFEST_PATH  — absolute Path to tests/harness/fixtures/MANIFEST.json
- DATASETS       — canonical 5-project tuple (order matches manifest)
- FixtureEntry   — frozen dataclass: (project, pkl_dir, calls_json, description?)
- load_manifest() — returns list[FixtureEntry] with absolute Paths; raises
                    FileNotFoundError if any listed pkl_dir or calls_json is absent.

sys.path bootstrap: inherited from tests/conftest.py when invoked via pytest.
Do NOT modify sys.path in this module.
"""
from __future__ import annotations

import json
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_PATH: Path = (
    Path(__file__).resolve().parent / "fixtures" / "MANIFEST.json"
)

# Canonical project tuple — matches manifest order.
DATASETS: tuple[str, ...] = (
    "mediastore",
    "teastore",
    "teammates",
    "bigbluebutton",
    "jabref",
)


# ---------------------------------------------------------------------------
# FixtureEntry dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FixtureEntry:
    """Pinned (project, pkl_dir, calls_json) triple for one gpt-5.4 baseline run."""

    project: str
    pkl_dir: Path
    calls_json: Path
    description: Optional[str] = None


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def load_manifest() -> list[FixtureEntry]:
    """Parse MANIFEST.json and return 5 FixtureEntry records.

    Each entry's pkl_dir and calls_json are resolved to absolute Paths against
    the repo root (MANIFEST_PATH.parents[3], which equals tests/conftest.py's
    ROOT = parents[1] of tests/conftest.py).

    Raises:
        FileNotFoundError: if any listed pkl_dir directory or calls_json file
            is absent on disk.  The error message includes the relative path
            from the manifest so the caller can identify the missing artefact.
    """
    # Repo root: tests/harness/fixtures/MANIFEST.json -> parents[3] = repo root
    repo_root: Path = MANIFEST_PATH.parents[3]

    raw: list[dict] = json.loads(MANIFEST_PATH.read_text())

    entries: list[FixtureEntry] = []
    for item in raw:
        pkl_dir = (repo_root / item["pkl_dir"]).resolve()
        calls_json = (repo_root / item["calls_json"]).resolve()

        if not pkl_dir.is_dir():
            raise FileNotFoundError(
                f"MANIFEST.json entry '{item['project']}': pkl_dir not found: "
                f"{item['pkl_dir']!r} (resolved to {pkl_dir})"
            )
        if not calls_json.is_file():
            raise FileNotFoundError(
                f"MANIFEST.json entry '{item['project']}': calls_json not found: "
                f"{item['calls_json']!r} (resolved to {calls_json})"
            )

        entries.append(
            FixtureEntry(
                project=item["project"],
                pkl_dir=pkl_dir,
                calls_json=calls_json,
                description=item.get("description"),
            )
        )

    return entries
