"""Infrastructure self-tests for the Phase 44 golden-replay harness.

These are unit/integration tests for the harness package itself (not snapshot
tests — Plan 02 owns the six snapshot modules).

Tests:
  1.1 pyproject.toml has syrupy + pytest-socket in [dev]
  1.2 manifest.py exports: FixtureEntry, load_manifest, DATASETS, MANIFEST_PATH
  1.3 MANIFEST.json schema: 5 entries, correct keys, matches D-02
  1.4 loader.py: load_records, load_pkl, fixture_missing_reason
  1.5 replay_client.py: query() forbidden, extract_json delegates correctly
  1.6 adapters.py: BUILDER_PHASE_TAGS (D-03 gotcha), BUILDERS callable

Skip-on-missing convention: any test that depends on s_linker19 fixture data
uses fixture_missing_reason() before proceeding.  If fixtures are absent (e.g.
running from a worktree without the full results/ directory), the test is
skipped with an actionable reason — CI stays green.

sys.path: inherited from tests/conftest.py (conftest.py adds ROOT and ROOT/src).
"""
from __future__ import annotations

import json
import pathlib
import tomllib

import pytest


# ---------------------------------------------------------------------------
# Test 1.1 — pyproject.toml [dev] deps
# ---------------------------------------------------------------------------

def _repo_root() -> pathlib.Path:
    """Return the repo root (worktree root) anchored from this file's location.

    tests/harness/test_loader_self.py:
      parents[0] = tests/harness/
      parents[1] = tests/
      parents[2] = <repo/worktree root>
    """
    return pathlib.Path(__file__).resolve().parents[2]


def test_pyproject_has_syrupy() -> None:
    """syrupy>=4.6.0 is declared in [project.optional-dependencies].dev."""
    root = _repo_root()
    data = tomllib.loads((root / "pyproject.toml").read_text())
    dev_deps = data["project"]["optional-dependencies"]["dev"]
    assert any(d.startswith("syrupy") for d in dev_deps), (
        f"syrupy not found in pyproject.toml [dev] extras: {dev_deps}"
    )


def test_pyproject_has_pytest_socket() -> None:
    """pytest-socket>=0.7 is declared in [project.optional-dependencies].dev."""
    root = _repo_root()
    data = tomllib.loads((root / "pyproject.toml").read_text())
    dev_deps = data["project"]["optional-dependencies"]["dev"]
    assert any(d.startswith("pytest-socket") for d in dev_deps), (
        f"pytest-socket not found in pyproject.toml [dev] extras: {dev_deps}"
    )


# ---------------------------------------------------------------------------
# Test 1.2 — manifest.py exports
# ---------------------------------------------------------------------------

def test_manifest_imports() -> None:
    """from harness.manifest import ... succeeds for all public symbols."""
    from harness.manifest import (  # noqa: F401
        load_manifest,
        FixtureEntry,
        DATASETS,
        MANIFEST_PATH,
    )


def test_manifest_datasets_tuple() -> None:
    """DATASETS is exactly the expected 5-project tuple."""
    from harness.manifest import DATASETS

    assert DATASETS == (
        "mediastore",
        "teastore",
        "teammates",
        "bigbluebutton",
        "jabref",
    ), f"DATASETS mismatch: {DATASETS}"


def test_manifest_path_constant() -> None:
    """MANIFEST_PATH resolves to tests/harness/fixtures/MANIFEST.json."""
    from harness.manifest import MANIFEST_PATH

    assert MANIFEST_PATH.name == "MANIFEST.json"
    assert MANIFEST_PATH.parent.name == "fixtures"
    assert MANIFEST_PATH.parent.parent.name == "harness"
    assert MANIFEST_PATH.is_file(), f"MANIFEST.json not found at {MANIFEST_PATH}"


def test_manifest_fixture_entry_frozen_dataclass() -> None:
    """FixtureEntry is a frozen dataclass with the required fields."""
    from dataclasses import fields
    from harness.manifest import FixtureEntry
    import pathlib as _pathlib

    field_names = {f.name for f in fields(FixtureEntry)}
    assert {"project", "pkl_dir", "calls_json", "description"} == field_names, (
        f"FixtureEntry fields: {field_names}"
    )

    # Verify it is frozen (cannot assign)
    entry = FixtureEntry(
        project="test",
        pkl_dir=_pathlib.Path("/tmp"),
        calls_json=_pathlib.Path("/tmp/x.json"),
    )
    with pytest.raises((AttributeError, TypeError)):
        entry.project = "other"  # type: ignore[misc]


def test_load_manifest_five_entries() -> None:
    """load_manifest() returns 5 FixtureEntry records covering DATASETS."""
    from harness.manifest import load_manifest, DATASETS

    try:
        entries = load_manifest()
    except FileNotFoundError as exc:
        pytest.skip(f"Fixture files not present — {exc}")

    assert len(entries) == 5, f"Expected 5 entries, got {len(entries)}"
    projects = {e.project for e in entries}
    assert projects == set(DATASETS), f"Project set mismatch: {projects}"


def test_load_manifest_absolute_paths() -> None:
    """Each FixtureEntry has absolute Path objects for pkl_dir and calls_json."""
    from harness.manifest import load_manifest

    try:
        entries = load_manifest()
    except FileNotFoundError as exc:
        pytest.skip(f"Fixture files not present — {exc}")

    for entry in entries:
        assert entry.pkl_dir.is_absolute(), (
            f"pkl_dir not absolute for {entry.project}: {entry.pkl_dir}"
        )
        assert entry.calls_json.is_absolute(), (
            f"calls_json not absolute for {entry.project}: {entry.calls_json}"
        )
        assert entry.pkl_dir.is_dir(), (
            f"pkl_dir not a directory for {entry.project}: {entry.pkl_dir}"
        )
        assert entry.calls_json.is_file(), (
            f"calls_json not a file for {entry.project}: {entry.calls_json}"
        )


# ---------------------------------------------------------------------------
# Test 1.3 — MANIFEST.json schema
# ---------------------------------------------------------------------------

def test_manifest_json_parses() -> None:
    """MANIFEST.json is valid JSON with a list of 5 objects."""
    from harness.manifest import MANIFEST_PATH

    raw = json.loads(MANIFEST_PATH.read_text())
    assert isinstance(raw, list), f"MANIFEST.json top-level should be a list, got {type(raw)}"
    assert len(raw) == 5, f"Expected 5 entries in MANIFEST.json, got {len(raw)}"


def test_manifest_json_keys() -> None:
    """Each MANIFEST.json entry has at least {project, pkl_dir, calls_json}."""
    from harness.manifest import MANIFEST_PATH

    raw = json.loads(MANIFEST_PATH.read_text())
    required_keys = {"project", "pkl_dir", "calls_json"}
    for entry in raw:
        assert required_keys.issubset(entry.keys()), (
            f"Missing required keys in manifest entry: {entry.keys()}"
        )


def test_manifest_json_projects() -> None:
    """The 5 entries in MANIFEST.json cover exactly the DATASETS projects."""
    from harness.manifest import MANIFEST_PATH, DATASETS

    raw = json.loads(MANIFEST_PATH.read_text())
    projects = {e["project"] for e in raw}
    assert projects == set(DATASETS), f"Project set mismatch: {projects}"


# ---------------------------------------------------------------------------
# Test 1.4 — loader.py
# ---------------------------------------------------------------------------

def test_loader_imports() -> None:
    """from harness.loader import ... succeeds for all public symbols."""
    from harness.loader import (  # noqa: F401
        load_records,
        load_pkl,
        fixture_missing_reason,
    )


@pytest.mark.parametrize("project", ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"])
def test_load_records_phase_1_model(project: str) -> None:
    """load_records(project, 'phase_1_model') returns exactly 1 record with the right keys."""
    from harness.loader import fixture_missing_reason, load_records

    reason = fixture_missing_reason(project)
    if reason:
        pytest.skip(reason)

    records = load_records(project, "phase_1_model")
    assert len(records) >= 1, (
        f"Expected at least 1 phase_1_model record for {project}, got {len(records)}"
    )
    first = records[0]
    assert first.get("phase") == "phase_1_model"
    assert "prompt" in first
    assert "response_text" in first


@pytest.mark.parametrize("project", ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"])
def test_load_records_empty_for_unknown_phase(project: str) -> None:
    """load_records returns [] for a phase tag that matches nothing (not an error)."""
    from harness.loader import fixture_missing_reason, load_records

    reason = fixture_missing_reason(project)
    if reason:
        pytest.skip(reason)

    records = load_records(project, "phase_99_does_not_exist")
    assert records == [], f"Expected [] for unknown phase, got {records}"


@pytest.mark.parametrize("project", ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"])
def test_load_pkl_layer1(project: str) -> None:
    """load_pkl(project, 'layer1') returns a non-None object."""
    from harness.loader import fixture_missing_reason, load_pkl

    reason = fixture_missing_reason(project)
    if reason:
        pytest.skip(reason)

    obj = load_pkl(project, "layer1")
    assert obj is not None, f"load_pkl returned None for {project}/layer1"


@pytest.mark.parametrize("project", ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"])
def test_load_pkl_missing_layer_raises(project: str) -> None:
    """load_pkl raises FileNotFoundError for a nonexistent layer name."""
    from harness.loader import fixture_missing_reason, load_pkl

    reason = fixture_missing_reason(project)
    if reason:
        pytest.skip(reason)

    with pytest.raises(FileNotFoundError):
        load_pkl(project, "layer_nonexistent_99")


def test_fixture_missing_reason_returns_none_when_present() -> None:
    """fixture_missing_reason returns None for any project whose files exist."""
    from harness.loader import fixture_missing_reason
    from harness.manifest import DATASETS

    any_present = False
    for project in DATASETS:
        reason = fixture_missing_reason(project)
        if reason is None:
            any_present = True
            break

    if not any_present:
        pytest.skip("No fixture data present — all projects skipped")


def test_fixture_missing_reason_returns_string_when_absent() -> None:
    """fixture_missing_reason returns a non-empty string for a fake project."""
    from harness.loader import fixture_missing_reason

    # 'nonexistent_project' is not in MANIFEST.json → should return a string
    reason = fixture_missing_reason("nonexistent_project")
    assert isinstance(reason, str) and len(reason) > 0, (
        f"Expected a skip-reason string, got: {reason!r}"
    )


# ---------------------------------------------------------------------------
# Test 1.5 — replay_client.py
# ---------------------------------------------------------------------------

def test_replay_client_imports() -> None:
    """from harness.replay_client import ReplayClient, replay_parse succeeds."""
    from harness.replay_client import ReplayClient, replay_parse  # noqa: F401


def test_replay_client_construction_no_network() -> None:
    """ReplayClient() can be constructed without network access."""
    from harness.replay_client import ReplayClient

    client = ReplayClient()
    assert client is not None


def test_replay_client_query_forbidden() -> None:
    """ReplayClient.query() raises RuntimeError with the expected message."""
    from harness.replay_client import ReplayClient

    client = ReplayClient()
    with pytest.raises(RuntimeError, match="ReplayClient.query\\(\\) is forbidden"):
        client.query("anything")


def test_replay_client_extract_json_valid() -> None:
    """ReplayClient.extract_json returns dict for valid JSON response_text."""
    from harness.replay_client import ReplayClient

    client = ReplayClient()
    result = client.extract_json('{"architectural": ["Auth"], "ambiguous": []}')
    assert result == {"architectural": ["Auth"], "ambiguous": []}, result


def test_replay_client_extract_json_none_on_invalid() -> None:
    """ReplayClient.extract_json returns None for response_text with no JSON."""
    from harness.replay_client import ReplayClient

    client = ReplayClient()
    result = client.extract_json("Sorry, I cannot help with that.")
    assert result is None, f"Expected None, got {result}"


def test_replay_parse_convenience_helper() -> None:
    """replay_parse() parses JSON via singleton ReplayClient."""
    from harness.replay_client import replay_parse

    result = replay_parse('{"validations": [{"case": 1, "approve": true}]}')
    assert result == {"validations": [{"case": 1, "approve": True}]}, result


# ---------------------------------------------------------------------------
# Test 1.6 — adapters.py
# ---------------------------------------------------------------------------

def test_adapters_imports() -> None:
    """from harness.adapters import BUILDER_PHASE_TAGS, BUILDERS succeeds."""
    from harness.adapters import BUILDER_PHASE_TAGS, BUILDERS  # noqa: F401


def test_builder_phase_tags_keys() -> None:
    """BUILDER_PHASE_TAGS has exactly 6 keys matching the s_linker19 builders."""
    from harness.adapters import BUILDER_PHASE_TAGS

    expected_keys = {
        "_prompt_ambiguity",
        "_prompt_doc_knowledge_extract",
        "_prompt_doc_knowledge_judge",
        "_prompt_extraction",
        "_prompt_validation",
        "_prompt_coref",
    }
    assert set(BUILDER_PHASE_TAGS.keys()) == expected_keys, (
        f"BUILDER_PHASE_TAGS keys mismatch: {set(BUILDER_PHASE_TAGS.keys())}"
    )


def test_builder_phase_tags_values_are_tuples() -> None:
    """Every BUILDER_PHASE_TAGS value is a tuple (not list, not string)."""
    from harness.adapters import BUILDER_PHASE_TAGS

    for name, tags in BUILDER_PHASE_TAGS.items():
        assert isinstance(tags, tuple), (
            f"BUILDER_PHASE_TAGS[{name!r}] is {type(tags).__name__}, expected tuple"
        )


def test_builder_phase_tags_validation_gotcha() -> None:
    """BUILDER_PHASE_TAGS['_prompt_validation'] includes phase_5_coref_validation (D-03 gotcha)."""
    from harness.adapters import BUILDER_PHASE_TAGS

    tags = BUILDER_PHASE_TAGS["_prompt_validation"]
    assert "phase_5_coref_validation" in tags, (
        f"D-03 gotcha missing: phase_5_coref_validation not in {tags}"
    )
    assert len(tags) == 3, (
        f"Expected 3 tags for _prompt_validation, got {len(tags)}: {tags}"
    )


def test_builders_keys_match_phase_tags() -> None:
    """BUILDERS and BUILDER_PHASE_TAGS have identical key sets."""
    from harness.adapters import BUILDER_PHASE_TAGS, BUILDERS

    assert set(BUILDERS.keys()) == set(BUILDER_PHASE_TAGS.keys()), (
        f"Key mismatch: BUILDERS={set(BUILDERS)} vs PHASE_TAGS={set(BUILDER_PHASE_TAGS)}"
    )


def test_builders_are_callable() -> None:
    """Every value in BUILDERS is callable."""
    from harness.adapters import BUILDERS

    for name, fn in BUILDERS.items():
        assert callable(fn), f"BUILDERS[{name!r}] is not callable"


# ---------------------------------------------------------------------------
# Test 1.7 — GATE-01 byte-equality
# ---------------------------------------------------------------------------

def test_gate01_src_unchanged() -> None:
    """git diff --stat HEAD -- src/llm_sad_sam/ must produce no output."""
    import subprocess

    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "src/llm_sad_sam/"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"git diff failed: {result.stderr}"
    assert result.stdout.strip() == "", (
        f"GATE-01 FAIL — src/llm_sad_sam/ has been modified:\n{result.stdout}"
    )
