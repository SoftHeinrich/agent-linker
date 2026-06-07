"""Snapshot tests for SLinker19._prompt_coref (phase_5_coref).

NOTE — D-03 gotcha: the coref-validation phase tag is handled by the validation
test module, NOT this module.  This module covers ONLY phase_5_coref.

Parametrized over (project, call_index) because coref fires multiple times
per project (once per batch of 10 sentences).  The pytest_generate_tests hook
builds the grid lazily so projects with no records produce clear skips.

Per-case behavior:
1. Load the record at (project, call_index) from the fixture _calls.json.
2. Reconstruct (comp_names, cases) from the prompt text.
3. Assert the rebuilt prompt is byte-equal to record["prompt"] (sanity gate).
4. Replay record["response_text"] through replay_parse.
5. Assert the parsed output equals the committed syrupy snapshot.
"""
from __future__ import annotations

import pytest
from tests.harness.loader import load_records, fixture_missing_reason
from tests.harness.adapters import BUILDERS, BUILDER_PHASE_TAGS
from tests.harness.replay_client import replay_parse
from tests.harness.manifest import DATASETS
from tests.harness.inputs import reconstruct_inputs

_BUILDER = "_prompt_coref"
_PHASE_TAG = BUILDER_PHASE_TAGS[_BUILDER][0]   # "phase_5_coref"


def pytest_generate_tests(metafunc):
    """Build (project, call_index) parametrize grid at collection time."""
    if "project" in metafunc.fixturenames and "call_index" in metafunc.fixturenames:
        params = []
        for project in DATASETS:
            reason = fixture_missing_reason(project)
            if reason is not None:
                params.append(
                    pytest.param(project, 0, id=f"{project}-call0")
                )
                continue
            records = load_records(project, _PHASE_TAG)
            if not records:
                params.append(
                    pytest.param(project, 0, id=f"{project}-call0")
                )
            else:
                for ci in range(len(records)):
                    params.append(
                        pytest.param(project, ci, id=f"{project}-call{ci}")
                    )
        metafunc.parametrize("project,call_index", params)


def test_coref_parsed_snapshot(project, call_index, snapshot):
    """Snapshot test: parsed structured output for _prompt_coref.

    Covers only phase_5_coref.
    See test_s_linker20_prompt_validation.py for the coref-validation phase tag
    (D-03 gotcha: that tag reuses _prompt_validation, not _prompt_coref).
    """
    reason = fixture_missing_reason(project)
    if reason is not None:
        pytest.skip(reason)

    records = load_records(project, _PHASE_TAG)
    if not records:
        pytest.skip(f"no records for {project!r}/{_PHASE_TAG!r}")
    if call_index >= len(records):
        pytest.skip(f"no record at {project}/{_PHASE_TAG}/{call_index}")

    record = records[call_index]

    args = reconstruct_inputs(_BUILDER, record, _PHASE_TAG)

    rebuilt_prompt = BUILDERS[_BUILDER](*args)
    assert rebuilt_prompt == record["prompt"], (
        f"Prompt rebuild mismatch for builder={_BUILDER!r} "
        f"project={project!r} phase_tag={_PHASE_TAG!r} call_index={call_index} — "
        f"first 200-char diff: rebuilt={rebuilt_prompt[:200]!r} "
        f"vs logged={record['prompt'][:200]!r}"
    )

    parsed = replay_parse(record["response_text"])
    assert parsed == snapshot
