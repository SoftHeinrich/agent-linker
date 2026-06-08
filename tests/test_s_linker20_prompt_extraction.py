"""Snapshot tests for SLinker19._prompt_extraction (phase_2_framing_c_pass1/pass2).

Parametrized over (project, phase_tag, call_index) because extraction fires multiple
times per project (once per batch of sentences).  The pytest_generate_tests hook
builds the grid lazily so projects/tags with no records produce clear skips.

Per-case behavior:
1. Load the record at (project, phase_tag, call_index) from the fixture _calls.json.
2. Reconstruct (comp_names, mappings, batch) from the prompt text.
3. Assert the rebuilt prompt is byte-equal to record["prompt"] (sanity gate).
4. Replay record["response_text"] through replay_parse.
5. Assert the parsed output equals the committed syrupy snapshot.
"""
from __future__ import annotations

import os

import pytest
from tests.harness.loader import load_records, fixture_missing_reason
from tests.harness.adapters import BUILDERS, BUILDER_PHASE_TAGS
from tests.harness.replay_client import replay_parse
from tests.harness.manifest import DATASETS
from tests.harness.inputs import reconstruct_inputs

_BUILDER = "_prompt_extraction"
_PHASE_TAGS = BUILDER_PHASE_TAGS[_BUILDER]   # ("phase_2_framing_c_pass1", "phase_2_framing_c_pass2")


def pytest_generate_tests(metafunc):
    """Build (project, phase_tag, call_index) parametrize grid at collection time.

    Projects with no fixture data are represented by a single (project, tag, 0)
    slot that skips inside the test body (not at collection time), so the module
    always collects cleanly.
    """
    if "project" in metafunc.fixturenames and "phase_tag" in metafunc.fixturenames:
        params = []
        for project in DATASETS:
            reason = fixture_missing_reason(project)
            for phase_tag in _PHASE_TAGS:
                if reason is not None:
                    # One stub slot per (project, phase_tag) so collection reports the skip
                    params.append(
                        pytest.param(
                            project, phase_tag, 0,
                            id=f"{project}-{phase_tag}-call0",
                        )
                    )
                    continue
                records = load_records(project, phase_tag)
                if not records:
                    params.append(
                        pytest.param(
                            project, phase_tag, 0,
                            id=f"{project}-{phase_tag}-call0",
                        )
                    )
                else:
                    for ci in range(len(records)):
                        params.append(
                            pytest.param(
                                project, phase_tag, ci,
                                id=f"{project}-{phase_tag}-call{ci}",
                            )
                        )
        metafunc.parametrize("project,phase_tag,call_index", params)


def test_extraction_parsed_snapshot(project, phase_tag, call_index, snapshot):
    """Snapshot test: parsed structured output for _prompt_extraction."""
    reason = fixture_missing_reason(project)
    if reason is not None:
        pytest.skip(reason)

    records = load_records(project, phase_tag)
    if not records:
        pytest.skip(f"no records for {project!r}/{phase_tag!r}")
    if call_index >= len(records):
        pytest.skip(f"no record at {project}/{phase_tag}/{call_index}")

    record = records[call_index]

    args = reconstruct_inputs(_BUILDER, record, phase_tag)

    rebuilt_prompt = BUILDERS[_BUILDER](*args)
    # Phase 46 D-01 gate: production-mode runs assert prompt-equality (no CI
    # regression); scratch-mode runs skip it so prompt cuts to tests/scratch/
    # don't trivially fail. Parsed-output snapshot below remains the active gate.
    if os.environ.get("SAD_SAM_LINKER_SOURCE", "production") != "scratch":
        assert rebuilt_prompt == record["prompt"], (
            f"Prompt rebuild mismatch for builder={_BUILDER!r} "
            f"project={project!r} phase_tag={phase_tag!r} call_index={call_index} — "
            f"first 200-char diff: rebuilt={rebuilt_prompt[:200]!r} "
            f"vs logged={record['prompt'][:200]!r}"
        )

    parsed = replay_parse(record["response_text"])
    assert parsed == snapshot
