"""Snapshot tests for SLinker19._prompt_ambiguity (phase_1_model).

For each project in DATASETS:
1. Load the phase_1_model record from the fixture _calls.json.
2. Reconstruct the builder inputs from the prompt text.
3. Assert the rebuilt prompt is byte-equal to record["prompt"] (sanity gate).
4. Replay record["response_text"] through replay_parse.
5. Assert the parsed output equals the committed syrupy snapshot.

All parametrized cases skip gracefully (per-project, not whole-module) when
fixture data is absent, so partial fixture sets keep CI green.
"""
from __future__ import annotations

import os

import pytest
from tests.harness.loader import load_records, fixture_missing_reason
from tests.harness.adapters import BUILDERS, BUILDER_PHASE_TAGS
from tests.harness.replay_client import replay_parse
from tests.harness.manifest import DATASETS
from tests.harness.inputs import reconstruct_inputs

_BUILDER = "_prompt_ambiguity"
_PHASE_TAG = BUILDER_PHASE_TAGS[_BUILDER][0]   # "phase_1_model"


@pytest.mark.parametrize("project", DATASETS, ids=lambda p: f"project={p}")
def test_ambiguity_parsed_snapshot(project, snapshot):
    """Snapshot test: parsed structured output for _prompt_ambiguity.

    Step 6 (byte-equality sanity gate): if the rebuilt prompt diverges from the
    logged prompt, the D-03 builder→tag mapping is wrong and every downstream
    snapshot assertion is meaningless.  The assertion message names the triple
    so failures are one-liner debuggable.
    """
    reason = fixture_missing_reason(project)
    if reason is not None:
        pytest.skip(reason)

    records = load_records(project, _PHASE_TAG)
    if not records:
        pytest.skip(f"no records for {project!r}/{_PHASE_TAG!r}")

    call_index = 0
    if call_index >= len(records):
        pytest.skip(f"no record at {project}/{_PHASE_TAG}/{call_index}")

    record = records[call_index]

    # Reconstruct builder inputs from the logged prompt.
    args = reconstruct_inputs(_BUILDER, record, _PHASE_TAG)

    # Step 6: byte-equality sanity gate.
    rebuilt_prompt = BUILDERS[_BUILDER](*args)
    # Phase 46 D-01 gate: production-mode runs assert prompt-equality (no CI
    # regression); scratch-mode runs skip it so prompt cuts to tests/scratch/
    # don't trivially fail. Parsed-output snapshot below remains the active gate.
    if os.environ.get("SAD_SAM_LINKER_SOURCE", "production") != "scratch":
        assert rebuilt_prompt == record["prompt"], (
            f"Prompt rebuild mismatch for builder={_BUILDER!r} "
            f"project={project!r} phase_tag={_PHASE_TAG!r} call_index={call_index} — "
            f"first 200-char diff: rebuilt={rebuilt_prompt[:200]!r} "
            f"vs logged={record['prompt'][:200]!r}"
        )

    # Steps 7–8: replay parse + snapshot assertion.
    parsed = replay_parse(record["response_text"])
    assert parsed == snapshot
