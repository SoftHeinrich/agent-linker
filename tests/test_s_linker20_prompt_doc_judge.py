"""Snapshot tests for SLinker19._prompt_doc_knowledge_judge (phase_1_doc_judge).

For each project in DATASETS:
1. Load the phase_1_doc_judge record from the fixture _calls.json.
2. Reconstruct (comp_names, mapping_list) from the prompt text.
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

_BUILDER = "_prompt_doc_knowledge_judge"
_PHASE_TAG = BUILDER_PHASE_TAGS[_BUILDER][0]   # "phase_1_doc_judge"


@pytest.mark.parametrize("project", DATASETS, ids=lambda p: f"project={p}")
def test_doc_judge_parsed_snapshot(project, snapshot):
    """Snapshot test: parsed structured output for _prompt_doc_knowledge_judge."""
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

    args = reconstruct_inputs(_BUILDER, record, _PHASE_TAG)

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

    parsed = replay_parse(record["response_text"])
    assert parsed == snapshot
