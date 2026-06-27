---
phase: 44
plan: "01"
plan_id: 44-01
subsystem: test-harness
tags: [fixture-infrastructure, replay-harness, snapshot-testing, zero-llm-calls]
dependency_graph:
  requires: []
  provides:
    - tests.harness.manifest.load_manifest
    - tests.harness.manifest.FixtureEntry
    - tests.harness.manifest.DATASETS
    - tests.harness.manifest.MANIFEST_PATH
    - tests.harness.loader.load_records
    - tests.harness.loader.load_pkl
    - tests.harness.loader.fixture_missing_reason
    - tests.harness.replay_client.ReplayClient
    - tests.harness.replay_client.replay_parse
    - tests.harness.adapters.BUILDER_PHASE_TAGS
    - tests.harness.adapters.BUILDERS
  affects:
    - pyproject.toml (dev extras)
tech_stack:
  added:
    - syrupy>=4.6.0 (snapshot testing library, [dev] extra)
    - pytest-socket>=0.7 (network-disable plugin, [dev] extra)
  patterns:
    - frozen dataclass for manifest entries
    - functools.lru_cache for lazy manifest/JSON loading
    - skip-on-missing convention from test_single_step_harness.py
    - D-03 phase-tag mapping encoded as module-level constants
key_files:
  created:
    - tests/harness/__init__.py
    - tests/harness/fixtures/__init__.py
    - tests/harness/fixtures/MANIFEST.json
    - tests/harness/manifest.py
    - tests/harness/loader.py
    - tests/harness/replay_client.py
    - tests/harness/adapters.py
    - tests/harness/test_loader_self.py
  modified:
    - pyproject.toml
decisions:
  - "LOCK-1: syrupy>=4.6.0 chosen as snapshot library (AmberSerializer default; pytest-regressions NOT installed)"
  - "LOCK-2: parser isolation = call SLinker19._prompt_* staticmethods directly + replay through LLMClient.extract_json"
  - "LOCK-3: per (project, call_index) parametrization for multi-call builders; per project for single-call"
  - "LOCK-4: MANIFEST.json schema = list of {project, pkl_dir, calls_json} — no sha256 in Phase 44"
  - "LOCK-5: fixtures under tests/harness/; test modules at tests/test_s_linker20_prompt_*.py (Plan 02)"
  - "LOCK-6: zero-LLM-call = ReplayClient.query raises RuntimeError + pytest-socket --disable-socket (Plan 02)"
  - "MANIFEST paths: relative (repo-root-relative); load_manifest() resolves to absolute Paths"
metrics:
  duration: "~25 minutes"
  completed: "2026-06-06T03:21:25Z"
  tasks_completed: 1
  tasks_total: 1
  files_created: 8
  files_modified: 1
---

# Phase 44 Plan 01: Fixture Infrastructure Summary

**One-liner:** `tests/harness/` package providing manifest-driven pkl/calls-json fixture loading + ReplayClient with forbidden .query() for zero-LLM-call Phase 44 harness.

## What Was Built

Complete fixture infrastructure for the Phase 44 golden-replay harness:

### 5 MANIFEST Entries (D-02 pinning)

| project | pkl_dir | calls_json |
|---|---|---|
| mediastore | `results/phase_cache/s_linker19/openai/mediastore/` | `results/llm_logs/s_linker19_openai_mediastore_20260605_134622_calls.json` |
| teastore | `results/phase_cache/s_linker19/openai/teastore/` | `results/llm_logs/s_linker19_openai_teastore_20260604_065824_calls.json` |
| teammates | `results/phase_cache/s_linker19/openai/teammates/` | `results/llm_logs/s_linker19_openai_teammates_20260604_070526_calls.json` |
| bigbluebutton | `results/phase_cache/s_linker19/openai/bigbluebutton/` | `results/llm_logs/s_linker19_openai_bigbluebutton_20260604_070639_calls.json` |
| jabref | `results/phase_cache/s_linker19/openai/jabref/` | `results/llm_logs/s_linker19_openai_jabref_20260605_134705_calls.json` |

All 5 paths verified on disk at task close from main checkout.

### Locked Decisions (LOCK-1 through LOCK-6)

- **LOCK-1:** Snapshot library = syrupy>=4.6.0 (AmberSerializer, parsed-output dict snapshots). pytest-regressions NOT installed.
- **LOCK-2:** Parser isolation = option (c): call `SLinker19._prompt_*` @staticmethods directly + replay `response_text` through `LLMClient.extract_json`. NO monkey-patching of s_linker19; NO refactor of builders.
- **LOCK-3:** Parametrization granularity = per `(project, call_index)` for multi-call builders; per `project` for single-call. Loader exposes records as ordered list; test modules choose granularity.
- **LOCK-4:** MANIFEST.json carries NO `sha256` field in Phase 44. Schema = list of `{project, pkl_dir, calls_json}`. Optional `description` added.
- **LOCK-5:** fixtures + loader under `tests/harness/`; six test modules at `tests/test_s_linker20_prompt_*.py` (Plan 02). Snapshots default to `tests/__snapshots__/` (syrupy default).
- **LOCK-6:** Zero-LLM-call enforcement = BOTH (a) `ReplayClient.query` raises RuntimeError + (b) `pytest-socket --disable-socket` flag (Plan 02 adds the flag).

### GATE-01 Status: PASS

`git diff --stat HEAD -- src/llm_sad_sam/` produces empty output. All files under `src/llm_sad_sam/` are byte-equal to HEAD: s_linker19.py, s_linker13_min.py, prompts_v5.py, data_types_v2.py, llm_client.py.

### syrupy + pytest-socket Installation Confirmation

Both installed successfully via `pip install -e ".[dev,openai]"`:
- `import syrupy` OK
- `import pytest_socket` OK
- pyproject.toml `[project.optional-dependencies].dev` confirmed by `tomllib.loads` assertion.

## Test Results

```
pytest tests/harness/test_loader_self.py -v
======================== 24 passed, 23 skipped in 0.10s ========================
```

- 24 passed: pyproject checks, manifest imports, MANIFEST.json schema, loader imports, replay client contract, adapter mapping, GATE-01
- 23 skipped: fixture-data-dependent tests (pkl_dir / calls_json not present in worktree; skip-on-missing fires correctly)

## Plan 02 Handoff

### BUILDERS map keys (6):
```python
"_prompt_ambiguity"
"_prompt_doc_knowledge_extract"
"_prompt_doc_knowledge_judge"
"_prompt_extraction"
"_prompt_validation"
"_prompt_coref"
```

### BUILDER_PHASE_TAGS schema:
```python
{
    "_prompt_ambiguity":            ("phase_1_model",),
    "_prompt_doc_knowledge_extract": ("phase_1_doc_extract",),
    "_prompt_doc_knowledge_judge":   ("phase_1_doc_judge",),
    "_prompt_extraction":            ("phase_2_framing_c_pass1", "phase_2_framing_c_pass2"),
    "_prompt_validation":            ("phase_4_twopass_p1", "phase_4_twopass_p2", "phase_5_coref_validation"),
    "_prompt_coref":                 ("phase_5_coref",),
}
```
D-03 gotcha encoded: `phase_5_coref_validation` is in `_prompt_validation`, not `_prompt_coref`.

### `replay_parse()` signature:
```python
def replay_parse(response_text: str) -> dict | None
```
Singleton ReplayClient wrapper; delegates to `LLMClient.extract_json` via `LLMResponse(text=response_text, success=True)`.

### `fixture_missing_reason()` signature:
```python
def fixture_missing_reason(project: str) -> str | None
```
Returns `None` if all 5 PKL layers + calls_json exist; returns skip-reason string otherwise. Use with `pytest.skip(fixture_missing_reason(project))` pattern.

## Deviations from Plan

**1. [Rule 1 - Bug] Fixed pyproject.toml root path in test_loader_self.py**
- **Found during:** Task 1 verification run
- **Issue:** `parents[3]` from `tests/harness/test_loader_self.py` resolves to `worktrees/` not the worktree root; correct depth is `parents[2]`
- **Fix:** Added `_repo_root()` helper using `parents[2]` for worktree root from test file location
- **Files modified:** `tests/harness/test_loader_self.py`
- **Commit:** db98fbe (included in task commit)

No other deviations — plan executed as written.

## Known Stubs

None. All public APIs are fully implemented and wired. No hardcoded empty values flowing to test output.

## Threat Flags

No new trust boundaries introduced beyond those in the plan's threat register. T-44-03 mitigation implemented: `ReplayClient.query()` raises RuntimeError; `ReplayClient.extract_json()` delegates to `LLMClient.extract_json` which is pure `json.loads`.

## Self-Check: PASS

Files confirmed:
- tests/harness/__init__.py: EXISTS
- tests/harness/fixtures/__init__.py: EXISTS
- tests/harness/fixtures/MANIFEST.json: EXISTS (5 entries, all D-02 paths)
- tests/harness/manifest.py: EXISTS (exports FixtureEntry, load_manifest, DATASETS, MANIFEST_PATH)
- tests/harness/loader.py: EXISTS (exports load_records, load_pkl, fixture_missing_reason)
- tests/harness/replay_client.py: EXISTS (exports ReplayClient, replay_parse)
- tests/harness/adapters.py: EXISTS (exports BUILDER_PHASE_TAGS, BUILDERS)
- tests/harness/test_loader_self.py: EXISTS (24 passed, 23 skipped)
- pyproject.toml: MODIFIED (syrupy + pytest-socket added)

Commit db98fbe: FOUND in git log.
GATE-01: PASS (git diff --stat HEAD -- src/llm_sad_sam/ = empty).
