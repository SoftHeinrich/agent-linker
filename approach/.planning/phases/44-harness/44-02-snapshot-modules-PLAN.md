---
phase: 44
plan: 02
plan_id: 44-02
type: execute
wave: 2
depends_on:
  - 44-01
files_modified:
  - tests/test_s_linker20_prompt_ambiguity.py
  - tests/test_s_linker20_prompt_doc_extract.py
  - tests/test_s_linker20_prompt_doc_judge.py
  - tests/test_s_linker20_prompt_extraction.py
  - tests/test_s_linker20_prompt_validation.py
  - tests/test_s_linker20_prompt_coref.py
  - tests/test_s_linker20_harness_invariants.py
  - tests/__snapshots__/test_s_linker20_prompt_ambiguity.ambr
  - tests/__snapshots__/test_s_linker20_prompt_doc_extract.ambr
  - tests/__snapshots__/test_s_linker20_prompt_doc_judge.ambr
  - tests/__snapshots__/test_s_linker20_prompt_extraction.ambr
  - tests/__snapshots__/test_s_linker20_prompt_validation.ambr
  - tests/__snapshots__/test_s_linker20_prompt_coref.ambr
autonomous: true
requirements:
  - REQ-V264-02
user_setup: []

must_haves:
  truths:
    - "Six pytest test modules exist at tests/test_s_linker20_prompt_{ambiguity,doc_extract,doc_judge,extraction,validation,coref}.py."
    - "Each module rebuilds the prompt for each record via SLinker19._prompt_* @staticmethod and asserts the rebuilt prompt equals record['prompt'] byte-for-byte (sanity gate that the D-03 builder→tag mapping is correct)."
    - "Each module replays record['response_text'] through ReplayClient.extract_json and asserts the parsed dict matches a syrupy snapshot."
    - "Initial snapshots captured from the byte-equal s19 baseline are committed under tests/__snapshots__/."
    - "Running `pytest tests/test_s_linker20_prompt_*.py --disable-socket` exits 0 with all snapshot tests passing and zero network calls."
    - "test_s_linker20_prompt_validation.py parametrizes over (project, phase_tag) for all three phase_4_twopass_p1/p2 + phase_5_coref_validation tags per D-03."
    - "test_s_linker20_prompt_extraction.py and test_s_linker20_prompt_coref.py parametrize over (project, phase_tag, call_index) because builders fire multiple times per project."
    - "tests/test_s_linker20_harness_invariants.py asserts (a) GATE-01 byte-equality on src/llm_sad_sam/{linkers/experimental/s_linker19.py, linkers/experimental/s_linker13_min.py, linkers/experimental/prompts_v5.py}, (b) ReplayClient.query raises RuntimeError, (c) zero references to LLMClient().query() or network modules in tests/harness/ and tests/test_s_linker20_prompt_*.py."
    - "Test modules skip per-project (not whole-module) when fixture_missing_reason returns a non-None reason, so partial fixture sets keep the rest of CI green."
  artifacts:
    - path: "tests/test_s_linker20_prompt_ambiguity.py"
      provides: "snapshot test for _prompt_ambiguity (phase_1_model), parametrized by project"
      min_lines: 40
    - path: "tests/test_s_linker20_prompt_doc_extract.py"
      provides: "snapshot test for _prompt_doc_knowledge_extract (phase_1_doc_extract), parametrized by project"
      min_lines: 40
    - path: "tests/test_s_linker20_prompt_doc_judge.py"
      provides: "snapshot test for _prompt_doc_knowledge_judge (phase_1_doc_judge), parametrized by project"
      min_lines: 40
    - path: "tests/test_s_linker20_prompt_extraction.py"
      provides: "snapshot test for _prompt_extraction (phase_2_framing_c_pass1/pass2), parametrized by (project, phase_tag, call_index)"
      min_lines: 45
    - path: "tests/test_s_linker20_prompt_validation.py"
      provides: "snapshot test for _prompt_validation (3 phase tags including phase_5_coref_validation per D-03 gotcha), parametrized by (project, phase_tag, call_index)"
      min_lines: 45
    - path: "tests/test_s_linker20_prompt_coref.py"
      provides: "snapshot test for _prompt_coref (phase_5_coref), parametrized by (project, call_index)"
      min_lines: 45
    - path: "tests/test_s_linker20_harness_invariants.py"
      provides: "GATE-01 byte-equality + zero-LLM-call invariants"
      min_lines: 50
    - path: "tests/__snapshots__/test_s_linker20_prompt_ambiguity.ambr"
      provides: "captured parsed-output snapshot for ambiguity"
      contains: "mediastore"
    - path: "tests/__snapshots__/test_s_linker20_prompt_doc_extract.ambr"
      provides: "captured parsed-output snapshot for doc_extract"
      contains: "mediastore"
    - path: "tests/__snapshots__/test_s_linker20_prompt_doc_judge.ambr"
      provides: "captured parsed-output snapshot for doc_judge"
      contains: "mediastore"
    - path: "tests/__snapshots__/test_s_linker20_prompt_extraction.ambr"
      provides: "captured parsed-output snapshot for extraction"
      contains: "mediastore"
    - path: "tests/__snapshots__/test_s_linker20_prompt_validation.ambr"
      provides: "captured parsed-output snapshot for validation (3 phase tags)"
      contains: "phase_5_coref_validation"
    - path: "tests/__snapshots__/test_s_linker20_prompt_coref.ambr"
      provides: "captured parsed-output snapshot for coref"
      contains: "mediastore"
  key_links:
    - from: "tests/test_s_linker20_prompt_*.py"
      to: "tests.harness.loader.load_records"
      via: "from tests.harness.loader import load_records"
      pattern: "load_records"
    - from: "tests/test_s_linker20_prompt_*.py"
      to: "tests.harness.adapters.BUILDERS"
      via: "BUILDERS['_prompt_X'](...)"
      pattern: "BUILDERS\\["
    - from: "tests/test_s_linker20_prompt_*.py"
      to: "tests.harness.replay_client.replay_parse"
      via: "replay_parse(record['response_text'])"
      pattern: "replay_parse"
    - from: "tests/test_s_linker20_prompt_*.py"
      to: "syrupy snapshot fixture"
      via: "assert parsed == snapshot"
      pattern: "snapshot"
    - from: "tests/test_s_linker20_harness_invariants.py"
      to: "git diff --stat HEAD -- src/llm_sad_sam/linkers/experimental/"
      via: "subprocess.run check"
      pattern: "git diff"
---

<objective>
Ship the six pytest snapshot modules that complete REQ-V264-02:
each module rebuilds a prompt from `tests/harness/` cached `(prompt, response_text)` records, replays the response through `ReplayClient.extract_json`, and asserts the parsed structured output equals a syrupy snapshot captured from the byte-equal s19 baseline. Additionally ship a harness-invariants module that bolts down GATE-01 byte-equality and the zero-LLM-call guarantee in CI.

Purpose: REQ-V264-02 — "Pytest + snapshot harness ships one test module per s19 prompt builder ... tests pass at REQ-V264-02 close." This plan delivers the test modules, captures the initial snapshots from the unmodified s19 baseline, and makes future prompt-change verification (Phases 45–49) a zero-LLM-call snapshot diff.

Output:
- 6 `tests/test_s_linker20_prompt_*.py` modules using `tests.harness.loader` + `tests.harness.adapters` + syrupy
- Initial snapshots committed under `tests/__snapshots__/`
- 1 harness-invariants module pinning GATE-01 + zero-network-egress
- Full suite green: `pytest tests/test_s_linker20_prompt_*.py tests/test_s_linker20_harness_invariants.py --disable-socket` exits 0
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/REQUIREMENTS.md
@.planning/phases/44-harness/44-CONTEXT.md
@.planning/phases/44-harness/44-PATTERNS.md
@.planning/phases/44-harness/44-01-SUMMARY.md

# Plan 01 outputs (the foundation this plan builds on)
@tests/harness/__init__.py
@tests/harness/manifest.py
@tests/harness/loader.py
@tests/harness/replay_client.py
@tests/harness/adapters.py
@tests/harness/fixtures/MANIFEST.json

# Frozen source artefacts (READ-ONLY — byte-equal at plan close)
@src/llm_sad_sam/linkers/experimental/s_linker19.py
@src/llm_sad_sam/linkers/experimental/prompts_v5.py
@src/llm_sad_sam/llm_client.py
@tests/conftest.py
@tests/test_v20_baseline_regression.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Build all six pytest snapshot modules (ambiguity, doc_extract, doc_judge, extraction, validation, coref) plus the inputs-reconstruction helper</name>

  <files>
    tests/test_s_linker20_prompt_ambiguity.py,
    tests/test_s_linker20_prompt_doc_extract.py,
    tests/test_s_linker20_prompt_doc_judge.py,
    tests/test_s_linker20_prompt_extraction.py,
    tests/test_s_linker20_prompt_validation.py,
    tests/test_s_linker20_prompt_coref.py,
    tests/harness/inputs.py
  </files>

  <read_first>
    - .planning/phases/44-harness/44-01-SUMMARY.md (Plan 01 outputs: BUILDERS map, BUILDER_PHASE_TAGS, replay_parse signature, fixture_missing_reason signature)
    - tests/harness/loader.py (load_records / load_pkl / fixture_missing_reason — exact API surface)
    - tests/harness/adapters.py (BUILDERS / BUILDER_PHASE_TAGS — the D-03 mapping)
    - tests/harness/replay_client.py (replay_parse — the parser path)
    - tests/test_v20_baseline_regression.py lines 37–48 (module-scoped fixture + DATASETS-parametrize pattern to mirror)
    - tests/test_single_step_harness.py lines 26–44 (per-project skip-on-missing pattern)
    - src/llm_sad_sam/linkers/experimental/s_linker19.py lines 263–377 (6 builder signatures — what arguments to reconstruct from the pkl)
    - src/llm_sad_sam/linkers/experimental/s_linker19.py lines 555–910 (the run methods that originally called each builder — shows what `comp_names`, `mappings`, `cases`, `batch`, `focus` look like at call time, and which pkl layer holds what)
    - src/llm_sad_sam/core/data_types_v2.py (DocumentKnowledge, ModelKnowledge — the dataclasses held by the pkls)
    - syrupy docs (briefly — for `snapshot` fixture and `--snapshot-update` flag; AmberSerializer is the default)
  </read_first>

  <behavior>
    Common module shape (mirror tests/test_v20_baseline_regression.py module-fixture pattern):

      ```python
      # Top of every module
      from __future__ import annotations
      import pytest
      from tests.harness.loader import load_records, fixture_missing_reason
      from tests.harness.adapters import BUILDERS, BUILDER_PHASE_TAGS
      from tests.harness.replay_client import replay_parse
      from tests.harness.manifest import DATASETS
      from tests.harness.inputs import reconstruct_inputs   # see below
      ```

    Per-module behavior (all 6 share this shape; differences are parametrization + builder lookup + which phase tags):

      For each (project, phase_tag, call_index) in the parametrize grid:
        1. `reason = fixture_missing_reason(project)`; if not None → `pytest.skip(reason)`.
        2. `records = load_records(project, phase_tag)`; if `call_index >= len(records)` → `pytest.skip(f"no record at {project}/{phase_tag}/{call_index}")`. Robust to projects with fewer batches.
        3. `record = records[call_index]`.
        4. Reconstruct builder inputs from the pkl payload via `tests/harness/inputs.py` (see below).
        5. `rebuilt_prompt = BUILDERS[builder_name](*reconstructed_args)`.
        6. **Sanity gate (NOT the snapshot)** — `assert rebuilt_prompt == record["prompt"]`, with an `assert msg` naming the (project, phase_tag, call_index) so a future D-03 mismatch is debuggable in one line. This is the critical correctness invariant for the harness — the snapshot tests are meaningless if the rebuilt prompt diverges from the logged prompt.
        7. `parsed = replay_parse(record["response_text"])`.
        8. `assert parsed == snapshot`  (syrupy assertion — fixture `snapshot` argument; AmberSerializer handles nested dicts/lists by default).

    Per-builder parametrization specifics (Tests 1.1 through 1.6):

      Test 1.1 — `test_s_linker20_prompt_ambiguity.py` (`_prompt_ambiguity` / tag `phase_1_model`):
        - `@pytest.mark.parametrize("project", DATASETS)` (single call per project — `call_index` always 0).
        - Inputs to reconstruct: `names: list[str]` — the component names list passed to the prompt. Pulled from `load_pkl(project, "layer1")` (which holds the ModelKnowledge with the component name set used at the time `set_phase("phase_1_model")` fired at line 561). See inputs.py.

      Test 1.2 — `test_s_linker20_prompt_doc_extract.py` (`_prompt_doc_knowledge_extract` / tag `phase_1_doc_extract`):
        - `@pytest.mark.parametrize("project", DATASETS)`.
        - Inputs: `(comp_names, doc_lines)` — comp_names from `layer1.pkl` ModelKnowledge, doc_lines from `layer1.pkl` (the doc-line list the doc-knowledge phase consumed). If the pkl doesn't carry them, fall back to parsing from `record["prompt"]` (reverse-extract the COMPONENTS line and DOCUMENT block — but this is a fallback; primary path is pkl).

      Test 1.3 — `test_s_linker20_prompt_doc_judge.py` (`_prompt_doc_knowledge_judge` / tag `phase_1_doc_judge`):
        - `@pytest.mark.parametrize("project", DATASETS)`.
        - Inputs: `(comp_names, mapping_list)` — pulled from `layer2.pkl` (DocumentKnowledge extracted state pre-judge) or reverse-extracted from `record["prompt"]`.

      Test 1.4 — `test_s_linker20_prompt_extraction.py` (`_prompt_extraction` / tags `phase_2_framing_c_pass1`, `phase_2_framing_c_pass2`):
        - Two-axis parametrize: project × phase_tag, then per-(project, phase_tag) the test re-parametrizes over `call_index` in `range(len(records))` (use `pytest_generate_tests` or `pytest.param` with id strings) — each batch gets its own snapshot.
        - Inputs: `(comp_names, mappings, batch)` — batch is a list of sentence objects; reverse-extract from `record["prompt"]` is the most reliable path because the per-batch slice is computed at runtime by `_iter_batches`.

      Test 1.5 — `test_s_linker20_prompt_validation.py` (`_prompt_validation` / tags `phase_4_twopass_p1`, `phase_4_twopass_p2`, **`phase_5_coref_validation`** — D-03 gotcha):
        - Three-tag parametrize: project × phase_tag × call_index.
        - Inputs: `(comp_names, cases, focus)` — `focus` is `""` for the two phase_4 tags and `COREF_VALIDATION_FOCUS` constant for `phase_5_coref_validation` (see s_linker19.py line 894 and prompts_v5.py — import constant from prompts_v5). Reverse-extract `cases` from `record["prompt"]` CASES block.

      Test 1.6 — `test_s_linker20_prompt_coref.py` (`_prompt_coref` / tag `phase_5_coref`):
        - `@pytest.mark.parametrize("project,call_index", ...)` — typically one call per project but tolerate multiple.
        - Inputs: `(comp_names, cases)` — `cases` is a list of dicts with `sent` (Sentence) and `context` (list[str]). Reverse-extract from the `--- Case N: SX ---` blocks in `record["prompt"]`.

    Inputs reconstruction helper (`tests/harness/inputs.py`):
      - This is the meaty bit. The harness needs to call the builder with the SAME inputs the builder originally received, so the byte-equality assertion at step 6 holds. Strategy:
        - **Primary:** parse the inputs back out of `record["prompt"]` because the prompt strings are deterministic f-strings with fixed scaffolding. Each builder has stable section markers (`COMPONENTS: ...`, `NAMES: ...`, `DOCUMENT:\n...`, `CASES:\n...`, `--- Case N: SX ---`). Write per-builder reverse-extractors:
            - `reconstruct_ambiguity_inputs(record) -> tuple[list[str]]`
            - `reconstruct_doc_extract_inputs(record) -> tuple[list[str], list[str]]`
            - `reconstruct_doc_judge_inputs(record) -> tuple[list[str], list[str]]`
            - `reconstruct_extraction_inputs(record) -> tuple[list[str], list[str], list[Sentence]]`
            - `reconstruct_validation_inputs(record, phase_tag) -> tuple[list[str], list[str], str]`
            - `reconstruct_coref_inputs(record) -> tuple[list[str], list[Case]]`  where Case = dict with `sent: Sentence`, `context: list[str]`
        - Reverse-extract MUST yield arguments that, when passed to the builder, produce a string == record["prompt"] (the step-6 assertion is the unit test of inputs.py).
        - `Sentence` dataclass: import from `llm_sad_sam.core.data_types_v2`. Only `.number` and `.text` are used by the prompt scaffolding; reverse-extract uses the `S{number}: {text}` pattern.
      - **Module-level entry point:** `reconstruct_inputs(builder_name: str, record: dict, phase_tag: str) -> tuple` dispatches to the per-builder helper. Tests call this single function.

    Module-scoped fixtures:
      - Each test module defines a module-scoped fixture `def manifest_entries() -> list[FixtureEntry]: return load_manifest()` if any module-scoped state is needed, OR relies directly on `load_records` lru_cache (Plan 01). Mirror tests/test_v20_baseline_regression.py "load once per module" pattern.

    Failure-message hygiene:
      - The prompt-equality assertion (step 6) MUST include a diff hint: when the assertion fires, the error message names which builder, which (project, phase_tag, call_index), and reports the first 200-char diff between rebuilt_prompt and record["prompt"]. This is the single most important debug surface — without it, a D-03 mistake produces a wall of unreadable text.
  </behavior>

  <action>
    1. Create `tests/harness/inputs.py` with the 6 per-builder reverse-extractors + `reconstruct_inputs` dispatch function. Module docstring documents that this module's correctness is asserted by the step-6 prompt-equality assertions in the 6 test modules — if a reverse-extractor is wrong, the tests fail loudly with a diff. Import `Sentence` from `llm_sad_sam.core.data_types_v2`. Import `COREF_VALIDATION_FOCUS` from `llm_sad_sam.linkers.experimental.prompts_v5` for the validation phase_5 tag dispatch.

    2. Create `tests/test_s_linker20_prompt_ambiguity.py` per Test 1.1 behavior. Module-scoped `_pinned_projects` constant `= DATASETS`. Single test function `test_ambiguity_parsed_snapshot(project, snapshot)` with `@pytest.mark.parametrize("project", DATASETS, ids=lambda p: f"project={p}")`. Asserts (a) prompt-rebuild byte-equality, (b) `parsed == snapshot`.

    3. Create `tests/test_s_linker20_prompt_doc_extract.py` per Test 1.2 behavior. Mirror module 2 with builder name `_prompt_doc_knowledge_extract` and phase tag `phase_1_doc_extract`.

    4. Create `tests/test_s_linker20_prompt_doc_judge.py` per Test 1.3 behavior. Mirror with builder `_prompt_doc_knowledge_judge` and phase tag `phase_1_doc_judge`.

    5. Create `tests/test_s_linker20_prompt_extraction.py` per Test 1.4 behavior. Two-tag (`phase_2_framing_c_pass1`, `phase_2_framing_c_pass2`). Use `pytest_generate_tests` to compute the (project, phase_tag, call_index) grid lazily so projects with no records for a tag get a clear skip (not a collection failure). ID format: `f"{project}-{phase_tag}-call{call_index}"`.

    6. Create `tests/test_s_linker20_prompt_validation.py` per Test 1.5 behavior. Three tags (`phase_4_twopass_p1`, `phase_4_twopass_p2`, `phase_5_coref_validation`). Same `pytest_generate_tests` pattern. Dispatch `focus` argument per phase_tag inside `reconstruct_validation_inputs`. Module docstring calls out the D-03 phase_5_coref_validation gotcha explicitly so future readers don't move the case to the coref module.

    7. Create `tests/test_s_linker20_prompt_coref.py` per Test 1.6 behavior. Single tag `phase_5_coref`. Same `(project, call_index)` grid.

    8. **Initial snapshot capture (manual one-shot, in this task):** run
         `pytest tests/test_s_linker20_prompt_*.py --snapshot-update --disable-socket -p no:cacheprovider`
       to produce the initial `tests/__snapshots__/test_s_linker20_prompt_*.ambr` files. This MUST be the only snapshot-update invocation in this task; subsequent runs assert against these committed snapshots.

       Sanity-check the captured snapshots: each .ambr file must include at least one snapshot per project that has fixtures present. For mediastore (which has both pkl_dir and calls_json per D-02), all 6 modules must produce at least one snapshot.

    9. Run the full test set without `--snapshot-update` to confirm green:
         `pytest tests/test_s_linker20_prompt_*.py --disable-socket -v`
       MUST exit 0 with every parametrized case either PASS or SKIP (skips only for missing fixtures or empty record lists).

    10. Commit the 6 .ambr files under `tests/__snapshots__/` so future runs can verify byte-equality against the s19 baseline.

    11. GATE-01 byte-equality verification: run `git diff --stat HEAD -- src/llm_sad_sam/` and assert zero output. This task touches nothing under src/.
  </action>

  <verify>
    <automated>
      pytest tests/test_s_linker20_prompt_ambiguity.py --disable-socket -v --tb=short
    </automated>
    <automated>
      pytest tests/test_s_linker20_prompt_doc_extract.py --disable-socket -v --tb=short
    </automated>
    <automated>
      pytest tests/test_s_linker20_prompt_doc_judge.py --disable-socket -v --tb=short
    </automated>
    <automated>
      pytest tests/test_s_linker20_prompt_extraction.py --disable-socket -v --tb=short
    </automated>
    <automated>
      pytest tests/test_s_linker20_prompt_validation.py --disable-socket -v --tb=short
    </automated>
    <automated>
      pytest tests/test_s_linker20_prompt_coref.py --disable-socket -v --tb=short
    </automated>
    <automated>
      pytest tests/ -k "s_linker20_prompt" --collect-only --disable-socket 2>&1 | grep -E "test_s_linker20_prompt_(ambiguity|doc_extract|doc_judge|extraction|validation|coref)" | sort -u | wc -l
      # MUST emit at least 6 (the six module roots; the collector lists module:: lines, count distinct modules)
    </automated>
    <automated>
      bash -c 'for f in tests/__snapshots__/test_s_linker20_prompt_{ambiguity,doc_extract,doc_judge,extraction,validation,coref}.ambr; do
        if [ ! -f "$f" ]; then echo "MISSING: $f"; exit 1; fi
        if [ ! -s "$f" ]; then echo "EMPTY:   $f"; exit 1; fi
      done
      echo "All 6 snapshot files exist and are non-empty"'
    </automated>
    <automated>
      grep -l "phase_5_coref_validation" tests/test_s_linker20_prompt_validation.py
      # MUST emit the file path (D-03 gotcha encoded literally in the validation module, not the coref module)
    </automated>
    <automated>
      bash -c '
        if grep -q "phase_5_coref_validation" tests/test_s_linker20_prompt_coref.py 2>/dev/null; then
          echo "ERROR: phase_5_coref_validation must NOT appear in the coref test module per D-03"
          exit 1
        fi
        echo "D-03 gotcha respected: phase_5_coref_validation absent from coref module"
      '
    </automated>
  </verify>

  <acceptance_criteria>
    - All six files `tests/test_s_linker20_prompt_{ambiguity,doc_extract,doc_judge,extraction,validation,coref}.py` exist.
    - `pytest tests/test_s_linker20_prompt_*.py --disable-socket` exits 0 (all PASS or SKIP, zero FAIL, zero ERROR).
    - `pytest tests/ -k "s_linker20_prompt" --collect-only --disable-socket` reports at least 6 distinct module entries (one per builder).
    - `tests/__snapshots__/test_s_linker20_prompt_*.ambr` exists for all 6 modules and is non-empty.
    - `grep "phase_5_coref_validation" tests/test_s_linker20_prompt_validation.py` returns 1+ match (D-03 gotcha encoded in validation module).
    - `grep "phase_5_coref_validation" tests/test_s_linker20_prompt_coref.py` returns 0 matches (D-03 gotcha NOT in coref module).
    - For each module: rebuild-prompt byte-equality assertion passes for every parametrized non-skipped case (asserted implicitly because no test FAILs).
    - `git diff --stat HEAD -- src/llm_sad_sam/` produces zero lines (GATE-01 byte-equal).
    - `grep -rE '\\.query\\(' tests/test_s_linker20_prompt_*.py | grep -v '^#'` returns zero matches (no LLM.query call paths reachable from test modules).
  </acceptance_criteria>

  <done>
    Six pytest snapshot modules ship, initial snapshots are committed under tests/__snapshots__/, the rebuild-prompt byte-equality gate proves the D-03 builder→tag mapping is correct, and the full suite passes under `--disable-socket` proving zero network egress. The validation module covers all three of phase_4_twopass_p1, phase_4_twopass_p2, and phase_5_coref_validation per D-03; the coref module covers only phase_5_coref. GATE-01 preserved.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Ship harness-invariants test (GATE-01 byte-equality, zero-network-egress, ReplayClient guard) and run full Phase 44 success-criteria check</name>

  <files>
    tests/test_s_linker20_harness_invariants.py
  </files>

  <read_first>
    - tests/harness/replay_client.py (Plan 01 — ReplayClient.query raises RuntimeError; this task asserts that contract from a test module)
    - .planning/REQUIREMENTS.md §HARNESS — REQ-V264-01 Success Criterion 4: "zero LLM API calls (verified by absence of network I/O or mock assertion)"
    - .planning/ROADMAP.md Phase 44 Success Criteria 1–4
    - .planning/PROJECT.md §Constraints — GATE-01 byte-equality on `s_linker19.py` and `s_linker13_min.py`
    - src/llm_sad_sam/linkers/experimental/s_linker19.py (read-only; only its existence + git status are checked)
    - src/llm_sad_sam/linkers/experimental/s_linker13_min.py (read-only; same)
    - src/llm_sad_sam/linkers/experimental/prompts_v5.py (read-only; same)
    - tests/conftest.py (subprocess invocation must keep ROOT consistent)
  </read_first>

  <behavior>
    Single test module `tests/test_s_linker20_harness_invariants.py` with these test functions:

    Test 2.1 — `test_gate_01_byte_equality_s19_s13min_prompts_v5`:
      - For each of {src/llm_sad_sam/linkers/experimental/s_linker19.py, src/llm_sad_sam/linkers/experimental/s_linker13_min.py, src/llm_sad_sam/linkers/experimental/prompts_v5.py}:
        - Run `git diff --stat HEAD -- <path>` in subprocess; assert stdout is empty.
      - If git is unavailable (CI without git), test skips with reason "git binary not on PATH".
      - This is the live in-CI GATE-01 check; ROADMAP Phase 44 Success Criterion (carried) and CONTEXT.md §Out of Scope item #1.

    Test 2.2 — `test_replay_client_query_forbidden`:
      - `from tests.harness.replay_client import ReplayClient`
      - `with pytest.raises(RuntimeError, match="ReplayClient.query.. is forbidden"): ReplayClient().query("any prompt")`.
      - Belt-and-suspenders even though Plan 01's test_loader_self also asserts this — keeps the invariant visible in the Phase 44 success-criteria suite.

    Test 2.3 — `test_no_llm_query_calls_in_harness_or_snapshot_modules`:
      - `grep -rnE '\.query\(' tests/harness/ tests/test_s_linker20_prompt_*.py tests/test_s_linker20_harness_invariants.py`:
        - Allowed matches: the ReplayClient.query definition line in tests/harness/replay_client.py and any docstring/comment mentioning "query() is forbidden".
        - Disallowed: any other invocation of `.query(`.
      - Run via `subprocess.run` so the test is robust to working-directory weirdness; parse output and assert only the allowlisted lines appear.

    Test 2.4 — `test_no_network_module_imports_in_test_layer`:
      - `grep -rnE '^(import|from) (openai|anthropic|requests|httpx|urllib)' tests/harness/ tests/test_s_linker20_prompt_*.py tests/test_s_linker20_harness_invariants.py`
      - Must emit zero matches (excluding comments — use `grep -v '^[[:space:]]*#'`).
      - Plan 01's ReplayClient imports `LLMClient` and `LLMResponse` from `llm_sad_sam.llm_client`, which transitively might import openai — but THAT module is in src/ not tests/, so this grep won't flag it. The test layer itself must contain no direct network-client imports.

    Test 2.5 — `test_full_harness_suite_green_under_disable_socket`:
      - Invoke `subprocess.run(["python", "-m", "pytest", "tests/test_s_linker20_prompt_ambiguity.py", "tests/test_s_linker20_prompt_doc_extract.py", "tests/test_s_linker20_prompt_doc_judge.py", "tests/test_s_linker20_prompt_extraction.py", "tests/test_s_linker20_prompt_validation.py", "tests/test_s_linker20_prompt_coref.py", "--disable-socket", "-q", "--no-header"], cwd=ROOT)`.
      - Assert `returncode == 0`.
      - Skip with reason "pytest-socket not installed; check pyproject.toml [dev]" if `--disable-socket` flag isn't recognized.
      - THIS is the Phase 44 Success Criterion 4 ("zero LLM API calls verified by absence of network I/O") enforced as a single bot-runnable test.
      - To prevent runaway test recursion (the outer pytest runs this test, which spawns an inner pytest), the test skips itself when the env var `_PHASE44_INNER=1` is set, and the subprocess invocation sets that env var. Pattern is the same one tests/test_single_step_harness.py uses for engine smoke tests.

    Module-level constants:
      - `FROZEN_BYTE_EQUAL_PATHS = ("src/llm_sad_sam/linkers/experimental/s_linker19.py", "src/llm_sad_sam/linkers/experimental/s_linker13_min.py", "src/llm_sad_sam/linkers/experimental/prompts_v5.py")`
      - `ROOT = Path(__file__).resolve().parents[1]` (mirrors tests/conftest.py)
  </behavior>

  <action>
    1. Create `tests/test_s_linker20_harness_invariants.py` implementing Tests 2.1 through 2.5 per the behavior contract. Use `subprocess.run` with `capture_output=True, text=True, timeout=120` for the git and grep checks.

    2. Add a module docstring naming each of the four Phase 44 ROADMAP success criteria and which test verifies which:
       - SC1 (fixture infrastructure exposes triples) → asserted transitively by the 6 snapshot modules passing.
       - SC2 (six pytest test modules exist) → asserted by Test 2.5's pytest collection invocation.
       - SC3 (all snapshot tests pass on unmodified s19 baseline) → asserted by Test 2.5.
       - SC4 (zero LLM API calls verified) → asserted by Tests 2.2, 2.3, 2.4, 2.5 (--disable-socket).

    3. Run the new module: `pytest tests/test_s_linker20_harness_invariants.py --disable-socket -v`. Must exit 0.

    4. Run the full Phase 44 suite end-to-end:
         `pytest tests/test_s_linker20_prompt_*.py tests/test_s_linker20_harness_invariants.py tests/harness/test_loader_self.py --disable-socket -v`
       MUST exit 0. This is the single command Phase 44 close auditing runs.

    5. Verify GATE-01 one last time: `git diff --stat HEAD -- src/llm_sad_sam/`. Zero lines.
  </action>

  <verify>
    <automated>
      pytest tests/test_s_linker20_harness_invariants.py --disable-socket -v --tb=short
    </automated>
    <automated>
      pytest tests/test_s_linker20_prompt_ambiguity.py tests/test_s_linker20_prompt_doc_extract.py tests/test_s_linker20_prompt_doc_judge.py tests/test_s_linker20_prompt_extraction.py tests/test_s_linker20_prompt_validation.py tests/test_s_linker20_prompt_coref.py tests/test_s_linker20_harness_invariants.py tests/harness/test_loader_self.py --disable-socket -v --tb=short
    </automated>
    <automated>
      bash -c '
        out=$(git diff --stat HEAD -- src/llm_sad_sam/)
        if [ -n "$out" ]; then
          echo "GATE-01 FAIL: $out"; exit 1
        fi
        echo "GATE-01 PASS — src/llm_sad_sam/ byte-equal"
      '
    </automated>
    <automated>
      bash -c '
        matches=$(grep -rnE "^[[:space:]]*(import|from) (openai|anthropic|requests|httpx|urllib)" tests/harness/ tests/test_s_linker20_prompt_*.py tests/test_s_linker20_harness_invariants.py 2>/dev/null | grep -v "^[^:]*:[0-9]*:[[:space:]]*#")
        if [ -n "$matches" ]; then
          echo "Network-module import found in test layer:"; echo "$matches"; exit 1
        fi
        echo "Test layer free of network-module imports"
      '
    </automated>
  </verify>

  <acceptance_criteria>
    - `tests/test_s_linker20_harness_invariants.py` exists.
    - `pytest tests/test_s_linker20_harness_invariants.py --disable-socket` exits 0.
    - `pytest tests/test_s_linker20_prompt_*.py tests/test_s_linker20_harness_invariants.py tests/harness/test_loader_self.py --disable-socket` exits 0 (the canonical Phase 44 close-audit command).
    - Test 2.1 actively runs `git diff --stat HEAD -- <path>` for s_linker19.py, s_linker13_min.py, and prompts_v5.py and asserts empty output.
    - Test 2.2 asserts `ReplayClient().query("x")` raises RuntimeError.
    - Test 2.3 grep finds zero non-allowlisted `.query(` invocations in the test layer.
    - Test 2.4 grep finds zero `import openai|anthropic|requests|httpx|urllib` in tests/harness/ or tests/test_s_linker20_*.py.
    - Test 2.5 spawns an inner pytest with `--disable-socket` and asserts returncode 0; uses `_PHASE44_INNER` env var to prevent infinite recursion.
    - `git diff --stat HEAD -- src/llm_sad_sam/` produces zero lines after this task (GATE-01 final check for Phase 44).
  </acceptance_criteria>

  <done>
    Phase 44 has a single bot-runnable success-criteria suite. Running `pytest tests/test_s_linker20_prompt_*.py tests/test_s_linker20_harness_invariants.py tests/harness/test_loader_self.py --disable-socket` returns 0 and proves: (a) the harness exposes triples for every builder × project, (b) all 6 snapshot modules exist and pass on the unmodified s19 baseline, (c) zero LLM API calls were made, (d) GATE-01 byte-equality on s_linker19.py + s_linker13_min.py + prompts_v5.py preserved. Phase 44 is closeable.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| disk → process (pickle.load via Plan 01 loader) | inherited from Plan 01; no new pickle reads in this plan |
| pytest snapshot files → process | new `.ambr` files committed in this plan; read at every test invocation |
| inner pytest subprocess → outer pytest | Test 2.5 spawns a child pytest; CI must not infinite-loop |
| test layer → src/llm_sad_sam/ | strictly read-only; GATE-01 verified in CI by Test 2.1 |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-44-06 | Tampering | snapshot files in `tests/__snapshots__/` are the test oracle | mitigate | Initial snapshots captured under controlled conditions: one `pytest --snapshot-update --disable-socket` run with src/llm_sad_sam/ confirmed byte-equal (Task 1 step 11). Snapshots are committed to git; any subsequent change is review-visible. Tampering with snapshots in a future commit cannot make a regressed s20 prompt look correct because GATE-01 still locks s19 byte-equal — the snapshots are derived from s19 alone. |
| T-44-07 | DoS / EoP | inner pytest subprocess in Test 2.5 could recurse if run in test-collection mode | mitigate | `_PHASE44_INNER=1` env var sentinel: outer test sets it before subprocess, inner test reads it at module top and skips. Pattern proven in tests/test_single_step_harness.py. Subprocess has `timeout=600` and `--disable-socket` so even a runaway can't cause network-side effects. |
| T-44-08 | Information Disclosure | `record["prompt"]` and `record["response_text"]` are committed in `_calls.json` and embedded in `.ambr` snapshots | accept | These files are already in the repo (pre-existing artefacts from v2.6.3 paper-replay phases) and contain no secrets — prompts are derived from open-source benchmark documents, responses are LLM completions. No new disclosure surface created by this phase. |
| T-44-09 | Tampering | accidental edits to s_linker19.py/s_linker13_min.py/prompts_v5.py in Phase 44 | mitigate | Test 2.1 runs `git diff --stat HEAD -- <path>` for each frozen file in CI and fails if non-empty. Even an inadvertent whitespace edit fails the test. This bolts GATE-01 into the same pytest suite that REQ-V264-02 requires green. |
| T-44-10 | DoS | `pytest-socket --disable-socket` could be flaky on some platforms (silent no-op if plugin not installed) | mitigate | Test 2.5 explicitly skips with actionable message ("pytest-socket not installed; check pyproject.toml [dev]") if `--disable-socket` is unrecognized. Plan 01 acceptance criterion ensures pytest-socket is in `[dev]`. Belt-and-suspenders: Test 2.2/2.3/2.4 enforce the no-network property via grep and the ReplayClient.query guard, which work regardless of pytest-socket availability. |
| T-44-SC | Tampering | npm/pip/cargo installs | mitigate | No new package installs in this plan — syrupy + pytest-socket already added in Plan 01 (44-01 threat register T-44-05 covers legitimacy). |
</threat_model>

<verification>
- `pytest tests/test_s_linker20_prompt_*.py --disable-socket` exits 0 (REQ-V264-02 Success Criterion 3).
- `pytest tests/ -k "s_linker20_prompt" --collect-only --disable-socket` reports at least 6 distinct module entries (REQ-V264-02 module count).
- `pytest tests/test_s_linker20_harness_invariants.py --disable-socket` exits 0 (Phase 44 Success Criteria 1–4 all asserted).
- `tests/__snapshots__/test_s_linker20_prompt_*.ambr` exists for all 6 modules and is committed to git.
- `git diff --stat HEAD -- src/llm_sad_sam/` produces zero lines (GATE-01 byte-equal).
- D-03 gotcha verification: `phase_5_coref_validation` appears in validation module, NOT in coref module.
- Network-egress: zero `import openai|anthropic|requests|httpx|urllib` in test layer (asserted by Test 2.4).
- Zero `.query(` invocations in test layer outside the allowlisted ReplayClient definition (asserted by Test 2.3).
</verification>

<success_criteria>
1. Six pytest test modules exist (`tests/test_s_linker20_prompt_{ambiguity,doc_extract,doc_judge,extraction,validation,coref}.py`), each importing from `tests.harness` and asserting (a) prompt-rebuild byte-equality, (b) parsed-output snapshot equality.
2. Initial snapshots captured from the byte-equal s19 baseline are committed under `tests/__snapshots__/`.
3. `pytest tests/test_s_linker20_prompt_*.py tests/test_s_linker20_harness_invariants.py tests/harness/test_loader_self.py --disable-socket` exits 0 — the canonical Phase 44 close-audit command.
4. `test_s_linker20_prompt_validation.py` covers all three of phase_4_twopass_p1, phase_4_twopass_p2, phase_5_coref_validation per D-03 (the coref-validation gotcha).
5. `test_s_linker20_prompt_coref.py` covers ONLY phase_5_coref (no phase_5_coref_validation overlap).
6. `test_s_linker20_harness_invariants.py` asserts GATE-01 byte-equality on s_linker19.py, s_linker13_min.py, prompts_v5.py via live `git diff --stat` checks.
7. `test_s_linker20_harness_invariants.py` asserts zero `.query(` invocations and zero network-module imports in the test layer.
8. Phase 44 ROADMAP Success Criteria 1–4 all verifiable from this single pytest invocation.
</success_criteria>

<output>
Create `.planning/phases/44-harness/44-02-SUMMARY.md` when done, listing:
- The 6 test module names + parametrize cardinalities (per-module test count after collection)
- The 6 snapshot file paths + their sizes
- D-03 gotcha verification result (phase_5_coref_validation lives in validation module only)
- GATE-01 final status for Phase 44 close (must be PASS)
- The canonical Phase 44 close-audit command (`pytest tests/test_s_linker20_prompt_*.py tests/test_s_linker20_harness_invariants.py tests/harness/test_loader_self.py --disable-socket`)
- Phase 45 handoff: the loader API is now the audit substrate — Phase 45's prompt-audit doc references load_records(project, phase_tag) as the way to inspect any prompt/response pair without re-running the LLM.
</output>

## Artifacts this phase produces

**New test modules (six per REQ-V264-02):**
- `tests/test_s_linker20_prompt_ambiguity.py`
- `tests/test_s_linker20_prompt_doc_extract.py`
- `tests/test_s_linker20_prompt_doc_judge.py`
- `tests/test_s_linker20_prompt_extraction.py`
- `tests/test_s_linker20_prompt_validation.py`
- `tests/test_s_linker20_prompt_coref.py`

**Phase 44 invariants module:**
- `tests/test_s_linker20_harness_invariants.py`

**New inputs-reconstruction helper:**
- `tests/harness/inputs.py`
  - `reconstruct_ambiguity_inputs(record) -> tuple[list[str]]`
  - `reconstruct_doc_extract_inputs(record) -> tuple[list[str], list[str]]`
  - `reconstruct_doc_judge_inputs(record) -> tuple[list[str], list[str]]`
  - `reconstruct_extraction_inputs(record) -> tuple[list[str], list[str], list[Sentence]]`
  - `reconstruct_validation_inputs(record, phase_tag) -> tuple[list[str], list[str], str]`
  - `reconstruct_coref_inputs(record) -> tuple[list[str], list[dict]]`
  - `reconstruct_inputs(builder_name, record, phase_tag) -> tuple` (dispatch)

**New snapshot files (committed):**
- `tests/__snapshots__/test_s_linker20_prompt_ambiguity.ambr`
- `tests/__snapshots__/test_s_linker20_prompt_doc_extract.ambr`
- `tests/__snapshots__/test_s_linker20_prompt_doc_judge.ambr`
- `tests/__snapshots__/test_s_linker20_prompt_extraction.ambr`
- `tests/__snapshots__/test_s_linker20_prompt_validation.ambr` (covers 3 phase tags per D-03)
- `tests/__snapshots__/test_s_linker20_prompt_coref.ambr`

**New test functions:**
- `test_ambiguity_parsed_snapshot(project, snapshot)`
- `test_doc_extract_parsed_snapshot(project, snapshot)`
- `test_doc_judge_parsed_snapshot(project, snapshot)`
- `test_extraction_parsed_snapshot(project, phase_tag, call_index, snapshot)`
- `test_validation_parsed_snapshot(project, phase_tag, call_index, snapshot)` — covers phase_5_coref_validation per D-03
- `test_coref_parsed_snapshot(project, call_index, snapshot)`
- `test_gate_01_byte_equality_s19_s13min_prompts_v5()`
- `test_replay_client_query_forbidden()`
- `test_no_llm_query_calls_in_harness_or_snapshot_modules()`
- `test_no_network_module_imports_in_test_layer()`
- `test_full_harness_suite_green_under_disable_socket()`

**Module-level constants (in test_s_linker20_harness_invariants.py):**
- `FROZEN_BYTE_EQUAL_PATHS` — tuple of three frozen source paths
- `ROOT` — repo root Path

**Canonical Phase 44 close-audit command:**
```
pytest tests/test_s_linker20_prompt_*.py tests/test_s_linker20_harness_invariants.py tests/harness/test_loader_self.py --disable-socket
```

**No new CLI flags, no new entry points, no schema changes.**
