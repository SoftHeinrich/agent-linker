---
phase: 44
plan: 01
plan_id: 44-01
type: execute
wave: 1
depends_on: []
files_modified:
  - tests/harness/__init__.py
  - tests/harness/fixtures/__init__.py
  - tests/harness/fixtures/MANIFEST.json
  - tests/harness/manifest.py
  - tests/harness/loader.py
  - tests/harness/adapters.py
  - tests/harness/replay_client.py
  - pyproject.toml
autonomous: true
requirements:
  - REQ-V264-01
user_setup: []

must_haves:
  truths:
    - "tests/harness/ exists as an importable Python package."
    - "tests/harness/fixtures/MANIFEST.json lists the 5 pinned (project, pkl_dir, calls_json) triples from D-02 with repo-root-relative paths."
    - "load_manifest() returns 5 FixtureEntry records and exposes the canonical DATASETS tuple."
    - "load_records(project, phase_tag) returns the JSON-list of {phase, prompt, response_text, ...} records the s19 _TracingLLMClient wrote, filtered by phase_tag, in original order."
    - "load_pkl(project, layer) deserializes results/phase_cache/s_linker19/openai/<project>/{layer1..4,final}.pkl as the dataclasses defined in src/llm_sad_sam/core/data_types_v2.py without any LLM call or network I/O."
    - "An adapter map BUILDER_PHASE_TAGS implements D-03 verbatim — including the gotcha that phase_5_coref_validation belongs to _prompt_validation."
    - "ReplayClient.query() raises RuntimeError. Only ReplayClient.extract_json is reachable, and it delegates to LLMClient.extract_json byte-equal."
    - "syrupy>=4.6.0 and pytest-socket>=0.7 are declared in pyproject.toml [project.optional-dependencies].dev."
    - "No file under src/llm_sad_sam/ is modified — s_linker19.py, s_linker13_min.py, prompts_v5.py, data_types_v2.py, llm_client.py byte-equal at plan close."
  artifacts:
    - path: "tests/harness/__init__.py"
      provides: "package marker"
    - path: "tests/harness/fixtures/__init__.py"
      provides: "fixtures subpackage marker"
    - path: "tests/harness/fixtures/MANIFEST.json"
      provides: "5-entry pinned pkl/calls-json ledger (D-02)"
      contains: "mediastore"
    - path: "tests/harness/manifest.py"
      provides: "load_manifest() -> list[FixtureEntry]; DATASETS tuple"
      exports: ["FixtureEntry", "load_manifest", "DATASETS", "MANIFEST_PATH"]
    - path: "tests/harness/loader.py"
      provides: "load_records / load_pkl / FixtureMissing skip helper"
      exports: ["load_records", "load_pkl", "fixture_missing_reason"]
    - path: "tests/harness/adapters.py"
      provides: "BUILDER_PHASE_TAGS dict + builder callable registry (D-03)"
      exports: ["BUILDER_PHASE_TAGS", "BUILDERS"]
    - path: "tests/harness/replay_client.py"
      provides: "ReplayClient — calls extract_json on cached response_text; .query() forbidden"
      exports: ["ReplayClient", "replay_parse"]
    - path: "pyproject.toml"
      provides: "syrupy + pytest-socket dev dependencies"
      contains: "syrupy"
  key_links:
    - from: "tests/harness/loader.py"
      to: "results/llm_logs/s_linker19_openai_<project>_*_calls.json"
      via: "json.loads(path.read_text())"
      pattern: "calls_json"
    - from: "tests/harness/loader.py"
      to: "results/phase_cache/s_linker19/openai/<project>/*.pkl"
      via: "pickle.load(open(path, 'rb'))"
      pattern: "phase_cache"
    - from: "tests/harness/replay_client.py"
      to: "src/llm_sad_sam/llm_client.py::LLMClient.extract_json"
      via: "delegating method call; no .query() ever invoked"
      pattern: "extract_json"
    - from: "tests/harness/adapters.py"
      to: "src/llm_sad_sam/linkers/experimental/s_linker19.py::SLinker19._prompt_*"
      via: "@staticmethod call by reference — no SLinker19 instantiation"
      pattern: "SLinker19\\._prompt_"
---

<objective>
Build the fixture-infrastructure foundation for the Phase 44 golden-replay harness:
a `tests/harness/` package that loads `results/phase_cache/s_linker19/openai/<project>/{layer1..4,final}.pkl` + `results/llm_logs/s_linker19_openai_<project>_<TIMESTAMP>_calls.json` paired through `tests/harness/fixtures/MANIFEST.json`, exposes `(prompt, response_text)` triples grouped by D-03 phase tag, and provides a `ReplayClient` whose `.query()` is unreachable so REQ-V264-01's "zero new LLM calls" property is structurally guaranteed.

Purpose: REQ-V264-01 — every Phase 44 test module reads from this single loader API. Locking the fixture wiring + LLM-isolation contract here lets Plan 02 focus on the six snapshot test modules without re-deciding fixture shape.

Output:
- `tests/harness/` package (5 modules + 2 init markers)
- `tests/harness/fixtures/MANIFEST.json` with the 5 pinned entries from CONTEXT D-02
- `pyproject.toml` dev extras gain `syrupy>=4.6.0` and `pytest-socket>=0.7`
- Zero modifications to any frozen source artefact (GATE-01 byte-equal preserved)
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
@CLAUDE.md

# Reference source artefacts (READ-ONLY — must remain byte-equal)
@src/llm_sad_sam/linkers/experimental/s_linker19.py
@src/llm_sad_sam/llm_client.py
@src/llm_sad_sam/core/data_types_v2.py
@tests/conftest.py
@tests/test_single_step_harness.py
@tests/test_v20_baseline_regression.py
@pyproject.toml
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Wire syrupy + pytest-socket dev deps and create tests/harness/ package skeleton (manifest + loader + ReplayClient + adapter map)</name>

  <files>
    pyproject.toml,
    tests/harness/__init__.py,
    tests/harness/fixtures/__init__.py,
    tests/harness/fixtures/MANIFEST.json,
    tests/harness/manifest.py,
    tests/harness/loader.py,
    tests/harness/replay_client.py,
    tests/harness/adapters.py
  </files>

  <read_first>
    - .planning/phases/44-harness/44-CONTEXT.md (D-01 paired fixtures, D-02 manifest schema, D-03 builder→phase-tag map, "Claude's Discretion" — lock decisions here)
    - .planning/phases/44-harness/44-PATTERNS.md (Analog A sys.path bootstrap; Analog B _calls.json record schema; Analog C skip-on-missing; Analog D builder signatures; Analog E phase-tag call sites)
    - src/llm_sad_sam/linkers/experimental/s_linker19.py lines 121–160 (`_TracingLLMClient.query` — wire format of every record in calls_json)
    - src/llm_sad_sam/linkers/experimental/s_linker19.py lines 263–377 (the 6 `@staticmethod` prompt builders — call signatures the adapter map registers)
    - src/llm_sad_sam/linkers/experimental/s_linker19.py lines 555–910 (the `set_phase` call sites at 561, 573, 604, 646, 793, 835, 894 — verifies D-03 mapping)
    - src/llm_sad_sam/llm_client.py lines 37–46 (`LLMResponse` dataclass) and lines 985–1019 (`LLMClient.extract_json` — the parser path the ReplayClient delegates to)
    - tests/conftest.py (sys.path bootstrap — inherited by tests/harness/ automatically)
    - tests/test_single_step_harness.py lines 26–44 (skip-on-missing fixture convention to mirror)
    - tests/test_v20_baseline_regression.py lines 37–48 (module-scoped fixture loading the manifest pattern)
    - pyproject.toml lines 19–26 ([project.optional-dependencies] dev block — exact in-file edit target)
    - results/llm_logs/s_linker19_openai_mediastore_20260605_134622_calls.json (sample one record to verify keys: phase, prompt, response_text, success, model, ts, elapsed_s, timeout, max_retries, error, latency_ms, token_usage)
    - .planning/PROJECT.md §Constraints — "standalone duplicated files over inheritance"; GATE-01 byte-equality
  </read_first>

  <behavior>
    Locked planner decisions (encode as code or asserts):
    - LOCK-1: snapshot library = syrupy>=4.6.0 (parsed-output dict snapshots; AmberSerializer default). pytest-regressions NOT installed.
    - LOCK-2: parser isolation = option (c) — call SLinker19._prompt_* @staticmethods directly + replay response_text through LLMClient.extract_json. NO monkey-patching of s_linker19; NO refactor of builders.
    - LOCK-3: parametrization granularity — per `(project, call_index)` for multi-call builders; per `project` for single-call. Loader exposes records as ordered list; test modules choose granularity.
    - LOCK-4: MANIFEST.json carries NO `sha256` field in Phase 44. Schema = list of {project, pkl_dir, calls_json}. Optional `description` allowed.
    - LOCK-5: layout — fixtures + loader under `tests/harness/`; six test modules at `tests/test_s_linker20_prompt_*.py` (Plan 02). Snapshots default location `tests/__snapshots__/` (syrupy).
    - LOCK-6: zero-LLM-call enforcement = BOTH (a) `ReplayClient.query` raises RuntimeError + (b) `pytest-socket --disable-socket` flag (Plan 02 asserts).

    Behavior contract per file:

    Test 1.1 — pyproject.toml edit:
      - Reading parsed pyproject.toml's [project.optional-dependencies].dev MUST contain "syrupy>=4.6.0" and "pytest-socket>=0.7" alongside the existing "pytest>=8.0.0" and "pytest-asyncio>=0.23.0" entries (no entries removed, only appended).

    Test 1.2 — manifest.py:
      - `from tests.harness.manifest import load_manifest, FixtureEntry, DATASETS, MANIFEST_PATH` succeeds.
      - `DATASETS == ("mediastore", "teastore", "teammates", "bigbluebutton", "jabref")` exactly.
      - `MANIFEST_PATH` resolves to `tests/harness/fixtures/MANIFEST.json` (Path object, repo-root anchored).
      - `load_manifest()` returns `list[FixtureEntry]` of length 5 with `e.project` covering DATASETS as a set.
      - Each entry: `e.pkl_dir` and `e.calls_json` are absolute Path objects resolved against repo root; `e.pkl_dir.is_dir()` and `e.calls_json.is_file()` hold for any present fixture (load_manifest does NOT raise when files exist; raises FileNotFoundError with the manifest's relative path included in the message when they don't).
      - `FixtureEntry` is a `@dataclass(frozen=True)` with fields `project: str`, `pkl_dir: Path`, `calls_json: Path`, `description: str | None = None`.

    Test 1.3 — MANIFEST.json contents (D-02 pinning):
      - File parses as JSON. Top-level value is a list of 5 objects.
      - Each object has keys exactly `{project, pkl_dir, calls_json}` (description optional).
      - The 5 (project, pkl_dir, calls_json) triples match D-02 verbatim:
          mediastore     → results/phase_cache/s_linker19/openai/mediastore/     + results/llm_logs/s_linker19_openai_mediastore_20260605_134622_calls.json
          teastore       → results/phase_cache/s_linker19/openai/teastore/       + results/llm_logs/s_linker19_openai_teastore_20260604_065824_calls.json
          teammates      → results/phase_cache/s_linker19/openai/teammates/      + results/llm_logs/s_linker19_openai_teammates_20260604_070526_calls.json
          bigbluebutton  → results/phase_cache/s_linker19/openai/bigbluebutton/  + results/llm_logs/s_linker19_openai_bigbluebutton_20260604_070639_calls.json
          jabref         → results/phase_cache/s_linker19/openai/jabref/         + results/llm_logs/s_linker19_openai_jabref_20260605_134705_calls.json
      - All 5 listed files exist on disk (smoke-checked at scout time and at task close).

    Test 1.4 — loader.py:
      - `load_records(project: str, phase_tag: str) -> list[dict]` filters the project's `_calls.json` by `record["phase"] == phase_tag`, preserves original order, returns the raw record dicts. Empty list if no record matches that tag (not an error).
      - `load_pkl(project: str, layer: str) -> object` calls `pickle.load(open(entry.pkl_dir / f"{layer}.pkl", "rb"))` where layer ∈ {"layer1", "layer2", "layer3", "layer4", "final"}; raises FileNotFoundError if absent. Deserialization succeeds because `tests/conftest.py` already places `src/` on sys.path; loader.py MUST NOT modify sys.path (rely on conftest).
      - `fixture_missing_reason(project: str) -> str | None` returns None when the project's pkl_dir contains layer1..4.pkl + final.pkl AND its calls_json exists; otherwise returns a human-readable string used as a `pytest.skip(reason=...)` argument, naming the missing file. Mirror the `tests/test_single_step_harness.py` style.
      - Loader functions are pure (no module-level filesystem reads; manifest loaded lazily on first call and cached via `functools.lru_cache`).

    Test 1.5 — replay_client.py:
      - `ReplayClient` constructor: takes no arguments. Internally instantiates `LLMClient(checkpoint_fallback="claude", checkpoint_fallback_model=None)` ONLY to bind its `extract_json` method; never calls .query(). Construction MUST NOT make network calls.
      - `ReplayClient.query(*args, **kwargs)` raises `RuntimeError("ReplayClient.query() is forbidden — Phase 44 harness must not contact any LLM backend")`. Asserted in unit tests.
      - `ReplayClient.extract_json(response_text: str) -> dict | None`: wraps response_text in `LLMResponse(text=response_text, success=True)` (importing LLMResponse from llm_sad_sam.llm_client) and delegates to `LLMClient.extract_json`. Returns the parsed dict or None for the no-JSON-found case (same semantics as production).
      - Module-level convenience: `replay_parse(response_text: str) -> dict | None` instantiates a singleton ReplayClient lazily and proxies. Imported by Plan 02's test modules.

    Test 1.6 — adapters.py:
      - `BUILDER_PHASE_TAGS: dict[str, tuple[str, ...]]` encodes D-03 verbatim:
          "_prompt_ambiguity": ("phase_1_model",)
          "_prompt_doc_knowledge_extract": ("phase_1_doc_extract",)
          "_prompt_doc_knowledge_judge": ("phase_1_doc_judge",)
          "_prompt_extraction": ("phase_2_framing_c_pass1", "phase_2_framing_c_pass2")
          "_prompt_validation": ("phase_4_twopass_p1", "phase_4_twopass_p2", "phase_5_coref_validation")
          "_prompt_coref": ("phase_5_coref",)
        Test asserts every value is a tuple, no list, no string — keeps the schema explicit.
      - `BUILDERS: dict[str, Callable]` maps each builder name to `SLinker19._prompt_<name>` (the `@staticmethod` reference). Import:
          `from llm_sad_sam.linkers.experimental.s_linker19 import SLinker19`
        Each value MUST be callable. The functions are NOT invoked in this plan — only registered. (Plan 02 invokes them under test.)
      - Imports are top-level; the `from llm_sad_sam...` import succeeds because `tests/conftest.py` puts `src/` on sys.path (verified for any test or pytest collection invocation).

    Test 1.7 — GATE-01 byte-equality assertion:
      - After all edits, `git diff --stat -- src/llm_sad_sam/` reports zero file changes. Test runs `git diff --stat HEAD -- src/llm_sad_sam/` and asserts the output is empty.
      - Snapshotted source set: s_linker19.py, s_linker13_min.py, prompts_v5.py, data_types_v2.py, llm_client.py — none modified.
  </behavior>

  <action>
    1. Edit pyproject.toml: under `[project.optional-dependencies] dev = [...]`, append `"syrupy>=4.6.0"` and `"pytest-socket>=0.7"`. Preserve the existing two entries and order. Do not touch the `openai`/`[project.scripts]`/`[tool.pytest.ini_options]` blocks.

    2. Create `tests/harness/__init__.py` and `tests/harness/fixtures/__init__.py` as empty files (package markers).

    3. Create `tests/harness/fixtures/MANIFEST.json` as a JSON list of 5 objects exactly matching D-02 (paths verbatim from the CONTEXT.md table). Include an optional `"description"` per entry equal to `"gpt-5.4 byte-equal baseline pinned for v2.6.4 Phase 44"`.

    4. Create `tests/harness/manifest.py` implementing per LOCK-4 and Test 1.2 behavior:
       - `MANIFEST_PATH = Path(__file__).resolve().parent / "fixtures" / "MANIFEST.json"`
       - `DATASETS: tuple[str, ...] = ("mediastore", "teastore", "teammates", "bigbluebutton", "jabref")`
       - `@dataclass(frozen=True) class FixtureEntry: project: str; pkl_dir: Path; calls_json: Path; description: str | None = None`
       - `def load_manifest() -> list[FixtureEntry]`: read MANIFEST_PATH, parse JSON, resolve `pkl_dir` and `calls_json` against the repo root (= `MANIFEST_PATH.parents[3]` — verify with the existing tests/conftest.py ROOT computation), return frozen entries in manifest order. Use `functools.lru_cache(maxsize=1)`.

    5. Create `tests/harness/loader.py` implementing Test 1.4 behavior. Imports: `pickle`, `json` not required if records are pre-parsed by the calls_json read (loader reads the entire calls_json once per project via lru_cache, then filters). Document in module docstring that sys.path bootstrap is delegated to `tests/conftest.py` and that this module MUST NOT modify sys.path.

    6. Create `tests/harness/replay_client.py` implementing Test 1.5 behavior. Import `LLMClient, LLMResponse` from `llm_sad_sam.llm_client`. ReplayClient holds `self._llm = LLMClient(checkpoint_fallback="claude")` and exposes `extract_json` via delegation; `query` raises RuntimeError as specified. Add `replay_parse` module-level helper using `functools.lru_cache(maxsize=1)` on a `_singleton()` factory.

    7. Create `tests/harness/adapters.py` implementing Test 1.6 behavior. Import `SLinker19` from `llm_sad_sam.linkers.experimental.s_linker19`. Define `BUILDER_PHASE_TAGS` and `BUILDERS` as module-level constants per the D-03 mapping verbatim.

    8. Add a unit-test module `tests/harness/test_loader_self.py` (lives in tests/harness/ so pytest collects it via `testpaths=["tests"]`) that exercises Tests 1.2/1.3/1.4/1.5/1.6. These are infrastructure self-tests, not snapshot tests — Plan 02 owns the six snapshot modules. Skip-on-missing-fixture convention applied per Test 1.4. Asserts ReplayClient.query() raises RuntimeError.

    9. Verify GATE-01: run `git diff --stat -- src/llm_sad_sam/` and confirm empty output. No edits to any file under src/. Per D-01/CLAUDE.md, this plan is read-only on s_linker19.py, prompts_v5.py, data_types_v2.py, llm_client.py, s_linker13_min.py.

    10. Install dev extras locally: `pip install -e ".[dev,openai]"` so the new syrupy + pytest-socket deps are available for Plan 02.
  </action>

  <verify>
    <automated>
      pytest tests/harness/test_loader_self.py -v --tb=short
    </automated>
    <automated>
      python -c "
import json, pathlib
m = json.loads(pathlib.Path('tests/harness/fixtures/MANIFEST.json').read_text())
assert isinstance(m, list) and len(m) == 5, f'manifest must have 5 entries, got {len(m)}'
projects = {e['project'] for e in m}
assert projects == {'mediastore','teastore','teammates','bigbluebutton','jabref'}, projects
for e in m:
    assert set(e.keys()) >= {'project','pkl_dir','calls_json'}, e
    assert pathlib.Path(e['pkl_dir']).is_dir(), e['pkl_dir']
    assert pathlib.Path(e['calls_json']).is_file(), e['calls_json']
print('MANIFEST.json OK')
"
    </automated>
    <automated>
      python -c "
import tomllib, pathlib
data = tomllib.loads(pathlib.Path('pyproject.toml').read_text())
dev = data['project']['optional-dependencies']['dev']
assert any(s.startswith('syrupy') for s in dev), dev
assert any(s.startswith('pytest-socket') for s in dev), dev
print('pyproject.toml [dev] OK')
"
    </automated>
    <automated>
      bash -c '
        diff_out=$(git diff --stat HEAD -- src/llm_sad_sam/)
        if [ -n "$diff_out" ]; then
          echo "GATE-01 FAIL — src/llm_sad_sam/ modified:"
          echo "$diff_out"
          exit 1
        fi
        echo "GATE-01 OK — src/llm_sad_sam/ byte-equal"
      '
    </automated>
    <automated>
      python -c "
import sys; sys.path.insert(0, 'tests'); sys.path.insert(0, 'src')
from harness.adapters import BUILDER_PHASE_TAGS, BUILDERS
assert BUILDER_PHASE_TAGS['_prompt_validation'] == ('phase_4_twopass_p1','phase_4_twopass_p2','phase_5_coref_validation'), BUILDER_PHASE_TAGS['_prompt_validation']
assert set(BUILDERS.keys()) == set(BUILDER_PHASE_TAGS.keys())
assert all(callable(b) for b in BUILDERS.values())
print('D-03 mapping verified')
"
    </automated>
  </verify>

  <acceptance_criteria>
    - tests/harness/__init__.py exists (empty package marker).
    - tests/harness/fixtures/MANIFEST.json parses as JSON, contains exactly 5 entries with required keys {project, pkl_dir, calls_json}, and all 5 (pkl_dir, calls_json) pairs exist on disk.
    - tests/harness/manifest.py defines load_manifest, FixtureEntry, DATASETS, MANIFEST_PATH at module level.
    - tests/harness/loader.py defines load_records, load_pkl, fixture_missing_reason at module level with the signatures described under behavior.
    - tests/harness/replay_client.py: `pytest tests/harness/test_loader_self.py::test_replay_client_query_forbidden` passes (verifying RuntimeError is raised).
    - tests/harness/adapters.py: BUILDER_PHASE_TAGS["_prompt_validation"] is a 3-tuple containing "phase_5_coref_validation" (D-03 gotcha verified in code, not just docs).
    - `pytest tests/harness/test_loader_self.py` exits 0.
    - `python -c "import tomllib; assert any(s.startswith('syrupy') for s in tomllib.loads(open('pyproject.toml').read())['project']['optional-dependencies']['dev'])"` exits 0.
    - `git diff --stat HEAD -- src/llm_sad_sam/` produces zero lines (GATE-01 byte-equal).
    - `grep -rE "openai|anthropic|requests\\.(get|post)|httpx|\\.query\\(" tests/harness/ --include="*.py" | grep -v "raises RuntimeError" | grep -v "extract_json" | grep -v '^#'` produces zero non-comment matches (network-egress smoke check — only the forbidden-.query() RuntimeError mention is allowed).
  </acceptance_criteria>

  <done>
    Fixture infrastructure is importable from tests/harness/, MANIFEST.json pins all 5 D-02 entries, ReplayClient enforces zero-LLM-call by raising on .query(), D-03 builder→phase-tag map is encoded in adapters.py with the coref-validation gotcha, snapshot library + zero-network test dep are wired into pyproject.toml [dev], and no source file under src/llm_sad_sam/ was modified. Plan 02 can build the six snapshot modules on top of this without further infrastructure decisions.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| disk → process (pickle.load) | `phase_cache/*.pkl` files deserialized in-process |
| disk → process (json.loads) | `MANIFEST.json` + `_calls.json` parsed in-process |
| pytest process → network | Any LLM client construction risks env-driven backend init |
| test code → frozen source | Read-only access to s_linker19.py, prompts_v5.py, llm_client.py |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-44-01 | Tampering | `pickle.load` on phase_cache/*.pkl | accept | Fixtures are git-committed under `results/phase_cache/s_linker19/openai/`; MANIFEST.json pins canonical paths. No attacker-controllable input. Pickle deserialization restricted to dataclasses defined in `src/llm_sad_sam/core/data_types_v2.py` (transitively reachable from sys.path bootstrap in tests/conftest.py). Threat surface = same as any test reading from results/. |
| T-44-02 | Tampering | `json.loads` on `_calls.json` and `MANIFEST.json` | accept | Same as T-44-01 — files are committed/manifest-pinned, not user-supplied at test runtime. Standard `json.loads` (no `object_hook`) cannot execute code. |
| T-44-03 | Information Disclosure / DoS | Accidental network egress from `LLMClient` construction in `ReplayClient` | mitigate | `ReplayClient.__init__` constructs `LLMClient(checkpoint_fallback="claude")` purely to bind its bound-method `extract_json`. `extract_json` does pure `json.loads` (verified at `llm_client.py:985–1019`) — no network I/O. `ReplayClient.query` raises `RuntimeError` to make any accidental call mode fail loud. Plan 02 adds `pytest --disable-socket` as a second seatbelt. |
| T-44-04 | Tampering / Repudiation | Silent mutation of frozen source files breaking GATE-01 | mitigate | Plan acceptance criterion runs `git diff --stat HEAD -- src/llm_sad_sam/` and fails if non-empty. CI-discoverable. |
| T-44-05 | Tampering | Snapshot library supply-chain (syrupy + pytest-socket are new dev deps) | mitigate | Both packages are mature, widely-used pytest plugins on PyPI. syrupy: maintained by Tony Narlock & contributors (1.5k+ stars, weekly releases). pytest-socket: maintained by Mike Pirnat (350+ stars, used in pytest stdlib testing patterns). Version pins (`>=4.6.0`, `>=0.7`) preserve forward compatibility. No `[ASSUMED]` / `[SUS]` / `[SLOP]` rating — both are textbook pytest plugins. Listed under `[dev]` extras only; not shipped with the production wheel. |
| T-44-SC | Tampering | npm/pip/cargo installs | mitigate | Two new pip dependencies (syrupy, pytest-socket) — both verifiable on pypi.org. Rated LEGITIMATE per T-44-05 audit; no blocking human checkpoint required because both are textbook pytest plugins with public install metrics. RESEARCH.md does not exist for this milestone, so the legitimacy classification is recorded inline in this threat register. |
</threat_model>

<verification>
- `pytest tests/harness/test_loader_self.py -v` exits 0.
- MANIFEST.json contains all 5 D-02 entries; every referenced file exists on disk.
- `git diff --stat HEAD -- src/llm_sad_sam/` is empty (GATE-01 byte-equality preserved).
- Network-egress smoke: only allowed reference to `.query(` in tests/harness/ is the `RuntimeError` raised by `ReplayClient.query` itself.
- pyproject.toml [project.optional-dependencies].dev contains both syrupy and pytest-socket.
- D-03 mapping encoded in code: `BUILDER_PHASE_TAGS["_prompt_validation"]` includes `"phase_5_coref_validation"`.
</verification>

<success_criteria>
1. tests/harness/ package importable from pytest test code (`from tests.harness.loader import load_records` works under `testpaths=["tests"]`).
2. load_manifest() returns 5 FixtureEntry records covering DATASETS, with each pkl_dir + calls_json existing on disk per CONTEXT D-02.
3. ReplayClient.query() raises RuntimeError (asserted in test_loader_self.py); ReplayClient.extract_json delegates to LLMClient.extract_json byte-equal.
4. BUILDER_PHASE_TAGS encodes D-03 verbatim including the phase_5_coref_validation → _prompt_validation gotcha.
5. pyproject.toml [project.optional-dependencies].dev contains syrupy>=4.6.0 and pytest-socket>=0.7.
6. GATE-01: zero modifications to any file under src/llm_sad_sam/ (verified by `git diff --stat HEAD -- src/llm_sad_sam/`).
</success_criteria>

<output>
Create `.planning/phases/44-harness/44-01-SUMMARY.md` when done, listing:
- The 5 MANIFEST entries created
- The locked decisions (LOCK-1 through LOCK-6 listed in <behavior>)
- GATE-01 status (must be PASS)
- syrupy + pytest-socket install confirmation
- Plan 02 handoff: BUILDERS map keys (6), BUILDER_PHASE_TAGS schema, replay_parse() signature, fixture_missing_reason() signature
</output>

## Artifacts this phase produces

**New files:**
- `tests/harness/__init__.py`
- `tests/harness/fixtures/__init__.py`
- `tests/harness/fixtures/MANIFEST.json`
- `tests/harness/manifest.py`
- `tests/harness/loader.py`
- `tests/harness/replay_client.py`
- `tests/harness/adapters.py`
- `tests/harness/test_loader_self.py`

**New Python symbols:**
- `tests.harness.manifest.FixtureEntry` (frozen dataclass: `project: str`, `pkl_dir: Path`, `calls_json: Path`, `description: str | None = None`)
- `tests.harness.manifest.MANIFEST_PATH` (Path constant)
- `tests.harness.manifest.DATASETS` (tuple constant: `("mediastore", "teastore", "teammates", "bigbluebutton", "jabref")`)
- `tests.harness.manifest.load_manifest() -> list[FixtureEntry]`
- `tests.harness.loader.load_records(project: str, phase_tag: str) -> list[dict]`
- `tests.harness.loader.load_pkl(project: str, layer: str) -> object`
- `tests.harness.loader.fixture_missing_reason(project: str) -> str | None`
- `tests.harness.replay_client.ReplayClient` (class with `extract_json` delegating to LLMClient.extract_json; `query` raises RuntimeError)
- `tests.harness.replay_client.replay_parse(response_text: str) -> dict | None`
- `tests.harness.adapters.BUILDER_PHASE_TAGS` (dict[str, tuple[str, ...]] encoding D-03)
- `tests.harness.adapters.BUILDERS` (dict[str, Callable] mapping builder names to `SLinker19._prompt_*` staticmethods)

**MANIFEST.json schema (frozen for Phase 44):**
- Top-level: JSON list
- Per-entry keys: `project` (str), `pkl_dir` (str — repo-root-relative), `calls_json` (str — repo-root-relative), `description` (str, optional)
- Entry count: 5
- Cardinality: one entry per project in DATASETS

**Edited files:**
- `pyproject.toml` — `[project.optional-dependencies].dev` gains `"syrupy>=4.6.0"` and `"pytest-socket>=0.7"` (no other changes).

**New CLI flags / scripts:** none. (Plan 02 may add a `--snapshot-update` invocation script if needed for initial capture.)
