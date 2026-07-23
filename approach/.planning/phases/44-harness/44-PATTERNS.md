# Phase 44: HARNESS — Pattern Map

**Mapped:** 2026-06-05
**Files analyzed:** 9 to be created/modified
**Analogs found:** 9 / 9

---

## File Classification

| New / Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `tests/harness/__init__.py` | package init | n/a | (none — empty marker) | n/a |
| `tests/harness/fixtures/MANIFEST.json` | config / manifest | static-load | `tests/fixtures/v2_0_baseline.json` | role-match (JSON ledger) |
| `tests/harness/loader.py` (fixture loader + manifest reader + per-builder adapters) | utility / fixture-infra | file-I/O + transform | `tests/conftest.py` (sys.path bootstrap) + `_TracingLLMClient` (record schema) + `tests/test_single_step_harness.py` (cache-presence skip) | role-match composite |
| `tests/test_s_linker20_prompt_ambiguity.py` | test (snapshot) | request-response (replay) | `tests/test_v20_baseline_regression.py` (JSON-fixture-driven pytest) | exact (JSON-fixture-driven, parametrized) |
| `tests/test_s_linker20_prompt_doc_extract.py` | test (snapshot) | request-response (replay) | same as above | exact |
| `tests/test_s_linker20_prompt_doc_judge.py` | test (snapshot) | request-response (replay) | same as above | exact |
| `tests/test_s_linker20_prompt_extraction.py` | test (snapshot, multi-call) | batch / request-response | same as above | exact |
| `tests/test_s_linker20_prompt_validation.py` | test (snapshot, multi-tag) | batch / request-response | same as above | exact (3 phase tags) |
| `tests/test_s_linker20_prompt_coref.py` | test (snapshot, batched) | batch | same as above | exact |
| `pyproject.toml` (edit `[project.optional-dependencies].dev`) | config | n/a | existing `[dev]` block lines 19–22 | exact in-file |

---

## Pattern Assignments

### `tests/harness/loader.py` (fixture-infra utility)

**Analog A — sys.path bootstrap (already inherited via `tests/conftest.py`, but harness scripts run outside pytest may need to replicate it):**
Source: `tests/conftest.py` (lines 1–10)
```python
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```
**Apply:** any snapshot-capture utility outside `tests/` must replicate; tests under `tests/harness/` inherit it via the top-level `conftest.py`. The pickle deserialization of `phase_cache/*.pkl` requires `src/` on `sys.path` because the pickles contain `llm_sad_sam.core.data_types_v2` dataclass references.

**Analog B — `_calls.json` record schema (the wire format the loader must decode):**
Source: `src/llm_sad_sam/linkers/experimental/s_linker19.py` lines 132–156 (`_TracingLLMClient.query`)
```python
def query(self, prompt: str, timeout: int = 180, max_retries: int = 3) -> LLMResponse:
    phase = _current_phase()
    t0 = time.time()
    resp = self._inner.query(prompt, timeout=timeout, max_retries=max_retries)
    record = {
        "phase": phase, "ts": t0,
        "elapsed_s": round(time.time() - t0, 3),
        "timeout": timeout, "max_retries": max_retries,
        "prompt": prompt,
        "response_text": getattr(resp, "text", None),
        "success": getattr(resp, "success", None),
        "error": getattr(resp, "error", None),
        "latency_ms": getattr(resp, "latency_ms", None),
        "model": getattr(resp, "model", None),
    }
    ...
    self._sink.append(record)
```
Verified by inspection: a real file is a JSON **list** of 14 such records (for mediastore); keys present are `{elapsed_s, error, latency_ms, max_retries, model, phase, prompt, response_text, success, timeout, token_usage, ts}`. The loader's per-builder adapter groups records by `record["phase"]` per the D-03 mapping, then exposes `(prompt, response_text)` pairs. Decoded `response_text` is fed back through `LLMClient.extract_json` (treated as the parser) by wrapping it in an `LLMResponse`:
```python
# pseudocode for replay adapter
from llm_sad_sam.llm_client import LLMResponse
fake_resp = LLMResponse(text=record["response_text"], success=True)
parsed = real_client.extract_json(fake_resp)   # canonical parser path
```
`LLMResponse` constructor (from `src/llm_sad_sam/llm_client.py` lines 37–46):
```python
@dataclass
class LLMResponse:
    text: str
    success: bool
    error: Optional[str] = None
    token_usage: Optional[TokenUsage] = None
    model: Optional[str] = None
    latency_ms: Optional[int] = None
```

**Analog C — manifest-driven fixture root + skip-on-missing convention:**
Source: `tests/test_single_step_harness.py` lines 26–44
```python
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

BASELINE_CACHE = ROOT / "results" / "phase_cache" / "s_linker13_clean" / "mediastore"
REQUIRED_PKLS = ("layer1.pkl", "layer2.pkl", "entity_candidates.pkl",
                 "entity_decisions.pkl", "final.pkl")

_baseline_present = all((BASELINE_CACHE / name).exists() for name in REQUIRED_PKLS)
requires_baseline = pytest.mark.skipif(
    not _baseline_present,
    reason=("baseline s_linker13_clean checkpoints missing under "
            f"{BASELINE_CACHE} — Task 2 smoke tests require Phase 10 fixtures"),
)
```
**Apply:** mirror the `REQUIRED_PKLS` + `pytest.mark.skipif` pattern for each manifest project to keep CI green when fixtures are absent. Adapt to iterate over the manifest entries.

**Analog D — builder signatures (what the adapter must call to rebuild prompts):**
Source: `src/llm_sad_sam/linkers/experimental/s_linker19.py` lines 263–377. All six are `@staticmethod` and take only plain-data inputs (lists of strings / sentence objects), which means the loader can call them without instantiating `SLinker19` (no LLM construction). Signatures:
```python
_prompt_ambiguity(names) -> str                                            # line 264
_prompt_doc_knowledge_extract(comp_names, doc_lines) -> str                # line 284
_prompt_doc_knowledge_judge(comp_names, mapping_list) -> str               # line 304
_prompt_extraction(comp_names, mappings, batch) -> str                     # line 321
_prompt_validation(comp_names, cases, focus) -> str                        # line 337
_prompt_coref(comp_names, cases) -> str                                    # line 352
```
**Apply:** because builders are pure functions of (a) the phase-cache state at call time and (b) the prompt constants imported from `prompts_v5`, the harness can rebuild prompts deterministically from the pkl payload + the static constants. The reconstructed prompt must equal `record["prompt"]` byte-for-byte for the builder–tag pairing in D-03 to be trustworthy; if it doesn't, the test is asserting the wrong invariant.

**Analog E — phase-tag call sites (drives D-03 mapping):**
Source: `s_linker19.py` lines 561, 573, 604, 646, 793, 835, 894
```python
self.llm.set_phase("phase_1_model")             # line 561   → _prompt_ambiguity
self.llm.set_phase("phase_1_doc_extract")       # line 573   → _prompt_doc_knowledge_extract
self.llm.set_phase("phase_1_doc_judge")         # line 604   → _prompt_doc_knowledge_judge (via _ask(phase=...))
self.llm.set_phase(phase_tag)                   # line 646   → _prompt_extraction       (tags: phase_2_framing_c_pass1/pass2)
self.llm.set_phase(phase_tag)                   # line 793   → _prompt_validation       (tags: phase_4_twopass_p1/p2)
self.llm.set_phase("phase_5_coref")             # line 835   → _prompt_coref
self.llm.set_phase("phase_5_coref_validation")  # line 894   → _prompt_validation again (D-03 gotcha)
```

---

### `tests/test_s_linker20_prompt_*.py` (six pytest snapshot modules)

**Analog: `tests/test_v20_baseline_regression.py`** (JSON-fixture-driven, parametrized, fixture-as-source-of-truth).

**Fixture-load pattern** (lines 37–48):
```python
FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "v2_0_baseline.json"
DATASETS = ("mediastore", "teastore", "teammates", "bigbluebutton", "jabref")


@pytest.fixture(scope="module")
def baseline():
    """Load the pinned v2.0 baseline JSON once per test module."""
    assert FIXTURE_PATH.exists(), (
        f"v2.0 baseline fixture missing at {FIXTURE_PATH} - "
        "Phase 10 Plan 01 Task 1 must run first."
    )
    return json.loads(FIXTURE_PATH.read_text())
```
**Apply:** each snapshot test module gets one module-scoped fixture that calls the `tests/harness/loader.py` API to load the MANIFEST and the records for that builder's phase tags. The five-project tuple matches `DATASETS` exactly — re-use the same constant name to keep convention.

**Per-project parametrization** (this codebase uses `pytest.mark.parametrize` with the dataset tuple; see e.g. similar pattern in `tests/test_v20_baseline_regression.py` further down). Snapshot library choice is deferred to planner, but for **syrupy** the call-site shape is:
```python
@pytest.mark.parametrize("project", DATASETS)
def test_ambiguity_parsed_snapshot(project, snapshot):
    record = load_records(project, phase="phase_1_model")[0]
    rebuilt_prompt = SLinker19._prompt_ambiguity(record["inputs"]["names"])
    assert rebuilt_prompt == record["prompt"], "GATE: rebuilt prompt must match logged prompt"
    parsed = replay_parse(record["response_text"])
    assert parsed == snapshot
```
For **pytest-regressions** swap `assert parsed == snapshot` for `data_regression.check(parsed)`. Either is compatible with the loader API.

**Multi-tag module (validation):** the `test_s_linker20_prompt_validation.py` module parametrizes over `(project, phase_tag)` where `phase_tag ∈ {"phase_4_twopass_p1", "phase_4_twopass_p2", "phase_5_coref_validation"}` per D-03.

**Multi-call module (extraction & coref):** records per `(project, phase_tag)` are a *list* (one per batch in `_iter_batches`). Parametrize as `(project, call_index)` to give each batched call its own snapshot.

---

### `pyproject.toml` (add snapshot lib to `[dev]`)

**Analog: existing `[project.optional-dependencies]` block at lines 19–22:**
```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]
```
**Apply (planner picks one):**
```toml
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "syrupy>=4.6.0",                  # option A
    # "pytest-regressions>=2.5.0",    # option B
]
```
Confirmed neither is currently installed (`python3 -c "import syrupy"` and `import pytest_regressions` both raise `ModuleNotFoundError`). The chosen lib MUST land here before tests can run.

---

### `tests/harness/fixtures/MANIFEST.json`

**Analog: `tests/fixtures/v2_0_baseline.json`** — JSON ledger consumed by a single test module, single source of truth for "what is the pinned state?" Per D-02 the minimum schema is:
```json
[
  {
    "project": "mediastore",
    "pkl_dir": "results/phase_cache/s_linker19/openai/mediastore/",
    "calls_json": "results/llm_logs/s_linker19_openai_mediastore_20260605_134622_calls.json"
  },
  ...
]
```
Paths are repo-root-relative (resolved against `ROOT` from `conftest.py`). Existence of all five `*_calls.json` filenames listed in D-02 confirmed by `ls` at scout time. Planner may add optional `expected_sha256` (per "Claude's Discretion") for drift detection / GATE-01 hook.

---

## Shared Patterns

### Read-only LLM-client / zero-API-call guarantee
**Source:** `src/llm_sad_sam/llm_client.py` lines 985–1015 (`extract_json`)
```python
def extract_json(self, response: LLMResponse) -> Optional[dict]:
    if not response.success or not response.text:
        return None
    text = response.text.strip()
    # Fast path: entire response is valid JSON
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # ... balanced-brace recovery ...
```
**Apply to:** every test module. The harness's parser path is `LLMClient.extract_json(LLMResponse(text=record["response_text"], success=True))`. **No `query()` is ever invoked.** A `LLMClient` instance can be constructed in checkpoint-fallback mode (so no API key is needed), or the loader can call `extract_json` as a bound method on a class-method-style helper. Confirmation that exit-code-0 + zero LLM calls (REQ-V264-01 / Success Criterion 4) is via the absence of any network I/O in the parser path — guaranteed structurally because `extract_json` only calls `json.loads`.

### Fixture-presence skip convention
**Source:** `tests/test_single_step_harness.py` lines 38–44 (shown above).
**Apply to:** all six test modules. If any project's `pkl_dir` or `calls_json` is missing, skip *that* project's parametrization (not the entire module) with an actionable reason string. This keeps CI green in clean checkouts and on developers' machines that haven't fetched the `phase_cache` artefacts.

### sys.path bootstrap
**Source:** `tests/conftest.py` (all 10 lines, shown above).
**Apply to:** inherited automatically by anything under `tests/`. The harness fixture loader does **not** need to re-bootstrap. Any snapshot-capture utility outside `tests/` (e.g., a one-shot `scripts/capture_s19_snapshots.py` if the planner introduces one) MUST replicate.

### GATE-01 byte-equality preservation
**Source:** CLAUDE.md (project rules) + REQ-V264-08 + GATE-01.
**Apply to:** the harness is read-only on `s_linker19.py`, `prompts_v5.py`, `s_linker13_min.py`, and any transitively-imported module. The `@staticmethod` decoration on the six prompt builders means we can `from llm_sad_sam.linkers.experimental.s_linker19 import SLinker19` and call `SLinker19._prompt_ambiguity(...)` without instantiating the class or touching the LLM client. This is the recommended adapter strategy (Claude's Discretion option (c) in CONTEXT.md) — it avoids the monkey-patching of option (a) and the refactor risk of option (b).

---

## No Analog Found

None. All nine files have at least a role-match analog in `tests/` or in the source tree.

---

## Metadata

**Analog search scope:** `tests/`, `tests/fixtures/`, `src/llm_sad_sam/linkers/experimental/s_linker19.py`, `src/llm_sad_sam/llm_client.py`, `results/llm_logs/` (schema verification), `results/phase_cache/s_linker19/openai/` (presence verification), `pyproject.toml`.
**Files scanned (Read):** 8 (`tests/conftest.py`, `tests/test_v20_baseline_regression.py`, `tests/test_single_step_harness.py`, `s_linker19.py` × 4 ranges, `llm_client.py` × 2 ranges, `pyproject.toml`, `REQUIREMENTS.md`, `ROADMAP.md` slice, `44-CONTEXT.md`, `CLAUDE.md`).
**Files probed (Bash/grep):** `results/llm_logs/`, `results/phase_cache/s_linker19/openai/`, snapshot-lib availability check.
**Pattern extraction date:** 2026-06-05.
