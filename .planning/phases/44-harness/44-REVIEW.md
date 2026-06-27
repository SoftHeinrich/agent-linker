---
phase: 44-harness
reviewed: 2026-06-07T00:00:00Z
depth: standard
files_reviewed: 17
files_reviewed_list:
  - tests/harness/__init__.py
  - tests/harness/adapters.py
  - tests/harness/fixtures/__init__.py
  - tests/harness/fixtures/MANIFEST.json
  - tests/harness/inputs.py
  - tests/harness/loader.py
  - tests/harness/manifest.py
  - tests/harness/replay_client.py
  - tests/harness/test_loader_self.py
  - tests/test_s_linker20_harness_invariants.py
  - tests/test_s_linker20_prompt_ambiguity.py
  - tests/test_s_linker20_prompt_coref.py
  - tests/test_s_linker20_prompt_doc_extract.py
  - tests/test_s_linker20_prompt_doc_judge.py
  - tests/test_s_linker20_prompt_extraction.py
  - tests/test_s_linker20_prompt_validation.py
  - tests/__snapshots__/* (6 .ambr files)
findings:
  critical: 1
  warning: 6
  info: 5
  total: 12
status: issues_found
---

# Phase 44: Code Review Report

**Reviewed:** 2026-06-07T00:00:00Z
**Depth:** standard
**Files Reviewed:** 17
**Status:** issues_found

## Summary

The Phase 44 harness establishes a golden-replay snapshot infrastructure with strong
zero-LLM-egress guards (`ReplayClient.query()` forbidden, multi-layer grep + pytest-socket
invariants). The contract surface (manifest, loader, adapters, replay_client) is
generally clean and well-documented.

Two structural defects warrant attention:

1. **Dual import paths for the same harness modules** (`tests.harness.loader` from
   snapshot tests vs. `harness.loader` from `test_loader_self.py` and from
   `loader.py` itself). Under pytest's default rootdir-mode sys.path injection,
   both import paths can resolve, producing two distinct module objects with
   independent `lru_cache` state and divergent `FixtureEntry` class identities.
   This is the single most consequential issue in this batch (BLOCKER for hash/
   identity-based callers; latent for the current test surface).

2. **`ReplayClient.__init__` instantiates a fully-configured `LLMClient`** that
   honors `LLM_BACKEND=openai` from `.env`, opens a log file, and creates a session
   directory under `~/.llm-sad-sam/sessions`. The stated rationale ("checkpoint
   fallback claude avoids OpenAI client construction") is incorrect — that flag
   only applies when `backend == CHECKPOINT`. The OpenAI client itself remains
   lazy so no network call fires, but the disk side-effects and the misleading
   comment are warnings worth fixing.

Five quality findings and several info items round out the report.

---

## Critical Issues

### CR-01: Dual import paths produce two module instances of the harness

**Files:**
- `tests/harness/loader.py:19` — `from harness.manifest import load_manifest, FixtureEntry`
- `tests/harness/test_loader_self.py:71,201,299,351,…` — `from harness.{manifest,loader,replay_client,adapters} import …`
- `tests/test_s_linker20_prompt_*.py:16-24` — `from tests.harness.{loader,adapters,…} import …`
- `tests/conftest.py:1-10` — sys.path bootstrap adds only `ROOT` and `ROOT/src`

**Issue:**
The conftest adds `<repo>/` and `<repo>/src/` to `sys.path`. Snapshot test modules
import via the dotted form `tests.harness.…`, which resolves because Python 3 treats
the `tests/` directory as an implicit namespace package (there is no
`tests/__init__.py`).

In parallel, pytest's rootdir-mode `rootdir` insertion adds the test file's parent
(`tests/`) to `sys.path`. That makes the SAME files importable under a *second*
dotted name: `harness.manifest`, `harness.loader`, `harness.replay_client`,
`harness.adapters`. Both `test_loader_self.py` and `tests/harness/loader.py` itself
use this short form (`from harness.manifest import …`).

Consequence: `tests.harness.manifest` and `harness.manifest` end up as two distinct
module objects in `sys.modules`. Each has its own copy of:

- `load_manifest()` (decorated with `@functools.lru_cache`) — manifest is parsed
  and validated twice; per-call cache is doubled.
- `_load_calls_json()` (also `lru_cache`-decorated) — calls.json is parsed twice
  per project; memory cost doubled.
- The `FixtureEntry` class itself — two separate classes with the same name.
  `isinstance(entry, tests.harness.manifest.FixtureEntry)` is False for an entry
  created via `harness.manifest.FixtureEntry`, breaking any future identity check.
- The `DATASETS` tuple — duplicated; harmless for equality but a footgun if any
  consumer ever does `is` instead of `==`.

The current test surface does not yet exercise an `isinstance` path, so functional
tests pass. But the violation is real and will silently bite the first time a
caller compares classes by identity, mocks one module path, or relies on
`lru_cache` warm-up.

Additionally, `tests/harness/loader.py:19` reading `from harness.manifest …`
implies the harness package is being designed as if `tests/` were on sys.path,
while the snapshot tests treat `tests/` as a package root. The two design choices
are mutually inconsistent.

**Fix:**
Pick one canonical import path and use it everywhere. Recommended: standardise
on `tests.harness.…` since the snapshot modules already do that and conftest
already adds ROOT to sys.path.

```python
# tests/harness/loader.py
from tests.harness.manifest import load_manifest, FixtureEntry  # was: from harness.manifest …

# tests/harness/test_loader_self.py — replace every:
from harness.manifest import …
# with:
from tests.harness.manifest import …
# (same for harness.loader, harness.replay_client, harness.adapters)
```

Then verify with `python -c "import sys; assert 'harness.manifest' not in sys.modules"`
after a test run.

Alternative: add `tests/__init__.py` and configure pytest with
`pythonpath = ["src"]` only (removing the rootdir injection of `tests/`). That
also forces a single canonical path but is a larger refactor.

---

## Warnings

### WR-01: `ReplayClient.__init__` opens log files and creates session directories

**File:** `tests/harness/replay_client.py:33-36`

**Issue:**
`LLMClient(checkpoint_fallback="claude")` triggers the full LLMClient constructor.
That constructor:

- reads `LLM_BACKEND` from the environment (`.env` sets it to `openai` per
  `CLAUDE.md`), so `self.backend` becomes `OPENAI` regardless of the
  `checkpoint_fallback` argument — see `llm_client.py:92-105` and `:129`
  (checkpoint_fallback is only honored when `self.backend == CHECKPOINT`).
- calls `_subprocess_cwd.mkdir(parents=True, exist_ok=True)` under
  `~/.llm-sad-sam/sessions/<session_id>` for every ReplayClient instance
  (`llm_client.py:155-159`).
- calls `_setup_logging()` which opens a log file handle when
  `enable_logging=True` (the default).

For a "replay-safe" client whose only job is to delegate to `extract_json`
(pure `json.loads`), these are unwanted side effects. They also contradict the
class docstring claim "Construction does NOT make network calls. LLMClient is
instantiated with checkpoint_fallback="claude" so no backend-specific
initialisation runs." — the second sentence is factually wrong.

**Fix:**
Either bypass `LLMClient` instantiation entirely and import the parsing helper
directly, or disable logging and force a no-op backend:

```python
def __init__(self) -> None:
    # extract_json is pure json.loads and does not need a configured backend.
    self._llm: LLMClient = LLMClient(
        backend=LLMBackend.CLAUDE,   # force a known no-network default
        enable_logging=False,        # no log file
    )
```

And update the docstring to match reality. If `LLMClient.__init__` cannot be
configured to skip the session-dir + logging, refactor `extract_json` into a
free function and call it directly without instantiating `LLMClient`.

---

### WR-02: `_singleton()` caches a process-global `ReplayClient` across pytest sessions

**File:** `tests/harness/replay_client.py:78-81`

**Issue:**
`functools.lru_cache(maxsize=1)` keeps the singleton alive for the lifetime of
the Python process. In a long-running pytest session that flips `LLM_BACKEND`
between tests (e.g., via monkeypatch), the cached client retains the original
backend. Worse: if the singleton was created under `enable_logging=True`, the
log file remains open until process exit (no `close()` mechanism).

**Fix:**
Add a `clear_cache()` helper for tests that need to reset state, or expose the
singleton creation behind a pytest fixture. At minimum, document the lifetime
explicitly:

```python
def _reset_singleton() -> None:
    """Used by tests that need a fresh ReplayClient (e.g., after monkeypatching env)."""
    _singleton.cache_clear()
```

---

### WR-03: Dead imports in `inputs.py` create confusion about the parser strategy

**File:** `tests/harness/inputs.py:34-35`

**Issue:**
```python
from llm_sad_sam.core.data_types_v2 import ModelKnowledge, DocumentKnowledge
from llm_sad_sam.linkers.experimental.prompts_v5 import COREF_VALIDATION_FOCUS
```

None of these symbols is referenced in the file. The `COREF_VALIDATION_FOCUS`
import is especially misleading because the docstring at lines 247-250 explicitly
states "we reconstruct it by reading it directly from the prompt rather than
importing the constant" — yet the import remains. A reader will assume the
constant is used somewhere and waste time grepping.

**Fix:** Remove the three unused imports.

---

### WR-04: `reconstruct_doc_extract_inputs` drops empty document lines

**File:** `tests/harness/inputs.py:120`

**Issue:**
```python
doc_lines = [ln for ln in doc_content.split("\n") if ln]
```

The builder emits `chr(10).join(doc_lines)`. If any sentence in `doc_lines`
were an empty string (legal for a degenerate sentence), the rebuild would
contain a blank line that the reverse extractor strips, breaking the step-6
byte-equality check. The current fixture set probably never hits this, but the
asymmetric handling between builder (preserves) and extractor (drops) is a
latent fragility.

**Fix:** Preserve every line so the round-trip is symmetric:

```python
doc_lines = doc_content.split("\n")
# Optionally strip a single trailing "" if .find("\n\nReturn JSON:") matched
# (the builder always ends doc_content without a trailing newline).
```

A similar issue exists in `reconstruct_extraction_inputs` at line 226
(`if not line.strip(): continue`), though Sentence text is less likely to be
empty in practice.

---

### WR-05: `reconstruct_validation_inputs` Case-segment splitter trims trailing blank lines

**File:** `tests/harness/inputs.py:308-327`

**Issue:**
The code does:
```python
while current_lines and not current_lines[-1].strip():
    current_lines.pop()
```
to strip trailing blank lines from each case before flushing. The builder
joins cases with `chr(10).join(cases)` (one newline between segments). If a
legitimate case ended with a blank line in the input, the round-trip would
diverge. The step-6 byte-equality check is the only safety net.

Additionally, the line-by-line `re.match(r"^Case \d+:", line)` check will
misclassify any line *inside* a case body that happens to start with
`Case 3:` (e.g., quoted documentation). Low probability but unbounded by
the parser.

**Fix:**
1. Keep blank lines verbatim and trust the splitter to use unambiguous
   sentinels.
2. Use a stricter regex anchored on a known sentinel (e.g., `^Case \d+: S\d+`)
   to reduce false matches.

---

### WR-06: `reconstruct_coref_inputs` terminal-marker substrings can match user content

**File:** `tests/harness/inputs.py:391-396`

**Issue:**
```python
terminal_markers = [
    "\nAre there any",      # part of ANTECEDENT_ALIAS_RULES
    "\nFor each anaphoric", # part of COREF_RULES
    "\nReturn JSON:",
    "\n\nJSON only:",
]
```
These are searched via `prompt.find(marker, header_end)`. The first three are
short, ambiguous English phrases. If any context sentence in a coref CASE
contains the literal text "Are there any …" (plausible for a documentation
sentence), the parser will declare the block ended prematurely and drop the
remainder of the cases.

**Fix:** Anchor on a delimiter the builder controls deterministically, e.g.
the exact prefix that precedes `COREF_RULES` in `prompts_v5.py`. Or, since
the prompt structure is fully known, locate `COREF_RULES.lstrip().split("\n", 1)[0]`
as the sentinel rather than guessing English fragments.

---

## Info

### IN-01: `test_doc_extract_parsed_snapshot` swallows step-6 failures as warnings

**File:** `tests/test_s_linker20_prompt_doc_extract.py:56-73`

**Issue:** The test emits `warnings.warn(...)` instead of failing when the
rebuilt prompt diverges. The dead `else: assert rebuilt_prompt == record["prompt"]`
branch on line 68 is unreachable (the assertion is preceded by the `if` check
on line 56). When the workaround is removed (after fixtures refresh), the dead
`else` branch will silently never fire if `prompt_equal` is True — it's a no-op
re-assertion. Either the warning should be a hard assert or the structure
should be flattened.

**Fix:** Once teastore/teammates/bigbluebutton fixtures are refreshed, remove
the conditional and always assert. Until then, simplify:

```python
if rebuilt_prompt != record["prompt"]:
    warnings.warn(f"[prompt-version-drift] {project!r}/{_PHASE_TAG!r}: …",
                  UserWarning, stacklevel=1)
# The else branch is a no-op; drop it.
```

---

### IN-02: `_REQUIRED_LAYERS` over-strict for snapshot-only tests

**File:** `tests/harness/loader.py:22-28`

**Issue:** `fixture_missing_reason` skips a project if any of `layer1..layer4`
or `final` pkl is missing, but the snapshot tests never call `load_pkl`. A user
with a partial fixture set (calls.json present, pkls missing) sees every
snapshot test skipped even though they could run from calls.json alone.

**Fix:** Either split the check into `calls_missing_reason` and
`pkls_missing_reason` (snapshot tests use the former only) or weaken the
existing function and add a separate `pkls_missing_reason` for tests that
genuinely need pickles.

---

### IN-03: `_is_allowlisted_query_match` allowlist is too broad

**File:** `tests/test_s_linker20_harness_invariants.py:144-188`

**Issue:** The allowlist returns `True` for any line in `replay_client.py`
(line 175-177: "Only the def query line is a definition; other .query( in that
file are docstring text"). This means an accidental future call to `.query()`
added to `replay_client.py` would be silently allowlisted. Similarly, the
"forbidden in code" string match on line 185 will allowlist any line that just
mentions the word "forbidden", regardless of whether it's a real call.

**Fix:** Tighten the allowlist to a closed enumeration of (file, line) pairs
or to specific regex patterns that match only docstring content (e.g.,
`"""` on the same line).

---

### IN-04: `LLMResponse` constructor coupling is fragile

**File:** `tests/harness/replay_client.py:70`

**Issue:** `fake_resp = LLMResponse(text=response_text, success=True)` hard-codes
two kwargs of an external dataclass. If `LLMResponse` ever grows a required
field, this construction breaks at runtime. The harness should depend on the
narrowest possible LLMClient API.

**Fix:** Either introduce a stable helper in `llm_client.py` (e.g.,
`LLMClient.extract_json_from_text(s)`) and call that, or wrap in a try/except
that surfaces a clear "harness vs LLMClient drift" error.

---

### IN-05: MANIFEST.json file paths use trailing slash for pkl_dir but not calls_json

**File:** `tests/harness/fixtures/MANIFEST.json`

**Issue:** Cosmetic inconsistency — `pkl_dir` ends with `/` while `calls_json`
does not. `Path.resolve()` normalises both, so no correctness impact, but
trailing slashes in JSON config invite confusion when humans diff manifests.

**Fix:** Drop trailing slashes from `pkl_dir` for consistency.

---

_Reviewed: 2026-06-07T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
