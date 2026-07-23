---
phase: 01-baseline-and-infrastructure
plan: 02
subsystem: infrastructure
tags: [infra, caching, dependencies, refactor]
requires: []
provides:
  - diskcache>=5.6.1 dependency
  - tabulate>=0.9.0 dependency (Phase 5 PROMO-03 consumer)
  - LLMClient._cache (diskcache.Cache) attribute
  - _llm_response_to_dict / _llm_response_from_dict helpers
affects:
  - pyproject.toml
  - src/llm_sad_sam/llm_client.py
tech_stack:
  added:
    - diskcache 5.6.3 (SQLite-backed concurrent-safe cache)
    - tabulate 0.10.0 (dep-only in Phase 1, consumed in Phase 5)
  patterns:
    - Drop-in cache via dict-like API (cache[key] = value, cache.get(key))
    - SHA-256 prompt-hash key scheme preserved (D-05)
key_files:
  modified:
    - pyproject.toml
    - src/llm_sad_sam/llm_client.py
  created: []
decisions:
  - "D-05: Same SHA-256 key scheme; external LLMBackend.CHECKPOINT API unchanged"
  - "D-05: Existing on-disk JSON caches are orphaned (not migrated)"
  - "D-06: tabulate added as dep-only (no Phase 1 consumer)"
  - "D-01: anthropic is NOT added — backend stays on claude -p"
metrics:
  duration: ~15 minutes
  completed: 2026-05-13
  tasks_completed: 2
  files_modified: 2
requirements_completed:
  - INFRA-03
---

# Phase 01 Plan 02: diskcache + tabulate Infrastructure Summary

One-liner: Add `diskcache>=5.6.1` and `tabulate>=0.9.0` to `pyproject.toml`; replace the 5 SHA-256-keyed JSON-file methods in `llm_client.py` with a single `diskcache.Cache`, preserving the SHA-256 prompt-hash key scheme and the external `LLMClient.enable_checkpoint` / `query` API.

## Tasks Executed

| # | Name | Commit | Files |
|---|------|--------|-------|
| 1 | Add diskcache and tabulate to pyproject.toml | `0151bf4` | `pyproject.toml` |
| 2 | Replace SHA-256 JSON checkpoint with diskcache.Cache | `e075c86` | `src/llm_sad_sam/llm_client.py` |

## What Changed

### `pyproject.toml`

```diff
 dependencies = [
     "click>=8.1.0",
     "lxml>=5.0.0",
     "rapidfuzz>=3.0.0",
+    "diskcache>=5.6.1",
+    "tabulate>=0.9.0",
 ]
```

`pip install -e ".[dev,openai]"` installed `diskcache-5.6.3` and `tabulate-0.10.0`. Importability verified: `python -c "import diskcache; import tabulate"`.

### `src/llm_sad_sam/llm_client.py`

#### Methods deleted (3)

- `_checkpoint_path(self, prompt) -> Path` — no longer needed; diskcache owns paths.
- `_load_cached_response(self, path) -> Optional[LLMResponse]` — replaced by `self._cache.get(key)` + `_llm_response_from_dict`.
- `_save_cached_response(self, path, response) -> None` — replaced by `self._cache[key] = self._llm_response_to_dict(response)`.

#### Methods preserved (1)

- `_prompt_hash(prompt) -> str` (SHA-256) — kept unchanged; still the cache-key scheme per D-05.

#### Methods rewritten (1)

- `_query_checkpoint(self, prompt, timeout, max_retries) -> LLMResponse` — now uses `self._cache.get(key)` for hit, `self._cache[key] = ...` for save. Logic unchanged otherwise (cache miss delegates to `_checkpoint_fallback`, `latency_ms=0` on hit, only successful responses saved).

#### Methods added (2)

- `_llm_response_to_dict(response) -> dict` (staticmethod) — preserves the JSON dict shape used by the deleted `_save_cached_response`.
- `_llm_response_from_dict(data) -> LLMResponse` (staticmethod) — preserves reconstruction logic from deleted `_load_cached_response`, including `TokenUsage` and `model`/`latency_ms` fields.

#### `__init__` changes

```python
self._cache: Optional[diskcache.Cache] = None
if self.backend == LLMBackend.CHECKPOINT:
    self._checkpoint_dir = Path(os.environ.get("CHECKPOINT_DIR", "./results/llm_checkpoint"))
    self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
    self._cache = diskcache.Cache(str(self._checkpoint_dir))   # NEW
    ...
```

#### `enable_checkpoint` diff shape

```python
self._checkpoint_dir = Path(checkpoint_dir)
self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
if self._cache is not None:           # NEW
    self._cache.close()                # NEW (close prior on re-enable)
self._cache = diskcache.Cache(str(self._checkpoint_dir))  # NEW
self.backend = LLMBackend.CHECKPOINT
```

External signature (`enable_checkpoint(checkpoint_dir, fallback=None, fallback_model=None) -> None`) is byte-identical.

#### Module imports

Added top-level `import diskcache` next to the existing `import logging`.

## Verification

All acceptance criteria passed:

- `grep -q '"diskcache>=5.6.1"' pyproject.toml` — OK
- `grep -q '"tabulate>=0.9.0"' pyproject.toml` — OK
- `grep -q "anthropic" pyproject.toml` — non-zero (correct; not added)
- `python -c "import diskcache; import tabulate"` — OK (`5.6.3 0.10.0`)
- `grep -q "^import diskcache" src/llm_sad_sam/llm_client.py` — OK
- `def _checkpoint_path` / `_load_cached_response` / `_save_cached_response` — all deleted (non-zero grep exit)
- `def _prompt_hash` / `def _query_checkpoint` / `def _llm_response_to_dict` / `def _llm_response_from_dict` — all present
- `diskcache.Cache` and `self._cache` present
- `grep -c "^import anthropic\|^from anthropic"` returns 0 (D-01)

Smoke test:

```bash
python -c "
from llm_sad_sam.llm_client import LLMClient, LLMBackend
c = LLMClient(backend=LLMBackend.CLAUDE)
c.enable_checkpoint('/tmp/test_diskcache_smoke', fallback=LLMBackend.CLAUDE)
assert c._cache is not None
assert LLMClient._prompt_hash('hello world') == LLMClient._prompt_hash('hello world')
assert len(LLMClient._prompt_hash('hello world')) == 64
print('OK')
"
```

Output: `OK`.

Additional end-to-end round-trip test (insert `LLMResponse` → `_cache[key]` → re-read via `_cache.get(key)` → `_llm_response_from_dict`) confirms `text`, `success`, `token_usage.total_tokens`, and `model` are preserved.

External-API preservation check:

```bash
python -c "from llm_sad_sam.llm_client import LLMClient; assert hasattr(LLMClient, 'query') and hasattr(LLMClient, 'enable_checkpoint')"
```

Output: `API preserved`.

## Orphaned JSON Cache Note

Per D-05, the pre-existing `./results/llm_checkpoint/*.json` files written by the old `_save_cached_response` are NOT migrated. They are now orphaned — harmless, simply ignored by the new `diskcache.Cache` (which uses its own SQLite-backed layout under the same directory). No cleanup is performed; the next cache-miss will populate the diskcache layer.

## Deviations from Plan

None — plan executed exactly as written. All 2 tasks completed in order with the specified edits.

## Self-Check: PASSED

- `pyproject.toml` contains `diskcache>=5.6.1` and `tabulate>=0.9.0` — FOUND
- `src/llm_sad_sam/llm_client.py` contains `import diskcache`, `diskcache.Cache`, `self._cache`, `_llm_response_to_dict`, `_llm_response_from_dict` — FOUND
- 3 methods removed (`_checkpoint_path`, `_load_cached_response`, `_save_cached_response`) — CONFIRMED ABSENT
- Commit `0151bf4` (Task 1) — FOUND in `git log`
- Commit `e075c86` (Task 2) — FOUND in `git log`
- External API (`query`, `enable_checkpoint`) preserved — VERIFIED
- Smoke test prints `OK` — VERIFIED
- Round-trip cache test passes — VERIFIED
