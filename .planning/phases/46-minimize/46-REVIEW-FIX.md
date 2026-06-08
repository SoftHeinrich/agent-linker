---
phase: 46
fixed_at: 2026-06-08
review_path: .planning/phases/46-minimize/46-REVIEW.md
iteration: 1
findings_in_scope: 6
fixed: 3
skipped: 3
status: partial
---

# Phase 46: Code Review Fix Report

**Fixed at:** 2026-06-08
**Source review:** `.planning/phases/46-minimize/46-REVIEW.md`
**Iteration:** 1
**Mode:** `--auto` (Warning scope by default; Info findings included only when trivial)

**Summary:**
- Findings in scope: 6 (2 Warning + 4 Info)
- Fixed: 3 (both Warnings + IN-04)
- Skipped: 3 (IN-01, IN-02, IN-03 — out-of-scope per "<30s each" rule)

**GATE-01:** verified clean. `git diff master..HEAD -- src/llm_sad_sam/linkers/experimental/` produces no output across all three fix commits.

## Fixed Issues

### WR-01: doc_extract step-6 gate has a tautological branch

**Files modified:** `tests/test_s_linker20_prompt_doc_extract.py`
**Commit:** `313bc06`
**Applied fix:** Dropped the dead `else` branch entirely. The follow-on `assert rebuilt_prompt == record["prompt"]` could only execute when `prompt_equal is True`, making it a tautology. The drift-soft-warn remains the only active signal for prompt-version mismatches; the parsed-output snapshot below is the hard gate. Replaced the dead branch with a brief comment explaining why no `else` is needed. Chose the "drop dead branch" option (not the env-var strict-mode variant) per "simpler fix that preserves intent."

### WR-02: dead `COREF_VALIDATION_FOCUS` import in inputs.py

**Files modified:** `tests/harness/inputs.py`
**Commit:** `c15f0d0`
**Applied fix:** Removed the unused `from llm_sad_sam.linkers.experimental.prompts_v5 import COREF_VALIDATION_FOCUS` import at line 35. The reverse-extractor derives `focus` from prompt text directly (line 293), so the import was load-bearing for nothing. Eliminates a latent cross-source coupling because the import targeted *production* prompts_v5 — under `SAD_SAM_LINKER_SOURCE=scratch` it would have been a stale reference. The docstring at line 245 (`focus = COREF_VALIDATION_FOCUS` in the per-phase table) was left intact because the surrounding paragraph already explains "we reconstruct it by reading it directly from the prompt rather than importing the constant" — the cross-reference is descriptive, not load-bearing.

### IN-04: empty `tests/scratch/__init__.py` lacks marker docstring

**Files modified:** `tests/scratch/__init__.py`
**Commit:** `5e06237`
**Applied fix:** Added the two-sentence docstring proposed verbatim in the review ("Phase 46 mutable scratch package..."). Pure discoverability improvement.

## Skipped Issues

### IN-01: `adapters.py` raises `RuntimeError` at import time on bad env value

**File:** `tests/harness/adapters.py:52-54`
**Reason:** Out of scope. The reviewer marked it optional. Restructuring to a `pytest_configure` hook or to `ValueError` is a >30s change touching conftest semantics, and the current behavior ("fail loudly at the toggle boundary") is explicitly defensible per the review. Defer to a future cleanup pass.
**Original issue:** RuntimeError at module import crashes pytest collection rather than producing a per-test skip/error on bad `SAD_SAM_LINKER_SOURCE` value.

### IN-02: `ACCEPTED_PREFIXES` is a function-local tuple — re-allocated per call

**File:** `tests/harness/inputs.py:279-282`
**Reason:** Out of scope per "<30s" rule. Strictly the hoist itself would take ~20s, but the rename `ACCEPTED_PREFIXES → _ACCEPTED_VALIDATION_PREFIXES` plus updating all references (including the error message at line 291) and re-checking the per-call usage site adds risk surface for a sub-microsecond perf gain. Deferred; can be cherry-picked into a follow-up if Phase 47 expands the prefix set.
**Original issue:** Constant tuple re-allocated on every call to `reconstruct_validation_inputs`; should live at module scope.

### IN-03: scratch source-toggle has no test coverage

**File:** `tests/harness/adapters.py:45-54`
**Reason:** Out of scope. A new test file using `importlib.reload` + `monkeypatch.setenv` to exercise all three toggle branches takes well >30s to design, write, and verify. Reviewer marked optional.
**Original issue:** The `SAD_SAM_LINKER_SOURCE` env-var toggle has three behaviors (`production`, `scratch`, else→raise) with no unit-test coverage.

## Test Evidence

Both production-mode and scratch-mode runs of `pytest tests/test_s_linker20_prompt_*.py -x --tb=short` collect 45 items and report `45 skipped in <1s` with **0 failures and 0 errors**. All 45 cases skip via `fixture_missing_reason()` because the per-project `_calls.json` and `*.pkl` fixtures are not checked into this worktree (regenerated locally via `s_linker19 --backend openai`). The clean collection and parametrization confirm that none of the three edits broke test loading or import-time wiring.

```
production mode:
  collected 45 items
  ssssssssssssssssssssssssssssssssssssssssssssss  [100%]
  45 skipped in 0.34s

scratch mode (SAD_SAM_LINKER_SOURCE=scratch):
  collected 45 items
  ssssssssssssssssssssssssssssssssssssssssssssss  [100%]
  45 skipped in 0.18s
```

Additionally verified `tests/harness/inputs.py` imports cleanly after the dead import was removed (`from tests.harness import inputs` succeeds; `hasattr(inputs, 'COREF_VALIDATION_FOCUS')` is `False`, confirming removal).

## GATE-01 Verification

`git diff master..HEAD --stat` across all three fix commits:

```
 tests/harness/inputs.py                     |  1 -
 tests/scratch/__init__.py                   |  4 ++++
 tests/test_s_linker20_prompt_doc_extract.py | 11 ++++-------
 3 files changed, 8 insertions(+), 8 deletions(-)
```

Restricted to `src/llm_sad_sam/linkers/experimental/`: **empty diff**. GATE-01 holds.

---

_Fixed: 2026-06-08_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
