---
phase: 46
review_depth: standard
files_reviewed: 11
findings:
  critical: 0
  warning: 2
  info: 4
  total: 6
status: findings
---

# Phase 46 (MINIMIZE) — Code Review

**Reviewed:** 2026-06-08
**Depth:** standard
**Reviewer:** Claude (gsd-code-reviewer)

## Summary

Reviewed 11 test/harness files plus the two scratch source copies. GATE-01 holds: `src/llm_sad_sam/linkers/experimental/{s_linker19.py,prompts_v5.py}` last touched at commit `a6f9c64` (2026-06-08 15:39), before the Phase 46 work window. The scratch package is properly initialized (empty `__init__.py`, Python treats it as a regular package), and `tests/scratch/s_linker19.py` correctly rewires its prompts import to `tests.scratch.prompts_v5` — meaning cuts to scratch prompts are exercised only when `SAD_SAM_LINKER_SOURCE=scratch`.

The step-6 gate `if os.environ.get("SAD_SAM_LINKER_SOURCE", "production") != "scratch":` is uniformly applied across all 6 prompt test modules, all 6 import `os`, and all 6 wrap the byte-equality `assert` (not the snapshot assertion) — so the parsed-output snapshot remains the active gate in scratch mode. The `ACCEPTED_PREFIXES` extension is backwards-compatible: the original opener stays at index 0 and the new one ("Validate components in a document.") is not a string-prefix of the production opener, so there is no shadowing.

Findings below are quality issues only — no correctness, security, or data-loss defects.

## Warnings

### WR-01: `doc_extract` step-6 gate has a tautological branch (effectively dead assertion)

**File:** `tests/test_s_linker20_prompt_doc_extract.py:62-79`
**Issue:** The branch structure is:
```python
prompt_equal = rebuilt_prompt == record["prompt"]
if not prompt_equal:
    warnings.warn(...)
else:
    assert rebuilt_prompt == record["prompt"], (...)   # always true
```
The `else` branch only executes when `prompt_equal is True`, so the `assert` inside it is a tautology and can never fail. The intent (per the comment "We assert byte-equality for projects whose fixtures match the current code") is unenforceable as written — drift cases silently warn, matching cases assert something already proven. The hard-assertion path is effectively dead.

This existed before Phase 46 in spirit (the drift-soft-warn was the v44 design) but the Phase 46 wrapping by `if ... != "scratch":` cements the dead branch without correcting it.

**Fix:** Either drop the dead `else` (the assertion is redundant) or, if a strict-mode option is wanted, gate it on an env var:
```python
if not prompt_equal:
    if os.environ.get("STRICT_PROMPT_REBUILD"):
        raise AssertionError(f"Prompt rebuild mismatch ...")
    warnings.warn(f"[prompt-version-drift] ...")
# (no else needed — pass-through when equal)
```

### WR-02: Unused import `COREF_VALIDATION_FOCUS` in `tests/harness/inputs.py`

**File:** `tests/harness/inputs.py:35`
**Issue:** `from llm_sad_sam.linkers.experimental.prompts_v5 import COREF_VALIDATION_FOCUS` is imported but never referenced in code — only mentioned in a docstring (line 245). The reverse-extractor on line 293 derives `focus` from the prompt text directly. The import is dead code.

This is mildly load-bearing for Phase 46 reasoning: the import targets *production* `prompts_v5`, not the scratch copy. In scratch mode, if the import had been used to reconstruct `focus`, it would have produced a wrong value (mismatched against the scratch builder's expectation). Removing the unused import eliminates this latent cross-source coupling.

**Fix:** Delete line 35. Update line 245 docstring to drop the `COREF_VALIDATION_FOCUS` cross-reference or note that it is reconstructed from prompt text only.

## Info

### IN-01: `adapters.py` raises `RuntimeError` at import time on bad env value

**File:** `tests/harness/adapters.py:52-54`
**Issue:** `RuntimeError` is raised at module import. When pytest collects test modules that transitively import `tests.harness.adapters`, an unset-or-typo env var (e.g., `SAD_SAM_LINKER_SOURCE=prod`) crashes test collection rather than failing individual tests with a clear skip/error. The error message is good (it shows the bad value), but `pytest` will report this as an internal collection error.

Not a bug — failing loudly at the toggle boundary is defensible — but it means a single typo aborts the entire prompt-test suite. Consider using `ValueError` (more specific) or catching at the conftest level for friendlier reporting.

**Fix (optional):** Move the env-var validation into a `pytest_configure` hook in `conftest.py`, or change to `ValueError` for category precision.

### IN-02: `ACCEPTED_PREFIXES` is a function-local tuple — re-allocated per call

**File:** `tests/harness/inputs.py:279-282`
**Issue:** The tuple is re-defined inside `reconstruct_validation_inputs` on every call. Validation runs many times per project; while the per-call cost is negligible (two short strings, a tuple), the tuple is a constant and should be module-level.

Module-level placement also makes it discoverable for unit tests / future expansion (Phase 47 may add more accepted openers).

**Fix:** Hoist to module scope alongside the dataclass:
```python
_ACCEPTED_VALIDATION_PREFIXES = (
    "Validate component references in a software architecture document.",
    "Validate components in a document.",
)
```

### IN-03: Scratch source-toggle has no test coverage

**File:** `tests/harness/adapters.py:45-54`
**Issue:** The `SAD_SAM_LINKER_SOURCE` toggle has three behaviors (`production`, `scratch`, `else → raise`) and no unit test exercises them. A future refactor that breaks the toggle (e.g., typo in the scratch import path) would only be caught by running the harness with the env var set, which is not part of CI by default.

**Fix (optional):** Add a tiny test that imports adapters under each value via `importlib.reload` + `monkeypatch.setenv`. This makes the toggle a first-class contract.

### IN-04: Empty `tests/scratch/__init__.py` lacks a marker docstring

**File:** `tests/scratch/__init__.py`
**Issue:** The file is 0 bytes. Functionally fine (Python treats it as a regular package), but a one-line docstring would document intent for readers stumbling on `tests/scratch/`:

```python
"""Phase 46 mutable scratch package — production-byte-equal copies with trial cuts applied.

Importable only when SAD_SAM_LINKER_SOURCE=scratch (see tests/harness/adapters.py).
"""
```

This is purely a discoverability nit.

## Verified-Clean Checks (no findings)

- **GATE-01 (byte-equal production):** Last commit touching `src/llm_sad_sam/linkers/experimental/{s_linker19.py,prompts_v5.py}` is `a6f9c64` at 15:39:12, before Phase 46 work began at 16:15+. Working tree clean. No leakage.
- **Step-6 gate uniformity:** All 6 `test_s_linker20_prompt_*.py` modules import `os`, all 6 wrap the byte-equality `assert` with `if os.environ.get("SAD_SAM_LINKER_SOURCE", "production") != "scratch":`, all 6 leave the snapshot assertion (`assert parsed == snapshot`) unguarded. Consistent.
- **Default behavior preserved:** `os.environ.get(..., "production")` and the `_SOURCE = os.environ.get(..., "production")` default in adapters.py both mean an unset env var → production import → original CI behavior unchanged.
- **`ACCEPTED_PREFIXES` backwards compatibility:** Order is `(production, scratch)`. Neither string is a prefix of the other (diverge at "component**s**" vs "component **references**"), so iteration order does not affect parsing. Production records continue to match index 0; scratch records match index 1.
- **Scratch import wiring:** `tests/scratch/s_linker19.py:107` imports from `tests.scratch.prompts_v5` (not production). Confirmed via diff against the production source — this is the only structural delta beyond the kept cuts.
- **No production-path leakage in scratch:** Scratch `prompts_v5.py` and `s_linker19.py` carry the kept cuts (AMB few-shot drop, DKJ few-shot drop, VAL/COR rewordings). No accidental edits to production paths.
- **D-03 mapping intact:** `BUILDER_PHASE_TAGS` in `adapters.py` still maps `_prompt_validation` to all three phase tags including `phase_5_coref_validation` — the gotcha documented in `test_s_linker20_prompt_validation.py` is preserved.

---

_Reviewed: 2026-06-08_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
