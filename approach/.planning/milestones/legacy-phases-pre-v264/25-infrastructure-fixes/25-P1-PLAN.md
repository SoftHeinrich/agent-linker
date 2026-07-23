# Plan: Phase 25-P1 — Infrastructure Fixes

**Phase**: 25 — Infrastructure Fixes
**Status**: complete
**Completed**: 2026-06-02

## Changes

### REQ-V25-01: Oracle cache key includes bank_content_hash

**File**: `scripts/voyager_train_tlr_v4_beta.py` line 455

**Before**:
```python
ck = _cache_key(text_path, project, backend_str, model_str, f"oracle_iter{iter_num}")
```

**After**:
```python
bch = _bank_content_hash(bank)
ck = f"{Path(text_path).stem}_{_comp_hash(project)}_{bch}_{backend_str}_{model_str}_oracle_iter{iter_num}"
```

Matches the pattern already used by the L cache (lines 281-282). Prevents split-3 TM from reusing mainline oracle outputs when bank state differs.

### REQ-V25-02: Probation min-commit threshold raised to delta >= 0.005

**File**: `scripts/voyager_train_tlr_v4_beta.py`

Added `MIN_COMMIT_DELTA = 0.005` constant (after line 88).

In `run_outer_pass()`, after L run delta computation, added early-exit block: if `not dry_run and delta < MIN_COMMIT_DELTA`, skip O+D+Gate steps and return no-op summary. No extra LLM calls. Summary includes `"below_min_commit_delta": True`.

### REQ-V25-03: D prompt underfilled-slot steering

**File**: `scripts/voyager_train_tlr_v4_beta.py`

Added `{underfilled_slots}` field to `D_PROMPT` after bank_summary section:
```
HIGH-PRIORITY SLOTS (zero patterns — propose for these first before adding to populated slots):
{underfilled_slots}
```

In `_run_distillator_d()`, computed:
```python
empty_slots = [s for s in SLOT_NAMES if not bank.get("slot_patterns", {}).get(s)]
underfilled_slots = ", ".join(empty_slots) if empty_slots else "(all slots have ≥1 pattern)"
```

GATE-06: no benchmark vocabulary in added text.

## Verification

- `MIN_COMMIT_DELTA = 0.005` confirmed in module
- D_PROMPT contains `{underfilled_slots}` placeholder
- Oracle key includes `bch = _bank_content_hash(bank)`
- 116/117 tests pass (pre-existing GATE-02 drift failure unrelated to Phase 25)
- Frozen artifacts unmodified: `s_linker13.py`, `prompts_v2.py`, `ilinker*.py`, `data_types_v2.py`, `document_loader_v2.py`, `pcm_parser_v2.py`, `s_linker13_min.py`
