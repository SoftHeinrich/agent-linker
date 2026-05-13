---
phase: 01-baseline-and-infrastructure
plan: 01
subsystem: planning-docs
tags: [doc-strike, requirements, roadmap, scope-change]
dependency_graph:
  requires: []
  provides:
    - "REQUIREMENTS.md: INFRA-02/04 visibly struck with D-01 reference"
    - "ROADMAP.md: Phase 1 entry reflects active 4-requirement scope"
  affects:
    - .planning/REQUIREMENTS.md
    - .planning/ROADMAP.md
tech_stack:
  added: []
  patterns:
    - "Markdown strikethrough (~~...~~) for paper-trail retention of dropped requirements"
    - "Dated audit-trail footer entry for scope changes"
key_files:
  created:
    - .planning/phases/01-baseline-and-infrastructure/01-01-SUMMARY.md
  modified:
    - .planning/REQUIREMENTS.md
    - .planning/ROADMAP.md
decisions:
  - "Strike-through (not delete) keeps the paper trail visible: reviewers can see what was originally scoped and why it was dropped (D-01)"
  - "Traceability table rows for INFRA-02/04 kept with status STRUCK (D-01) rather than removed — maintains row count for downstream tooling"
metrics:
  duration: ~10 min
  completed: 2026-05-13
requirements_completed:
  - INFRA-01-doc-prereq (D-01a — documentation alignment)
---

# Phase 1 Plan 01: Doc Strike (D-01a) — Mark INFRA-02/04 STRUCK Summary

Updated `REQUIREMENTS.md` and `ROADMAP.md` to reflect locked user decision D-01: INFRA-02 (SDK migration) and INFRA-04 (post-SDK F1 unchanged) are dropped from scope. Backend stays on `claude -p` subprocess. This unblocks the downstream Phase 1 plans (01-02 through 01-05) by aligning the active requirements surface with the locked decision.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Strike INFRA-02 and INFRA-04 in REQUIREMENTS.md | 13af918 | .planning/REQUIREMENTS.md |
| 2 | Update ROADMAP.md §Phase 1: drop INFRA-02/04, remove void success criterion | 0473cc3 | .planning/ROADMAP.md |

## Changes — REQUIREMENTS.md

### Baseline & Infrastructure section

**Before** (lines 13, 15):
```
- [ ] **INFRA-02**: `anthropic>=0.40.0` SDK added to `pyproject.toml`; `llm_client.py` migrated from `claude -p` subprocess to direct SDK call with `temperature=0.0` and prompt caching
- [ ] **INFRA-04**: Baseline F1 unchanged (within run-to-run variance) after SDK migration, confirmed on hard tier first, then full 5-project sweep
```

**After**:
```
- [ ] ~~**INFRA-02**: `anthropic>=0.40.0` SDK added to `pyproject.toml`; `llm_client.py` migrated from `claude -p` subprocess to direct SDK call with `temperature=0.0` and prompt caching~~ — **STRUCK per D-01** (Phase 1 CONTEXT, 2026-04-24): backend stays on `claude -p` subprocess; Claude CLI exposes no temperature flag and no caller-controlled cache headers, so this requirement cannot be met via CLI. Not deferred — dropped.
- [ ] ~~**INFRA-04**: Baseline F1 unchanged (within run-to-run variance) after SDK migration, confirmed on hard tier first, then full 5-project sweep~~ — **STRUCK per D-01**: depends on INFRA-02 which is struck.
```

### Traceability table (rows 76, 78)

**Before**:
```
| INFRA-02 | Phase 1 | Pending |
| INFRA-04 | Phase 1 | Pending |
```

**After**:
```
| INFRA-02 | Phase 1 | STRUCK (D-01) |
| INFRA-04 | Phase 1 | STRUCK (D-01) |
```

### Coverage footer

**Before**:
```
- v1 requirements: 21 total (5 INFRA + 6 VAR + 4 PROMO + 6 GATE)
- Mapped to phases: 21
```

**After**:
```
- v1 requirements: 21 total (5 INFRA — 2 struck per D-01 + 6 VAR + 4 PROMO + 6 GATE) → 19 active
- Mapped to phases: 19 active (+ 2 struck)
```

### Appended audit-trail entry (above trailing horizontal rule)
```
*2026-04-25: INFRA-02 and INFRA-04 struck per Phase 1 CONTEXT decision D-01 — SDK migration removed from scope; backend stays on `claude -p` subprocess.*
```

## Changes — ROADMAP.md §Phase 1

### Goal line (line 35)

**Before**:
> A reproducible `s_linker12c` baseline is captured, **the LLM SDK is migrated to direct API calls**, and `s_linker13a` (Spike 001 trailing-word LLM enrichment) passes the dual floor — giving a clean starting point for the entire ablation chain.

**After**:
> A reproducible `s_linker12c` baseline is captured, **per-variant checkpoint namespacing is in place**, and `s_linker13a` (Spike 001 trailing-word LLM enrichment) passes the dual floor — giving a clean starting point for the entire ablation chain.

### Requirements line

**Before**:
```
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04, INFRA-05, VAR-01
```

**After**:
```
**Requirements**: INFRA-01, INFRA-03, INFRA-05, VAR-01 (INFRA-02 and INFRA-04 struck per D-01 — see Phase 1 CONTEXT)
```

### Success Criteria (renumbered 5 → 4 items)

**Before**:
```
  1. `results/ablation_results/` contains a 12c baseline run with per-dataset F1, FP/FN counts, and a JSON result file
  2. `run_ablation.py` invokes the Anthropic SDK directly (no `claude -p` subprocess); `temperature=0.0` confirmed in logs; prompt-caching headers present
  3. `diskcache>=5.6.1` and `tabulate>=0.9.0` are in `pyproject.toml`; `anthropic>=0.40.0` is in `pyproject.toml`
  4. `s_linker13a` registered in `run_ablation.py`; hard-tier run (teammates + BBB) completes with no regression >1pp vs 12c; full 5-project sweep confirms macro F1 ≥ 93% and no dataset >2pp below 12c baseline
  5. Each variant's `_checkpoint_dir` uses its own `_VARIANT_NAME` constant (no hardcoded `"s_linker12c"` string in 13a)
```

**After**:
```
  1. `results/ablation_results/` contains a 12c baseline run with per-dataset F1, FP/FN counts, and a JSON result file
  2. `diskcache>=5.6.1` and `tabulate>=0.9.0` are in `pyproject.toml`
  3. `s_linker13a` registered in `run_ablation.py`; hard-tier run (teammates + BBB) completes with no regression >1pp vs 12c; full 5-project sweep confirms macro F1 ≥ 93% and no dataset >2pp below 12c baseline
  4. Each variant's `_checkpoint_dir` uses its own `_VARIANT_NAME` constant (no hardcoded `"s_linker12c"` string in 13a)
```

Criterion #2 (SDK migration) removed. The old criterion #3 was split: the `anthropic>=0.40.0` clause was removed, leaving the diskcache/tabulate clause as the new criterion #2. Old #4 and #5 became new #3 and #4.

## Verification Results

REQUIREMENTS.md:
- `grep -c "STRUCK per D-01" .planning/REQUIREMENTS.md` → 2 (PASS, ≥ 2 required)
- `grep -q "~~\*\*INFRA-02\*\*"` → exit 0 (PASS)
- `grep -q "~~\*\*INFRA-04\*\*"` → exit 0 (PASS)
- `grep -c "STRUCK (D-01)"` → 2 (PASS, one per traceability row)
- `grep -q "INFRA-01|INFRA-03|INFRA-05|VAR-01"` all present (PASS)
- `grep -q "19 active"` → exit 0 (PASS)

ROADMAP.md:
- `grep -q "INFRA-01, INFRA-03, INFRA-05, VAR-01"` → exit 0 (PASS)
- `grep -q "INFRA-02, INFRA-03, INFRA-04"` → exit non-zero (PASS — old comma-separated string removed)
- `grep -q "Anthropic SDK directly"` → exit non-zero (PASS — criterion #2 removed)
- `grep -q "anthropic>=0.40.0"` → exit non-zero (PASS — clause removed)
- `grep -q "diskcache>=5.6.1\` and \`tabulate>=0.9.0\` are in \`pyproject.toml\`"` → exit 0 (PASS)
- `grep -q "struck per D-01"` → exit 0 (PASS)
- Numbered list shows exactly four entries `1.`, `2.`, `3.`, `4.` (PASS)

Note: ROADMAP.md still contains the string `INFRA-02` in two places — both intentional and required by the plan: (a) the parenthetical "(INFRA-02 and INFRA-04 struck per D-01 — see Phase 1 CONTEXT)" on the Requirements line; (b) the existing plan listing "01-01-PLAN.md — Doc strike (D-01a): mark INFRA-02/04 STRUCK ...". The plan's specific automated check looks for the old comma-separated sequence, which is removed.

## Deviations from Plan

None - plan executed exactly as written.

## Decisions Made

- Strike-through retained (per plan) rather than deletion — preserves paper trail for reviewers
- Plan-listing line at line 44 of ROADMAP.md kept as-is — it self-references this plan and would be self-defeating to alter

## Self-Check: PASSED

Verifications performed:
- File `.planning/REQUIREMENTS.md` exists: FOUND
- File `.planning/ROADMAP.md` exists: FOUND
- Commit `13af918` exists: FOUND in `git log` (Task 1)
- Commit `0473cc3` exists: FOUND in `git log` (Task 2)
- All acceptance criteria for both tasks: PASS (see Verification Results above)
