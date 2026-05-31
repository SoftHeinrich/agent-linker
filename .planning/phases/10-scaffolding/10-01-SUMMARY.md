---
phase: 10-scaffolding
plan: 01
subsystem: testing-infra
tags: [gate-02, regression, fixture, canonical-variants]
requires:
  - run_ablation.CANONICAL_VARIANTS (45 entries)
  - results/ablation_results/*.json (Claude Sonnet sweeps)
provides:
  - tests/fixtures/v2_0_baseline.json (pinned 5-dataset per-variant P/R/F1)
  - tests/test_v20_baseline_regression.py (GATE-02 contract test)
affects:
  - Phase 11/12/13 promotion workflow (must run this test before any promotion)
tech-stack:
  added: []
  patterns: [pytest.parametrize, fixture-driven static regression]
key-files:
  created:
    - tests/fixtures/v2_0_baseline.json
    - tests/test_v20_baseline_regression.py
  modified: []
decisions:
  - "Claude Sonnet is the sole backend for the GATE-02 fixture. GPT-5.4 cross-model sweeps (ablation_20260531_063446.json, ablation_20260531_055235.json) intentionally excluded; they live under the GATE-01 cross-model contract instead."
  - "s_linker13 is the canonical promotion of s_linker13f (per v2.0 audit COMBINE retro-designation). Per-dataset values for s_linker13 are sourced from the s_linker13f sweep ablation_20260529_215932.json."
  - "Partial-coverage variants and zero-coverage variants are placed in fixture['missing']. The regression test xfails those slots so the gap is visible in CI without breaking the build."
  - "tolerance_abs_f1 = 1e-4 pinned in the fixture as the contract for Phase 13 PROMPT-03 live-run comparison."
metrics:
  duration: ~15min
  completed: 2026-05-31
---

# Phase 10 Plan 01: Baseline Regression Test (GATE-02) Summary

Pinned the v2.0-close Claude Sonnet F1 baseline for every `CANONICAL_VARIANTS` entry into `tests/fixtures/v2_0_baseline.json`, then shipped `tests/test_v20_baseline_regression.py` as the GATE-02 frozen-compat contract: any drift between the registry and the fixture is a loud, actionable failure before any v2.1 promotion can ship.

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Snapshot v2.0 baseline F1 fixture | `966df82` | tests/fixtures/v2_0_baseline.json |
| 2 | Write GATE-02 regression test | `98cdca2` | tests/test_v20_baseline_regression.py |

## Fixture: tests/fixtures/v2_0_baseline.json

- `schema_version`: 1.0
- `frozen_at`: v2.0-close 2026-05-31
- `tolerance_abs_f1`: 1.0e-4 (consumed by Phase 13 PROMPT-03)
- `datasets`: `["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]`
- **Coverage: 30 pinned + 15 missing = 45 / 45** (100% of CANONICAL_VARIANTS)

### s_linker13 anchor (audit baseline)

- Stored macro F1: **0.950585** (round-trip of per-dataset F1 values)
- Audit-claimed anchor: 0.9509 (within 5e-3 tolerance per plan acceptance — actual delta is 5e-4)
- Per-dataset: mediastore 0.9841 | teastore 1.0000 | teammates 0.9474 | bigbluebutton 0.8214 | jabref 1.0000
- Source: `ablation_20260529_215932.json` (the v2.0 audit's `s_linker13f` final-sweep file; `s_linker13` is the canonical promotion of `s_linker13f` per COMBINE-02 retro-designation, so the two share identical per-dataset values).

### Source ablation files (pinned variants)

Stitched per (variant, dataset) using the most recent Claude Sonnet sweep available on disk, excluding GPT-5.4 cross-model runs (May 31). Most variants come from a single sweep; a few pulled from 2 files due to incremental dev-loop runs:

| variant | source file(s) |
|---------|----------------|
| i1 | ablation_20260221_203819.json |
| i2 | ablation_20260221_234737.json |
| s_linker2 | ablation_20260314_164411.json, ablation_20260316_232048.json |
| s_linker3 | ablation_20260314_190233.json |
| s_linker4 | ablation_20260314_232944.json |
| s_linker5 | ablation_20260314_232944.json |
| s_linker6 | ablation_20260316_232048.json |
| s_linker7 | ablation_20260317_124610.json |
| s_linker7a | ablation_20260317_142244.json |
| s_linker7b | ablation_20260317_142948.json |
| s_linker8 | ablation_20260317_234308.json |
| s_linker9 | ablation_20260318_232308.json |
| s_linker9a | ablation_20260319_104128.json |
| s_linker9c | ablation_20260319_103911.json |
| s_linker9d | ablation_20260319_135405.json, … |
| s_linker9e | ablation_20260319_135345.json, … |
| s_linker10 | ablation_20260320_122306.json |
| s_linker10a | ablation_20260320_083055.json |
| s_linker11 | ablation_20260323_185246.json |
| s_linker11a | ablation_20260327_112242.json, … |
| s_linker12b | ablation_20260513_192513.json, … |
| s_linker12c | ablation_20260513_192513.json, … |
| s_linker12d | ablation_20260513_192513.json, … |
| s_linker12e | ablation_20260404_160434.json |
| s_linker13a | ablation_20260528_173020.json |
| s_linker13b | ablation_20260528_190916.json |
| s_linker13c | ablation_20260528_201851.json |
| s_linker13e | ablation_20260529_201324.json |
| s_linker13f | ablation_20260529_215932.json |
| s_linker13 | ablation_20260529_215932.json (mirrors s_linker13f) |

(Exact `source_files` array per variant is recorded inside the fixture JSON.)

### Variants forced into `missing` (no v2.0-close 5-dataset Claude Sonnet sweep on disk)

| variant | reason |
|---------|--------|
| `i3` | No `i3` measurement in any ablation file under `results/ablation_results/` |
| `s_linker` | Only `mediastore` appears in ancient dev-loop files (Mar 13) |
| `s_linker9b` | Never appears in any ablation file |
| `s_linker11b` | Only `mediastore` (single dev-loop file, Mar 30) |
| `s_linker11c` | Only `mediastore` (Mar 30) |
| `s_linker11d` | Only `mediastore` (Mar 30) |
| `s_linker11e` | Only `mediastore` (Mar 31) |
| `s_linker12a` | Only `mediastore` (Mar 31) |
| `s_linker13d` | Only `teammates` + `bigbluebutton` (D-21d parity-spike runs) |
| `s_linker13g_pre` | Only `teammates` + `bigbluebutton` (EXT-01 hard-tier dev loop, ablation_20260530_115900.json) |
| `s_linker13g_sem` | Only `teammates` + `bigbluebutton` (EXT-01 hard-tier, ablation_20260530_121014.json) |
| `s_linker13g_pre_alias` | Zero per-dataset numeric metrics on disk (only diff/Jaccard files; gate05 file lists TM+BBB but in a non-standard schema) |
| `s_linker13g_sem_alias` | Same as above |
| `s_linker13g_pre_full` | Same as above |
| `s_linker13g_sem_full` | Same as above |

These are XFAIL slots in the regression test, not hard failures. Future fixture refreshes (or dropping these variants from `CANONICAL_VARIANTS`) clear the slots.

## Test: tests/test_v20_baseline_regression.py

- 50 items collected (1 + 1 + 30 parametrize + 1 + 1 + 15 parametrize + 1) — runs in **0.08s** wall-clock.
- **35 passed, 15 xfailed** on `master` HEAD.
- Zero imports from `llm_sad_sam.linkers.*` (verified by `grep -c`).
- Module docstring contains the literal tokens `GATE-02`, `frozen-compat`, `CANONICAL_VARIANTS`, `v2.0 baseline JSON` (grep-discoverable).
- Smoke-checked: injecting a fake variant into `CANONICAL_VARIANTS` causes Test 1 to fail with a message naming the variant (verified out-of-tree; no test or code change committed for the smoke).

### Behaviors covered

1. CANONICAL_VARIANTS ↔ fixture subset/superset (registry drift = GATE-02 fail).
2. Pinned ∩ missing = ∅.
3. Per-pinned-variant: 5 datasets present, P/R/F1 floats in [0, 1], macro_f1 round-trips mean(F1) within 1e-6.
4. `s_linker13` macro F1 anchors to 0.9509 ± 5e-3 via `math.isclose`.
5. `tolerance_abs_f1` == 1e-4.
6. Missing-variant slots → XFAIL with diagnostic message.
7. Module docstring grep-discoverable.

## Verification Results

| Check | Status |
|-------|--------|
| `python -m pytest tests/test_v20_baseline_regression.py -v` exits 0 | PASS (35 passed, 15 xfailed) |
| `cat tests/fixtures/v2_0_baseline.json | python -m json.tool > /dev/null` exits 0 | PASS |
| CANONICAL_VARIANTS coverage = 100% (pinned + missing) | PASS (30 + 15 = 45 / 45) |
| `grep -c "GATE-02" tests/test_v20_baseline_regression.py` ≥ 1 | PASS (15) |
| `grep -E "Reencoding\|FreeSWITCH\|kurento\|Redis\|Recording" tests/fixtures/v2_0_baseline.json` empty | PASS |
| `grep -c "from llm_sad_sam.linkers" tests/test_v20_baseline_regression.py` == 0 | PASS |
| Full pytest runtime < 5s | PASS (0.08s) |
| `s_linker13` macro F1 within 5e-3 of 0.9509 | PASS (0.9506, delta 5e-4) |

## Deviations from Plan

None auto-fixed (no Rule 1/2/3 events). Two clarifying choices logged here:

1. **GPT-5.4 cross-model file exclusion (interpretation, not deviation).** The plan instructs locating "the most recent ablation_*.json" per variant. The May 31 sweeps (`ablation_20260531_063446.json`, `ablation_20260531_055235.json`) would be most-recent for `s_linker13` by date but reproduce GPT-5.4 cross-model numbers (macro F1 0.9077, matching the GATE-01 cross-model baseline in REQUIREMENTS.md exactly). Per the plan's explicit cross-check ("audit lists `ablation_20260529_215932.json` as the s_linker13 final sweep with macro F1 0.9509 — use grep on the file to confirm before pinning") I used `ablation_20260529_215932.json` for `s_linker13` and excluded both May 31 files from the stitching pool. This preserves the GATE-02 contract that the fixture pins the **Claude Sonnet** baseline (REQUIREMENTS.md GATE-02 wording + plan Action step 5 concrete value pin note).

2. **s_linker13 := s_linker13f data.** Per the v2.0 audit (`COMBINE-02` retro-designation: "s_linker13 retro-designated as the COMBINE artifact... source JSON ablation_20260529_215932.json"), `s_linker13` is the canonical promotion of `s_linker13f` with no code delta. The fixture pins the same per-dataset values for both names. This is explicitly documented in the fixture's `notes` field.

## Authentication Gates

None. Test is fully static (no LLM calls, no .env loading).

## Self-Check: PASSED

- File `tests/fixtures/v2_0_baseline.json`: FOUND
- File `tests/test_v20_baseline_regression.py`: FOUND
- Commit `966df82`: FOUND
- Commit `98cdca2`: FOUND
- Pytest run: 35 passed, 15 xfailed (acceptable per GATE-02 contract)
