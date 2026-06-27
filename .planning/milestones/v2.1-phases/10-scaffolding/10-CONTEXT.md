# Phase 10: Scaffolding - Context

**Gathered:** 2026-05-31
**Status:** Ready for planning
**Mode:** Infrastructure phase — smart discuss skipped (scaffolding + file/registration/test/gate criteria, no user-facing behavior)

<domain>
## Phase Boundary

Stand up the v2.1 cleanup foundation: ship `s_linker13_clean.py` as a standalone canonical variant, extract versioned helper modules (`helper_v3.py` family) without touching the v2.0-frozen siblings, codify GATE-01 cross-model (gpt-5.4 ≥ 0.9077 within ≤ 1pp), and add a regression test that pins every `CANONICAL_VARIANTS` entry to the v2.0 baseline JSON. End state: subsequent trim/promotion phases (11–13) can iterate without ever risking a frozen variant.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices at Claude's discretion — pure infrastructure phase. Constraints already pinned in REQUIREMENTS.md and ROADMAP:

- `s_linker13.py`, `prompts_v2.py`, `data_types_v2.py`, `document_loader_v2.py`, `pcm_parser_v2.py`, `ilinker*` — FROZEN. Do not edit.
- New variant lands at `src/llm_sad_sam/linkers/experimental/s_linker13_clean.py` (standalone — duplicate code is fine per user pref).
- Helper modules grouped by concern under versioned siblings (`helper_v3.py` or `helper_v3_<concern>.py` if grouping benefits clarity). v2.0 helpers untouched.
- Cross-model tolerance T = 1pp (matches REQUIREMENTS `≤ 1pp`). Logged in PROJECT.md Key Decisions and STATE.md Standing Gates.
- Regression test asserts identical-or-equivalent F1 (within harness float tolerance) for every entry in `CANONICAL_VARIANTS` against v2.0 baseline JSON. Test entry point lives next to existing test infra (`tests/test_v20_baseline_regression.py` or equivalent) and is wired into the ablation harness.
- Parity criterion for CLEAN-01: `s_linker13_clean` produces F1 identical to `s_linker13` on all 5 datasets via Claude Sonnet.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `s_linker13.py` — frozen, importable. Copy/refactor into `s_linker13_clean.py`.
- `prompts_v2.py` — frozen. Cleaned prompts later land in `prompts_v3.py` (Phase 12).
- Helpers currently live in: `core/data_types_v2.py`, `core/document_loader_v2.py`, `pcm_parser_v2.py`, plus inlined helpers inside `s_linker13.py`. New cleanup helpers land in versioned siblings (`helper_v3.py` family).
- `run_ablation.py` already maintains `CANONICAL_VARIANTS` list + `VARIANT_SPECS` dict. New variant registers here.
- `tests/test_s_linker13d_parity.py` — existing parity test pattern usable as a template for v2.0 baseline regression.
- `results/ablation_results/*.json` — existing per-variant baseline JSONs; v2.0 close should have produced canonical baseline (locate exact path in plan-phase).

### Established Patterns
- One standalone `.py` per variant (no inheritance chains — explicit user preference).
- Variants registered in `CANONICAL_VARIANTS` (list) + `VARIANT_SPECS` (dict with `module=` import path).
- Structured docstring + GATE-07 canonical registration per promoted variant.
- Tests under `tests/` use `test_*.py` pattern; existing parity test is the closest analog.

### Integration Points
- `run_ablation.py` — variant registration, ablation entry.
- `PROJECT.md` Key Decisions table — gate definition.
- `.planning/STATE.md` Standing Gates — gate logging.
- `tests/` — regression test home.

</code_context>

<specifics>
## Specific Ideas

- The v2.0 baseline JSON used for the regression test must be the canonical `CANONICAL_VARIANTS` snapshot from v2.0 close. If not already pinned as a fixture, snapshot it now under a stable path (e.g. `tests/fixtures/v2_0_baseline.json`).
- Helper concern groupings to evaluate during planning (final shape decided at plan-phase): (a) document knowledge / enrichment, (b) coreference / alias mentions, (c) ambiguity classification, (d) misc utilities. If a single `helper_v3.py` is enough, one file is fine.
- Tolerance value committed as exactly `1.0pp` (not "≤ 1pp" — needs a concrete number for the gate check).
- Frozen-compat regression test must run before any future variant promotion (PROMPT-03 in Phase 13).

</specifics>

<deferred>
## Deferred Ideas

None — phase scope is well-bounded by REQUIREMENTS CLEAN-01/CLEAN-02/GATE-01/GATE-02.

</deferred>
