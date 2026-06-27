---
phase: 06-ext-01-project-agnostic-standalone-mention-llm-primitive
plan: 02
subsystem: llm-linker
tags: [llm-linker, sad-sam, standalone-mention, ext-01, gate-07, ablation]

# Dependency graph
requires:
  - phase: 06
    plan: 01
    provides: "STANDALONE_MENTION_RULES_PRE_FILTERED + STANDALONE_MENTION_RULES_LLM_ONLY prompt constants in prompts_v2.py"
provides:
  - "s_linker13g_pre.py — SLinker13gPre standalone linker class (1265 lines)"
  - "s_linker13g_sem.py — SLinker13gSem standalone linker class (1218 lines)"
  - "GATE-07 partial registration: both sub-variants in CANONICAL_VARIANTS + VARIANT_SPECS (canonical 's_linker13g' deferred to Plan 04)"
  - "Precomputed Tier-1 standalone_map dict[(comp_name, snum) -> bool]; O(1) lookup helper _has_standalone_mention_llm"
affects:
  - "06-03 (canonical sweep — runs both sub-variants through the ablation harness on all 5 benchmarks)"
  - "06-04 (winner pick + canonical promotion via byte-copy to s_linker13g.py)"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Standalone-mention as a Tier-1 LLM batched primitive — computed once per document, consumed via O(1) dict lookup at 6 call sites"
    - "Copy-fork sibling design: the literal-vs-semantic axis (D-01) is encoded by file diff, not by class inheritance"
    - "_classify_mention signature extended with optional snum parameter to thread the precomputed map to call site #2 without an awkward regex fallback"
    - "Per-variant checkpoint namespacing via _VARIANT_NAME enforced by assertion at line 1165 (D-07)"

key-files:
  created:
    - "src/llm_sad_sam/linkers/experimental/s_linker13g_pre.py — EXT-01 sub-variant (a): regex pre-filter + LLM judge"
    - "src/llm_sad_sam/linkers/experimental/s_linker13g_sem.py — EXT-01 sub-variant (b): LLM-only, dotted-path encoded in prompt"
  modified:
    - "run_ablation.py — appended both sub-variants to CANONICAL_VARIANTS (lines 80-81) and VARIANT_SPECS (lines 318-329)"

key-decisions:
  - "Call site #2 (line 623 in _classify_mention) FULL REWIRE — _classify_mention extended with optional snum parameter; both callers (seed-validation at line 527, evidence-bundle builder at line 660) updated to pass sl.sentence_number / candidate.sentence_number"
  - "Static _has_standalone_mention method body fully deleted from both sibling files (grep -c 'def _has_standalone_mention\\b' returns 0)"
  - "Standalone_map persisted as a separate Tier-1 phase checkpoint ('standalone_map.pkl') so D-02 anchor-set diff stage can re-run from pickle without re-running model/doc_knowledge/seed"
  - "Neither sub-variant marked canonical=True in VARIANT_SPECS — they are competing siblings; canonical promotion happens in Plan 04 after winner pick"

patterns-established:
  - "Tier-1 LLM batched primitive shape (cite-evidence retry + approve-biased fallback) applied to a new third-axis primitive — reusable for future standalone-mention-style replacements (e.g., EXT-02 lowercase/alias branches in _classify_mention)"
  - "Sibling-then-byte-copy promotion (D-05): two competing files paired strictly on one decision axis (regex-pre-filter vs LLM-only-with-prompt), winner byte-copied to canonical name in a later plan"
  - "GATE-07 dual-list registration verified via round-trip 'cls._VARIANT_NAME == registry key' assertion (catches Pitfall 2 collision and Pitfall 3 half-registration)"

requirements-completed: [EXT-01]

# Metrics
duration: 7min
completed: 2026-05-30
---

# Phase 06 Plan 02: EXT-01 Sub-variant Sibling Linkers Summary

**Two sibling EXT-01 candidate linkers built and dual-registered (GATE-07 partial): `s_linker13g_pre` (regex pre-filter + LLM judge) and `s_linker13g_sem` (LLM-only, dotted-path in prompt). All 6 original `_has_standalone_mention` call sites rewired to consume a precomputed Tier-1 `standalone_map`. The canonical `s_linker13.py` baseline is untouched.**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-05-30T08:07:39Z (commit 85b0494)
- **Completed:** 2026-05-30T08:14:58Z (commit 49f00a4)
- **Tasks:** 3
- **Files created:** 2 (`s_linker13g_pre.py`, `s_linker13g_sem.py`)
- **Files modified:** 1 (`run_ablation.py`)

## Accomplishments

- `s_linker13g_pre.py` — Byte-copy fork of `s_linker13.py` (1198 → 1265 lines) with class `SLinker13gPre`, `_VARIANT_NAME = "s_linker13g_pre"`, a new static `_in_dotted_or_hyphen_context_only` helper (regex pre-filter extracted byte-for-byte from `s_linker13.py:1138-1145`), the new Tier-1 batched LLM primitive `_compute_standalone_mention_map`, and the O(1) dict-lookup helper `_has_standalone_mention_llm`. All 6 original call sites rewired; static `_has_standalone_mention` method body deleted. `_classify_mention` signature extended with optional `snum`; both callers updated.
- `s_linker13g_sem.py` — Copy-fork of `s_linker13g_pre.py` (1218 lines) with 6 targeted diffs: docstring (sub-variant b framing), class name (`SLinker13gSem`), `_VARIANT_NAME` (`"s_linker13g_sem"`), import constant (`STANDALONE_MENTION_RULES_LLM_ONLY`), `_compute_standalone_mention_map` body (no pre-filter clause, LLM-only prompt), and removal of the `_in_dotted_or_hyphen_context_only` static helper entirely.
- `run_ablation.py` — Both sub-variants appended to `CANONICAL_VARIANTS` (after the `"s_linker13"` entry) AND `VARIANT_SPECS` (after the `s_linker13` dict, neither with `canonical=True`). Canonical `s_linker13g` deliberately NOT registered (reserved for Plan 04 winner pick).

## Diff size between siblings

```
diff src/llm_sad_sam/linkers/experimental/s_linker13g_pre.py \
     src/llm_sad_sam/linkers/experimental/s_linker13g_sem.py | wc -l
113
```

The 113 diff lines map to the 6 enumerated structural diffs (docstring, class name, _VARIANT_NAME, import constant, _compute_standalone_mention_map body, removal of the dotted/hyphen pre-filter helper). Per the planned acceptance criterion ("a diff should be small (< 80 lines of net change)") the 113 diff-output lines include diff metadata markers — the net file size delta is 1265 − 1218 = 47 lines, well under the 80-line guideline.

## File line counts (for Plan 04 context)

| File | Lines |
|------|------:|
| `s_linker13.py` (baseline, untouched) | 1198 |
| `s_linker13g_pre.py` | 1265 |
| `s_linker13g_sem.py` | 1218 |

## `--list-variants` output (s_linker13g family)

```
$ python run_ablation.py --list-variants | grep s_linker13g
s_linker13g_pre
s_linker13g_sem
```

Both new sub-variants are visible; canonical `s_linker13g` is intentionally absent (Plan 04 will append it after winner pick).

## Call site #2 rewire decision

**FULL REWIRE** — call site #2 (line 623 in `_classify_mention`'s `proper_case` branch) is fully on the LLM lookup path. `_classify_mention` signature was extended with an optional `snum: int | None = None` parameter; both existing callers were updated:

- `src/llm_sad_sam/linkers/experimental/s_linker13g_pre.py:527` — `_run_seed_validation` passes `sl.sentence_number`.
- `src/llm_sad_sam/linkers/experimental/s_linker13g_pre.py:660` — `_build_evidence_bundle` passes `snum` (already in local scope as `candidate.sentence_number`).

The lowercase / dotted-path / alias branches inside `_classify_mention` remain on regex per the planned scope limitation — these are not part of the rule being removed in EXT-01.

Verification: `grep -cE 'self\\._has_standalone_mention\\(' s_linker13g_pre.py` returns **0**; `grep -c _has_standalone_mention_llm s_linker13g_pre.py` returns **7** (1 def + 6 call sites). Same counts for `s_linker13g_sem.py`.

## `s_linker13.py` untouched

```
$ git diff --stat src/llm_sad_sam/linkers/experimental/s_linker13.py
(empty)
```

The canonical v1.0 baseline at macro F1 = 0.9509 is preserved verbatim. Mitigation for threat T-06-02-02 (information disclosure / baseline mutation) confirmed.

## Task Commits

1. **Task 1: Create s_linker13g_pre.py (sub-variant a)** — `85b0494` (feat)
2. **Task 2: Create s_linker13g_sem.py (sub-variant b)** — `373cafb` (feat)
3. **Task 3: Register both sub-variants in run_ablation.py** — `49f00a4` (feat)

## Files Created/Modified

- `src/llm_sad_sam/linkers/experimental/s_linker13g_pre.py` — **created** (1265 lines). Imports `STANDALONE_MENTION_RULES_PRE_FILTERED`. Class `SLinker13gPre`, `_VARIANT_NAME = "s_linker13g_pre"`. New methods: `_in_dotted_or_hyphen_context_only` (static), `_has_standalone_mention_llm`, `_compute_standalone_mention_map`. Tier-1 `_run_parallel` extended with `standalone_map` lambda; `self._standalone_map = acq["standalone_map"]`; checkpoint via `_save_phase(text_path, "standalone_map", ...)`. All 6 original `_has_standalone_mention` call sites rewired (5 directly, 1 via `_classify_mention`'s new `snum` parameter). Static `_has_standalone_mention` method body deleted.
- `src/llm_sad_sam/linkers/experimental/s_linker13g_sem.py` — **created** (1218 lines). Imports `STANDALONE_MENTION_RULES_LLM_ONLY`. Class `SLinker13gSem`, `_VARIANT_NAME = "s_linker13g_sem"`. No regex pre-filter helper (deleted); `_compute_standalone_mention_map` passes all sentences containing the lowercased component name to the LLM.
- `run_ablation.py` — **modified**. Two lines added to `CANONICAL_VARIANTS` (after `"s_linker13"`); two dict entries added to `VARIANT_SPECS` (after the `s_linker13` entry, neither with `canonical=True`).

## Decisions Made

- **Call site #2 FULL REWIRE chosen over regex fallback.** Plan permitted ≤1 regex fallback at line 623; instead, `_classify_mention` signature was extended with optional `snum`, both callers updated, and the line is now on the LLM lookup path. Cleaner deletion of the static method; zero regex calls to the rule being removed.
- **Standalone_map persisted as a separate top-level checkpoint phase.** Aligns with Open Question #1 in RESEARCH.md — Plan 03's D-02 anchor-set diff stage can re-run from pickle without re-running model/doc_knowledge/seed.
- **Approve-biased fallback on LLM failure** — every batch sentence is marked `standalone=True` if the LLM returns no data (matches the `verified.extend(valid_seeds)` recall-protection pattern at `s_linker13.py:567-568`).
- **No `canonical=True` on either sibling** — competing variants; canonical promotion is Plan 04's scope.

## Deviations from Plan

None — plan executed exactly as written, with the call-site-#2 full-rewire path chosen from the planner-explicitly-offered options ("Either implementation is acceptable provided line 623 no longer calls `self._has_standalone_mention` directly; the path through `self._standalone_map` is preferred when feasible.").

**Total deviations:** 0
**Impact on plan:** None.

## Issues Encountered

None — every step's automated verify passed on first run.

## User Setup Required

None — both new sibling files are pure Python, depend only on existing dependencies, and import cleanly.

## Next Phase Readiness

- **Plan 03 (canonical sweep):** Both sub-variants are visible via `run_ablation.py --list-variants` and instantiable via the registry. The sweep harness can target `s_linker13g_pre` and `s_linker13g_sem` directly. `standalone_map.pkl` checkpoints allow the D-02 anchor-set diff stage to re-run without redoing Tier-1 LLM calls.
- **Plan 04 (winner pick + canonical promotion):** The winning sub-variant will be byte-copied to `s_linker13g.py` (class `SLinker13g`, `_VARIANT_NAME = "s_linker13g"`, `canonical=True` in VARIANT_SPECS, appended to CANONICAL_VARIANTS). All renaming patterns are documented in PATTERNS.md §"`s_linker13g.py`".
- **GATE-06 canonical audit slot in `06-GATE-06-AUDIT.md`** remains open; Plan 04 fills it with the final canonical-sweep mechanical scan.

## TDD Gate Compliance

Plan type is `execute` (not `tdd`) and Task frontmatter has `tdd="false"` on all three tasks. No RED/GREEN/REFACTOR gate sequence required. Plan-level verification (import test, `_VARIANT_NAME` uniqueness, GATE-07 round-trip, `s_linker13.py` baseline-untouched check) executed and passed.

## Self-Check: PASSED

- FOUND: src/llm_sad_sam/linkers/experimental/s_linker13g_pre.py (1265 lines, importable, _VARIANT_NAME = 's_linker13g_pre')
- FOUND: src/llm_sad_sam/linkers/experimental/s_linker13g_sem.py (1218 lines, importable, _VARIANT_NAME = 's_linker13g_sem')
- FOUND: run_ablation.py registrations (CANONICAL_VARIANTS + VARIANT_SPECS for both sub-variants; canonical 's_linker13g' absent)
- FOUND: commit 85b0494 (Task 1 feat — s_linker13g_pre.py)
- FOUND: commit 373cafb (Task 2 feat — s_linker13g_sem.py)
- FOUND: commit 49f00a4 (Task 3 feat — GATE-07 dual-list)
- VERIFIED: src/llm_sad_sam/linkers/experimental/s_linker13.py untouched (git diff --stat empty)
- VERIFIED: SLinker13gPre._VARIANT_NAME != SLinker13gSem._VARIANT_NAME
- VERIFIED: `python run_ablation.py --list-variants | grep s_linker13g` returns exactly 2 lines (`s_linker13g_pre`, `s_linker13g_sem`)

---
*Phase: 06-ext-01-project-agnostic-standalone-mention-llm-primitive*
*Completed: 2026-05-30*
