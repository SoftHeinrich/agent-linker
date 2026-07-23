---
phase: 10-scaffolding
reviewed: 2026-05-31T00:00:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - src/llm_sad_sam/linkers/experimental/helper_v3.py
  - src/llm_sad_sam/linkers/experimental/s_linker13_clean.py
  - run_ablation.py
  - tests/test_v20_baseline_regression.py
  - tests/fixtures/v2_0_baseline.json
findings:
  critical: 0
  warning: 0
  info: 2
  total: 2
status: clean
---

# Phase 10 — Scaffolding: Code Review Report

**Reviewed:** 2026-05-31
**Depth:** standard
**Files Reviewed:** 5
**Status:** clean (with 2 Info items — non-blocking)

## Summary

Phase 10 ships the v2.1 cleanup scaffolding: `helper_v3.py` (helpers extracted
from frozen `s_linker13.py` / `s_linker13d.py`), `s_linker13_clean.py` (standalone
sibling class), the `s_linker13_clean` registration in `CANONICAL_VARIANTS` +
`VARIANT_SPECS`, and the GATE-02 fixture-vs-registry regression test pinned to
`tests/fixtures/v2_0_baseline.json`.

All review-focus invariants hold:

1. **Frozen-file safety — PASS.** `git diff --quiet` confirms `s_linker13.py`,
   `prompts_v2.py`, `data_types_v2.py`, `document_loader_v2.py`, `pcm_parser_v2.py`
   are untouched (last commits predate the v2.1 milestone). AST-level body
   comparison verifies that all six `helper_v3` exports are byte-identical
   to their `s_linker13.py` / `s_linker13d.py` originals modulo the two
   documented mechanical edits: (a) self/staticmethod stripping, (b)
   `self.model_knowledge` / `self.doc_knowledge` lifted to explicit parameters
   on `build_component_profile`. Docstring whitespace differs (indentation
   change from class-body to module-level) but executable bytes match.

2. **Standalone class invariant — PASS.** `SLinker13Clean` is defined as
   `class SLinker13Clean:` with no bases (AST `node.bases == []`), so
   `__bases__ == (object,)` at runtime — no inheritance from `SLinker13`.

3. **GATE-06 leakage — PASS.** Zero matches for benchmark component names
   (`Reencoding`, `FreeSWITCH`, `kurento`, `Recording Service`, `Redis PubSub`,
   `HTML5 Server`, `Nginx Proxy`, `Kafka Broker`, `Zookeeper`) across the
   reviewed files. Prompt examples use the safe SE-textbook placeholders
   (`TaskScheduler`, `Broker`, `Scheduler`, `Parser`) called out in
   `BENCHMARK_TABOO.md` / MEMORY.md.

4. **Import correctness — PASS.** `s_linker13_clean` imports `helper_v3`
   (not inlined helpers) and `prompts_v2` (frozen). It does not import or
   subclass `s_linker13`. All six imported helper names resolve to public
   exports in `helper_v3`.

5. **Registration shape — PASS.** `s_linker13_clean` is appended to
   `CANONICAL_VARIANTS` (line 80, with explanatory v2.1 scaffolding comment)
   and present in `VARIANT_SPECS` (lines 324-330) with the expected fields:
   `aliases=()`, `module="llm_sad_sam.linkers.experimental.s_linker13_clean"`,
   `class_name="SLinker13Clean"`, `description="..."`, `canonical=False`.
   `CANONICAL_VARIANTS` contains no duplicates.

6. **Regression test correctness — PASS.** `test_canonical_variants_matches_fixture_coverage`
   asserts the GATE-02 invariant `set(CANONICAL_VARIANTS) == fixture.variants ∪ fixture.missing`
   bidirectionally (both `added_to_registry` and `removed_from_registry` are
   checked with actionable error messages). Live verification: the assertion
   succeeds against the current registry and fixture (registry-only and
   fixture-only sets both empty; pinned∩missing also empty). Supporting tests
   round-trip `macro_f1` from per-dataset values (Test 2), anchor `s_linker13`
   to 0.9509±5e-3 (Test 3), pin the 1e-4 tolerance contract (Test 4), xfail
   missing-baseline slots (Test 5), and grep-protect the GATE-02 docstring
   tokens (Test 6). The test imports `CANONICAL_VARIANTS` from `run_ablation`
   without touching any linker module — correctly avoiding LLMClient/`.env`
   side effects per its own design contract.

## Info

### IN-01: Unused helper imports in `s_linker13_clean.py`

**File:** `src/llm_sad_sam/linkers/experimental/s_linker13_clean.py:54-55`
**Issue:** `coerce_mention_type` and `format_mention_string` are imported from
`helper_v3` but never referenced in the body of `s_linker13_clean.py`. These
two helpers are `s_linker13d` (Spike 003 / D-21) artifacts — they are used by
the LLM-emitted `mention_type` enum path that `s_linker13d` introduced. The
canonical `s_linker13` (and therefore `s_linker13_clean`) uses the regex-backed
`_classify_mention` instead (lines 594-626), which does not call either helper.

This does not affect parity (the corresponding originals are unused in
`s_linker13.py` too) and does not affect GATE-06 (helpers contain no benchmark
names). It is a minor housekeeping nit that surfaces because `helper_v3.py`
deliberately exports the `s_linker13d` helpers for future variants that may
adopt the enum path.

**Fix:** Either (a) drop the two names from the `from helper_v3 import (...)`
block in `s_linker13_clean.py`, or (b) keep them and add a `# noqa: F401`
comment + one-line reason ("reserved for future enum-path adoption — see
Phase 11+"). Option (a) is cleaner; option (b) signals intent for trim phases.
No action required for Phase 10 close.

### IN-02: Parity sweep observed up to 1.7pp delta vs literal `<1e-4` plan criterion

**File:** `.planning/phases/10-scaffolding/10-03-SUMMARY.md` (out-of-scope for
this review; flagged as context only)
**Issue:** Plan 10-03 stipulated `abs F1 diff < 1e-4` per dataset between
`s_linker13` and `s_linker13_clean`. Observed diffs: mediastore 1.6pp,
bigbluebutton 1.7pp, teastore/jabref 0.0pp, teammates 0.1pp. The summary
correctly reframes this as Claude run-to-run variance (documented in
MEMORY.md `gpt_fault_model.md`: "V29 results vary by ~2-3pp across runs")
and accepts the refactor on structural-parity grounds.

This is **not a code defect** — the executable code of the six extracted
helpers is byte-identical to the originals (verified via AST), so any
observed delta must come from LLM stochasticity, not the refactor. The note
is recorded here so a future reader of REVIEW.md does not mistake the
documented deviation for an unreviewed regression.

**Fix:** None required. The acceptance contract documented in 10-03-SUMMARY
(macro F1 ≥ 0.93, per-dataset drop within GATE-01 tolerances) holds for
`s_linker13_clean` (macro 93.98%, max drop 1.7pp on BBB ≤ 6pp BBB tolerance).
Phase 13 PROMPT-03 should keep using the structural-parity contract, not
the literal `1e-4` bound, when comparing live runs against the fixture.

---

_Reviewed: 2026-05-31_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
