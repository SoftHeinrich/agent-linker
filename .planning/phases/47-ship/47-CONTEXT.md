# Phase 47: SHIP - Context

**Gathered:** 2026-06-09
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — discuss skipped)

<domain>
## Phase Boundary

Create `src/llm_sad_sam/linkers/experimental/s_linker20.py` as a self-contained standalone
variant with the minimized prompt constants (from Phase 46) inlined directly in the file,
register it in the runner (`run_ablation.py`), and verify it does NOT touch the byte-equal
state of `s_linker19.py` or `s_linker13_min.py`.

Delivers (success criteria from ROADMAP):
1. `s_linker20.py` exists with `experimental=True`, `canonical=False`, no inheritance from
   `s_linker19`, all minimized prompt constants inlined directly in the file.
2. `run_ablation.py --variants s_linker20` runs without error (dry-run/cached mode sufficient;
   no LLM calls required).
3. `git diff` on `s_linker19.py` and `s_linker13_min.py` (vs v2.6.3 close hashes) is empty —
   GATE-01 verified.
4. Constants imported by `s_linker19` are unchanged on disk (byte-equal) — paper RQ1–RQ4
   replay determinism preserved.

Out of scope: running any gpt-5.4 sweep (that is Phase 48 SWEEP, which spends LLM budget).
</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase. Use the
ROADMAP phase goal, success criteria, the Phase 46 MINIMIZE-LOG, and the frozen scratch
artifacts as the authoritative inputs.
</decisions>

<code_context>
## Existing Code Insights

### Minimized prompt set (Phase 46 output — the source of truth for inlining)
- `tests/scratch/s_linker19.py` and `tests/scratch/prompts_v5.py` — the FROZEN minimized
  variant. 12 cuts kept, 14 LOC saved, 1 benchmark-leak eliminated. s_linker20 inlines this
  minimized prompt set into a single standalone file.
- `.planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md` — 19 rows (17 trial + 2 protected);
  the `kept` rows' after-text is what gets inlined. Documents per-cut replacement vocabulary
  decisions (e.g. `components` bare, `noun phrase that refers back`,
  `grouping that encompasses multiple elements`).
- `.planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md` — per-constant generality verdicts;
  `CUT-{TAG}-NN` ids are the foreign key into the minimize log.

### Runner registration pattern (`run_ablation.py`)
- Variant names listed in a registry list near line ~103 (alongside s_linker15/15b/15c/17a…).
- Full variant config dicts near line ~523: `dict(module="llm_sad_sam.linkers.experimental.<name>", ...)`.
- s_linker20 must be added in both the list and the config-dict registry, mirroring s_linker19's entry.

### GATE-01 byte-equal protection
- `s_linker19.py` and `s_linker13_min.py` are BYTE-EQUAL FROZEN (paper RQ1–RQ4 replay determinism).
- s_linker20 must be a NEW standalone file with NO inheritance from s_linker19 and must not edit
  the constants s_linker19 imports. Verify via `git diff` / SHA-256 vs v2.6.3 close hashes.
- Standing gates carried in: GATE-01 PASS, GATE-06 PASS (zero benchmark vocabulary).

### Environment note
- Repo owned by uid 1001; session runs as dev (1000). Permissions fixed this session
  (chgrp+g+w + system git safe.directory). Writes and git commits work.
</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase. The minimized prompt set in `tests/scratch/`
and the MINIMIZE-LOG kept-rows are authoritative for what to inline.
</specifics>

<deferred>
## Deferred Ideas

None — Phase 48 SWEEP (the behavioral validation that spends LLM budget) is the next phase and
is intentionally gated behind explicit user go-ahead.
</deferred>
