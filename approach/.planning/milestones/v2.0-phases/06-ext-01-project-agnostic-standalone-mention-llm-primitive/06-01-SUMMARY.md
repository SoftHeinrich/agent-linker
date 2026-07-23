---
phase: 06-ext-01-project-agnostic-standalone-mention-llm-primitive
plan: 01
subsystem: llm-prompt
tags: [llm-prompt, traceability, standalone-mention, gate-06, prompts_v2]

# Dependency graph
requires:
  - phase: 05-promotion
    provides: "s_linker13.py baseline with _has_standalone_mention still in place (RISKY KEEP from v1.0)"
provides:
  - "STANDALONE_MENTION_RULES_PRE_FILTERED prompt constant (sub-variant a: 889 bytes)"
  - "STANDALONE_MENTION_RULES_LLM_ONLY prompt constant (sub-variant b: 1245 bytes, dotted-path negative example included)"
  - "06-GATE-06-AUDIT.md pre-clearance record (Plan 04 will append final canonical audit)"
affects:
  - "06-02 (sibling linker construction — both linkers import these two prompts)"
  - "06-03 (canonical sweep — uses these prompts via the Plan 02 linker files)"
  - "06-04 (final audit — appends to 06-GATE-06-AUDIT.md)"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-sibling prompt design encodes the literal-vs-semantic axis (D-01) as data, not as code branches"
    - "Pre-filter sub-variant (a) deliberately omits dotted-path teaching since the regex pre-filter handles it before the LLM sees the sentence"
    - "LLM-only sub-variant (b) carries the compiler.parser.ASTBuilder dotted-identifier negative example to defuse the 13d TeaMMates regression (Pitfall 1)"

key-files:
  created:
    - ".planning/phases/06-ext-01-project-agnostic-standalone-mention-llm-primitive/06-GATE-06-AUDIT.md - GATE-06 pre-clearance record (mechanical taboo scan + reviewer-defensibility table)"
  modified:
    - "src/llm_sad_sam/linkers/experimental/prompts_v2.py - appended two new prompt constants under new section header 'Tier 1 — Standalone-Mention Detection (EXT-01)'"

key-decisions:
  - "Helper extraction (_in_dotted_or_hyphen_context_only) deliberately deferred to Plan 02 (D-Discretion) to keep Plan 01's review surface limited to prompt text + audit only"
  - "Both prompts use N_INTEGER placeholder consistent with existing s_linker13.py JSON template convention (parseable by _parse_snum)"
  - "Substring-match artefact in plan verify command (ui inside ASTBuilder) documented in audit; correct check uses grep -iwE (word boundaries)"

patterns-established:
  - "GATE-06 pre-clearance: any new prompt requires (a) mechanical BENCHMARK_TABOO scan AND (b) reviewer-defensibility table — pre-clearance file slot left open for the canonical-sweep audit by a later plan"
  - "Sibling-prompt isolation: when two competing sub-variants share one decision axis, the prompts are authored together in one plan to ensure they are paired strictly on the axis under study (here: dotted-path encoding)"

requirements-completed: [EXT-01]

# Metrics
duration: 3min
completed: 2026-05-30
---

# Phase 06 Plan 01: Project-Agnostic Standalone-Mention Prompts (EXT-01) Summary

**GATE-06 pre-clearance: PASS — two standalone-mention prompt constants added to prompts_v2.py, paired on the dotted-path-encoding axis, with zero benchmark-derived terms confirmed by both mechanical scan and reviewer-defensibility check.**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-05-30T08:02:06Z
- **Completed:** 2026-05-30T08:05:01Z
- **Tasks:** 2
- **Files modified:** 1 (`prompts_v2.py`)
- **Files created:** 1 (`06-GATE-06-AUDIT.md`)

## Accomplishments

- Two new prompt constants (`STANDALONE_MENTION_RULES_PRE_FILTERED`, `STANDALONE_MENTION_RULES_LLM_ONLY`) landed under a new section header in `prompts_v2.py`, both importable and following the established `<title>\nRULES:\n...\nReturn JSON: ...\nJSON only:"""` shape.
- Sub-variant (b) carries the `compiler.parser.ASTBuilder` dotted-identifier negative example required to defuse the historical 13d TeaMMates regression (Pitfall 1 from RESEARCH.md).
- GATE-06 pre-clearance file records both the mechanical taboo scan stdout and a reviewer-defensibility table covering all 4 example sentences, with a `TBD — written by Plan 04` slot for the canonical-sweep audit.

## Task Commits

Each task was committed atomically:

1. **Task 1: Append two standalone-mention prompt constants to prompts_v2.py** — `0ade81b` (feat)
2. **Task 2: Write GATE-06 pre-clearance audit for the two new prompts** — `ec87653` (docs)

Deviation commits:

3. **Substring-match artefact documentation in audit** — `9ebd681` (docs)

## Files Created/Modified

- `src/llm_sad_sam/linkers/experimental/prompts_v2.py` — Appended new section header and two prompt constants (lines 225-258 in the new file layout). Existing constants (`AMBIGUITY_*`, `DOC_KNOWLEDGE_*`, `WORD_USAGE_PROMPT`, `ENTITY_EXTRACTION_RULES`, `VALIDATION_RULES`, `COREF_RULES`, `SEED_DISAMBIGUATION_RULES`) untouched.
- `.planning/phases/06-ext-01-project-agnostic-standalone-mention-llm-primitive/06-GATE-06-AUDIT.md` — New file: GATE-06 pre-clearance audit with mechanical scan command, scan stdout, reviewer-defensibility table, decision, and open items for Plan 04.

## GATE-06 mechanical scan output (for Plan 04 reference)

Narrow-scope command (the operative GATE-06 check on the two new constants):

```
awk '/STANDALONE_MENTION_RULES_PRE_FILTERED|STANDALONE_MENTION_RULES_LLM_ONLY/,/^"""$/' \
  src/llm_sad_sam/linkers/experimental/prompts_v2.py \
  | grep -iE "(<full BENCHMARK_TABOO regex — see 06-GATE-06-AUDIT.md>)" \
  || echo "NO HITS"
```

Stdout (literal): `NO HITS`.

Word-bounded variant (defensive — correct against substring overlaps):

```
... | grep -iwE "(logic|UI|client|storage|common|cache|registry|persistence|facade|recording|cascade|conversion|dedicated|adapter|processor|kurento|freeswitch|redis|bbb|html5|preferences|globals|watermark|reencoding|recommender|datastore)" || echo "NO HITS (word-bounded)"
```

Stdout (literal): `NO HITS (word-bounded)`.

## Prompt byte-length signal (for Plan 04 cost context)

| Constant | Bytes | Notes |
|----------|-------|-------|
| `STANDALONE_MENTION_RULES_PRE_FILTERED` | 889 | 4 rules, no domain examples — relies on Plan 02 regex pre-filter for dotted/hyphen cases |
| `STANDALONE_MENTION_RULES_LLM_ONLY` | 1245 | 5 rules, 4 inline examples (Parser, compiler.parser.ASTBuilder, Parser-style grammar, FileSystem) |

The 356-byte gap is the dotted-path teaching budget — Plan 02's regex pre-filter has to be cheaper than ~356 bytes of per-call prompt tokens × call volume to win on cost.

## Decisions Made

- **Helper extraction deferred to Plan 02** — per D-Discretion, `_in_dotted_or_hyphen_context_only` lands alongside the sub-variant (a) linker file in Plan 02 to keep Plan 01's review surface to "prompt text + audit only".
- **Token placeholder = `N_INTEGER`** — matches existing convention in `s_linker13.py:772, 1068`; parser `_parse_snum` already handles this token.
- **No edits to `s_linker13.py`, `run_ablation.py`, or any linker file** — strictly out-of-scope per Plan 01 success criteria.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Plan verification command produces false-positive substring match**
- **Found during:** Plan-level verification (after Task 2 commit)
- **Issue:** The `<verification>` block uses `grep -iE` without word boundaries; this matches `ui` as a substring inside `ASTBuilder` (`AST**Bui**lder`). `ASTBuilder` is on the BENCHMARK_TABOO.md confirmed-safe whitelist (line 62, Compiler design), so this is a false positive, not a real GATE-06 hit.
- **Fix:** Documented the artefact in `06-GATE-06-AUDIT.md` with the correct word-bounded check (`grep -iwE`) — its stdout is `NO HITS (word-bounded)`. Flagged for Plan 04 to use `-iwE` in the canonical sweep.
- **Files modified:** `.planning/phases/06-ext-01-project-agnostic-standalone-mention-llm-primitive/06-GATE-06-AUDIT.md`
- **Verification:** Word-bounded re-run returns `NO HITS (word-bounded)`; the substring artefact (`ui` inside `ASTBuilder`) is the only match and is a known-safe example.
- **Committed in:** `9ebd681` (separate docs commit so the rationale is auditable)

---

**Total deviations:** 1 auto-fixed (1 Rule 1 — false-positive in plan verify command)
**Impact on plan:** Zero impact on prompt content or GATE-06 decision. The substring-overlap artefact is an artefact of the verification regex's lack of word boundaries, not a real benchmark-leakage event. Documentation handed to Plan 04 so the canonical sweep uses the correct regex.

## Issues Encountered

None — the substring-match artefact above was caught by the plan-level verification step and documented before the SUMMARY landed.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `prompts_v2.STANDALONE_MENTION_RULES_PRE_FILTERED` and `prompts_v2.STANDALONE_MENTION_RULES_LLM_ONLY` are importable and ready for Plan 02's two sibling linker files (`s_linker13g_pre.py`, `s_linker13g_sem.py`) to consume.
- `06-GATE-06-AUDIT.md` records pre-clearance PASS with the canonical-audit slot left open for Plan 04.
- No edits required to `s_linker13.py` or `run_ablation.py` by this plan — Plan 02 takes that scope.

## TDD Gate Compliance

Plan type is `execute`, not `tdd` — no RED/GREEN/REFACTOR gate sequence required. Plan-level verification (import test + grep checks) executed and passed.

## Self-Check: PASSED

- FOUND: src/llm_sad_sam/linkers/experimental/prompts_v2.py (modified, both constants importable)
- FOUND: .planning/phases/06-ext-01-project-agnostic-standalone-mention-llm-primitive/06-GATE-06-AUDIT.md (created)
- FOUND: commit 0ade81b (Task 1 feat — prompt constants)
- FOUND: commit ec87653 (Task 2 docs — GATE-06 pre-clearance)
- FOUND: commit 9ebd681 (deviation docs — substring artefact note)

---
*Phase: 06-ext-01-project-agnostic-standalone-mention-llm-primitive*
*Completed: 2026-05-30*
