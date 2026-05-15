---
phase: 01-baseline-and-infrastructure
plan: 05
subsystem: infra
tags: [ablation, spike-001, trailing-word, variant, gate-05]

requires:
  - phase: 01-baseline-and-infrastructure
    provides: 12c baseline (Plan 04), _VARIANT_NAME pattern (Plan 03), diskcache (Plan 02)
provides:
  - s_linker13a variant file (registered in runner)
  - Spike 001 LLM trailing-word primitive exercised end-to-end in live pipeline
  - Hard-tier gate measurement (GATE-05 outcome: hard reject)

affects: phase-02-ambiguity-cleanup, phase-03-mention-classifier-migration

tech-stack:
  added: []
  patterns: [Spike-001 fully_llm_driven trailing-word alias discovery]

key-files:
  created:
    - src/llm_sad_sam/linkers/experimental/s_linker13a.py
  modified:
    - run_ablation.py

key-decisions:
  - "Plan/Research-vs-code drift: s_linker12c.py NO LONGER contains _split_component_name or _enrich_trailing_words (cleaned in commit a8aae92 'clean s_linker12c for ICSE'). The plan/RESEARCH/spike-002-audit text predates that clean. 13a was therefore implemented as 12c-copy + NEW _enrich_trailing_words (Spike 001 pattern) wired into _learn_document_knowledge_enriched."
  - "Taboo audit script substring-match false positive: 'ambiguity' contained 'gui' (taboo for JabRef). Replaced 'no ambiguity' with 'no overlap' in the inline prompt without semantic loss."
  - "GATE-05 hard reject on bigbluebutton (delta = -0.048 << -0.01 threshold). Halt before full sweep per user checkpoint policy."

requirements-completed: []  # VAR-01 NOT satisfied — hard-tier regression > 1pp on BBB

duration: 38 min
completed: 2026-05-15
---

# Phase 01 / Plan 05 — SUMMARY

**Hard-tier rejected: s_linker13a regresses BigBlueButton F1 by 4.8pp vs 12c baseline. VAR-01 not satisfied. Phase 1 remains incomplete.**

## Performance

- **Duration:** 38 min (code: ~5 min, hard-tier ablation: ~24 min, analysis + writeup: ~9 min)
- **Started:** 2026-05-15T18:17Z (Task 1 commit `e1dc990`)
- **Completed:** 2026-05-15T18:42Z (hard-tier results landed)
- **Tasks completed:** 2 of 4 (Task 3 checkpoint blocked, Task 4 not started)
- **Files modified:** 2 (`s_linker13a.py` new, `run_ablation.py` registration)

## What Shipped (Task 1)

- `src/llm_sad_sam/linkers/experimental/s_linker13a.py` (1198 lines)
  - Class: `SLinker13a` with `_VARIANT_NAME = "s_linker13a"`
  - Structured module docstring with `REMOVED_FROM:` and `RULES_REMOVED:`
  - Inline prompt constant `LLM_ONLY_TRAILING_WORD_PROMPT` (taboo-clean — TaskScheduler/Scheduler placeholders)
  - New method `_enrich_trailing_words(knowledge, sentences, components)` — Spike 001 pattern, single LLM call + evidence-sentence guardrail
  - Wired into `_learn_document_knowledge_enriched` (single call at end, before `return knowledge`)
- `run_ablation.py`: 13a appended to `CANONICAL_VARIANTS` and `VARIANT_SPECS` (last entry after `s_linker12e`)
- All Task 1 acceptance criteria PASS (file exists, class renamed, no `_split_component_name`, prompt constant, evidence-sentence guardrail, no OrderProcessor, TaskScheduler present, no `s_linker12c` path-literal leak, runner registration smoke test green).
- BENCHMARK_TABOO audit: `TABOO AUDIT CLEAN` after fixing the `gui`-in-`ambiguity` substring false positive (replaced "no ambiguity" → "no overlap").

**Task 1 commit:** `e1dc990` (feat(01-05): add s_linker13a with Spike 001 LLM trailing-word enrichment)

## Hard-Tier Results (Task 2 — GATE-05)

**Ablation JSON:** `results/ablation_results/ablation_20260515_184127.json`

| Dataset       | 12c F1 (Plan 04) | 13a F1 | delta (pp) | 12c TP/FP/FN | 13a TP/FP/FN | 13a time |
|---------------|------------------:|-------:|-----------:|--------------:|--------------:|---------:|
| teammates     | 0.938            | **0.931** | **-0.007** | 53 /  3 /  4 | 54 /  5 /  3 |  899 s |
| bigbluebutton | 0.844            | **0.796** | **-0.048** | 46 /  1 / 16 | 43 /  3 / 19 |  547 s |

**Hard-tier macro:** 13a = (0.931 + 0.796) / 2 = **0.864** vs 12c hard-tier macro = (0.938 + 0.844) / 2 = **0.891**.

## GATE-05 Decision: HARD REJECT

**Checkpoint policy (executor):**

| Condition | Outcome |
|-----------|---------|
| delta_TM ≥ -0.01 AND delta_BBB ≥ -0.01 | auto-approve → proceed to full sweep |
| -0.01 < delta_TM ≤ -0.02 OR -0.01 < delta_BBB ≤ -0.02 | marginal → flag, surface as blocker, halt before Task 4 |
| **delta < -0.02 on either dataset** | **hard reject, halt, surface failure mode** |

- delta_TM = **-0.007** (within auto-approve tolerance)
- delta_BBB = **-0.048** (HARD REJECT — 4.8pp below 12c, 4× the rejection threshold)

**Task 4 (full 5-project sweep) NOT executed.** Per plan and policy, no full sweep is run when hard-tier rejects.

## Failure-Mode Analysis

### Mechanism: zero trailing-word aliases discovered + Claude run-to-run drift in Tier 2

1. **Spike 001 `_enrich_trailing_words` produced 0 new aliases on both teammates and bigbluebutton.** Grepping the hard-tier run log shows the existing alias-discovery step (`_learn_document_knowledge_enriched` body) still produces 11 aliases for BBB (`fsels → FSESL`, `Kurento Media Server → kurento`, etc.), but NO line tagged `"Alias (trailing-word, LLM):"` appears. The new LLM call asks "what single-word reference is shorthand for a multi-word component," and on these documents either (a) the LLM finds no candidate that passes rule-2 ("not a generic role" — words like `Server` / `Client` are rejected), or (b) the candidate fails the evidence-sentence guardrail.

2. **Per-source FP/FN deltas (BBB):**
   - 12c: TP=46, FP=1, FN=16
   - 13a: TP=43, FP=3, FN=19
   - Δ = -3 TP, +2 FP, +3 FN. The added FPs are `Redis PubSub @ S27, S31` (seed source) and `Recording Service @ S50` (entity source). The lost TPs are all `HTML5 Client` / `HTML5 Server` partial-name mentions in the same band 12c also missed (S6, S10, etc.).

3. **No causal link from 13a's new code to the failures.** The added FPs come from the seed and entity pipelines, which are byte-identical to 12c. The added FNs are in the same multi-word-partial-name failure mode 12c exhibits. This matches MEMORY.md's documented variance pattern: "GPT has massive run-to-run variance (±5-12 links)…"; Claude is normally tighter, but BBB's HTML5 Client/Server / WebRTC-SFU partial mentions are an unstable boundary even on Claude.

4. **Indirect causal channel: the new LLM call's place in the prompt-cache stream.** Adding any new LLM call to `_learn_document_knowledge_enriched` changes the order/timing of subsequent calls, which can perturb seed-validation and entity-pipeline outcomes through Claude's run-to-run drift even with identical prompts. This is the same kind of non-deterministic cascade documented in MEMORY.md.

### What this DOES validate

- Spike 001's `fully_llm_driven` LLM-only prompt is **safe** (zero false positives — strict adherence to rule 2 / rule 3 / evidence-sentence guardrail).
- The new method is **not actively harming** through wrong aliases (it produced no aliases on the harder dataset).
- The pipeline integration (D-07 path discipline) holds: `results/phase_cache/s_linker13a/{teammates,bigbluebutton}/` exist; `results/phase_cache/s_linker12c/` has no new subdirs.

### What this does NOT validate

- VAR-01 ("13a passes the dual floor on the full 5-project sweep"). Not measured — gate rejected before Task 4.
- That the trailing-word LLM primitive actually **adds value** in the live pipeline. On these two datasets it added zero aliases. The Spike's parity test (`_test_parity_with_current`) ran on synthetic fixtures with handcrafted single-word mentions; real benchmark docs may simply not contain the pattern Spike 001 targets.

## Acceptance Criteria Check (Plan 05)

| Criterion | Status |
|-----------|--------|
| Task 1: s_linker13a.py exists, class renamed, prompt constant, no `_split_component_name`, registered in runner | PASS |
| Task 1: BENCHMARK_TABOO audit clean | PASS (after `gui`-in-`ambiguity` fix) |
| Task 2: latest ablation JSON has 13a rows for teammates + bigbluebutton | PASS |
| Task 2: results/phase_cache/s_linker13a/{teammates,bigbluebutton}/ exist | PASS |
| Task 2: results/phase_cache/s_linker12c/ no new subdirs | PASS |
| Task 2: deltas vs Plan 04 baseline computed and reported | PASS |
| Task 3 (checkpoint): GATE-05 hard-tier review | **REJECT** (delta_BBB = -0.048) |
| Task 4 (full sweep): macro F1 ≥ 0.93, per-dataset within 2pp | **NOT EXECUTED** |
| VAR-01 satisfied | **NO** |

## Phase 1 Status

- Plans 01–04: complete (doc strike, diskcache, _VARIANT_NAME, 12c baseline).
- Plan 05: **incomplete — GATE-05 hard reject**. VAR-01 unmet. Phase 1 success criterion 3 ("s_linker13a registered…hard-tier with no regression >1pp vs 12c") **FAILS**.

Phase 1 cannot be marked complete on this run. Recommended next-step routing (for caller to choose):

1. **Re-run hard-tier for variance check.** MEMORY.md documents ±5-12 link variance on Claude for BBB. A second cache-cleared run could land in the 0.84-0.86 envelope and pass the marginal band. Cost: ~24 min. Cheapest test.
2. **Promote Spike 001 to per-document gate** — only call `_enrich_trailing_words` when the existing alias-discovery step is silent or thin, reducing the timing-perturbation surface.
3. **Strengthen Spike 001 prompt rule 2.** Add a positive case for multi-word components whose trailing word is generic but unambiguous in context (e.g., `Recording Service → Recording` when "Service" is uniquely owned). The current rule 2 ("not a generic role") may be over-rejecting.
4. **Accept that VAR-01 is unreachable with Spike 001 as designed** and re-scope Phase 1 to drop VAR-01 (and pull a different first-rule-removal into 13a). This is a decision for the project owner.

## Commands Executed

```bash
cp src/llm_sad_sam/linkers/experimental/s_linker12c.py src/llm_sad_sam/linkers/experimental/s_linker13a.py
# Edits: docstring, class rename, _VARIANT_NAME, prompt constant, _enrich_trailing_words, banner, wiring.
python <<'PYEOF'  # taboo audit
…
PYEOF
python -c "from llm_sad_sam.linkers.experimental.s_linker13a import SLinker13a; …"   # smoke test PASS
python -c "from run_ablation import CANONICAL_VARIANTS, VARIANT_SPECS, build_linker; …"   # registration PASS
git commit -m "feat(01-05): add s_linker13a with Spike 001 LLM trailing-word enrichment"   # e1dc990
nohup python run_ablation.py --variants s_linker13a --datasets teammates bigbluebutton > /tmp/13a_hardtier.log 2>&1 &
# 24 min elapsed → ablation_20260515_184127.json
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 — Bug / Plan-vs-code drift] Plan and RESEARCH assume `s_linker12c.py` contains `_split_component_name` and `_enrich_trailing_words`; the file does not.**
- **Found during:** Task 1 (copy + edit step).
- **Issue:** Plan §Step 3 says "Delete `_split_component_name` (the static method around L292-298 in the 12c source — decorator + signature + body, 7 lines)." That method does not exist in the current 12c. Likewise Step 5 "Replace the body of `_enrich_trailing_words` (currently L420-482 in 12c source)." Spike 002's AUDIT.md says "Source: `llm_sad_sam/linkers/experimental/s_linker12c.py` (1211 lines)" — our 12c is 1159 lines. Commit `a8aae92` ("clean s_linker12c for ICSE: unified aliases, parallel extraction, structural guardrails", Apr 3 2026) removed both methods from 12c. Those methods now live only in `s_linker12d` and `s_linker12e`.
- **Fix:** Honor the plan's `must_haves` (which are code-state truths, not action steps):
  - `_split_component_name` not present in 13a → satisfied trivially (12c copy never had it).
  - `_enrich_trailing_words` uses Spike 001 pattern → **ADD** the method (not "rewrite"), wire it into `_learn_document_knowledge_enriched` before `return knowledge`.
  - All other Task 1 acceptance criteria (REMOVED_FROM docstring, RULES_REMOVED docstring, prompt constant, taboo audit, registration) satisfied verbatim.
- **Verification:** All 13 Task 1 acceptance criteria pass.
- **Committed in:** `e1dc990`

**2. [Rule 1 — Bug] Taboo audit substring false positive: `gui` matched inside `ambiguity`.**
- **Found during:** Task 1 step 7 (BENCHMARK_TABOO audit).
- **Issue:** The audit script does naive substring matching. The string `"no ambiguity"` in rule 3 of the prompt contains `gui` (JabRef component name). The Spike 001 source has the same string and was shipped without this audit; the plan's script is stricter than reality requires.
- **Fix:** Replaced `"no ambiguity"` with `"no overlap"` in rule 3 of `LLM_ONLY_TRAILING_WORD_PROMPT`. Semantic content unchanged; rule 3 still asserts "no other listed component ends with the same short word."
- **Verification:** Re-ran the audit script → `TABOO AUDIT CLEAN`.
- **Committed in:** `e1dc990`

**3. [Rule 1 — Bug] `__init__` print banner still said `"SLinker12c (12b - Tier 2, intersection voting)"`.**
- **Found during:** Task 1 step 8 (import smoke test).
- **Issue:** The 12c constructor prints a self-identification banner; the cp copy carried that banner into 13a.
- **Fix:** Updated banner to `"SLinker13a (12c + Spike 001 LLM trailing-word enrichment)"`.
- **Verification:** Smoke-test banner now correctly identifies the variant.
- **Committed in:** `e1dc990`

**4. [Rule 1 — Bug] Stray smoke-test pickle dir leaked into 13a cache.**
- **Found during:** Task 2 (pickle cache hygiene check after hard-tier).
- **Issue:** The Task 1 step 8 smoke test instantiated `SLinker13a` with `text_path='/tmp/fake_dataset.txt'`, which made `_checkpoint_dir` create `results/phase_cache/s_linker13a/fake_dataset/` as a side effect.
- **Fix:** Removed the stray dir after Task 2 (`rm -rf results/phase_cache/s_linker13a/fake_dataset`).
- **Verification:** `ls results/phase_cache/s_linker13a/` now shows only `bigbluebutton` and `teammates`.
- **Not separately committed** (no source change — runtime artifact cleanup).

---

**Total deviations:** 4 auto-fixed (4× Rule 1 bugs).
**Impact on plan:** Deviation #1 is a planner-state drift that changes the semantic of "rule removal" from "delete + rewrite" to "add" — material because the 13a contribution becomes "exercise Spike 001 primitive end-to-end," not "remove a load-bearing rule." This is recorded for the milestone audit but does not affect the GATE-05 decision (which is purely an F1 measurement).

## Issues Encountered

**1. GATE-05 hard reject — BBB regression 4.8pp.** This is the blocker. Failure-mode analysis above. Recommended re-routes presented to caller.

## Authentication Gates

None.

## User Setup Required

None.

## Next Phase Readiness

**Phase 1 NOT ready to close.** Specifically:

- Phase 1 success criterion 3 (`s_linker13a registered…hard-tier run completes with no regression >1pp vs 12c; full 5-project sweep confirms macro F1 ≥ 93%…`) FAILS on hard-tier.
- VAR-01 requirement unsatisfied.
- ROADMAP Phase 1 cannot be marked `[x] Complete`.

Phase 2 (Ambiguity Cleanup — 13b + 13c) is **blocked** on Phase 1 closure: Phase 2 plans branch off the 13a variant pattern, so they need a passing 13a (or an alternate first-removal variant) to inherit from.

**Blocking decision needed from caller:** which re-route (1-4 in §Phase 1 Status) to take.

---
*Phase: 01-baseline-and-infrastructure*
*Plan: 05*
*Completed: 2026-05-15 (with GATE-05 hard reject)*
