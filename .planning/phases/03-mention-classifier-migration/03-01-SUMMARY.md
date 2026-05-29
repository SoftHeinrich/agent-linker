---
phase: 03-mention-classifier-migration
plan: 01
subsystem: linker-variants
tags: [s_linker13d, ablation, mention-classifier, spike-003, gate-05-fail, llm-regression]

requires:
  - phase: 02-ambiguity-cleanup
    provides: "13b parent variant (clean GATE-01 pass)"
provides:
  - "s_linker13d standalone variant — 13b with `_classify_mention` regex replaced by Spike 003 LLM enum"
  - "Hard-tier evidence that LLM mention classifier produces a -18.8pp TM regression vs 12c (-19.7pp vs 13b parent), driven by 33 entity-source FPs on dotted-path references (`ui.website`, `logic.api`, `storage.entity`)"
  - "Failure dossier for Phase 3 closure decision"

key-files:
  created:
    - src/llm_sad_sam/linkers/experimental/s_linker13d.py
    - .planning/phases/03-mention-classifier-migration/03-01-SUMMARY.md
  modified:
    - run_ablation.py

requirements-completed: []  # VAR-04 NOT satisfied — TM regression > 1pp on hard tier

duration: ~45 min (code + hard-tier ablation)
completed: 2026-05-29
---

# Phase 03 / Plan 01 — SUMMARY

**Status: HARD REJECT at GATE-05.** s_linker13d's LLM mention classifier (Spike 003 pattern, STRICT enum coercion) ships per Task 1, but on the hard-tier benchmark gate it regresses Teammates F1 by 18.8pp vs 12c (and 19.7pp vs the 13b parent baseline). Per the standing checkpoint policy (`delta_TM < -0.02 → hard reject`), Task 4 (full sweep) was NOT executed.

## What Shipped (Task 1)

- `src/llm_sad_sam/linkers/experimental/s_linker13d.py` — standalone copy of `s_linker13b.py` with:
  - `_VARIANT_NAME = "s_linker13d"`
  - `_classify_mention` 4-branch regex replaced by `LLM_CLASSIFY_MENTION_PROMPT` + STRICT enum coercion (`_coerce_mention_type` raises `ValueError` on unknown).
  - Structured module docstring (REMOVED_FROM, RULES_REMOVED).
  - Registered in `run_ablation.py`. BENCHMARK_TABOO audit clean.
  - Smoke test + import OK.
- Commit: `5133d7b feat(03-01): add s_linker13d — replace _classify_mention with LLM enum (Spike 003)`

## Hard-Tier Results (Task 2, GATE-05)

JSON: `results/ablation_results/ablation_20260529_110532.json`

| Dataset       | F1_12c | F1_13b | F1_13d |   Δ vs 12c |   Δ vs 13b | TP | FP | FN | source breakdown          |
|---------------|-------:|-------:|-------:|-----------:|-----------:|---:|---:|---:|---------------------------|
| teammates     |  0.938 |  0.947 | **0.750** | **-0.188** | **-0.197** | 54 | **33** |  3 | sources: seed=45 entity=39 coref=3; FP-by-source: entity=33 |
| bigbluebutton |  0.844 |  0.839 |  0.832 |  -0.012    |  -0.007    | 47 |  4 | 15 | sources: seed=44 entity=5 coref=2; FP-by-source: seed=3 entity=1 |

- **teammates:** delta = -0.188 vs 12c. EXCEEDS the hard-reject threshold (`< -0.02`) by 9.4×. **HARD REJECT.**
- **bigbluebutton:** delta = -0.012 (within marginal-but-acceptable band). Healthy.

## Gate Decision

**HARD REJECT at Task 3 checkpoint per standing policy from Phase 1 closure (2026-05-28):**
> `delta_TM < -0.02 OR delta_BBB < -0.06 → hard reject, halt, write failure SUMMARY, return blocker`

Task 4 (full 5-project sweep) NOT executed.

## Failure-Mode Analysis

All 33 TM FPs are entity-source. Sample (S22-S26, S84-S86, S125):

```
S22 UI (entity): logic, ui.website, ui.controller represent an application of Model-View-Controller
S22 Logic (entity): logic, ui.website, ui.controller represent an application of Model-View-Controller
S23 UI (entity): ui.website is not a real package
S26 UI (entity): ui.website is not a Java package
S84 Logic (entity): Package overview contains logic.api, logic.core
S85 Logic (entity): logic.api provides the API of the component to be accessed by the UI
S86 Logic (entity): logic.core contains the core logic of the system
S125 Storage (entity): Classes in the storage.entity package are not visible outside this component
```

Mechanism: TM documentation extensively uses **dotted-path package references** (`ui.website`, `logic.api`, `storage.entity`). The original regex `_classify_mention` in 13b correctly tagged these as `indirect` references (matched the `_in_dotted_path` pattern), which downstream filtering then rejected as architecturally-meaningful mentions.

The new LLM enum classifier evidently emits a "concrete" mention type for these dotted-path tokens (treating `ui.website` as a literal reference to the `UI` component), allowing them through entity validation and producing 33 spurious entity-source TPs that are actually FPs in the gold standard.

This matches the documented MEMORY.md observation:
> **LLM CAN replace P8c boundary filters**: Convention-aware filter (3-step reasoning guide) catches 11 FPs vs 5 regex, 0 TPs killed. Key: protect partial_inject links from the filter.

But:
> **LLM can't replace P8b partial injection**: Kills all TPs even with ±2 sentence context. Partial-name disambiguation too hard without project-specific knowledge.

The `_classify_mention` regex was load-bearing for TM. Dotted-path classification is closer to "partial injection disambiguation" than to "boundary filtering" — the regex encodes deep project-specific knowledge (Java package convention `parent.child.leaf` = indirect reference, not concrete component) that an LLM can't reliably reproduce without explicit per-pattern training data.

## What Did Not Run

- Task 4 (full 5-project sweep) — skipped per hard-reject policy.
- ROADMAP Phase 3 NOT marked complete.
- STATE.md and ROADMAP.md left unchanged.

## Recovery Paths (For User)

1. **Strengthen the LLM prompt** to explicitly reject dotted-path references (`ui.website`, `logic.api`) as indirect — add a rule like "if the mention contains a `.` and looks like a package path, classify as `indirect`." Cost: ~30 min code + re-run hard tier. Risk: prompt-engineering arms race; MEMORY warns these prompts are at a local optimum and simplifications/changes regress.
2. **Hybrid classifier**: keep the regex as a pre-filter for clearly indirect mentions (dotted paths, lowercase fragments), only call LLM for the ambiguous cases. Cost: ~1 hour code + re-run. Defeats the "remove the rule" purity claim but preserves TM precision.
3. **Drop VAR-04 — retire Spike 003 for the mention-classifier slot**. Conclude that `_classify_mention` is load-bearing structural code that does not survive LLM substitution. Move VAR-04 to a deferred-items list. Phase 3 closes empty; Phase 4 begins next.
4. **Replan Phase 3 around a different rule removal** (e.g., a non-classification heuristic in 13b that's more likely to survive LLM replacement). Significant scope creep.

**Recommendation: path 3.** The regression magnitude (-19pp TM) is far outside any variance band documented to date and matches a known LLM weakness pattern. Forcing 13d through a stronger prompt would likely produce a fragile pattern-matcher with prompt-engineering hidden in it — undesirable for the "defensible no-hand-crafted-rules" thesis. Cleanest record is to log this empirically: "LLM mention classifier rejected on dotted-path classification regression."

## Commands Executed

```bash
cp src/llm_sad_sam/linkers/experimental/s_linker13b.py src/llm_sad_sam/linkers/experimental/s_linker13d.py
# Edits: docstring, class rename, _VARIANT_NAME, prompt constant, _classify_mention rewrite, _coerce_mention_type, _format_mention_string, registration.
python -c "from llm_sad_sam.linkers.experimental.s_linker13d import SLinker13d; ..."   # smoke OK
python -c "from run_ablation import build_linker; ..."   # registration OK
# BENCHMARK_TABOO audit: CLEAN
git commit -m "feat(03-01): add s_linker13d — replace _classify_mention with LLM enum (Spike 003)"   # 5133d7b
nohup python run_ablation.py --variants s_linker13d --datasets teammates bigbluebutton > /tmp/13d_hardtier.log 2>&1 &
# 27 min elapsed → ablation_20260529_110532.json
```

## BENCHMARK_TABOO Audit

PASS. Inline prompt uses safe SE-textbook examples.

## Pickle Cache Hygiene (D-07)

- `results/phase_cache/s_linker13d/{teammates,bigbluebutton}/` exist. No leakage into other variant namespaces.

---
*Phase: 03-mention-classifier-migration*
*Plan: 01*
*Completed: 2026-05-29 (with GATE-05 HARD REJECT — VAR-04 unmet)*
