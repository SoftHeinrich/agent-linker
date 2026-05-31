# Roadmap: llm-sad-sam-v45

## Milestones

- ✅ **v1.0 — Rule-to-LLM Ablation** (`s_linker12c` → `s_linker13`) — Phases 1–5 — shipped 2026-05-29. Final macro F1 0.9509. See [`milestones/v1.0-ROADMAP.md`](milestones/v1.0-ROADMAP.md).
- ✅ **v2.0 — Complete Rule Removal + Cross-Model — Generality First** — Phases 6–9 — shipped 2026-05-31. EXT-01 closed empty (negative), CROSS evidence published on gpt-5.4. See [`milestones/v2.0-ROADMAP.md`](milestones/v2.0-ROADMAP.md) and [`milestones/v2.0-MILESTONE-AUDIT.md`](milestones/v2.0-MILESTONE-AUDIT.md).
- 🚧 **v2.1 — Cleanup + Prompt Simplification** — Phases 10–13 — in progress.

## Phases

<details>
<summary>✅ v1.0 — Rule-to-LLM Ablation (Phases 1–5) — SHIPPED 2026-05-29</summary>

Phases 1–5 complete. See `milestones/v1.0-ROADMAP.md` for full detail.

</details>

<details>
<summary>✅ v2.0 — Complete Rule Removal + Cross-Model (Phases 6–9) — SHIPPED 2026-05-31</summary>

Phases 6–9 complete. See `milestones/v2.0-ROADMAP.md` for full detail.

</details>

### 🚧 v2.1 — Cleanup + Prompt Simplification (In Progress)

**Milestone Goal:** Produce a slimmed standalone variant (`s_linker13_clean` / `s_linker13_min`) using new versioned helpers and `prompts_v3.py`, gated by Claude Sonnet macro F1 ≥ 0.93 AND gpt-5.4 macro ≥ 0.9077 within ≤ 1pp tolerance. Nothing currently runnable breaks.

- [ ] **Phase 10: Scaffolding** — Standalone clean variant, versioned helpers, frozen-compat regression test, and cross-model gate definition
- [ ] **Phase 11: Research** — Literature + web survey on prompt minimization techniques to inform trim strategy
- [ ] **Phase 12: Trim Ablation** — Per-prompt rule-trim variants in `prompts_v3.py` gated by Claude Sonnet AND gpt-5.4 floors; generality re-audit of every retained trim
- [ ] **Phase 13: Promotion & Wrap** — Final `s_linker13_min.py` sweep (or negative result publication) + ABLATION-TABLE addendum and .tex regen

## Phase Details

### Phase 10: Scaffolding
**Goal**: The clean variant, versioned helper modules, regression safeguard, and gated cross-model definition are in place so that all subsequent trim and promotion work has a verified, non-breaking foundation.
**Depends on**: Nothing (first phase of v2.1; v2.0 shipped)
**Requirements**: CLEAN-01, CLEAN-02, GATE-01, GATE-02
**Success Criteria** (what must be TRUE):
  1. `s_linker13_clean.py` is importable via `run_ablation.py`, registered in `CANONICAL_VARIANTS` / `VARIANT_SPECS`, and produces F1 identical to `s_linker13.py` on Claude Sonnet across all 5 datasets.
  2. Factored helper modules (`helper_v3.py` or equivalent versioned siblings) exist; `s_linker13_clean` imports them rather than inlining; old helper modules remain untouched.
  3. A regression test asserts that every variant in `CANONICAL_VARIANTS` produces F1 matching the v2.0 baseline JSON; the test passes before any further promotion.
  4. The v2.1 cross-model gate (gpt-5.4 macro ≥ 0.9077 within ≤ 1pp tolerance, tolerance value committed) is formally defined and logged in `PROJECT.md` Key Decisions and `STATE.md` Standing Gates.
**Plans**: TBD

### Phase 11: Research
**Goal**: A short, concrete survey of prompt-minimization harness techniques is available to directly inform which rules to trim first and how to redesign prompts in `prompts_v3.py`.
**Depends on**: Phase 10 (scaffolding must exist before trim strategy is actionable)
**Requirements**: PROMPT-05
**Success Criteria** (what must be TRUE):
  1. `.planning/research/PROMPT-HARNESS-SURVEY.md` exists with ≥ 3 concrete techniques (e.g., Chain-of-Thought, ReAct, rubric distillation) each scored for fit to `s_linker13`.
  2. Each technique entry states whether it is compatible with GATE-06 (project-agnostic, no benchmark-derived phrasing) and estimates the expected impact on rule count in the relevant prompts.
  3. The survey concludes with a recommended trim-order or technique prioritization that the Phase 12 ablation can act on directly.
**Plans**: TBD

### Phase 12: Trim Ablation
**Goal**: Every prompt imported by `s_linker13_clean` has had its rule set trimmed as a separate ablation variant; each trim is accepted only if it passes both the Claude Sonnet gate and the gpt-5.4 cross-model gate AND clears the GATE-06 generality audit; rejected trims are documented.
**Depends on**: Phase 11 (survey findings drive trim strategy)
**Requirements**: PROMPT-01, PROMPT-02, PROMPT-04
**Success Criteria** (what must be TRUE):
  1. `prompts_v3.py` exists alongside `prompts_v2.py`; `prompts_v2.py` is unchanged; a committed v2→v3 mapping table documents every prompt as kept / dropped / renamed with rationale.
  2. Each accepted trim variant achieves Claude Sonnet macro F1 ≥ 0.93 (BBB ≤ 6pp below baseline, others ≤ 2pp below baseline) AND gpt-5.4 macro ≥ 0.9077 within the ≤ 1pp tolerance defined in Phase 10.
  3. Each rejected trim is documented in the milestone summary with the specific failing gate and the dataset(s) that failed.
  4. Every retained trim passes GATE-06: a BENCHMARK_TABOO.md scan returns clean AND a reviewer-defensibility note confirms no project-tailored phrasing.
**Plans**: TBD

### Phase 13: Promotion & Wrap
**Goal**: Either `s_linker13_min.py` ships as a promoted, canonically registered variant after passing both gates on a full 5-dataset sweep, or a documented negative result is published — and either way the ABLATION-TABLE is updated with v2.1 rows.
**Depends on**: Phase 12 (trim results determine what can be promoted)
**Requirements**: PROMPT-03, GATE-03
**Success Criteria** (what must be TRUE):
  1. If promotion holds: `s_linker13_min.py` exists, is registered in `CANONICAL_VARIANTS` / `VARIANT_SPECS`, and its full 5-dataset F1 sweep (Claude Sonnet AND gpt-5.4) is on record, passing both gates.
  2. If promotion does not hold: a negative result document is committed explaining which gates failed and on which datasets, framed as a publishable finding per the project thesis.
  3. `ABLATION-TABLE.md` contains a v2.1 addendum row for `s_linker13_clean`, at least one per-trim variant row (kept or rejected), and `s_linker13_min` if promoted; existing v1.0/v2.0 rows are unchanged.
  4. The corresponding `.tex` artifact is regenerated to include all v2.1 rows.
**Plans**: TBD

## Progress

**Execution Order:** 10 → 11 → 12 → 13

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 10. Scaffolding | v2.1 | 0/TBD | Not started | - |
| 11. Research | v2.1 | 0/TBD | Not started | - |
| 12. Trim Ablation | v2.1 | 0/TBD | Not started | - |
| 13. Promotion & Wrap | v2.1 | 0/TBD | Not started | - |
