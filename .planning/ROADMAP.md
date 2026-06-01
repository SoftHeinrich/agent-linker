# Roadmap: llm-sad-sam-v45

## Milestones

- ✅ **v1.0 — Rule-to-LLM Ablation** (`s_linker12c` → `s_linker13`) — Phases 1–5 — shipped 2026-05-29. Final macro F1 0.9509. See [`milestones/v1.0-ROADMAP.md`](milestones/v1.0-ROADMAP.md).
- ✅ **v2.0 — Complete Rule Removal + Cross-Model — Generality First** — Phases 6–9 — shipped 2026-05-31. EXT-01 closed empty (negative), CROSS evidence published on gpt-5.4. See [`milestones/v2.0-ROADMAP.md`](milestones/v2.0-ROADMAP.md) and [`milestones/v2.0-MILESTONE-AUDIT.md`](milestones/v2.0-MILESTONE-AUDIT.md).
- ✅ **v2.1 — Cleanup + Prompt Simplification** — Phases 10–13 — shipped 2026-06-01. Final canonical: `s_linker13_min` (Claude macro 0.9506, gpt-5.4 macro 0.9069). 10/10 requirements complete; 4 standing gates held. See `.planning/phases/13-promotion-wrap/13-03-MILESTONE-SUMMARY.md`.

## Phases

<details>
<summary>✅ v1.0 — Rule-to-LLM Ablation (Phases 1–5) — SHIPPED 2026-05-29</summary>

Phases 1–5 complete. See `milestones/v1.0-ROADMAP.md` for full detail.

</details>

<details>
<summary>✅ v2.0 — Complete Rule Removal + Cross-Model (Phases 6–9) — SHIPPED 2026-05-31</summary>

Phases 6–9 complete. See `milestones/v2.0-ROADMAP.md` for full detail.

</details>

### ✅ v2.1 — Cleanup + Prompt Simplification (Shipped 2026-06-01)

**Milestone Goal:** Produce a slimmed standalone variant (`s_linker13_clean` / `s_linker13_min`) using new versioned helpers and `prompts_v3.py`, gated by Claude Sonnet macro F1 ≥ 0.93 AND gpt-5.4 macro ≥ 0.9077 within ≤ 1pp tolerance. Nothing currently runnable breaks.

- [x] **Phase 10: Scaffolding** — Standalone clean variant, versioned helpers, frozen-compat regression test, and cross-model gate definition
- [x] **Phase 11: Research** — Literature + web survey on prompt minimization techniques to inform trim strategy
- [x] **Phase 12: Trim Ablation** — Per-prompt rule-trim variants in `prompts_v3.py` gated by Claude Sonnet AND gpt-5.4 floors; generality re-audit of every retained trim
- [x] **Phase 13: Promotion & Wrap** — Final `s_linker13_min.py` PROMOTED on both backends + ABLATION-TABLE v2.1 addendum (+.tex regen)

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
**Plans**: 4 plans
- [x] 10-01-baseline-regression-test-PLAN.md — snapshot v2.0 baseline JSON fixture + GATE-02 regression test (Wave 1)
- [x] 10-02-helper-v3-extraction-PLAN.md — extract pure helpers from s_linker13 into helper_v3.py versioned sibling (Wave 1) ✅ 2026-05-31
- [ ] 10-03-s-linker13-clean-PLAN.md — standalone s_linker13_clean.py + CANONICAL_VARIANTS/VARIANT_SPECS registration + 5-dataset parity sweep vs s_linker13 (Wave 2, depends on 10-02)
- [x] 10-04-cross-model-gate-codify-PLAN.md — codify GATE-01 cross-model T=1.0pp in PROJECT.md Key Decisions + STATE.md Standing Gates (Wave 1)

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
**Plans**: 7 plans
Plans:
- [x] 12-00-gpt54-baseline-sweep-PLAN.md — one-time gpt-5.4 baseline sweep on s_linker13_clean to establish cross-model anchor (Wave 1)
- [x] 12-01-prompts-v3-scaffold-PLAN.md — create prompts_v3.py with 9 active constants + s_linker13_clean_v3 thin sibling + v2→v3 mapping table (Wave 1)
- [x] 12-02-single-step-ablation-harness-PLAN.md — checkpoint-loaded single-step ablation engine + CLI + dependency contract (Wave 1)
- [ ] 12-03-step1-judge-trim-PLAN.md — Step 1 trim: distill DOC_KNOWLEDGE_JUDGE_RULES via Technique 3 + 8; ablate 5 datasets × 2 backends (Wave 2)
- [ ] 12-04-step2-entval-merge-PLAN.md — Step 2 trim: merge ENTITY_EXTRACTION_RULES + VALIDATION_RULES via Technique 3; ablate 5 datasets × 2 backends (Wave 2)
- [ ] 12-05-step3-runtime-rubric-PLAN.md — Step 3 trim: replace DOC_KNOWLEDGE_JUDGE_RULES with inference-time rubric builder (supplement Techniques 2+3); ablate 5 datasets × 2 backends (Wave 2)
- [ ] 12-06-gate06-defensibility-audit-PLAN.md — full BENCHMARK_TABOO sweep + reviewer-defensibility audit on every retained trim + FINAL v2→v3 mapping (Wave 3)

### Phase 13: Promotion & Wrap
**Goal**: Either `s_linker13_min.py` ships as a promoted, canonically registered variant after passing both gates on a full 5-dataset sweep, or a documented negative result is published — and either way the ABLATION-TABLE is updated with v2.1 rows.
**Depends on**: Phase 12 (trim results determine what can be promoted)
**Requirements**: PROMPT-03, GATE-03
**Success Criteria** (what must be TRUE):
  1. If promotion holds: `s_linker13_min.py` exists, is registered in `CANONICAL_VARIANTS` / `VARIANT_SPECS`, and its full 5-dataset F1 sweep (Claude Sonnet AND gpt-5.4) is on record, passing both gates.
  2. If promotion does not hold: a negative result document is committed explaining which gates failed and on which datasets, framed as a publishable finding per the project thesis.
  3. `ABLATION-TABLE.md` contains a v2.1 addendum row for `s_linker13_clean`, at least one per-trim variant row (kept or rejected), and `s_linker13_min` if promoted; existing v1.0/v2.0 rows are unchanged.
  4. The corresponding `.tex` artifact is regenerated to include all v2.1 rows.
**Plans**: 3 plans
- [x] 13-01-s-linker13-min-promotion-PLAN.md — Compose trim1 + trim9 → s_linker13_min; 5-dataset sweep both backends; promote on PASS (Wave 1) ✅ 2026-06-01
- [x] 13-02-ablation-table-addendum-PLAN.md — Append v2.1 rows to ABLATION-TABLE.md + regenerate .tex (Wave 2) ✅ 2026-06-01
- [x] 13-03-milestone-summary-PLAN.md — Write milestone v2.1 summary + Phase 13 VERIFICATION + STATE update (Wave 2) ✅ 2026-06-01

## Progress

**Execution Order:** 10 → 11 → 12 → 13

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 10. Scaffolding | v2.1 | 4/4 | Complete | 2026-05-31 |
| 11. Research | v2.1 | 1/1 | Complete | 2026-05-31 |
| 12. Trim Ablation | v2.1 | 13/13 | Complete | 2026-06-01 |
| 13. Promotion & Wrap | v2.1 | 3/3 | Complete | 2026-06-01 |
