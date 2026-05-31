# Requirements: llm-sad-sam-v45 — Milestone v2.1

**Defined:** 2026-05-31
**Milestone:** v2.1 Cleanup + Prompt Simplification
**Core Value:** Every rule removed from `s_linker13`/its prompts must hold macro F1 ≥ 0.93 on Claude Sonnet AND gpt-5.4 macro within tolerance of v2.0 baseline (0.9077) — or be rejected. Every retained prompt + helper must read as project-agnostic to a reviewer (GATE-06). Nothing currently runnable may break.

## v2.1 Requirements

Requirements for this milestone. Each maps to roadmap phases.

### CLEAN — Code Cleanup

- [ ] **CLEAN-01**: Standalone `s_linker13_clean.py` variant ships, importable via `run_ablation.py` and registered in `CANONICAL_VARIANTS` / `VARIANT_SPECS`. Original `s_linker13.py` remains frozen and untouched, continues to import `prompts_v2`.
- [ ] **CLEAN-02**: Factored helper modules (`helper_v3.py`, `helper_v4.py`, …) carry the cleaned helpers grouped by concern. `s_linker13_clean` imports them rather than inlining. Old helper modules stay live so existing variants continue importing them unchanged.

### PROMPT — Prompt Simplification

- [ ] **PROMPT-01**: `prompts_v3.py` ships side-by-side with `prompts_v2.py`. Only prompts actually invoked by `s_linker13_clean` carry over. `prompts_v2.py` left untouched. A v2→v3 mapping table is committed (prompt name → kept / dropped / renamed + rationale).
- [ ] **PROMPT-02**: Per-prompt rule-trim ablation — every prompt's rule set trimmed as a separate variant under GATE-01 (Claude) AND the v2.1 cross-model gate (gpt-5.4 macro ≥ 0.9077 within tolerance). Trims accepted only if BOTH gates hold. Rejected trims documented in the milestone summary with the failing dataset(s).
- [ ] **PROMPT-03**: Final minimal-prompt variant `s_linker13_min.py` ships only after a full 5-dataset sweep on Claude Sonnet AND gpt-5.4 passes both gates. If neither configuration holds, the milestone publishes a negative result instead of promoting.
- [ ] **PROMPT-04**: Generality re-audit (GATE-06 + `BENCHMARK_TABOO.md` scan + reviewer-defensibility check) on every retained trim. Any trim that introduces benchmark-derived phrasing is rejected regardless of F1.
- [ ] **PROMPT-05**: Literature + web survey on "reduce rules, let LLM reason more" prompt-minimization harness techniques (Chain-of-Thought, ReAct, deliberation/self-consistency, rubric distillation, plan-then-execute, etc.). Output: short report under `.planning/research/PROMPT-HARNESS-SURVEY.md` with ≥ 3 concrete techniques scored for fit-to-`s_linker13`. Findings inform PROMPT-02 trim strategy and prompt redesigns in `prompts_v3`.

### GATE — Validation Infra

- [ ] **GATE-01**: Cross-model gate codified — `gpt-5.4 macro ≥ 0.9077 within tolerance T` (T defined in milestone, e.g. ≤ 1pp regression) formalized as a standing v2.1 gate. Logged in `PROJECT.md` Key Decisions table and `STATE.md` Standing Gates section.
- [ ] **GATE-02**: Frozen-compat regression test that asserts every variant in `CANONICAL_VARIANTS` produces F1 identical to the v2.0 baseline JSON. Runs before any promotion. Test lives next to existing test infra and is wired into `run_ablation.py` or a dedicated entry point.
- [ ] **GATE-03**: `ABLATION-TABLE.md` addendum + `.tex` artifact regenerated to include v2.1 rows (`s_linker13_clean`, per-trim variants kept after PROMPT-02, and `s_linker13_min` if shipped). Existing v1.0/v2.0 rows unchanged.

## Deferred to Future Milestones

Tracked but not in v2.1 roadmap.

### EXT (carried from v1.0/v2.0)

- **EXT-04**: Emit-biased boundary prompting on alias-discovery (BBB variance band tightening 3pp → 1pp). Variance work, not rule removal.
- **EXT-upstream**: Upstream-tier rule removal targeting extraction/coref tier where v2.0 EXT-01 evidence located the BBB recall gap.

### ADAPTER

- **ADAPTER-01**: Multi-model adapter exploration — project-agnostic backend-adaptive harness layer. Requires fresh GATE-06 thinking.

## Out of Scope

Explicitly excluded from v2.1.

| Feature | Reason |
|---------|--------|
| Changes to `s_linker13.py` itself | Frozen as v2.0 production artifact. Cleanup ships as new `s_linker13_clean.py` next to it. |
| Changes to `prompts_v2.py` | Frozen for ablation chain compatibility. All trim work lands in `prompts_v3.py`. |
| Changes to old helper modules (`data_types_v2`, `document_loader_v2`, `pcm_parser_v2`, `ilinker*`) | Frozen for back-compat. Cleanup lands in new versioned helper modules. |
| New seed/linker approaches (ILinker3+, new pipelines) | Not a rule-reduction or cleanup task; would re-open thesis scope. |
| Non-SAD-SAM tasks (SAM-Code, SAD-Code) | Out of dataset scope (carry from v1.0/v2.0). |
| Cost optimization | User policy unchanged: no LLM budget limit; rule-replaceability/cleanness is the constraint. |
| GPT-5.2 (separate from gpt-5.4) re-evaluation | v2.1 cross-model gate uses gpt-5.4 baseline already established in v2.0. |
| EXT-04 variance-band work | Deferred — variance work, not cleanup/trimming. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CLEAN-01 | Phase 10 | Pending |
| CLEAN-02 | Phase 10 | Pending |
| GATE-01 | Phase 10 | Pending |
| GATE-02 | Phase 10 | Pending |
| PROMPT-05 | Phase 11 | Pending |
| PROMPT-01 | Phase 12 | Pending |
| PROMPT-02 | Phase 12 | Pending |
| PROMPT-04 | Phase 12 | Pending |
| PROMPT-03 | Phase 13 | Pending |
| GATE-03 | Phase 13 | Pending |

**Coverage:**
- v2.1 requirements: 10 total
- Mapped to phases: 10 (100%)
- Unmapped: 0

---
*Requirements defined: 2026-05-31*
*Last updated: 2026-05-31 — v2.1 roadmap created*
