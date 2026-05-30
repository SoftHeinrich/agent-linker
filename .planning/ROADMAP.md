# Roadmap: llm-sad-sam-v45 Rule-to-LLM Ablation

## Milestones

- ✅ **v1.0 Rule-to-LLM Ablation** — Phases 1-5 (shipped 2026-05-29; re-audit `passed` 2026-05-30)
- 📋 **v2.0** — planned (EXT-01..EXT-04 candidate scope)

## Phases

<details>
<summary>✅ v1.0 Rule-to-LLM Ablation (Phases 1-5) — SHIPPED 2026-05-29</summary>

Final artifact: `s_linker13.py` — macro F1 = 0.9509 (+1.04 pp vs `s_linker12c` baseline 0.9405).

- [x] **Phase 1: Baseline and Infrastructure** (5/5 plans) — completed 2026-05-28. 13a macro 0.9364 under user-loosened BBB 4pp.
- [x] **Phase 2: Ambiguity Cleanup** (2/2 plans) — completed 2026-05-29. 13b macro +0.0114; 13c macro 0.9314 under BBB 6pp.
- [~] **Phase 3: Mention Classifier Migration** (1/1 closed empty) — completed 2026-05-29. VAR-04 retired after 13d TM regression (-19pp from dotted-path FPs); milestone-level negative result.
- [x] **Phase 4: Alias Scope and Coref Fold** (2/2 plans) — completed 2026-05-29. 13e macro 0.9380; 13f macro 0.9509 — best in chain.
- [x] **Phase 5: Promote and Ablation Artifact** (3/3 plans) — completed 2026-05-29. s_linker13 promoted; ABLATION-TABLE.md/.tex; METHODOLOGY.md; BBB-ROOT-CAUSE.md / BBB-DEEP-SEMANTIC-ANALYSIS.md appended 2026-05-30.

See `milestones/v1.0-ROADMAP.md` for full phase details, `milestones/v1.0-REQUIREMENTS.md` for requirements, `milestones/v1.0-MILESTONE-AUDIT.md` for audit report, and `MILESTONES.md` for accomplishments.

</details>

### 📋 v2.0 (Planned)

Candidate scope (deferred from v1.0):

- [ ] **EXT-01** — Spike on replacing `_has_standalone_mention` with LLM primitive (relaxed budget; cost-benefit re-analysis under v2)
- [ ] **EXT-02** — Drop dotted-path guard in `_has_standalone_mention` (narrower follow-up to EXT-01)
- [ ] **EXT-03** — GPT-5.2 cross-model re-evaluation of `s_linker13`
- [ ] **EXT-04** — Emit-biased boundary prompting on alias-discovery to shrink BBB borderline-4 variance band from ~3pp to ~1pp

Scope to be defined via `/gsd-new-milestone`.

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Baseline and Infrastructure | v1.0 | 5/5 | Complete | 2026-05-28 |
| 2. Ambiguity Cleanup | v1.0 | 2/2 | Complete | 2026-05-29 |
| 3. Mention Classifier Migration | v1.0 | 1/1 (closed empty) | Complete (VAR-04 retired) | 2026-05-29 |
| 4. Alias Scope and Coref Fold | v1.0 | 2/2 | Complete | 2026-05-29 |
| 5. Promote and Ablation Artifact | v1.0 | 3/3 | Complete | 2026-05-29 |
