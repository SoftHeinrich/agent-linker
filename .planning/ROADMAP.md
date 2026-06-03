# Roadmap: llm-sad-sam-v45

## Milestones

- ✅ **v1.0 — Rule-to-LLM Ablation** (`s_linker12c` → `s_linker13`) — Phases 1–5 — shipped 2026-05-29. Final macro F1 0.9509. See [`milestones/v1.0-ROADMAP.md`](milestones/v1.0-ROADMAP.md).
- ✅ **v2.0 — Complete Rule Removal + Cross-Model — Generality First** — Phases 6–9 — shipped 2026-05-31. EXT-01 closed empty (negative), CROSS evidence published on gpt-5.4. See [`milestones/v2.0-ROADMAP.md`](milestones/v2.0-ROADMAP.md) and [`milestones/v2.0-MILESTONE-AUDIT.md`](milestones/v2.0-MILESTONE-AUDIT.md).
- ✅ **v2.1 — Cleanup + Prompt Simplification** — Phases 10–13 — shipped 2026-06-01. `s_linker13_min` PROMOTED (Claude macro 0.9506, gpt-5.4 macro 0.9069). 3 trims shipped (Step 0 dead-code + trim1 distillation + trim9 runtime seed rubric) + 7 frontier variants documented. Voyager-TLR methodology validated for v2.2. See [`milestones/v2.1-ROADMAP.md`](milestones/v2.1-ROADMAP.md) and [`milestones/v2.1-MILESTONE-AUDIT.md`](milestones/v2.1-MILESTONE-AUDIT.md).
- ✅ **v2.2 — Probe-Wave Trimmed Close** — `s_linker13_min` unchanged + Probe D opt-in (gpt-5.4 only) — shipped 2026-06-01. 4 probes ran; 1 strong survivor (Probe D upstream coref rubric) shipped as opt-in carve-out. Voyager v4 multi-role + per-backend cache infrastructure + Probe A' vocab fix carried to v2.3 as proven prereqs. See [`milestones/v2.2-ROADMAP.md`](milestones/v2.2-ROADMAP.md) and [`milestones/v2.2-MILESTONE-AUDIT.md`](milestones/v2.2-MILESTONE-AUDIT.md).
- ✅ **v2.3 — Trained Multi-Role Prompt Replacement (β architecture)** — Phases 14–19 — shipped 2026-06-01. WEAK verdict (cross-split macro 90.5%, gpt-5.4). `s_linker14_voyager` ships experimental=True. See [`milestones/v2.3-ROADMAP.md`](milestones/v2.3-ROADMAP.md) and [`milestones/v2.3-MILESTONE-AUDIT.md`](milestones/v2.3-MILESTONE-AUDIT.md).
- ✅ **v2.4 — Probation Gate Fix + Axiom Improvements + v4 Re-run** — Phases 20–24 — shipped 2026-06-01. WEAK verdict (cross-split macro 90.5%, gpt-5.4). Oracle cache contamination + slot steering gap identified; v2.5 scoped to fix both. See [`milestones/v2.4-ROADMAP.md`](milestones/v2.4-ROADMAP.md) and [`milestones/v2.4-MILESTONE-AUDIT.md`](milestones/v2.4-MILESTONE-AUDIT.md).
- ✅ **v2.5 — Oracle Cache Fix + 15-Slot Expansion + Re-run** — Phases 25–30 — shipped 2026-06-02. WEAK verdict (cross-split macro 89.1%, gpt-5.4). Oracle cache fix validated (split-2 committed Pass 1); 15-slot expansion: 5/6 new slots with committed patterns. See [`milestones/v2.5-ROADMAP.md`](milestones/v2.5-ROADMAP.md) and [`milestones/v2.5-MILESTONE-AUDIT.md`](milestones/v2.5-MILESTONE-AUDIT.md).
- ❄️ **v2.6 — ILinker4 + LLM-Driven Training + Axiom Re-run** — Phases 31–37 — close FROZEN behind v2.6.1 (2026-06-02). Phase 37 close tasks (GATE-06 'Persistence' fix + audit) deferred. See [`milestones/v2.6-ROADMAP.md`](milestones/v2.6-ROADMAP.md).
- ✅ **v2.6.1 — No-Training Axiom Linker (s_linker15) + Axiom FP Fixes (PATCH)** — shipped 2026-06-03. Dropped Voyager training; `s_linker15` = axiom-only standalone (inlined prompts + 3 FP fixes), no bank. macro 89.1% gpt-5.4 / 92.7% Claude. Finding: training adds nothing (s15 = trained s14 on gpt); FP fixes fire on Claude, inert on gpt. See [`milestones/v2.6.1-ROADMAP.md`](milestones/v2.6.1-ROADMAP.md) and [`milestones/v2.6.1-MILESTONE-AUDIT.md`](milestones/v2.6.1-MILESTONE-AUDIT.md).
- ✅ **v2.6.2 — Multi-Framing Extraction Design (s_linker17a/17b)** — shipped 2026-06-03. 17a (rename-only) ≈ s15 within GPT variance (validates ICSE Framing A/B/C naming). 17b (k=2 unified) regresses TM −4.2pp / BBB −7.4pp (k=2 too conservative). ICSE decision: use 17a naming for paper. See [`milestones/v2.6.2-ROADMAP.md`](milestones/v2.6.2-ROADMAP.md) and [`milestones/v2.6.2-MILESTONE-AUDIT.md`](milestones/v2.6.2-MILESTONE-AUDIT.md).
- ❄️ **v2.7 — BBB Recall Closure** — Phases 38–42 — FROZEN behind v2.6.2 (2026-06-03). ⚠ Phases 40–41 (recall-oracle training redesign + training re-runs) need re-evaluation vs the v2.6.1 no-training finding before execution. See [`milestones/v2.7-ROADMAP.md`](milestones/v2.7-ROADMAP.md).

## Phases

<details>
<summary>✅ v1.0 — Rule-to-LLM Ablation (Phases 1–5) — SHIPPED 2026-05-29</summary>

Phases 1–5 complete. See `milestones/v1.0-ROADMAP.md` for full detail.

</details>

<details>
<summary>✅ v2.0 — Complete Rule Removal + Cross-Model (Phases 6–9) — SHIPPED 2026-05-31</summary>

Phases 6–9 complete. See `milestones/v2.0-ROADMAP.md` for full detail.

</details>

<details>
<summary>✅ v2.1 — Cleanup + Prompt Simplification (Phases 10–13) — SHIPPED 2026-06-01</summary>

Phases 10–13 complete. See `milestones/v2.1-ROADMAP.md` for full detail.

</details>

<details>
<summary>✅ v2.2 — Probe-Wave Trimmed Close — SHIPPED 2026-06-01</summary>

Probe wave (4 mechanisms) + trimmed close. No new canonical promoted; `s_linker13_min` carried forward unchanged. Probe D ships as opt-in gpt-5.4-only carve-out (`s_linker14_probe_d_upstream_clean`). See `milestones/v2.2-ROADMAP.md`.

</details>

<details>
<summary>✅ v2.3 — Trained Multi-Role Prompt Replacement (β architecture) — SHIPPED 2026-06-01 — WEAK verdict (90.5%)</summary>

### v2.3 Phase Summary

- [x] **Phase 14: β Training Harness Infrastructure** — All code (L/O/D/P modules, bank schema, cache adapter, `s_linker14_voyager` linker, GATE-06 helpers) implemented and unit-tested. Zero LLM budget consumed. (completed 2026-06-01)
- [x] **Phase 15: Probe Tier** — 1–2 outer passes on mainline split (gpt-5.4, $5–10). Verdict: CONTINUE (train macro 0.9152 after pass 1). (completed 2026-06-01)
- [x] **Phase 16: Range Tier** — Train to convergence (macro ≥ 0.90 or pass 5 cap), 5-dataset evaluation, 3-tier verdict. (completed 2026-06-01, verdict=WEAK, macro=89.8%)
- [x] **Phase 17: Confirmation Tier** — 3-split sweep, cross-split aggregation, final evaluation, dual-artifact registration. (completed 2026-06-01, verdict=WEAK, cross-split macro=90.5%)
- [x] **Phase 19: Milestone Close** — Milestone audit, requirements close-out, archive. (completed 2026-06-01)

Key findings: split-fragility (BBB in training → split2 empty bank), minimal cross-split consensus (2 patterns survive), probation gate broken (6 bugs). 3 debts carried to v2.4. See `.planning/milestones/v2.3-ROADMAP.md` and `.planning/milestones/v2.3-MILESTONE-AUDIT.md`.

</details>

<details>
<summary>✅ v2.4 — Probation Gate Fix + Axiom Improvements + v4 Re-run — SHIPPED 2026-06-01 — WEAK verdict (90.5%)</summary>

### v2.4 Phase Summary

- [x] **Phase 20: Infrastructure Prep** — Fixed traceability gate (Gate A + Gate B), improved axioms (COREF_RULES + SEED_DISAMBIGUATION_RULES), 5-dataset pre-training baseline eval. (completed 2026-06-01)
- [x] **Phase 21: Probe Tier** — β training with fixed gate + improved axioms on mainline split. Verdict: CONTINUE. (completed 2026-06-01)
- [x] **Phase 22: Range Tier** — Full convergence run; WEAK verdict (macro 90.5%). (completed 2026-06-01)
- [x] **Phase 23: Confirmation Tier** — 3-split sweep; WEAK verdict (cross-split 90.5%). Split-2 zero commits — oracle cache contamination identified as root cause. (completed 2026-06-01)
- [x] **Phase 24: Milestone Close** — Milestone audit, requirements close-out, archive. (completed 2026-06-01)

Key findings: oracle cache contamination (all splits got identical D proposals), slot steering gap (ENTITY_EXTRACTION_RULES never proposed), 2 infrastructure bugs catalogued for v2.5. See `.planning/milestones/v2.4-ROADMAP.md` and `.planning/milestones/v2.4-MILESTONE-AUDIT.md`.

</details>

<details>
<summary>✅ v2.5 — Oracle Cache Fix + 15-Slot Expansion + Re-run — SHIPPED 2026-06-02 — WEAK verdict (89.1%)</summary>

### v2.5 Phase Summary

- [x] **Phase 25: Infrastructure Fixes** — Oracle cache key fix (`bank_content_hash`), probation threshold raised to `delta >= 0.005`, D prompt underfilled-slot steering. (completed 2026-06-02)
- [x] **Phase 26: 15-Slot Expansion** — 6 new slot constants in `prompts_v3_axiom.py`, `ILinker3Injected` subclass, 4 inline prompts wired as slots, harness updated for 15-slot schema. (completed 2026-06-02)
- [x] **Phase 27: Probe Tier** — 2-pass mainline run; oracle fix validated; CONTINUE verdict. (completed 2026-06-02)
- [x] **Phase 28: Range Tier** — Convergence run; WEAK verdict (macro 89.3%); 5/6 new slots received patterns. (completed 2026-06-02)
- [x] **Phase 29: Confirmation Tier** — 3-split sweep; WEAK verdict (cross-split macro 89.1%); split-2 committed 12 patterns (oracle fix confirmed). (completed 2026-06-02)
- [x] **Phase 30: Milestone Close** — Milestone audit, requirements close-out, archive. (completed 2026-06-02)

Key findings: oracle cache fix validated (split-2 committed 12 patterns vs 0/5 in v2.4), 15-slot expansion: 5/6 slots populated (SEED_EXTRACTION_RULES + SEED_ACTOR_RULES empty — ILinker4 needed), lift +1.5pp over axiom-only (87.6%). Debts carried to v2.6: ILinker4, LLM Assessor gate redesign, 3 axiom gaps. See `.planning/milestones/v2.5-ROADMAP.md` and `.planning/milestones/v2.5-MILESTONE-AUDIT.md`.

</details>

<details>
<summary>❄️ v2.6 — ILinker4 + LLM-Driven Training + Axiom Re-run — FROZEN (close deferred)</summary>

### v2.6 Phase Summary

- [x] **Phase 31: ILinker4 + Prompt Hygiene** — `ilinker4.py` built (Voyager-native standalone, first-class SEED slots); 0 inline behavioral rules needing migration; wired into `s_linker14_voyager.py`. GATE-06 clean. (completed 2026-06-02)
- [x] **Phase 32: LLM-Driven Training Loop (v5)** — `voyager_train_tlr_v5.py` built: OD role (O+D merged), LLM Assessor (accept/reject/revise, max 1 revision), [TRAIN]/[TEST] log separation, `run_confirmation` with axiom-only baseline. dry-run verified. (completed 2026-06-02)
- [x] **Phase 33: Axiom Gap Fixes** — `prompts_v4_axiom.py` created; COREF_RULES (Gap 1 SCN + Gap 3 alias), SEED_DISAMBIGUATION_RULES (Gap 2 gerund), ALIAS_SCOPE_RULES + ANTECEDENT_ALIAS_RULES axiomized; s_linker14_voyager imports v4. (completed 2026-06-02)
- [x] **Phase 34: Probe Tier** — KILL verdict. [TEST] macro=84.86% < 87% threshold. Assessor active (9 decisions pass 1). v4 Gap 2 gerund rule over-aggressive (teammates train 81.97%). Phases 35+36 SKIPPED. (completed 2026-06-02)
- [ ] **Phase 35: Range Tier** — CONDITIONAL (Phase 34 CONTINUE). Convergence run, max 5 passes, 5-dataset eval, 3-tier verdict. Budget ≤ $25.
- [ ] **Phase 36: Confirmation Tier** — CONDITIONAL (Phase 35 ≥ 0.87). 3-split cross-validation, axiom-only baseline per held-out, final table vs v2.5 (89.1%) and canonical (90.69%). Budget ≤ $60.
- [ ] **Phase 37: Milestone Close** — Unconditional. Milestone audit, requirements close-out, archive.

See `.planning/milestones/v2.6-ROADMAP.md` for full phase details, plans, success criteria, and requirement coverage.

</details>

## Progress Table (v2.5 — COMPLETE)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 25. Infrastructure Fixes | 1/1 | ✅ Complete | 2026-06-02 |
| 26. 15-Slot Expansion | 1/1 | ✅ Complete | 2026-06-02 |
| 27. Probe Tier | 1/1 | ✅ Complete | 2026-06-02 |
| 28. Range Tier | 1/1 | ✅ Complete — WEAK verdict (macro 89.3%) | 2026-06-02 |
| 29. Confirmation Tier | 1/1 | ✅ Complete — WEAK verdict (cross-split 89.1%) | 2026-06-02 |
| 30. Milestone Close | 1/1 | ✅ Complete | 2026-06-02 |

## Progress Table (v2.6 — IN PROGRESS)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 31. ILinker4 + Prompt Hygiene | 1/1 | ✅ Complete | 2026-06-02 |
| 32. LLM-Driven Training Loop (v5) | 1/1 | ✅ Complete | 2026-06-02 |
| 33. Axiom Gap Fixes | 1/1 | ✅ Complete | 2026-06-02 |
| 34. Probe Tier | 1/1 | ✅ Complete — KILL verdict ([TEST] 84.86% < 87%) | 2026-06-02 |
| 35. Range Tier | — | ⏭ SKIPPED (Phase 34 KILL) | — |
| 36. Confirmation Tier | — | ⏭ SKIPPED (Phase 34 KILL) | — |
| 35. Range Tier | 0/TBD | Not started (conditional on Phase 34 CONTINUE) | — |
| 36. Confirmation Tier | 0/TBD | Not started (conditional on Phase 35 ≥ 0.87) | — |
| 37. Milestone Close | 0/TBD | Not started | — |

## Next Milestone

**v2.6 active — Phases 31–37.** Start with Phase 31 (ILinker4 + Prompt Hygiene): build `ilinker4.py` as a standalone Voyager-native seed extractor and audit all static prompts. No LLM training budget consumed during Phase 31. See `.planning/milestones/v2.6-ROADMAP.md` for full detail.

Requirements: `.planning/REQUIREMENTS.md` (v2.6 active section).
