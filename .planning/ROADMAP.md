# Roadmap: llm-sad-sam-v45

## Milestones

- ✅ **v1.0 — Rule-to-LLM Ablation** (`s_linker12c` → `s_linker13`) — Phases 1–5 — shipped 2026-05-29. Final macro F1 0.9509. See [`milestones/v1.0-ROADMAP.md`](milestones/v1.0-ROADMAP.md).
- ✅ **v2.0 — Complete Rule Removal + Cross-Model — Generality First** — Phases 6–9 — shipped 2026-05-31. EXT-01 closed empty (negative), CROSS evidence published on gpt-5.4. See [`milestones/v2.0-ROADMAP.md`](milestones/v2.0-ROADMAP.md) and [`milestones/v2.0-MILESTONE-AUDIT.md`](milestones/v2.0-MILESTONE-AUDIT.md).
- ✅ **v2.1 — Cleanup + Prompt Simplification** — Phases 10–13 — shipped 2026-06-01. `s_linker13_min` PROMOTED (Claude macro 0.9506, gpt-5.4 macro 0.9069). 3 trims shipped (Step 0 dead-code + trim1 distillation + trim9 runtime seed rubric) + 7 frontier variants documented. Voyager-TLR methodology validated for v2.2. See [`milestones/v2.1-ROADMAP.md`](milestones/v2.1-ROADMAP.md) and [`milestones/v2.1-MILESTONE-AUDIT.md`](milestones/v2.1-MILESTONE-AUDIT.md).
- ✅ **v2.2 — Probe-Wave Trimmed Close** — `s_linker13_min` unchanged + Probe D opt-in (gpt-5.4 only) — shipped 2026-06-01. 4 probes ran; 1 strong survivor (Probe D upstream coref rubric) shipped as opt-in carve-out. Voyager v4 multi-role + per-backend cache infrastructure + Probe A' vocab fix carried to v2.3 as proven prereqs. See [`milestones/v2.2-ROADMAP.md`](milestones/v2.2-ROADMAP.md) and [`milestones/v2.2-MILESTONE-AUDIT.md`](milestones/v2.2-MILESTONE-AUDIT.md).
- ✅ **v2.3 — Trained Multi-Role Prompt Replacement (β architecture)** — Phases 14–19 — shipped 2026-06-01. WEAK verdict (cross-split macro 90.5%, gpt-5.4). `s_linker14_voyager` ships experimental=True. See [`milestones/v2.3-ROADMAP.md`](milestones/v2.3-ROADMAP.md) and [`milestones/v2.3-MILESTONE-AUDIT.md`](milestones/v2.3-MILESTONE-AUDIT.md).
- 🔲 **v2.4 — Probation Gate Fix + Axiom Improvements + v4 Re-run** — Phases 20–24 — in progress 2026-06-01. Fix 3 v2.3 debts (broken gate D-1 + axiom vocabulary ceiling D-2/D-3); re-run β training to close −0.19pp gap to canonical and target STRONG (≥ 0.9173). See [`milestones/v2.4-ROADMAP.md`](milestones/v2.4-ROADMAP.md).

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
<summary>🔲 v2.4 — Probation Gate Fix + Axiom Improvements + v4 Re-run — IN PROGRESS 2026-06-01</summary>

### v2.4 Phase Summary

- [ ] **Phase 20: Infrastructure Prep** — Fix traceability gate (Gate A + Gate B) in `voyager_train_tlr_v4_beta.py`; improve axioms (COREF_RULES + SEED_DISAMBIGUATION_RULES) + code change in `s_linker14_voyager.py`; 5-dataset eval as pre-training baseline. Zero LLM training budget. 3 independent plans.
- [ ] **Phase 21: Probe Tier** — β training with fixed gate + improved axioms on mainline split (2 passes, gpt-5.4, $5–10). Cheap-kill: train macro < 0.87 after pass 2 → document gate fix as finding.
- [ ] **Phase 22: Range Tier** — CONDITIONAL (Phase 21 CONTINUE). Full convergence run (macro ≥ 0.90, max 5 passes), 5-dataset eval, 3-tier verdict (gpt-5.4, $15–25).
- [ ] **Phase 23: Confirmation Tier** — CONDITIONAL (Phase 22 ≥ 0.87). 3-split sweep, cross-split aggregation, final verdict + dual-artifact registration (gpt-5.4, $40–60). Validates split-2 (BBB) commits ≥1 pattern in 3/5 passes.
- [ ] **Phase 24: Milestone Close** — Unconditional. Milestone audit, requirements close-out, archive.

See `.planning/milestones/v2.4-ROADMAP.md` for full phase details, plans, success criteria, and requirement coverage.

</details>

## Progress Table (v2.3 — COMPLETE)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 14. β Training Harness Infrastructure | 6/6 | ✅ Complete | 2026-06-01 |
| 15. Probe Tier | 2/2 | ✅ Complete | 2026-06-01 |
| 16. Range Tier | 2/2 | ✅ Complete — WEAK verdict (macro 89.8%) | 2026-06-01 |
| 17. Confirmation Tier | 2/2 | ✅ Complete — WEAK verdict (cross-split 90.5%) | 2026-06-01 |
| 18. Compact-B Fallback | N/A | Not triggered (Phase 16 ≥ 0.87) | — |
| 19. Milestone Close | 1/1 | ✅ Complete | 2026-06-01 |

## Progress Table (v2.4 — IN PROGRESS)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 20. Infrastructure Prep | 0/3 | Not started | — |
| 21. Probe Tier | 0/2 | Not started (depends on Phase 20) | — |
| 22. Range Tier | 0/TBD | Not started (conditional on Phase 21 CONTINUE) | — |
| 23. Confirmation Tier | 0/TBD | Not started (conditional on Phase 22 ≥ 0.87) | — |
| 24. Milestone Close | 0/TBD | Not started | — |

## Next Milestone

**v2.4 active — Phases 20–24.** Start with Phase 20 (Infrastructure Prep): 3 independent plans (gate redesign, axiom improvements, pre-training eval). No LLM training budget consumed during Phase 20. See `.planning/milestones/v2.4-ROADMAP.md` for full detail.

Kickoff seed (locked decisions): `.planning/v2.4-prep/v2.4-KICKOFF-SEED.md`.
Requirements: `.planning/milestones/v2.4-REQUIREMENTS.md`.
