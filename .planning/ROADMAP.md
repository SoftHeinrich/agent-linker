# Roadmap: llm-sad-sam-v45

## Milestones

- ✅ **v1.0 — Rule-to-LLM Ablation** (`s_linker12c` → `s_linker13`) — Phases 1–5 — shipped 2026-05-29. Final macro F1 0.9509. See [`milestones/v1.0-ROADMAP.md`](milestones/v1.0-ROADMAP.md).
- ✅ **v2.0 — Complete Rule Removal + Cross-Model — Generality First** — Phases 6–9 — shipped 2026-05-31. EXT-01 closed empty (negative), CROSS evidence published on gpt-5.4. See [`milestones/v2.0-ROADMAP.md`](milestones/v2.0-ROADMAP.md) and [`milestones/v2.0-MILESTONE-AUDIT.md`](milestones/v2.0-MILESTONE-AUDIT.md).
- ✅ **v2.1 — Cleanup + Prompt Simplification** — Phases 10–13 — shipped 2026-06-01. `s_linker13_min` PROMOTED (Claude macro 0.9506, gpt-5.4 macro 0.9069). 3 trims shipped (Step 0 dead-code + trim1 distillation + trim9 runtime seed rubric) + 7 frontier variants documented. Voyager-TLR methodology validated for v2.2. See [`milestones/v2.1-ROADMAP.md`](milestones/v2.1-ROADMAP.md) and [`milestones/v2.1-MILESTONE-AUDIT.md`](milestones/v2.1-MILESTONE-AUDIT.md).
- ✅ **v2.2 — Probe-Wave Trimmed Close** — `s_linker13_min` unchanged + Probe D opt-in (gpt-5.4 only) — shipped 2026-06-01. 4 probes ran; 1 strong survivor (Probe D upstream coref rubric) shipped as opt-in carve-out. Voyager v4 multi-role + per-backend cache infrastructure + Probe A' vocab fix carried to v2.3 as proven prereqs. See [`milestones/v2.2-ROADMAP.md`](milestones/v2.2-ROADMAP.md) and [`milestones/v2.2-MILESTONE-AUDIT.md`](milestones/v2.2-MILESTONE-AUDIT.md).
- 🔲 **v2.3 — Trained Multi-Role Prompt Replacement (β architecture)** — Phases 14–19 — in progress 2026-06-01. β harness (L + O + D-with-CoT-A + P) trains per-slot JSON bank on gpt-5.4; `s_linker14_voyager` (experimental=True) ships finding regardless of polarity. See [`milestones/v2.3-ROADMAP.md`](milestones/v2.3-ROADMAP.md).

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
<summary>🔲 v2.3 — Trained Multi-Role Prompt Replacement (β architecture) — IN PROGRESS 2026-06-01</summary>

### v2.3 Phase Summary

- [ ] **Phase 14: β Training Harness Infrastructure** — All code (L/O/D/P modules, bank schema, cache adapter, `s_linker14_voyager` linker, GATE-06 helpers) implemented and unit-tested. Zero LLM budget consumed. Covers REQ-V23-01 through REQ-V23-04, REQ-V23-09 through REQ-V23-12, GATE-02, GATE-06, GATE-07.
- [ ] **Phase 15: Probe Tier** — 1–2 outer passes on mainline split (gpt-5.4, $5–10). Cheap-kill gate: train macro < 0.87 after pass 2 → kill v4 → Phase 18. Covers REQ-V23-07, REQ-V23-13, REQ-V23-14 (Probe).
- [ ] **Phase 16: Range Tier** — Train to convergence (macro ≥ 0.90 or pass 5 cap), 5-dataset evaluation, 3-tier verdict (gpt-5.4, $15–25). Covers REQ-V23-05, REQ-V23-07, REQ-V23-13, REQ-V23-14 (Range), REQ-V23-15.
- [ ] **Phase 17: Confirmation Tier** — CONDITIONAL (Phase 16 ≥ 0.87). 3-split sweep, cross-split aggregation, final evaluation, dual-artifact registration, ABLATION-TABLE update (gpt-5.4, $40–60). Covers REQ-V23-06, REQ-V23-07, REQ-V23-08 (pass), REQ-V23-14 (Confirmation), REQ-V23-15, GATE-01, GATE-07, GATE-08.
- [ ] **Phase 18: Compact-B Fallback** — CONDITIONAL (Phase 15 or 16 KILL). Implement + probe + range Compact-B (R345 single CoT role); ship positive or negative finding artifact ($10–20). Covers REQ-V23-08 (fail), REQ-V23-14 (Compact-B).
- [ ] **Phase 19: Milestone Close** — Unconditional. Milestone audit, requirements close-out, archive. Covers REQ-V23-05 (audit), GATE-01, GATE-08.

See `.planning/milestones/v2.3-ROADMAP.md` for full phase details, plans, success criteria, and requirement coverage.

</details>

## Progress Table (v2.3)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 14. β Training Harness Infrastructure | 6/6 | ✅ Complete | 2026-06-01 |
| 15. Probe Tier | 0/2 | Planned (2 plans, Wave 1 + Wave 2) | - |
| 16. Range Tier | 0/TBD | Not started (conditional on Ph 15 CONTINUE) | - |
| 17. Confirmation Tier | 0/TBD | Not started (conditional on Ph 16 ≥ 0.87) | - |
| 18. Compact-B Fallback | 0/TBD | Not started (conditional on Ph 15/16 KILL) | - |
| 19. Milestone Close | 0/TBD | Not started | - |

## Next Milestone

**v2.3 active — Phases 14–19.** Start with Phase 14 (β Training Harness Infrastructure): all code deliverables before any LLM budget is spent. See `.planning/milestones/v2.3-ROADMAP.md` for full detail.

Architecture spec: `.planning/v2.3-prep/v2.3-ARCHITECTURE.md`.
Kickoff seed (resolved decisions): `.planning/v2.3-prep/v2.3-KICKOFF-SEED.md`.
Requirements: `.planning/REQUIREMENTS.md`.
