# Roadmap: llm-sad-sam-v45

## Milestones

- ✅ **v1.0 — Rule-to-LLM Ablation** (`s_linker12c` → `s_linker13`) — Phases 1–5 — shipped 2026-05-29. Final macro F1 0.9509. See [`milestones/v1.0-ROADMAP.md`](milestones/v1.0-ROADMAP.md).
- ✅ **v2.0 — Complete Rule Removal + Cross-Model — Generality First** — Phases 6–9 — shipped 2026-05-31. EXT-01 closed empty (negative), CROSS evidence published on gpt-5.4. See [`milestones/v2.0-ROADMAP.md`](milestones/v2.0-ROADMAP.md) and [`milestones/v2.0-MILESTONE-AUDIT.md`](milestones/v2.0-MILESTONE-AUDIT.md).
- ✅ **v2.1 — Cleanup + Prompt Simplification** — Phases 10–13 — shipped 2026-06-01. `s_linker13_min` PROMOTED (Claude macro 0.9506, gpt-5.4 macro 0.9069). 3 trims shipped (Step 0 dead-code + trim1 distillation + trim9 runtime seed rubric) + 7 frontier variants documented. Voyager-TLR methodology validated for v2.2. See [`milestones/v2.1-ROADMAP.md`](milestones/v2.1-ROADMAP.md) and [`milestones/v2.1-MILESTONE-AUDIT.md`](milestones/v2.1-MILESTONE-AUDIT.md).
- ✅ **v2.2 — Probe-Wave Trimmed Close** — `s_linker13_min` unchanged + Probe D opt-in (gpt-5.4 only) — shipped 2026-06-01. 4 probes ran; 1 strong survivor (Probe D upstream coref rubric) shipped as opt-in carve-out. Voyager v4 multi-role + per-backend cache infrastructure + Probe A' vocab fix carried to v2.3 as proven prereqs. See [`milestones/v2.2-ROADMAP.md`](milestones/v2.2-ROADMAP.md) and [`milestones/v2.2-MILESTONE-AUDIT.md`](milestones/v2.2-MILESTONE-AUDIT.md).

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

## Next Milestone

**v2.3 — Voyager v4 Multi-Role (anchored).** v2.2 archived 2026-06-01. v2.3 anchor: Voyager v4 multi-role architecture (R1–R5) deferred from v2.2 with two PROVEN prerequisites carried forward — DO NOT re-explore:

1. **Per-backend cache infrastructure** — per-(text_stem, comp_hash, backend, model) cache-key for runtime LLM rubrics. Verified SANITY_PASS in `s_linker14_probe_d_upstream_clean.py` (Probe D cache-fix wave 2026-06-01). See [`v2.2-prep/probe-D-cachekey-fix-SUMMARY.md`](v2.2-prep/probe-D-cachekey-fix-SUMMARY.md).
2. **Probe A' vocab-aligned R3** — discourse/syntactic vocabulary (subject-position, anaphora, qualifier clause, ...) replaces textbook SE vocabulary, narrowing the R3/R5 deadlock that falsified original Probe A. Mediastore STRONG_PASS (+1.69pp); BBB WEAK_PASS (R5 0/8, F1 -0.24pp) — v4 is mediastore-viable, BBB-inactive on gpt-5.4. See [`v2.2-prep/probe-A-prime-vocab-aligned-SUMMARY.md`](v2.2-prep/probe-A-prime-vocab-aligned-SUMMARY.md) and [`v2.2-prep/v2.2-RANGE-A-PRIME-BBB-AND-CACHE-FIX-SUMMARY.md`](v2.2-prep/v2.2-RANGE-A-PRIME-BBB-AND-CACHE-FIX-SUMMARY.md).

Full v2.3 kickoff seed: [`v2.3-prep/v2.3-KICKOFF-SEED.md`](v2.3-prep/v2.3-KICKOFF-SEED.md).

Additional v2.3 candidates (carried from v2.1 + v2.2): ADAPTER-01, EXT-04, link provenance data structure, Extended Thinking on judge stages, Claude Probe D re-test with the new cache fix, Self-Refine contingent.
