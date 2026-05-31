# Roadmap: llm-sad-sam-v45

This file tracks the active milestone only. Shipped milestones are archived under `.planning/milestones/`.

## Milestones

- ✅ **v1.0 — Rule-to-LLM Ablation** (`s_linker12c` → `s_linker13`) — Phases 1–5 — shipped 2026-05-29. Final macro F1 0.9509. See [`milestones/v1.0-ROADMAP.md`](milestones/v1.0-ROADMAP.md).
- ✅ **v2.0 — Complete Rule Removal + Cross-Model — Generality First** — Phases 6–9 — shipped 2026-05-31. EXT-01 closed empty (negative), CROSS evidence published on gpt-5.4. See [`milestones/v2.0-ROADMAP.md`](milestones/v2.0-ROADMAP.md) and [`milestones/v2.0-MILESTONE-AUDIT.md`](milestones/v2.0-MILESTONE-AUDIT.md).
- ⏸ **Next** — no active milestone. Run `/gsd-new-milestone` to start v2.1+.

## Active Phase

None. v2.0 lifecycle complete; ready for next milestone definition.

## Production Artifact

`src/llm_sad_sam/linkers/experimental/s_linker13.py` — v1.0 final artifact, retained through v2.0 (v2.0 closed without modifying it). macro F1 = **0.9506 on Claude Sonnet**, **0.9077 on gpt-5.4** (5-dataset benchmark). canonical=True in `run_ablation.py`.

## Standing Gates (carry forward to future milestones)

- **GATE-01** — Dual floor: macro F1 ≥ 0.93 AND BBB ≤ 6pp below `s_linker12c` BBB AND each other dataset ≤ 2pp below its `s_linker12c` baseline.
- **GATE-05** — Hard-tier-first dev loop: regress >1pp on BBB or TM vs parent → no full sweep.
- **GATE-06** — Generality audit: every new prompt + helper passes BENCHMARK_TABOO scan AND reviewer-defensibility check; recorded per phase in SUMMARY.md.
- **GATE-07** — Every promoted variant registered in `CANONICAL_VARIANTS` + `VARIANT_SPECS`; standalone file; structured docstring with `REMOVED_FROM` / `RULES_REMOVED`.
