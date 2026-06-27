---
gsd_state_version: 1.0
milestone: v2.6.6
milestone_name: Standalone RQ3/RQ4 Eval Infra (s_linker20_union)
status: executing
stopped_at: 2026-06-21 — Phase 51 prep waves 1-2 done (51-01/02/03), GATE-01 EVIDENCE PASS; paused before live sweep 51-04. Resume with /gsd:execute-phase 51 (it skips done plans, lands on the 51-04 spend gate).
last_updated: "2026-06-21T21:08:45.539Z"
last_activity: 2026-06-21 -- Phase 51 waves 1-2 executed, GATE-01 passed, paused at 51-04 spend gate
progress:
  total_phases: 6
  completed_phases: 1
  total_plans: 6
  completed_plans: 1
  percent: 17
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-21 for v2.6.6 kickoff)

**Core value:** A small, fully self-contained eval bundle under `../working/` that deterministically replays the frozen `s_linker20_union` per-run checkpoints (both backends, N≥3) to compute paper RQ3 (validator contribution) and RQ4 (per-module + Full-vs-No-Knowledge) ablation results as full-detailed CSVs + SUMMARY.md, reproducible from that directory alone.
**Current focus:** Phase 51 — noknow

## Current Position

Phase: 51 (noknow) — EXECUTING (paused at 51-04 spend gate)
Plan: 3 of 5 complete (51-01/02/03 done; 51-04/05 pending)
Status: Prep waves 1-2 complete; GATE-01 EVIDENCE: PASS. Paused before the live ~$50-65 / ~7.5h No-Knowledge sweep (51-04) by user choice.
Last activity: 2026-06-21 -- Phase 51 waves 1-2 executed, GATE-01 passed, paused at spend gate

```
Progress: v2.6.6 [█████░░░░░░░░░░░░░░░░░░░░░░░░░] 1/6 phases — 17%
          Phase 50 EXTRACT      [x] bridge s20_union caches → neutral JSON — DONE
          Phase 51 NOKNOW       [ ] knowledge-disable path + No-Knowledge runs (bounded LLM)
          Phase 52 METRIC CORE  [ ] stdlib metric core + self-contained bundle scaffold
          Phase 53 RQ3          [ ] 4-config validator-contribution ablation
          Phase 54 RQ4          [ ] module decomposition + Full-vs-No-Knowledge A/B
          Phase 55 PACKAGE      [ ] audit CSV + SUMMARY.md + bundle + parity/determinism gate
```

## Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260627-mot | Investigate whether paper RQ3/RQ4 should report RQ2 size-aware metrics (grounded on real runs) | 2026-06-27 | `8388472` | [260627-mot-investigate-whether-paper-rq3-rq4-should](./quick/260627-mot-investigate-whether-paper-rq3-rq4-should/) |

**260627-mot verdict:** RQ4 — yes (size-aware deltas amplify each linker 2.5–3× over file-F1, consistent across both backends/all 3 runs). RQ3 — no (retained RQ2 metrics move against the validators or flip sign across backends; only noise rate captures the benefit and it was dropped from the suite). Surfaced a non-robust `results.tex` worst-component claim to cut. Deliverable: `../transarc-emp/mini-rq34/RQ2_LENS_DECISION.md`.

## Decisions (Phase 50)

- coref.raw/validated serialized as lists (never dict-collapse); preserves dup-(s,c) entries in 8/30 cells
- raw_resolution stripped from coref.metadata and final.provenance.coref_meta by default
- doc_knowledge.aliases serialized as list-of-records [{term,component,scope}] (runtime AliasEntry, not str)
- final.links read from final.pkl final list directly; never re-derived from coref_decisions dict (Landmine)
- results/v2.6.6_extracts/ gitignored; Phase 52 vendors via script re-run or copy

## v2.6.6 Roadmap Summary

| Phase | Goal | Key REQs | LLM? |
|-------|------|----------|------|
| 50 — EXTRACT | Frozen s20_union phase_caches → neutral stdlib JSON | EXTRACT-01/02/03 | No |
| 51 — NOKNOW | Knowledge-disable path + No-Knowledge runs (5×{gpt,sonnet}×N≥1) | NOKNOW-01/02 | **Yes (bounded)** |
| 52 — METRIC CORE | Stdlib metric core + self-contained bundle scaffold + parity | METRIC-01/02 | No |
| 53 — RQ3 | Validator ablation (Full/NoEntityValid/NoCitation/NoValidator) | RQ3-01/02 | No |
| 54 — RQ4 | Module decomposition (entity/coref/union + UpSet) + knowledge A/B | RQ4-01/02 | No |
| 55 — PACKAGE | Per-link audit + SUMMARY.md + self-contained bundle + parity gate | OUTPUT-01/02, BUNDLE-01/02 | No |

**Source of truth:** `s_linker20_union` per-run phase_caches — gpt `results/v2.6.5_s20union/gpt/run{1..N}/phase_cache`, sonnet `results/v2.6.5_s20union_sonnet/run{1..N}/phase_cache`. **Not s19.**
**Output target:** `../working/out/` — rq3_detail.csv, rq3_summary.csv, rq4_detail.csv, rq4_summary.csv, per-link audit CSV, SUMMARY.md.

## Standing Gates (into v2.6.6)

- **GATE-01**: canonical/paper artifacts untouched — `s_linker13_min.py`, `s_linker19.py`, and full-knowledge `s_linker20_union.py` byte-/snapshot-stable (the No-Knowledge path is flag-gated; off = unchanged). 🔄
- **GATE-06**: no benchmark-derived vocabulary introduced in any new code. 🔄
- **PARITY**: standalone Full-config macro reproduces the frozen `s_linker20_union` run numbers within tolerance; `run.py` reruns bit-identical. 🔄

## Key Design Facts (verified 2026-06-21)

- Both backends have per-run phase_cache (gpt + sonnet); gpt also has N=6 full runs (`results/v2.6.5/full_s_linker20_union_run{1..6}/`).
- `layer3`: entity `candidates`→`validated` + `decisions{(s,c):{approved,p1,p2,path,stage}}` → entity two-pass validator replayable.
- `layer4`: `coref_raw`→`coref_validated` + `coref_decisions{(s,c):{approved,path}}` → coref/citation validator replayable.
- `layer1`: `model_knowledge` + `doc_knowledge` (knowledge layer, present but cannot be ablated by replay → Phase 51 live run).
- `final`: `final` + `final_provenance`; `*_links.csv` carries a per-link `source` tag (`entity`/`coreference`).
- Gold: `…/ardoco/core/tests-base/target/classes/benchmark/<proj>/goldstandards/goldstandard_sad_YYYY-sam_YYYY.csv` (doc-to-model).

## Spike Findings (captured 2026-06-27, NOT in active v2.6.6 scope)

Orthogonal axis — **no-REASONING** (effort-0), not v2.6.6's no-knowledge. Captured as a
promotable seed; does not block the active milestone.

- **Spikes 004 + 005 — COMPLETE.** Shipped opt-in `s_linker20_union_layered` (Mode 5
  justification + Mode 1 claim rubric, entity-lenient/coref-strict): recovers thinking-on
  *precision* at effort-0 — Sonnet +1.1 (89.7→90.8), gpt-5.4 +3.8 (89.4→93.2), zero
  implicit-recall cost; FP parity entity 25/coref 7. Modes 2 & 4 rejected. GATE-01 holds.
- **Spike 005 verdict — STOP on extraction.** The same mechanism does NOT transfer to the
  extraction step: extraction-bound gap is 6.2% of gold, 68% run-variance, 44% non-verbatim
  inference. Asymmetry: thinking = precision-discriminator at gates (reconstructable, shipped)
  vs recall-generator at extraction (not reconstructable by a justification field). bbb LLM
  probe NOT worth running.
- **Seed (promotable to milestone / paper RQ):** `.planning/notes/2026-06-27-s20union-layered-SEED.md`

## Prior Milestone Context (carried, not active)

- **v2.6.4 — PAUSED** after Phase 48 (SWEEP). Negative result: `s_linker20` minimized prompts regressed gpt-5.4 macro to 88.9% (later shown to be variance; s20 TRUE macro ≈0.903). Phase 49 CLOSE intentionally NOT run. Archived docs: `.planning/milestones/v2.6.4-REQUIREMENTS.md` + `v2.6.4-ROADMAP.md`. Phase dirs `44–48` retained under `.planning/phases/`.
- **v2.6.5 (variance remediation)** — informal track (quick tasks 260610-lio, 260620-s2r, 260620-u2s, 260620-ycl). Found the s20-family tied within ±1.4pp noise; `s_linker20_union` is the mild best (macro ≈0.906 via BBB recall) and the **ship candidate** — the source for this milestone's RQ3/RQ4. Artifacts under `results/v2.6.5*/` + `logs/v2.6.5*/`.
- **v2.6.3 — SHIPPED** 2026-06-05. Paper RQ1–RQ4 cells populated via s19 checkpoint replay (now superseded for RQ3/RQ4 by s20_union per this milestone).

## v2.7 / Frozen

- **v2.7 (Phases 38–42)** — FROZEN. Resume after the v2.6.x line settles.
- **v2.6.4 close (Phase 49)** — deferred pending remediation disposition.

## Session Continuity

Last session: 2026-06-21T19:44:14.618Z
Stopped at: Phase 51 prep complete — 51-01 (no_knowledge flag + variant), 51-02 (GATE-01 harness), 51-03 (sweep scripts) all done & committed; GATE-01 EVIDENCE: PASS (structural + frozen-cache 30/30). Paused before 51-04 live sweep by user choice (prep-waves-only).
Resume file: .planning/phases/51-noknow/51-04-PLAN.md (wave 3 — live sweep, autonomous:false spend gate)
Next action: /clear then /gsd:execute-phase 51 — resumes at the 51-04 spend gate. Before approving: ensure OPENAI_API_KEY is set (gpt sweep) and the claude/sonnet backend is configured (sonnet sweep). Cost ~$50–65, ~7.5h unattended, resumable via per-(run,dataset) .done markers. Launch scripts: run_s20union_noknow_gpt_n3.sh then run_s20union_noknow_sonnet_n3.sh. After both .ALL_DONE, run 51-05 (extractor extension).
