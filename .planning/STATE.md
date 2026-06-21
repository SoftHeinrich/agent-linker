---
gsd_state_version: 1.0
milestone: v2.6.6
milestone_name: Standalone RQ3/RQ4 Eval Infra (s_linker20_union)
status: executing
stopped_at: 2026-06-21 — Phase 50 (EXTRACT) COMPLETE. Plan 50-01 executed: 30-cell pickle→neutral-JSON extractor, 30/30 PASS, deterministic.
last_updated: "2026-06-21T16:00:00.000Z"
last_activity: 2026-06-21 -- Phase 50 plan 50-01 complete (EXTRACT-01/02/03 satisfied)
progress:
  total_phases: 6
  completed_phases: 1
  total_plans: 1
  completed_plans: 1
  percent: 17
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-21 for v2.6.6 kickoff)

**Core value:** A small, fully self-contained eval bundle under `../working/` that deterministically replays the frozen `s_linker20_union` per-run checkpoints (both backends, N≥3) to compute paper RQ3 (validator contribution) and RQ4 (per-module + Full-vs-No-Knowledge) ablation results as full-detailed CSVs + SUMMARY.md, reproducible from that directory alone.
**Current focus:** Phase 50 — extract

## Current Position

Phase: 50 (extract) — COMPLETE
Plan: 1 of 1 (DONE)
Status: Ready for Phase 51 (NOKNOW)
Last activity: 2026-06-21 -- Phase 50 plan 50-01 complete (EXTRACT-01/02/03 satisfied)

```
Progress: v2.6.6 [█████░░░░░░░░░░░░░░░░░░░░░░░░░] 1/6 phases — 17%
          Phase 50 EXTRACT      [x] bridge s20_union caches → neutral JSON — DONE
          Phase 51 NOKNOW       [ ] knowledge-disable path + No-Knowledge runs (bounded LLM)
          Phase 52 METRIC CORE  [ ] stdlib metric core + self-contained bundle scaffold
          Phase 53 RQ3          [ ] 4-config validator-contribution ablation
          Phase 54 RQ4          [ ] module decomposition + Full-vs-No-Knowledge A/B
          Phase 55 PACKAGE      [ ] audit CSV + SUMMARY.md + bundle + parity/determinism gate
```

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

## Prior Milestone Context (carried, not active)

- **v2.6.4 — PAUSED** after Phase 48 (SWEEP). Negative result: `s_linker20` minimized prompts regressed gpt-5.4 macro to 88.9% (later shown to be variance; s20 TRUE macro ≈0.903). Phase 49 CLOSE intentionally NOT run. Archived docs: `.planning/milestones/v2.6.4-REQUIREMENTS.md` + `v2.6.4-ROADMAP.md`. Phase dirs `44–48` retained under `.planning/phases/`.
- **v2.6.5 (variance remediation)** — informal track (quick tasks 260610-lio, 260620-s2r, 260620-u2s, 260620-ycl). Found the s20-family tied within ±1.4pp noise; `s_linker20_union` is the mild best (macro ≈0.906 via BBB recall) and the **ship candidate** — the source for this milestone's RQ3/RQ4. Artifacts under `results/v2.6.5*/` + `logs/v2.6.5*/`.
- **v2.6.3 — SHIPPED** 2026-06-05. Paper RQ1–RQ4 cells populated via s19 checkpoint replay (now superseded for RQ3/RQ4 by s20_union per this milestone).

## v2.7 / Frozen

- **v2.7 (Phases 38–42)** — FROZEN. Resume after the v2.6.x line settles.
- **v2.6.4 close (Phase 49)** — deferred pending remediation disposition.

## Session Continuity

Last session: 2026-06-21 (Phase 50 execution)
Stopped at: Phase 50 COMPLETE. Plan 50-01 executed: 30/30 cells extracted + PASS, deterministic, GATE-01 clean.
Resume file: None
Next action: Phase 51 (NOKNOW) — knowledge-disable path in s_linker20_union + bounded live runs.
