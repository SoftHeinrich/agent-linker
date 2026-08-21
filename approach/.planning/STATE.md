---
gsd_state_version: 1.0
milestone: v2.6.6
milestone_name: milestone
status: paused
stopped_at: context exhaustion at 75% (2026-08-21)
last_updated: "2026-08-21T09:50:22.081Z"
last_activity: 2026-06-28 -- Quick 260628-dnl COMPLETE (see below)
progress:
  total_phases: 7
  completed_phases: 6
  total_plans: 28
  completed_plans: 26
  percent: 86
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
Last activity: 2026-06-28 -- Quick 260628-dnl COMPLETE (see below)

> **Quick 260628-dnl (COMPLETE):** Promoted `s_linker20_union_layered` → canonical
> **`s_linker21`** (paper Full, supersedes s13_min in reported results). Ran live gpt-5.4
> N=3 sweeps (Full + No-Knowledge, 0 GIVEUP, extract 15/15 PASS each) and scored all four
> RQs. S21 gpt-5.4 **macro-F1 0.936** (+4.2pp over s20_union no-reasoning); top doc→model
> system; best on every RQ2 size-aware metric (worst 0.753 / harmonic 0.884); RQ3 validators
> +9.3pp; RQ4 knowledge +5.79pp. Tables: `quick/260628-dnl-…/260628-dnl-RESULTS.md`.

> **Decision — D-04 REVISED (2026-06-28): GPT-5.4 is now the MAIN backend.**
> Supersedes the original D-04 ("main body Claude, appendix GPT-5.4 mirror",
> `.planning/milestones/v2.6.3-ROADMAP.md:57`). New rule: the paper reports **GPT-5.4**
> in the body for every RQ; **Claude Sonnet** moves to the appendix mirror (RQ3/RQ4) and the
> second RQ1 row. Rationale: S21 was scored on gpt-5.4 first and gpt is the primary reported
> system; per-project breakdowns + all sonnet results go to the appendix (space). Paper edits:
> `working/appendix/detailed-results.tex` (the Claude/per-project appendix; formerly
> `rq3-rq4-mirror.tex`) reframed for a GPT-5.4 body + the
> author's own note at the top of `working/sections/results.tex`. **Pending:** the float
> CONTENT swap (GPT s21 → body, Claude s21 → appendix) is the numbers pass, blocked on the
> sonnet run below.
>
> **In flight — Claude/Sonnet s21 sweep (launched 2026-06-28, harness job `bs6ne5nc7`):**
> `run_s21_sonnet_n3.sh && run_s21_noknow_sonnet_n3.sh` (backend=claude/sonnet,
> `CLAUDE_DISABLE_THINKING=1` = reasoning-off, the layered-validator requirement). Writes
> `results/v2.6.6_s21_sonnet/run{1,2,3}/` (Full) + `results/v2.6.6_s21_noknow_sonnet/run{1,2,3}/`
> (No-Knowledge); progress `logs/v2.6.6_s21_sonnet/PROGRESS.log`, completion `logs/*/.ALL_DONE`.
> Post-sweep: extract → score exactly like the gpt s21 path (260628-dnl SUMMARY steps 1–5,
> swap the `_sonnet` slots) to populate the Claude appendix mirror + RQ1 Claude row.

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

| # | Description | Date | Commit | Status | Directory |
|---|-------------|------|--------|--------|-----------|
| 260627-mot | Investigate whether paper RQ3/RQ4 should report RQ2 size-aware metrics (grounded on real runs) | 2026-06-27 | `8388472` | | [260627-mot-investigate-whether-paper-rq3-rq4-should](./quick/260627-mot-investigate-whether-paper-rq3-rq4-should/) |
| 260628-dnl | Promote s20U_layered → canonical `s_linker21` (Full) + run RQ1–4 results (gpt-5.4, N=3 live) | 2026-06-28 | `34b3239` | | [260628-dnl-promote-s20u-layered-to-s-linker21-canon](./quick/260628-dnl-promote-s20u-layered-to-s-linker21-canon/) |
| 260701-ld4 | Promote agent_router chain (agentic_router.py + GTP proposer) → `s_linker21_agentrouter`; register run_ablation variant; archive pilot/ → `.planning/archive/`; rewrite CLAUDE.md | 2026-07-01 | `3a06248` | Verified | [260701-ld4-promote-the-finalized-agent-router-based](./quick/260701-ld4-promote-the-finalized-agent-router-based/) |

**260701-ld4 verdict:** Promoted the bounded-autonomy agentic router (not the higher-scoring non-agentic named+routed config — user's explicit choice) as `SLinker21AgentRouter`, subclassing `s_linker21` with a gate-floored augmentation pass (can never regress below s21) plus CODE-routed candidates wired through `DirectCodeLinker`/`DirectLinkJudge` behind an optional `acm_path` kwarg (not yet plumbed by `run_ablation.py`'s harness — future work). GATE-01 held (s_linker21.py byte-identical). `pilot/` fully archived to `.planning/archive/router-pilot-260701/`; `CLAUDE.md` rewritten for the `router` branch. Plan-checked + verified (6/6 must-haves independently re-checked against live repo state).

**260628-dnl verdict:** S21 (layered no-reasoning validator) promoted to canonical Full. Live gpt-5.4 N=3 sweeps (Full + No-Knowledge) → S21 macro-F1 **0.936** (+4.2pp over s20_union no-reasoning 0.894): top doc→model system, best on all RQ2 size-aware metrics (worst 0.753 / harmonic 0.884), RQ3 validators +9.3pp combined, RQ4 knowledge module +5.79pp. Tables: `quick/260628-dnl-…/260628-dnl-RESULTS.md`.

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

Last session: 2026-08-21T09:50:22.078Z
Stopped at: context exhaustion at 75% (2026-08-21)
Resume file: None
Next action: /clear then /gsd:execute-phase 51 — resumes at the 51-04 spend gate. Before approving: ensure OPENAI_API_KEY is set (gpt sweep) and the claude/sonnet backend is configured (sonnet sweep). Cost ~$50–65, ~7.5h unattended, resumable via per-(run,dataset) .done markers. Launch scripts: run_s20union_noknow_gpt_n3.sh then run_s20union_noknow_sonnet_n3.sh. After both .ALL_DONE, run 51-05 (extractor extension).
