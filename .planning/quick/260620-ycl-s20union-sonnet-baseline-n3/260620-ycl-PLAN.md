---
quick_id: 260620-ycl
slug: s20union-sonnet-baseline-n3
date: 2026-06-20
type: experiment
status: in-progress
backend: claude (CLAUDE_MODEL=sonnet)
variant: s_linker20_union
N: 3
---

# PLAN — s20union (s_linker20_union) Sonnet baseline, N=3

## Goal

Establish the **missing Sonnet baseline** for `s_linker20_union` (the v2.6.5 union
ship candidate). It currently has gpt-5.4 numbers only (macro 0.906 @ N=6); the
prior quick task 260620-u2s flagged "the s20 family has no Sonnet validation" as the
open gap and recommended exactly this run. Produce a credible N=3 Sonnet macro-F1
(mean ± spread) per project, with **all intermediate results saved**.

## Decisions (locked)

- **Variant:** `s_linker20_union`, run **unmodified** (baseline integrity — no edits
  to the variant's internal DAG/concurrency).
- **Backend/model:** `LLM_BACKEND=claude`, `CLAUDE_MODEL=sonnet` (the linker's own
  defaults; set explicitly anyway).
- **N=3 = three independent invocations.** `run_ablation.py` has no `--N` flag, and
  the phase cache is keyed by `(variant, backend, dataset)` — so each run MUST use a
  **separate `PHASE_CACHE_DIR`**, else runs 2–3 replay run 1 and variance collapses
  to zero.
- **Datasets:** all 5 (mediastore, teastore, jabref, bigbluebutton, teammates),
  ordered light→heavy so failures surface early and the slow ones (bbb, teammates)
  run last.
- **"Save all intermediate results":** per-run isolated `--results-dir`,
  `PHASE_CACHE_DIR`, `LLM_LOG_DIR`, `CHECKPOINT_DIR` under
  `results/v2.6.5_s20union_sonnet/run{1,2,3}/` (links CSVs, timestamped ablation
  JSON, phase cache, raw LLM call logs). Per-invocation stdout/stderr to
  `logs/v2.6.5_s20union_sonnet/`.

## Execution strategy (per user: "run, but no massive parallel — slowly is fine")

- **Strictly sequential** at the orchestration level: one dataset at a time, one run
  at a time. Never run multiple datasets/runs concurrently.
- **Cooldowns:** 90 s between datasets, 240 s between runs — mitigates the documented
  failure mode (the `claude` CLI shares quota with the live session and degrades to
  empty responses under sustained load; see 260620-u2s "Environment finding").
- **Per-dataset retry + resume:** detect empty/failed datasets (python non-zero exit
  OR empty links CSV); retry once after a 300 s cooldown. Already-completed
  (run, dataset) pairs are skipped on re-launch (resume), so a partial sweep can be
  restarted cheaply.
- Runs as a **background job**; orchestrator monitors `PROGRESS.log` and aggregates.

## Tasks

1. **Driver script** `run_s20union_sonnet_n3.sh` — sequential N=3 × 5-dataset sweep
   with isolated intermediate dirs, cooldowns, retry, and resume markers.
2. **Execute** the sweep (background) and monitor to completion, retrying any
   empty-response datasets.
3. **Aggregate** per-run/per-dataset F1 into a macro table (mean ± spread), copy the
   machine-readable aggregate into this quick dir (raw `results/` is gitignored).
4. **Summarize** → `260620-ycl-SUMMARY.md`; update STATE.md; commit artifacts.

## Done when

- 3 runs × 5 datasets attempted; per-dataset F1 recorded (failures explicitly noted,
  not silently dropped).
- Sonnet macro mean ± spread for `s_linker20_union` reported per project + overall,
  compared against the existing gpt-5.4 union numbers and the canonical
  `s_linker13_min` Claude reference.
- All intermediate artifacts on disk under `results/v2.6.5_s20union_sonnet/`.
