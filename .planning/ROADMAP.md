# Roadmap: llm-sad-sam-v45 — Milestone v2.6.6

**Milestone:** v2.6.6 — Standalone RQ3/RQ4 Eval Infra (s_linker20_union)
**Created:** 2026-06-21
**Phase numbering:** continues from v2.6.4 (last numbered phase 48; 49 reserved for v2.6.4 CLOSE) → v2.6.6 starts at **Phase 50**.

## Goal

Build a small, fully self-contained eval bundle under `../working/` that deterministically replays the frozen `s_linker20_union` per-run checkpoints (both backends, N≥3) to compute RQ3 (validator contribution) and RQ4 (per-module + Full-vs-No-Knowledge) ablation results as full-detailed CSVs + SUMMARY.md, reproducible from that directory alone.

**Data reality (verified 2026-06-21):**
- Per-run phase_caches exist for **both backends** — gpt `results/v2.6.5_s20union/gpt/run{1..N}/phase_cache`, sonnet `results/v2.6.5_s20union_sonnet/run{1..N}/phase_cache`; plus gpt N=6 full runs in `results/v2.6.5/full_s_linker20_union_run{1..6}/`.
- `layer3` = entity `candidates`/`validated`/`decisions{(s,c):{approved,p1,p2,path,stage}}`; `layer4` = coref `coref_raw`/`coref_validated`/`coref_decisions{(s,c):{approved,path}}`; `layer1` = `model_knowledge`+`doc_knowledge`; `final` = `final`+`final_provenance`.
- Gold standards: `…/ardoco/core/tests-base/target/classes/benchmark/<proj>/goldstandards/goldstandard_sad_YYYY-sam_YYYY.csv`.
- **No-Knowledge is not on disk and not replayable** → Phase 51 produces it via a bounded live run.

## Phases

| # | Phase | Goal | Requirements | LLM? | Success Criteria |
|---|-------|------|--------------|------|------------------|
| 50 | EXTRACT | Bridge frozen s20_union phase_caches → neutral stdlib JSON | EXTRACT-01/02/03 | No | 4 |
| 51 | NOKNOW | Knowledge-disable path + No-Knowledge runs (5×{gpt,sonnet}×N≥1) | NOKNOW-01/02 | **Yes (bounded)** | 3 |
| 52 | METRIC CORE | Stdlib metric core + self-contained bundle scaffold | METRIC-01/02 | No | 4 |
| 53 | RQ3 | Validator-contribution ablation (4 configs) + detail CSVs | RQ3-01/02 | No | 4 |
| 54 | RQ4 | Module decomposition + Full-vs-No-Knowledge A/B + detail CSVs | RQ4-01/02 | No | 4 |
| 55 | PACKAGE | Per-link audit + SUMMARY.md + self-contained bundle + parity/determinism gate | OUTPUT-01/02, BUNDLE-01/02 | No | 5 |

**Dependency order:** 50 → {51, 52} → 53 → 54 (needs 51 + 52) → 55. Phase 51 (live runs) should be kicked off early since it is the only LLM-bound, latency-sensitive work; its extract feeds RQ4-02.

---

### Phase 50 — EXTRACT

**Goal:** Convert every frozen `s_linker20_union` per-run phase_cache into neutral, stdlib-loadable JSON so the downstream bundle never needs the linker classes or pickle.

**Requirements:** EXTRACT-01, EXTRACT-02, EXTRACT-03

**Success criteria:**
1. Running the extraction script in `agent-linker` produces one JSON per (backend × run × project) covering all of gpt + sonnet, every N run, all 5 projects — no missing cells.
2. Each JSON contains entity candidates/validated/decisions (with p1/p2), coref raw/validated/decisions, knowledge layer (model_knowledge + doc_knowledge), and final links + provenance/source.
3. The final-link set re-derived from each JSON is byte-equal (as a set) to that run's own `*_links.csv` / `ablation_*.json` — a printed per-run PASS/FAIL faithfulness check shows all PASS.
4. The extraction script depends only on the cache + linker classes (no network, no LLM) and is re-runnable deterministically.

---

### Phase 51 — NOKNOW

**Goal:** Make the No-Knowledge ablation real — add a knowledge-disable path to `s_linker20_union` and run it on all 5 projects on both backends, then extract it into the same neutral format.

**Requirements:** NOKNOW-01, NOKNOW-02

**Success criteria:**
1. `s_linker20_union` runs with a knowledge-disable flag/variant that skips the alias table and ambiguity map; with the flag off, a snapshot/golden check confirms full-knowledge behavior is unchanged (GATE-01).
2. A No-Knowledge run completes for 5 projects × {gpt, sonnet} × N≥1, with outputs + phase_cache captured under `results/` and the live-call cost logged.
3. The No-Knowledge runs are extracted into neutral JSON identical in shape to the Phase-50 Full extracts (so the scorer treats them uniformly).

---

### Phase 52 — METRIC CORE

**Goal:** Stand up the self-contained `../working/` bundle scaffold and a stdlib-only metric core, parity-checked against the project's existing metrics.

**Requirements:** METRIC-01, METRIC-02

**Success criteria:**
1. `../working/` exists with vendored sad→sam gold (5 projects), vendored neutral extracts, and a stdlib-only `src/` metric module — no imports from `agent-linker`/`transarc-emp`.
2. Link-level P/R/F1 + TP/FP/FN computed by the metric core on the Full config matches each run's own `ablation_*.json` F1 within a stated tolerance (parity printout).
3. RQ-metric primitives — per-component F1 distribution, sentence coverage, noise rate, UpSet set-overlap — are implemented and unit-exercised on at least one project.
4. The metric core runs under a clean `python3` with stdlib only (no third-party deps).

---

### Phase 53 — RQ3 (Validator Contribution)

**Goal:** Compute the four-config validator ablation by replay and report each validator's contribution.

**Requirements:** RQ3-01, RQ3-02

**Success criteria:**
1. Full / NoEntityValid / NoCitation / NoValidator are each scored per project × run × backend by toggling the cached validator decisions (no LLM).
2. Per-validator TP-preserved vs FP-removed (entity two-pass; coref/citation) and net ΔF1-if-removed are reported, with a per-component distribution.
3. `rq3_detail.csv` (per-run × per-config × per-project) and `rq3_summary.csv` (macro + per-project, N≥3 mean ± range, both backends) are written to `../working/out/`.
4. The four configs are internally consistent (NoValidator ⊇ NoEntityValid, NoCitation link sets; Full ⊆ each), verified by an assertion.

---

### Phase 54 — RQ4 (Module Contribution)

**Goal:** Compute the redesigned RQ4 — symmetric per-linker-module decomposition plus the Full-vs-No-Knowledge A/B.

**Requirements:** RQ4-01, RQ4-02

**Success criteria:**
1. Entity-only / coref-only / union(full) are scored per project × run × backend, reporting F1, unique TPs, UpSet overlap (|only_E|/|both|/|only_C|), sentence coverage, and noise rate, with N≥3 mean ± range.
2. Full vs No-Knowledge ΔF1 (+ coverage/noise deltas) is computed from the Phase-51 No-Knowledge extracts, per project, both backends.
3. `rq4_detail.csv` and `rq4_summary.csv` are written to `../working/out/` with both the module-decomposition and knowledge-A/B blocks.
4. UpSet cells reconcile with the totals (|only_E| + |both| + |only_C| equals the distinct union TP count), verified by an assertion.

---

### Phase 55 — PACKAGE

**Goal:** Emit the full-detailed audit, the human-readable SUMMARY, and finalize the self-contained, reproducible bundle behind a parity/determinism gate.

**Requirements:** OUTPUT-01, OUTPUT-02, BUNDLE-01, BUNDLE-02

**Success criteria:**
1. A per-link audit CSV lists every link with sentence, component, source-module, validator decision (p1/p2/approved), gold-match, and RQ3/RQ4 config membership.
2. `SUMMARY.md` shows RQ3 (validator contributions) and RQ4 (module contributions + knowledge A/B) headline tables for both backends, with N and variance noted.
3. `../working/` runs end-to-end via a single `run.py` with **no path dependency on sibling repos**, and a README documents one-command reproduction.
4. `run.py` produces bit-identical output across two consecutive runs (determinism check passes).
5. The Full-config macro reproduces the frozen `s_linker20_union` run numbers within the stated tolerance (parity gate passes).

## Coverage Validation

| Requirement | Phase | Status |
|-------------|-------|--------|
| EXTRACT-01 | 50 | Pending |
| EXTRACT-02 | 50 | Pending |
| EXTRACT-03 | 50 | Pending |
| NOKNOW-01 | 51 | Pending |
| NOKNOW-02 | 51 | Pending |
| METRIC-01 | 52 | Pending |
| METRIC-02 | 52 | Pending |
| RQ3-01 | 53 | Pending |
| RQ3-02 | 53 | Pending |
| RQ4-01 | 54 | Pending |
| RQ4-02 | 54 | Pending |
| OUTPUT-01 | 55 | Pending |
| OUTPUT-02 | 55 | Pending |
| BUNDLE-01 | 55 | Pending |
| BUNDLE-02 | 55 | Pending |

**All 15 requirements mapped across 6 phases. Coverage: 100% ✓**

---
*Roadmap created: 2026-06-21*
*Last updated: 2026-06-21 — v2.6.6 initial roadmap (6 phases, 50–55).*
