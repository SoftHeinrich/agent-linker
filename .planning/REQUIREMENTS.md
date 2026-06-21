# Requirements: llm-sad-sam-v45 — Milestone v2.6.6

**Defined:** 2026-06-21
**Milestone:** v2.6.6 — Standalone RQ3/RQ4 Eval Infra (s_linker20_union)
**Core Value:** A small, fully self-contained eval bundle under `../working/` that deterministically replays the frozen `s_linker20_union` per-run checkpoints (both backends, N≥3) to compute paper RQ3 (validator contribution) and RQ4 (per-module + knowledge A/B) ablation results as full-detailed CSVs + SUMMARY.md — reproducible by a reviewer from that directory alone.

**Source of truth:** `s_linker20_union` (the v2.6.5 ship candidate), **not** s19. Frozen per-run phase_caches at
`results/v2.6.5_s20union/gpt/run{1..N}/phase_cache` (gpt) and
`results/v2.6.5_s20union_sonnet/run{1..N}/phase_cache` (sonnet).

**Scope boundary:** Deterministic replay (zero LLM) for RQ3 + RQ4-modules. The **only** non-replay scope is a bounded live No-Knowledge run for the RQ4 knowledge A/B axis. Output is CSVs + SUMMARY.md; the paper `.tex` is untouched.

## v2.6.6 Requirements

### Extraction Bridge (EXTRACT)

- [x] **EXTRACT-01**: An extraction script (run inside `agent-linker`, where the linker classes exist) dumps every `s_linker20_union` per-run phase_cache (`layer1`–`layer4` + `final`) into neutral, stdlib-loadable JSON — both backends (gpt + sonnet), all N runs, all 5 projects.
- [x] **EXTRACT-02**: The extracted JSON captures every ablation-relevant field: entity `candidates`/`validated`/`decisions` (incl. `p1`/`p2` evidence gates), coref `coref_raw`/`coref_validated`/`coref_decisions`, the knowledge layer (`model_knowledge` + `doc_knowledge`), and the `final` links with per-link `source`/provenance.
- [x] **EXTRACT-03**: Extraction faithfulness is verified — the final-link set re-derived from each extract equals that run's own `*_links.csv` / `ablation_*.json`, per project × run × backend.

### No-Knowledge Ablation (NOKNOW)

- [ ] **NOKNOW-01**: `s_linker20_union` gains a knowledge-disable path (no alias table, no ambiguity map) behind a flag/variant; with the flag off, full-knowledge behavior is unchanged (snapshot-stable — GATE-01).
- [ ] **NOKNOW-02**: A No-Knowledge run executes on 5 projects × {gpt, sonnet} × N≥1; its outputs + phase_cache are captured under `results/` and extracted into the same neutral JSON format used for the Full runs.

### Metric Core (METRIC)

- [ ] **METRIC-01**: A stdlib-only metric core computes link-level Precision / Recall / F1 + TP/FP/FN against the sad→sam gold standard, parity-checked against each run's own `ablation_*.json` F1 on the Full config (within a stated tolerance).
- [ ] **METRIC-02**: Stdlib RQ-metric primitives are implemented: per-component F1 distribution, sentence coverage, noise rate, and set-overlap (UpSet `|only_E|` / `|both|` / `|only_C|`).

### RQ3 — Validator Contribution (RQ3)

- [ ] **RQ3-01**: The four validator configs — Full / NoEntityValid / NoCitation / NoValidator — are computed by replay over the extracts, per project × run × backend.
- [ ] **RQ3-02**: RQ3 reports per-validator TP-preserved vs FP-removed (entity two-pass; coref/citation), net ΔF1-if-removed, and per-component distribution — aggregated to N≥3 mean ± range (macro + per-project, both backends) in `rq3_detail.csv` + `rq3_summary.csv`.

### RQ4 — Module Contribution (RQ4)

- [ ] **RQ4-01**: Per-linker-module decomposition — entity-only / coref-only / union(full) — reports F1, unique TPs, UpSet overlap (`|only_E|`/`|both|`/`|only_C|`), sentence coverage, and noise rate, per project × run × backend, with N≥3 mean ± range, in `rq4_detail.csv` + `rq4_summary.csv`.
- [ ] **RQ4-02**: Knowledge A/B reports **Full vs No-Knowledge** ΔF1 (+ sentence-coverage and noise-rate deltas) from the NOKNOW runs, per project, both backends.

### Output + Bundle (OUTPUT / BUNDLE)

- [ ] **OUTPUT-01**: A per-link audit CSV is emitted — every link with `sentence`, `component`, `source-module`, validator decision (`p1`/`p2`/`approved`), gold-match, and which RQ3/RQ4 configs include it.
- [ ] **OUTPUT-02**: `SUMMARY.md` presents human-readable RQ3 (validator contributions) and RQ4 (module contributions + knowledge A/B) headline tables for **both backends**, with N and variance noted.
- [ ] **BUNDLE-01**: `../working/` is fully self-contained — vendored neutral extracts + vendored sad→sam gold (5 projects) + ported stdlib metric core, a single `run.py`, **no path dependency on sibling repos**, and a README documenting one-command reproduction.
- [ ] **BUNDLE-02**: Determinism + parity gate — `run.py` reruns are bit-identical, and the Full-config macro reproduces the frozen `s_linker20_union` run numbers within the stated tolerance.

## Future Requirements

Deferred; not in this milestone's roadmap.

- **TEX-01**: Render the computed RQ3/RQ4 numbers into the paper `tables/`+`figures/` `.tex` (rq3-validators, rq4-agents, rq3-validator figure, rq4-upset). *(Explicitly out of this milestone — output is CSVs + SUMMARY.md only.)*
- **NOKNOW-N**: Raise No-Knowledge to N≥3 per backend for variance bands on the knowledge axis (this milestone requires N≥1).
- **RQ12-01**: Fold RQ1/RQ2 (sad→sam + sad→code link/architecture metrics) into the same self-contained bundle.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Recomputing from s19 checkpoints | Source is the v2.6.5 ship candidate `s_linker20_union`, per user direction. |
| Paper `.tex` table/figure rendering | This milestone outputs CSVs + SUMMARY.md; TeX rendering is a separate downstream step (TEX-01). |
| doc-to-code (sad→code) RQ3/RQ4 | The validators/modules are framed on the SAD→SAM task; sad→code ablations are deferred. |
| N≥3 live runs for No-Knowledge | Replay covers RQ3 + RQ4-modules at N≥3; No-Knowledge is a bounded N≥1 live ablation this milestone. |
| Re-tuning / re-running Full s20_union | Full runs are frozen inputs; we replay them, not regenerate them. |
| Modifying canonical/paper linkers' behavior | GATE-01 — `s_linker13_min`, `s_linker19`, full-knowledge `s_linker20_union` stay byte-/snapshot-stable. |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| EXTRACT-01 | Phase 50 | Complete |
| EXTRACT-02 | Phase 50 | Complete |
| EXTRACT-03 | Phase 50 | Complete |
| NOKNOW-01 | Phase 51 | Pending |
| NOKNOW-02 | Phase 51 | Pending |
| METRIC-01 | Phase 52 | Pending |
| METRIC-02 | Phase 52 | Pending |
| RQ3-01 | Phase 53 | Pending |
| RQ3-02 | Phase 53 | Pending |
| RQ4-01 | Phase 54 | Pending |
| RQ4-02 | Phase 54 | Pending |
| OUTPUT-01 | Phase 55 | Pending |
| OUTPUT-02 | Phase 55 | Pending |
| BUNDLE-01 | Phase 55 | Pending |
| BUNDLE-02 | Phase 55 | Pending |

**Coverage:**

- v2.6.6 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-21*
*Last updated: 2026-06-21 after initial v2.6.6 definition*
