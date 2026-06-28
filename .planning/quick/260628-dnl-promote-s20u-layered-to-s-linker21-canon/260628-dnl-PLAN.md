---
quick_id: 260628-dnl
slug: promote-s20u-layered-to-s-linker21-canon
date: 2026-06-28
type: promote+run
status: in-progress
---

# Quick Task 260628-dnl: Promote s20U_layered → s_linker21 (canonical Full) + run RQ1–4 (gpt-5.4)

## Description

Two parts:
1. **Promote** the proven spike-004 layered validator (`s_linker20_union_layered`,
   experimental) into a top-level **canonical** linker `s_linker21` (the new paper
   "Full" variant, superseding `s_linker13_min` in reported results).
2. **Run RQ1–4 results** for S21 on the **gpt-5.4** backend (no-reasoning, N=3),
   wiring the outputs into the existing RQ scorers.

## Decisions (from user, locked)

- **S21 status:** `canonical=True, experimental=False` — new paper Full. (s_linker13_min's
  flag left untouched; multiple canonical entries already coexist — additive change.)
- **Run depth:** launch live sweeps now (real spend authorized) — gated behind a 1-dataset
  smoke check first.
- **Backend:** gpt-5.4 ONLY. No-reasoning (layered requires reasoning off; for gpt that
  means `OPENAI_REASONING_EFFORT` unset/none — matches the proven `run_s20union_gpt_n3.sh`).

## Key facts (verified)

- `s_linker20_union_layered.py` = 94-line subclass of `SLinker20Union` overriding only
  `_prompt_validation` (LAYERED_ENTITY_RULES lenient / LAYERED_COREF_RULES strict, Mode 5
  justification). Spike-004 result: gpt-5.4 89.4→93.2 (+3.8), every dataset up, zero
  implicit-recall cost. Taboo-clean.
- Registry: `CANONICAL_VARIANTS` list (run_ablation.py:40) + `VARIANT_SPECS` dict (:135).
  `build_linker` reads `module`/`class_name`/`kwargs`. `no_knowledge=True` is a supported
  ctor kwarg (used by `s_linker20_union_noknow`).
- Phase_cache nests `{PHASE_CACHE_DIR}/{_VARIANT_NAME}/{backend_tag}/{dataset}/` — S21 will
  write under `s_linker21`.
- **RQ1/RQ2** ← `evaluation/mini-src/rq12.py`, approach gpt rows from
  `sota/recovered-links/{model-doc/aalinker,doc-code/aalinker-composed}/gpt-5.4_full/{run}/{proj}.csv`.
  Produced via `scripts/extract_s20union_caches.py` + sota `build_unified.py`/`normalize.py`.
- **RQ3/RQ4** ← `evaluation/mini-rq34/rq34.py`; phase_cache path uses hardcoded
  `VARIANT="s_linker20_union"` (:67) and `$RQ34_OPENAI_SLOT` (:74). Needs `VARIANT` made
  env-overridable for S21.
- No `test_*.py` exist (CLAUDE.md smoke-test ref is stale) → validate via Python import/build.
- GATE-01 (canonical files byte-stable) — S21 is NEW, no conflict. GATE-06 (no benchmark
  vocab) — rubric copied verbatim from taboo-clean layered file.

## Tasks

### Task 1 — Promote S21 (code, $0, atomic commit)
- Create `src/llm_sad_sam/linkers/experimental/s_linker21.py` = verbatim copy of
  `s_linker20_union_layered.py` with ONLY: class `SLinker20UnionLayered`→`SLinker21`,
  `_VARIANT_NAME`→`"s_linker21"`, docstring → canonical-promotion language. (Verbatim copy
  guarantees behavioural identity with the proven config.)
- Register in `run_ablation.py`:
  - `CANONICAL_VARIANTS`: add `"s_linker21"`, `"s_linker21_noknow"`.
  - `VARIANT_SPECS["s_linker21"]`: module/class SLinker21, `canonical=True`,
    `experimental=False`.
  - `VARIANT_SPECS["s_linker21_noknow"]`: same module/class, `kwargs=dict(no_knowledge=True)`,
    `canonical=False, experimental=True`.
- Verify: `python -c "import run_ablation; run_ablation.build_linker('s_linker21')"` and
  `--list-variants` shows both. Commit.

### Task 2 — Run harness + scoring wiring (code, $0, atomic commit)
- `run_s21_gpt_n3.sh` (mirror run_s20union_gpt_n3.sh): VARIANT=s_linker21,
  BASE=results/v2.6.6_s21_gpt, reasoning unset (=no-reasoning), N=3.
- `run_s21_noknow_gpt_n3.sh` (mirror noknow): VARIANT=s_linker21_noknow,
  BASE=results/v2.6.6_s21_noknow_gpt, N=3.
- `evaluation/mini-rq34/rq34.py`: `VARIANT = os.environ.get("RQ34_VARIANT","s_linker20_union")`
  (stdlib-only; evaluation repo).
- Commit (per-repo: approach + transarc-emp).

### Task 3 — Smoke gate (live, ~$1–2)
- Run S21 on jabref only, gpt-5.4 no-reasoning, 1 run. Confirm: process exits 0, links CSV
  has >1 line, phase_cache/{layer3,layer4,final}.pkl written, macro sane (spike jabref
  layered→100). Report result + projected full-sweep cost before launching.

### Task 4 — Launch full sweeps (live, $$$, background)
- Background-launch `run_s21_gpt_n3.sh` then `run_s21_noknow_gpt_n3.sh`. Monitor via
  PROGRESS.log + .ALL_DONE markers.

### Task 5 — Score RQ1–4 + finalize (after sweeps)
- RQ3/RQ4: `RQ34_VARIANT=s_linker21 RQ34_OPENAI_SLOT=.../v2.6.6_s21_gpt python3 mini-rq34/rq34.py`
  (+ rq34_rq2.py). No-knowledge A/B via s21_noknow final-link macro delta.
- RQ1/RQ2: extract S21 caches → compose → sota dump (gpt-5.4_full slot) → `mini-src/rq12.py`.
- Write outputs (CSVs), SUMMARY.md, update STATE.md. Commit.

## must_haves
- truths: S21 registered as canonical=True; build_linker('s_linker21') works; S21 behaviour
  byte-identical to s_linker20_union_layered; rq34 VARIANT env-overridable.
- artifacts: s_linker21.py, run_s21_gpt_n3.sh, run_s21_noknow_gpt_n3.sh, RQ1–4 result CSVs,
  SUMMARY.md.
- key_links: run_ablation.py registry; evaluation/mini-rq34/rq34.py; evaluation/mini-src/rq12.py.

## Spend gate
Tasks 1–2 are $0 and land first. Task 3 (smoke) is the gate before Task 4's full spend
(~$50–65 per sweep × 2, per Phase-51 estimate). Report smoke + projection at the gate.
