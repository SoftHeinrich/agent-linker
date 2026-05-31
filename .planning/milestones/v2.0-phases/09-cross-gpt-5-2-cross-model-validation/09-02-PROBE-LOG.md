# Phase 9 Plan 02 — BBB Probe Log (s_linker13 on gpt-5.4)

**Plan:** 09-02
**Purpose:** D-03 Step 1 reasonableness gate before committing to full 5-dataset sweep (Plan 09-03).
**Cancellation rule:** D-10 — halt if BBB F1 < 0.6 OR persistent harness errors.

---

## Invocation Command (exact, with env vars)

```bash
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.4 \
PHASE_CACHE_DIR=./results/phase_cache_gpt54 \
python run_ablation.py --datasets bigbluebutton --variants s_linker13 \
  --results-dir results/ablation_results 2>&1 | tee results/ablation_results/09-02-bbb-probe-stdout.log
```

Three env overrides applied at the shell, no harness code edits (per Plan 09-01 CLEAN
GATE-06 audit and Plan 09-02 §<action>):

- `LLM_BACKEND=openai` — selects OpenAI dispatch in `run_ablation.py:get_backend()` (line 427).
- `OPENAI_MODEL_NAME=gpt-5.4` — overrides default `gpt-5.2` at three `setdefault`/`get` sites
  (`run_ablation.py:436`, `llm_client.py:112`, `s_linker13.py:171`). All three honour external env.
- `PHASE_CACHE_DIR=./results/phase_cache_gpt54` — namespaces s_linker13's per-phase pickle cache
  (`s_linker13.py:1160`) away from Claude's default `./results/phase_cache`. Verified post-run
  (see "Cache Directory" below) — pickles landed under the gpt54 root, no collision.

Backend descriptor printed by runner: `openai (gpt-5.4)` (confirms env override took effect).

---

## Timestamps

- **Start (wall):** 2026-05-31T03:51:47Z
- **End (wall):**   2026-05-31T03:52:35Z (approx — JSON filename `ablation_20260531_055235.json`
  uses local time; wall-clock elapsed ≈ 48 s per runner output)
- **Elapsed:**      48.4 s (per JSON `time` field for BBB block)

---

## BBB Metrics (from JSON)

Source JSON: `results/ablation_results/ablation_20260531_055235.json`
JSON key path: `bigbluebutton.s_linker13`

| Metric | Value |
|---|---|
| **F1** | **0.8037 (80.4%)** |
| **P (precision)** | 0.9556 (95.6%) |
| **R (recall)** | 0.6935 (69.4%) |
| **TP** | 43 |
| **FP** | 2 |
| **FN** | 19 |
| **n_links emitted** | 45 |
| **runtime** | 48.4 s |

FP breakdown (from runner stdout):
- `coreference` source: 1 FP
- `seed` source: 1 FP

Link sources (positives + FPs combined): `{'seed': 37, 'entity': 5, 'coreference': 3}`.

---

## Reference: Claude Sonnet Baseline (same dataset, same variant)

| Run | BBB F1 | TP | FP | FN | Source |
|---|---|---|---|---|---|
| Claude Sonnet | 0.821 | — | — | — | `ablation_20260529_215932.json` (PROJECT.md baseline) |
| gpt-5.4 (this probe) | **0.804** | 43 | 2 | 19 | this run |

Delta: **−1.7 pp BBB F1** vs Claude Sonnet baseline. Well within the BBB jitter band (~4pp per
project memory). Precision improved (95.6% gpt-5.4 vs Claude baseline range), recall regressed.
This is a model-provider-property observation, not a defect (per v2.0 D-05 framing) — to be
discussed at length in Plan 09-04 comparison report.

---

## Go/Halt Decision (D-03 Step 2, D-10)

**Sanity floor (D-10):** BBB F1 ≥ 0.6 → cleared (0.8037 ≥ 0.6).

**Harness errors:** None observed. One transient `Coref batch: empty response, retrying...`
notice in stdout — the existing retry path handled it; no halt-class condition. No 401/403/429
authentication or rate-limit errors. No model-id rejection (gpt-5.4 accepted as valid OpenAI
model id at the request layer).

**Pipeline completion:** All 3 Tiers (Knowledge Acquisition, Entity Pipeline + Coreference, Link
Consolidation) ran to completion. Checkpoints written: `layer1`, `entity_candidates`,
`entity_decisions`, `layer2`, `final`. Final dedup: 45 links from 82 raw.

**Verdict:** **GO — proceed to Plan 09-03 full 5-dataset sweep on gpt-5.4. Reuse this BBB probe
JSON; no BBB re-run needed.** (D-03 Step 3 path.)

---

## Paths

- **Probe JSON:** `results/ablation_results/ablation_20260531_055235.json`
- **Probe stdout log:** `results/ablation_results/09-02-bbb-probe-stdout.log`
- **BBB pickle cache directory:** `results/phase_cache_gpt54/s_linker13/bigbluebutton/`
  - `layer1.pkl`, `entity_candidates.pkl`, `entity_decisions.pkl`, `layer2.pkl`, `final.pkl`
- **CSV link export:** `results/ablation_results/s_linker13_bigbluebutton_links.csv` (per
  `export_links_csv` in `run_ablation.py:613`)
- **Phase-3 LLM log:** `results/llm_logs/s_linker13_bigbluebutton_20260531_055235.json`

---

## Retries / Transient Errors / Rate-Limit Waits

Observed in stdout:

- 1 × empty-response retry on coreference batch (`Coref batch: empty response, retrying...`).
  Handled by existing retry path; no halt condition.
- 0 × HTTP 4xx/5xx errors.
- 0 × authentication errors.
- 0 × model-id rejections.
- 0 × rate-limit waits.

Pipeline finished in 48.4 s wall-clock; no degradation versus Claude Sonnet runtime envelope.

---

## Cache Directory Verification

Per-phase pickle pickles confirmed under fresh root (post-run `ls`):

```
results/phase_cache_gpt54/s_linker13/bigbluebutton/
├── entity_candidates.pkl  (21536 B, 2026-05-31 05:52)
├── entity_decisions.pkl   ( 1303 B)
├── final.pkl              ( 2519 B)
├── layer1.pkl             ( 3550 B)
└── layer2.pkl             ( 8285 B)
```

`./results/phase_cache/` (Claude's default) was NOT touched by this run — namespacing held.

---

## User Adjudication

- **Option selected:** `go`
- **Timestamp (UTC):** 2026-05-31T04:29:30Z
- **BBB result summary:** F1 = 0.8037 (P 0.9556, R 0.6935; TP 43, FP 2, FN 19; runtime 48.4 s; cache namespacing under `results/phase_cache_gpt54/` verified, no collision with Claude's `results/phase_cache/`).
- **Gating-rule citation:** D-03 Step 2 sanity floor (BBB F1 ≥ 0.6) cleared; D-10 cancellation rule cleared on both legs (F1 floor met AND no persistent harness errors — only one transient empty-response retry on a coreference batch, handled by the existing retry path; no auth, rate-limit, model-id, or 4xx/5xx failures).
- **Decision effect:** Proceed to Plan 09-03 full 5-dataset sweep on gpt-5.4. BBB probe JSON (`results/ablation_results/ablation_20260531_055235.json`) is reused as the bigbluebutton row in the sweep — no BBB re-run (D-03 Step 3 path).

