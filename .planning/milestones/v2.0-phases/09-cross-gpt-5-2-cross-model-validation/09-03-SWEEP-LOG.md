# Phase 9 Plan 03 — Full Sweep Log (s_linker13 on gpt-5.4, 4 datasets + reused BBB)

**Plan:** 09-03
**Purpose:** D-03 Step 3 full 5-dataset cross-model evaluation of `s_linker13` on `gpt-5.4`.
**Outcome:** All 5 datasets resolved; 5-dataset macro F1 = **0.9077**.

---

## Mode Chosen

**Option B** — sweep 4 datasets (mediastore, teastore, teammates, jabref), reuse Plan 09-02
BBB probe JSON for the bigbluebutton row.

**Rationale (1 sentence):** Plan 09-02's BBB probe was harness-healthy (BBB F1 0.8037, single
recovered coref retry, no anomalous variance signal), so D-03 Step 3's default no-retest policy
applies and BBB API cost is saved.

---

## Invocation Command (exact, with env vars)

```bash
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.4 \
PHASE_CACHE_DIR=./results/phase_cache_gpt54 \
python run_ablation.py \
  --datasets mediastore teastore teammates jabref \
  --variants s_linker13 \
  --results-dir results/ablation_results 2>&1 \
  | tee results/ablation_results/09-03-sweep-stdout.log
```

Same three env overrides as Plan 09-02 — `LLM_BACKEND=openai` selects OpenAI dispatch at
`run_ablation.py:get_backend()` (line 427); `OPENAI_MODEL_NAME=gpt-5.4` wins over the
`setdefault("gpt-5.2")` at `run_ablation.py:436` (and at `llm_client.py:112`, `s_linker13.py:171`);
`PHASE_CACHE_DIR=./results/phase_cache_gpt54` namespaces s_linker13's per-phase pickle cache
(`s_linker13.py:1160`) so Claude's `./results/phase_cache/` is not touched.

Runner-printed backend descriptor (line 3 of stdout log): `openai (gpt-5.4)` — env override
confirmed effective.

**No edits to** `run_ablation.py`, `llm_client.py`, `s_linker13.py`, or any `prompts*.py` file
during this plan — env-override surface only (verified via `git status --short` post-run).

---

## Timestamps

| Event | UTC time |
|---|---|
| Sweep start (wall) | 2026-05-31T04:32:10Z |
| Sweep end (wall)   | 2026-05-31T04:34:46Z (per JSON filename `ablation_20260531_063446.json`, runner local→UTC) |
| Sweep elapsed (sum of dataset `time` fields) | **154.5 s** (≈ 2 m 35 s) |

Per-dataset elapsed (from JSON `time` field):

| Dataset | Wall time | n_links emitted |
|---|---|---|
| mediastore | 29.5 s | 31 |
| teastore   | 39.9 s | 27 |
| teammates  | 66.7 s | 74 |
| jabref     | 18.4 s | 19 |
| **subtotal** | **154.5 s** | 151 |

BBB (reused from Plan 09-02, not part of this invocation): 48.4 s, 45 links emitted.

---

## Per-Dataset F1 Table (5 rows) — with JSON-path provenance

| Dataset | F1 | P | R | TP | FP | FN | JSON source (basename) |
|---|---|---|---|---|---|---|---|
| mediastore    | **0.9677** | 0.9677 | 0.9677 | 30 | 1  | 1  | `ablation_20260531_063446.json` (this sweep) |
| teastore      | **1.0000** | 1.0000 | 1.0000 | 27 | 0  | 0  | `ablation_20260531_063446.json` (this sweep) |
| teammates     | **0.7939** | 0.7027 | 0.9123 | 52 | 22 | 5  | `ablation_20260531_063446.json` (this sweep) |
| bigbluebutton | **0.8037** | 0.9556 | 0.6935 | 43 | 2  | 19 | `ablation_20260531_055235.json` (reused from Plan 09-02 probe) |
| jabref        | **0.9730** | 0.9474 | 1.0000 | 18 | 1  | 0  | `ablation_20260531_063446.json` (this sweep) |

Full file paths:

- This sweep: `results/ablation_results/ablation_20260531_063446.json`
- Plan 09-02 probe (BBB reused): `results/ablation_results/ablation_20260531_055235.json`
- Stdout log (this sweep): `results/ablation_results/09-03-sweep-stdout.log`

---

## Macro F1 (mean of 5 per-dataset F1)

```
Macro F1 = (0.9677 + 1.0000 + 0.7939 + 0.8037 + 0.9730) / 5
         = 4.5384 / 5
         = 0.9077
```

**Macro F1 = 0.9077.**

Plan 09-04 will recompute this from the JSON files (this log is informational, the JSONs are
authoritative). The compute code used here is reproducible via:

```bash
python3 -c "
import json, glob, os
jsons = sorted(glob.glob('results/ablation_results/ablation_*.json'),
               key=os.path.getmtime, reverse=True)
seen = {}
for jp in jsons:
    try: d = json.load(open(jp))
    except Exception: continue
    if not isinstance(d, dict): continue
    for ds, vmap in d.items():
        if isinstance(vmap, dict) and isinstance(vmap.get('s_linker13'), dict) and ds not in seen:
            seen[ds] = (vmap['s_linker13']['F1'], jp)
required = ['mediastore','teastore','teammates','bigbluebutton','jabref']
for ds in required:
    f1, jp = seen[ds]; print(f'{ds:14s} F1={f1:.4f}  source={os.path.basename(jp)}')
print(f'Macro F1 = {sum(seen[d][0] for d in required)/5:.4f}')
"
```

---

## GATE-01 Cross-Model Check (informational; Plan 09-04 owns the verdict)

| Gate | Threshold | Cross-model observed (gpt-5.4) | Verdict |
|---|---|---|---|
| GATE-01 | macro F1 ≥ 0.93 | 0.9077 | **DOES NOT HOLD** |

Per Phase 9 framing (D-05): a cross-model macro F1 below GATE-01 is a **model-provider-property
finding**, not a defect in `s_linker13` — both "holds cross-model" and "does not hold —
model-provider-property finding" are acceptable Plan 09-04 conclusions. **No fix-it action items
will be added in Plan 09-04** per the v2.0 standing policy.

The Claude Sonnet baseline (`ablation_20260529_215932.json`) is `0.9506` macro on the same
5 datasets, so the cross-model gap is **−4.3 pp** (Claude → gpt-5.4). This is consistent with
project memory's documented V32 cross-model gap (`MEMORY.md §GPT-5.2 Compatibility`:
V32 Claude 94.5% → gpt-5.2 90.6% = −3.9 pp; V32 → gpt-5.4 = 87.7%) — i.e., the s_linker13
gpt-5.4 gap (−4.3 pp) is in the same family as prior cross-provider gaps measured on a
different artifact (V32).

---

## Retries / Transient Errors / Rate-Limit Waits

| Event | Where | Action | Halt-class? |
|---|---|---|---|
| `Coref batch: empty response, retrying...` | mediastore Tier 2 coreference | Handled by existing retry path; coref produced 3 links in final result | No (documented `approve-biased fallback on LLM failure` pattern; same event type observed in Plan 09-02 BBB) |
| 0 × 401/403 auth errors | — | — | n/a |
| 0 × 429 rate-limit waits | — | — | n/a |
| 0 × 5xx server errors | — | — | n/a |
| 0 × model-id rejections | — | — | n/a |
| 0 × dropped datasets | All 4 ran to completion | — | n/a |

Grep audit of `09-03-sweep-stdout.log` for `(transient|empty response|retry|error|Error|ERROR|429|401|403|5xx)`
returns only the one MS coref retry line above. **Sweep is harness-healthy.**

---

## Pipeline Completion Verification (per-dataset checkpoint trail)

All 4 swept datasets completed all 3 Tiers (Knowledge Acquisition, Entity Pipeline +
Coreference, Link Consolidation) with all 5 checkpoints written
(`layer1`, `entity_candidates`, `entity_decisions`, `layer2`, `final`):

```
results/phase_cache_gpt54/s_linker13/
├── bigbluebutton/  (from Plan 09-02)
├── jabref/
├── mediastore/
├── teammates/
└── teastore/
```

Each dataset directory contains all 5 expected `.pkl` files. Claude's
`./results/phase_cache/` was NOT modified by this sweep — namespacing held cleanly.

---

## Notable Per-Dataset Observations (informational only; not deviations)

- **teastore**: P=R=F1=1.000 (TP=27, FP=0, FN=0) — gpt-5.4 perfect on teastore.
- **mediastore**: F1=0.9677, single coref FP, single FN.
- **jabref**: F1=0.9730, single coref FP, perfect recall.
- **teammates**: F1=0.7939 — 22 FP (17 from seed source, 4 coref, 1 entity), 5 FN. This is the
  dataset that drags the macro F1 down below GATE-01. Pattern is consistent with project
  memory's observation that GPT over-extracts seeds on doc-heavy projects (teammates has 198
  sentences, the largest in the benchmark). To be discussed in Plan 09-04 comparison report as a
  model-provider-property finding (per D-05); no fix-it action items per v2.0 standing policy.
- **bigbluebutton (reused)**: F1=0.8037, P=0.9556, R=0.6935 — high precision, lower recall vs
  Claude.

---

## Consumed JSON Files (Plan 09-04 input pointers)

1. `results/ablation_results/ablation_20260531_063446.json` — this sweep: MS, TS, TM, JAB
2. `results/ablation_results/ablation_20260531_055235.json` — Plan 09-02 BBB probe (reused)
3. `results/ablation_results/09-03-sweep-stdout.log` — this sweep stdout (provenance trail)
4. `results/llm_logs/s_linker13_{mediastore,teastore,teammates,jabref}_<ts>.json` — per-dataset
   phase-3 LLM call logs (sweep)
5. `results/ablation_results/s_linker13_{mediastore,teastore,teammates,jabref}_links.csv` —
   per-dataset emitted link CSVs (sweep)

---

## CROSS-01 + CROSS-02 Closure

Per D-02 single-arm Phase 8 retro-designation, **this single sweep satisfies both
CROSS-01 (s_linker13 on cross-model, 5 datasets) and CROSS-02** (s_linker14 = no-op; collapsed
into s_linker13 per Phase 8). Per-dataset F1 evidence exists for all 5 datasets on gpt-5.4 with
no backend-specific prompt tailoring (env-override surface only, GATE-06 audit input for the
Plan 09-04 comparison report).

