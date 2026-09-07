# s_linker20_union — N=3 backend comparison

Variant **`s_linker20_union`** (union-of-sources linker, v2.6.5; registered in `run_ablation.py`),
run **3 independent times per backend** across all 5 ARDoCo benchmark projects
(mediastore, teastore, jabref, bigbluebutton, teammates).

## Where things are saved

This folder is the single home for both backends — 3 Sonnet runs + 3 gpt runs:

```
approach/results/v2.6.5_s20union/
├── README.md                       <- this file
├── sonnet/   -> ../v2.6.5_s20union_sonnet   (symlink; 3 runs; COMPLETE)
│   ├── run1/  run2/  run3/
│   │   └── <dataset>/
│   │       ├── ablation_<timestamp>.json     <- P / R / F1 + tp/fp/fn + fp_details/fn_details
│   │       ├── s_linker20_union_<dataset>_links.csv
│   │       └── .done                         <- resume marker (dataset finished OK)
│   ├── phase_cache/  llm_logs/  llm_checkpoint/   (per-run, for genuine N=3 independence)
└── gpt/                            (3 runs; COMPLETE — fresh live gpt-5.4 calls, 2026-06-21)
    └── run1/  run2/  run3/  (same per-dataset layout as sonnet/)
```

> The `sonnet/` entry is a **symlink** to the original sweep dir `results/v2.6.5_s20union_sonnet/`
> (left in place so existing references keep working). Resolve it transparently —
> `sonnet/run1/mediastore/...` works as if it were a real subfolder.

Logs (progress + per-dataset stdout) live **outside** this folder, under `approach/logs/`:

| Backend | Results dir | Log dir |
|---|---|---|
| Sonnet | `results/v2.6.5_s20union_sonnet/` (= `sonnet/`) | `logs/v2.6.5_s20union_sonnet/` |
| gpt-5.4 | `results/v2.6.5_s20union/gpt/` | `logs/v2.6.5_s20union_gpt/` |

`logs/<sweep>/PROGRESS.log` is the authoritative timeline; `.ALL_DONE` in the log dir = whole sweep finished.

## Which backend each slot used

| Slot | Backend env | Model | Launcher | Status |
|---|---|---|---|---|
| `sonnet/` | `LLM_BACKEND=claude` | `CLAUDE_MODEL=sonnet` (Claude Sonnet, via `claude` CLI subprocess) | `run_s20union_sonnet_n3.sh` | ✅ complete (2026-06-20) |
| `gpt/` | `LLM_BACKEND=openai` | `OPENAI_MODEL_NAME=gpt-5.4` (needs `OPENAI_API_KEY`) | `run_s20union_gpt_n3.sh` | ✅ complete (2026-06-21) |

> **gpt slot = fresh live calls.** Older gpt-5.4 runs of this variant exist at
> `results/v2.6.5/full_s_linker20_union_run1..6/` (2026-06-10) but are **deliberately not reused** here —
> this folder's `gpt/` is produced by a new live sweep matched to the Sonnet methodology
> (per-run isolated `phase_cache`, identical launcher, N=3).

Both launchers are strictly sequential, with cooldowns between datasets/runs, retry-once on an
empty/failed dataset, and resume via per-`(run,dataset)` `.done` markers (safe to re-run after a crash).

## Sonnet results (complete)

N=3 mean **macro-F1 = 0.9276** (per-run 0.9408 / 0.9074 / 0.9345; sd 0.0145). No failures or retries.

| dataset | run1 | run2 | run3 | mean F1 |
|---|:--:|:--:|:--:|:--:|
| mediastore | 0.9841 | 0.9492 | 0.9677 | 0.9670 |
| teastore | 0.9818 | 0.9310 | 0.9643 | 0.9590 |
| jabref | 1.0000 | 0.9730 | 1.0000 | 0.9910 |
| bigbluebutton | 0.8257 | 0.7810 | 0.8364 | 0.8143 |
| teammates | 0.9123 | 0.9027 | 0.9043 | 0.9064 |

bigbluebutton is the floor; jabref is essentially solved.

## gpt-5.4 results (complete — fresh live sweep)

N=3 mean **macro-F1 = 0.8939** (per-run 0.8914 / 0.8963 / 0.8940; sd 0.0020 — very stable). 51 min wall, no failures.

| dataset | run1 | run2 | run3 | mean F1 |
|---|:--:|:--:|:--:|:--:|
| mediastore | 0.9667 | 0.9508 | 0.9508 | 0.9561 |
| teastore | 0.9811 | 0.9811 | 0.9811 | 0.9811 |
| jabref | 0.9143 | 0.9412 | 0.9412 | 0.9322 |
| bigbluebutton | 0.7547 | 0.7736 | 0.7619 | 0.7634 |
| teammates | 0.8403 | 0.8348 | 0.8348 | 0.8366 |

## Head-to-head (N=3 means)

| dataset | Sonnet | gpt-5.4 | Δ (S − G) |
|---|:--:|:--:|:--:|
| mediastore | 0.9670 | 0.9561 | +0.0109 |
| teastore | 0.9590 | 0.9811 | **−0.0221** |
| jabref | 0.9910 | 0.9322 | +0.0588 |
| bigbluebutton | 0.8143 | 0.7634 | +0.0509 |
| teammates | 0.9064 | 0.8366 | +0.0698 |
| **macro-F1** | **0.9276** | **0.8939** | **+0.0337** |

**Sonnet wins overall** (+3.4pp macro), leading on 4/5 datasets — biggest gaps on teammates (+7.0pp) and jabref (+5.9pp). gpt-5.4's only win is teastore (+2.2pp), and it is markedly more stable run-to-run (sd 0.002 vs 0.015). bigbluebutton is the floor for both (Sonnet 0.814, gpt 0.763).

## How to (re)produce / score

Re-run either sweep from `approach/` (idempotent — `.done` markers skip finished work; delete a
run's `.done` to force a re-run):

```bash
cd approach
bash run_s20union_sonnet_n3.sh       # Sonnet  -> results/v2.6.5_s20union_sonnet/ (= sonnet/)
OPENAI_API_KEY=... bash run_s20union_gpt_n3.sh   # gpt-5.4 -> results/v2.6.5_s20union/gpt/
```

Score any sweep slot (reads the per-dataset `ablation_*.json` `F1` fields):

```bash
python3 - <<'PY'
import json, glob, statistics as st
SLOT="results/v2.6.5_s20union/sonnet"   # or .../gpt
DS=["mediastore","teastore","jabref","bigbluebutton","teammates"]; V="s_linker20_union"
f1=lambda r,ds: json.load(open(sorted(glob.glob(f"{SLOT}/{r}/{ds}/ablation_*.json"))[-1]))[ds][V]["F1"]
macro=[st.mean(f1(r,ds) for ds in DS) for r in ("run1","run2","run3")]
print("per-run macro:", [round(m,4) for m in macro], "| mean:", round(st.mean(macro),4))
PY
```

Each `ablation_*.json` also carries `fp_details` / `fn_details` (sentence, component, confidence, text)
for per-link error analysis.
