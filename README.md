# agent-linker

LLM-driven SAD–SAM trace link recovery for the ARDoCo benchmark suite.

## Prerequisites

- Python 3.11+
- [Claude CLI](https://github.com/anthropics/claude-code) installed and authenticated (`claude` on PATH)
- The [ardoco](https://github.com/ArDoCo/Core) repo cloned as a sibling directory:

```
adc/
  agent-linker/   ← this repo
  ardoco/         ← ARDoCo Core (provides benchmark data)
```

Clone ARDoCo if you don't have it:

```bash
git clone https://github.com/ArDoCo/Core.git ../ardoco
```

Install Python dependencies (lxml, rapidfuzz):

```bash
pip install lxml rapidfuzz
```

## Running S-Linkers

All runs go through `run_ablation.py`, which adds `src/` to the path automatically — no install step needed.

### Basic run (all 5 datasets)

```bash
python run_ablation.py --variants s_linker11
```

### Single dataset

```bash
python run_ablation.py --variants s_linker11 --datasets mediastore
```

### Multiple variants side-by-side

```bash
python run_ablation.py --variants s_linker10 s_linker11 --datasets mediastore teastore
```

### Available datasets

`mediastore`, `teastore`, `teammates`, `bigbluebutton`, `jabref`

### Current linkers

| Variant | Description | Macro F1 |
|---------|-------------|----------|
| `s_linker10` | ICSE submission — fully LLM-driven, no magic thresholds | ~93.7% |
| `s_linker11` | Uniform validation — seed links go through same 2-pass check as entity candidates | ~93.7% |
| `s_linker11a` | S-Linker11 + role-verification validation | — |

## LLM Backend

The default backend calls `claude` as a subprocess. Select a backend via `LLM_BACKEND`:

```bash
# Default — calls claude CLI
python run_ablation.py --variants s_linker11 --datasets mediastore

# Cache-through — serves cached responses, falls back to claude on miss
LLM_BACKEND=checkpoint python run_ablation.py --variants s_linker11 --datasets mediastore

# OpenAI
LLM_BACKEND=openai OPENAI_API_KEY=sk-... python run_ablation.py --variants s_linker11
```

Always uses **Claude Sonnet** (set automatically). To override:

```bash
CLAUDE_MODEL=sonnet python run_ablation.py --variants s_linker11
```

### Checkpoint backend

The checkpoint backend (`LLM_BACKEND=checkpoint`) caches every LLM response by prompt hash under `results/llm_checkpoint/`. Cache misses fall back to claude and are saved for future runs — subsequent runs on the same dataset are fully cached and near-instant.

```bash
CHECKPOINT_DIR=./results/llm_checkpoint   # default cache location
CHECKPOINT_FALLBACK=claude                # fallback backend on miss (default: claude)
```

## Output

Results are written to:

```
results/
  ablation_results/   — per-dataset link CSVs + summary JSON
  phase_cache/        — per-phase pkl checkpoints (enables fast re-runs)
  llm_logs/           — per-run JSONL query logs with token counts
  llm_checkpoint/     — prompt-keyed JSON cache (checkpoint backend)
```

Phase checkpoints in `results/phase_cache/` persist across runs, so re-running the same variant/dataset skips already-completed phases automatically.

## Benchmark data layout

Benchmark files are resolved relative to this repo:

```
../ardoco/core/tests-base/src/main/resources/benchmark/
  mediastore/
    text_2016/mediastore.txt
    model_2016/pcm/ms.repository
    goldstandards/goldstandard_sad_2016-sam_2016.csv
  teastore/  teammates/  bigbluebutton/  jabref/  ...
```

TransArc baseline CSVs (`sadSamTlr_*.csv`) are optional. If absent, the TransArc baseline row is skipped — linker results are still fully evaluated against the gold standard.
