# Main branch has been messy, see s20u branch for the clean, consolidated version
# agent-linker

Main branch has been messy, 

LLM-driven SAD-SAM trace link recovery for the ARDoCo benchmark suite.

The active repo now keeps:

- `ILinker1`, `ILinker2`, `ILinker3`
- `SLinker` through `SLinker11a`
- the prompt files and v2-stack modules those retained linkers need

Archived families such as `agent_linker*`, `cnr*`, `alinker`, and the older `ilinker2_v*` line live under [archive/README.md](/home/yu/project/adc/agent-linker/archive/README.md).

## Prerequisites

- Python 3.11+
- [Claude CLI](https://github.com/anthropics/claude-code) installed and authenticated if you use the default backend
- The [ardoco](https://github.com/ArDoCo/Core) repo cloned as a sibling directory:

```text
adc/
  agent-linker/
  ardoco/
```

Install dependencies:

```bash
pip install -e ".[dev,openai]"
```

## Running

[run_ablation.py](/home/yu/project/adc/agent-linker/run_ablation.py) is lightweight now and only supports the retained variants.

```bash
# Default retained variant
python run_ablation.py

# Single dataset
python run_ablation.py --datasets mediastore

# Explicit retained variants
python run_ablation.py --variants i1 i2 i3
python run_ablation.py --variants s_linker s_linker7 s_linker11a --datasets teastore

# Show supported variants
python run_ablation.py --list-variants
```

Supported dataset names: `mediastore`, `teastore`, `teammates`, `bigbluebutton`, `jabref`

Variant naming notes:

- `s_linker1` is accepted as an alias for `s_linker`
- `ilinker1`, `ilinker2`, and `ilinker3` are accepted as aliases for `i1`, `i2`, and `i3`

## LLM Backend

The default backend calls `claude` as a subprocess.

```bash
# Default
python run_ablation.py --datasets mediastore

# Checkpoint cache with fallback
LLM_BACKEND=checkpoint python run_ablation.py --datasets mediastore

# OpenAI
LLM_BACKEND=openai OPENAI_API_KEY=sk-... python run_ablation.py --datasets mediastore
```

The retained workflow defaults to `CLAUDE_MODEL=sonnet` and `OPENAI_MODEL_NAME=gpt-5.2`.

## Outputs

Results are written to:

```text
results/
  ablation_results/   per-dataset link CSVs and summary JSON
  phase_cache/        linker checkpoints
  llm_logs/           per-run JSONL query logs
  llm_checkpoint/     prompt cache for checkpoint backend
```

## Tests

The active test suite is intentionally small and lives under `tests/`.

```bash
pytest
```

Historical experiment scripts remain archived under [archive/test_scripts](/home/yu/project/adc/agent-linker/archive/test_scripts).
