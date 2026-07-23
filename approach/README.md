# agent-linker — s20U branch

Minimal slice of the experimental linker repo, trimmed to **only** what is
needed to run the `s_linker20_union` ("s20U") SAD-SAM sweep against the ARDoCo
benchmark suite. Everything unrelated (other linker families, planning docs,
logs, results, archives, tests, analysis scripts) has been removed on this
branch — see `git log master` for the full history.

## What `s20U` is

`s_linker20_union` (class `SLinker20Union`) is the v2.6.5 **minimized-prompt,
standalone** SAD-SAM linker: Framing C two-pass UNION consensus, no inheritance
from any other linker, all constants inlined. The `s_linker20_union_noknow`
variant reuses the same module with a knowledge-disable flag.

## Prerequisites

- Python 3.11+
- The [ardoco](https://github.com/ArDoCo/Core) repo cloned as a sibling so the
  benchmark inputs resolve at `../ardoco/core/tests-base/src/main/resources/benchmark`.
- An OpenAI key (`OPENAI_API_KEY`) for the `gpt-5.4` backend, or a Claude CLI for
  the Sonnet backend. `.env` selects the active backend (`LLM_BACKEND=openai`).

Install:

```bash
pip install -e ".[openai]"
```

## Running

```bash
# single dataset, one run
python run_ablation.py --variants s_linker20_union --datasets mediastore

# list registered variants
python run_ablation.py --list-variants
```

Supported datasets: `mediastore`, `teastore`, `teammates`, `bigbluebutton`, `jabref`.

### N=3 sweep runners

The retained shell scripts drive the strictly-sequential N=3 sweeps with
cooldowns, retry-once, and per-`(run,dataset)` resume markers:

| script | variant | backend |
| --- | --- | --- |
| `run_s20union_gpt_n3.sh` | `s_linker20_union` | gpt-5.4 |
| `run_s20union_sonnet_n3.sh` | `s_linker20_union` | Sonnet |
| `run_s20union_gpt_re_medium_n3.sh` | `s_linker20_union` | gpt-5.4 (reasoning=medium) |
| `run_s20union_noknow_gpt_n3.sh` | `s_linker20_union_noknow` | gpt-5.4 |
| `run_s20union_noknow_sonnet_n3.sh` | `s_linker20_union_noknow` | Sonnet |

```bash
OPENAI_API_KEY=sk-... ./run_s20union_gpt_n3.sh
```

## Outputs

Each run writes link CSVs, summary JSON, and per-run caches/logs under the
`results/` and `logs/` directories created by the runner (both are gitignored).
