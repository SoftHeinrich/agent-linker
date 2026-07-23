# Source manifest

This package vendors file snapshots rather than using submodules. Consequently,
the repository remains independently clonable even if an upstream repository is
moved, made private, or advances after this package is released. The Git commit
that contains this manifest is the package release identifier.

| Package path | Upstream repository | Pinned source commit | Source date |
| --- | --- | --- | --- |
| `approach/` | `https://github.com/SoftHeinrich/agent-linker.git` | `d9e2a711725fa5f057718f2f20dac4c9bdfe13b7` | 2026-07-09 |
| `evaluation/` | `https://github.com/SoftHeinrich/transarc-emp.git` | `bb98a60522a4964c8530527624484cf988f67d95` | 2026-07-09 |
| `paper/` | `https://github.com/SoftHeinrich/alinker-paper.git` | `988e9b12234974b5a17039d2061197b5e54c7062` | 2026-07-16 |
| `sota-links/` | `https://github.com/SoftHeinrich/sota-recovered-links.git` | `837df98931086178f1ce71dc8ec0c25cf6b148dc` | 2026-07-10 |
| `benchmark/` | `https://github.com/ardoco/ardoco.git` (`core/tests-base/src/main/resources/benchmark`) | `2fd8bee662950c9ebec6b537de09651dc5cc43e2` | 2026-07-16 |

Only the benchmark directory required by the approach and evaluation is copied
from ARDoCo; this is intentional. The package does not require the full ARDoCo
build to reproduce its recorded metrics.

The approach runner has one packaging-only adaptation: it defaults to the
vendored `benchmark/` directory and accepts `ALINKER_BENCHMARK` and
`ALINKER_CLI_RESULTS` overrides. No linker logic, prompts, recorded links, or
evaluation logic was changed.
