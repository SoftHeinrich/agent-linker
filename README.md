# ALinker replication package

This repository is a self-contained, Git-tracked snapshot for the ALinker
trace-link-recovery study. It vendors the exact source and input data used for
the package, so cloning this repository is sufficient to reproduce the
deterministic evaluation and to build the paper; it does not depend on sibling
checkouts or Git submodules.

## Contents

| Path | Contents |
| --- | --- |
| `approach/` | ALinker SAD-to-SAM implementation and experiment runner |
| `evaluation/` | Stdlib-only metric studies plus the recorded TransArC results |
| `sota-links/` | Normalized recovered links and provenance for SOTA baselines |
| `benchmark/` | ARDoCo benchmark inputs and gold standards required by evaluation |
| `paper/` | LaTeX source, figures, tables, and bibliography |

The exact upstream revisions and vendoring policy are recorded in
[`docs/SOURCE_MANIFEST.md`](docs/SOURCE_MANIFEST.md). For a complete, runnable
workflow and its limits, see [`docs/REPLICATION.md`](docs/REPLICATION.md).

## Quick start

```bash
git clone <PACKAGE-URL> alinker-replication-package
cd alinker-replication-package
./scripts/verify.sh
```

`verify.sh` uses only Python 3 and runs the frozen evaluation panel against the
vendored benchmark and recorded results. It must finish with `PASS`.

To prepare the optional live LLM experiment environment:

```bash
./scripts/bootstrap-approach.sh
export OPENAI_API_KEY=...        # only for a new live run
.venv/bin/python approach/run_ablation.py --list-variants
```

To compile the manuscript (with a TeX installation that provides `latexmk`):

```bash
./scripts/build-paper.sh
```
