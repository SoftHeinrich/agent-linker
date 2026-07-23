# Replication guide

## What is reproducible offline

The package contains the benchmark inputs, gold standards, recovered links, and
recorded run results needed to reproduce the paper's deterministic metrics. No
network access, API key, sibling checkout, or untracked data is needed for:

```bash
./scripts/verify.sh
```

The script sets `TRANSARC_BENCHMARK` and `TRANSARC_RESULTS_DIR` to vendored
paths, runs the frozen golden-panel regression check, then prints the SAD-Code
and SAD-SAM tables. The check is the acceptance criterion: it ends with
`PASS: mini-src/metrics.py reproduces the frozen golden panel`.

SOTA inputs are preserved in `sota-links/`. Their provenance, normalization,
coverage, and scoring caveats are documented in `sota-links/README.md`. The
evaluation scripts and their recorded data are preserved in `evaluation/`.

## Environment

Required for the deterministic replication:

- Git
- Python 3.11 or newer (the evaluation itself uses only the standard library)

Required only to compile the paper:

- a TeX Live installation with `latexmk`, `pdflatex`, and BibTeX

Required only to run a new ALinker experiment:

- Python 3.11 or newer
- an OpenAI key for the configured OpenAI backend, or a configured Claude CLI
- network access for API calls and Python package installation

Create the live-experiment environment with `./scripts/setup.sh`. It installs
the approach dependencies and, if `approach/.env` does not already exist,
creates it with an absolute `ALINKER_BENCHMARK` setting pointing to this
repository's vendored `benchmark/`. Existing `.env` files are never overwritten.
Override that setting with `ALINKER_BENCHMARK=/path/to/benchmark` if needed.
Results and logs are written under `approach/` and are ignored by Git.

## Live experiment caveat

The recorded links and all downstream metrics are exactly reproducible from this
repository. A fresh LLM run is *not* guaranteed to be bit-identical: hosted
models, provider-side defaults, and API availability can change. Treat a fresh
run as a robustness rerun and compare its normalized links with the preserved
artifacts in `sota-links/`; do not replace those artifacts when reproducing the
reported results.

## Paper build

```bash
./scripts/build-paper.sh
```

The generated `paper/main.pdf` is intentionally ignored. The committed TeX,
figures, tables, bibliography, and evaluation outputs are the authoritative
paper inputs.
