# ALinker replication package

This repository is a Git-tracked replication package for the ALinker
trace-link-recovery study. It vendors the exact source and input data used for
the deterministic evaluation. The manuscript is maintained in the
[`alinker-paper`](https://github.com/SoftHeinrich/alinker-paper) Git repository
and is pinned here as the `paper/` submodule.

## Contents

| Path | Contents |
| --- | --- |
| `approach/` | ALinker SAD-to-SAM implementation and experiment runner |
| `evaluation/` | Stdlib-only pipeline that scores the runs and renders the paper's tables |
| `studies/` | Side analyses that informed the paper's choices but write none of its tables |
| `sota-links/` | Normalized recovered links and provenance for SOTA baselines |
| `benchmark/` | ARDoCo benchmark inputs and gold standards required by evaluation |
| `paper/` | Pinned paper submodule: LaTeX source, figures, tables, and bibliography |

The exact upstream revisions and vendoring policy are recorded in
[`docs/SOURCE_MANIFEST.md`](docs/SOURCE_MANIFEST.md). For a complete, runnable
workflow and its limits, see [`docs/REPLICATION.md`](docs/REPLICATION.md).
For the distinction between S21, the router, and the S23 verification variants,
see [`docs/VARIANT_COMPARISON.md`](docs/VARIANT_COMPARISON.md).
The next deliberately narrow recall experiment is specified in
[`docs/S24_PROPOSAL.md`](docs/S24_PROPOSAL.md).

## Quick start

```bash
git clone --recurse-submodules <PACKAGE-URL> alinker-replication-package
cd alinker-replication-package
./scripts/verify.sh
```

`verify.sh` uses only Python 3 and runs the frozen evaluation panel against the
vendored benchmark and recorded results. It must finish with `PASS`.

To prepare the optional live LLM experiment environment and configure its
vendored benchmark path:

```bash
./scripts/setup.sh
export OPENAI_API_KEY=...        # only for a new live run
.venv/bin/python approach/run_ablation.py --list-variants
```

To compile the manuscript (with a TeX installation that provides `latexmk`):

```bash
./scripts/build-paper.sh
```

## Paper and Overleaf synchronization

The paper submodule has two local remotes: `origin` is the GitHub repository
that makes the pinned submodule commit clonable, and `overleaf` is the private
Overleaf Git remote. Configure the latter locally with the project URL and a
credential helper or token supplied by your environment; credentials are not
stored in `.gitmodules`:

```bash
git submodule update --init paper
git -C paper remote add overleaf https://git.overleaf.com/<project-id>
./scripts/install-hooks.sh
```

After the hook is installed, a commit in `paper/` checks both remotes and pushes
the commit to `origin/main` and `overleaf/main` when each remote is an ancestor
of the new commit. A remote that is ahead or has diverged stops the sync and
requires an explicit reconciliation. Use `./scripts/sync-paper-overleaf.sh
--check` to inspect readiness or run the command without a hook.

After a successful paper commit in an initialized submodule, the child hook
stages only the `paper` gitlink and creates a parent commit named
`chore: update paper submodule`. It refuses to overwrite a different staged
pointer or an unresolved parent merge. The hook does not push the parent
repository; push that parent commit explicitly. Git does not clone hooks, so
run `./scripts/install-hooks.sh` once in each fresh checkout.
