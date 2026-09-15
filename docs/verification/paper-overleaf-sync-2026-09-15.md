# Paper/Overleaf synchronization verification

**Date:** 2026-09-15
**Package source before migration:** `4b6db31c8b0a77a08b8a699e82aae9a608b496fb`
**Paper commit:** `15af9fac53389ad47792c3cb7bc5f61438b05007`

## Configuration checked

- The parent repository records `paper/` as a submodule at
  `git@github.com:SoftHeinrich/alinker-paper.git`, branch `main`.
- The paper repository has local remotes named `origin` and `overleaf`.
- The credential-bearing Overleaf URL is local Git configuration only and is
  intentionally not recorded here.
- Both repositories use a tracked `.githooks/post-commit` directory after
  `./scripts/install-hooks.sh` sets `core.hooksPath=.githooks`.

## Commands and results

### Paper migration and remote identity

The package paper tree was compared with the new submodule checkout using
`rsync -rcn --delete`, excluding only LaTeX build outputs and Python cache
directories.

```text
Paper source comparison: PASS (backup and submodule match outside ignored build artifacts).
Submodule worktree: clean.
```

The remote heads were then checked with:

```bash
git -C paper ls-remote origin refs/heads/main
git -C paper ls-remote overleaf refs/heads/main
```

```text
15af9fac53389ad47792c3cb7bc5f61438b05007  refs/heads/main
15af9fac53389ad47792c3cb7bc5f61438b05007  refs/heads/main
```

With `core.hooksPath=.githooks` enabled in a temporary paper checkout, the
paper post-commit hook ran during the final paper commit and reported:

```text
Pushed 15af9fac53389ad47792c3cb7bc5f61438b05007 to origin/main.
Pushed 15af9fac53389ad47792c3cb7bc5f61438b05007 to overleaf/main.

The first parent post-commit invocation exposed the inherited parent Git index
context when the hook called into the submodule:

```text
fatal: .git/index: index file open failed: Not a directory
Paper worktree is not clean; commit the paper before syncing.
```

The paper sync script now clears the inherited repository context before its
paper-repository checks. The subsequent paper commit exercised the nested hook
and pushed the corrected commit to both remotes.
```

### Hook and sync checks

```bash
bash -n .githooks/post-commit paper/.githooks/post-commit \
  paper/scripts/sync-paper.sh scripts/install-hooks.sh \
  scripts/sync-paper-overleaf.sh
./scripts/sync-paper-overleaf.sh --check
./scripts/sync-paper-overleaf.sh --dry-run
```

```text
PASS: hook and sync scripts parse
origin/main already contains 15af9fac53389ad47792c3cb7bc5f61438b05007.
overleaf/main already contains 15af9fac53389ad47792c3cb7bc5f61438b05007.
origin/main already contains 15af9fac53389ad47792c3cb7bc5f61438b05007.
overleaf/main already contains 15af9fac53389ad47792c3cb7bc5f61438b05007.
```

The same script was exercised against two local bare remotes with
`PAPER_BRANCH=sync-test OVERLEAF_BRANCH=sync-test`:

```text
origin/sync-test already contains 15af9fac53389ad47792c3cb7bc5f61438b05007.
overleaf/sync-test already contains 15af9fac53389ad47792c3cb7bc5f61438b05007.
Dry-run push would update origin/sync-test to f85be34b89ab1d8f4e0a2288c5c76cde8114ef68.
Dry-run push would update overleaf/sync-test to f85be34b89ab1d8f4e0a2288c5c76cde8114ef68.
origin/sync-test is behind HEAD; sync is ready.
overleaf/sync-test and HEAD have diverged; reconcile them before syncing.
divergence_exit=1
```

The divergence result is intentional: the script refuses to overwrite a
remote that is ahead or has a different history.

### Deterministic package verification

```bash
./scripts/verify.sh
```

```text
OK arm-default every generator reports arm 's120' (7/7 found)
PASS: mini-src/metrics.py reproduces the frozen golden panel (10 cells, sad-code + sad-sam).
Command exit status: 0
```

The run printed all five project rows for both tasks and completed without a
data or evaluation change.

### Local paper build

```bash
./scripts/build-paper.sh
```

```text
latexmk is required to build the paper (install TeX Live with latexmk).
Command exit status: 1
```

The local environment has no `latexmk`, `pdflatex`, `bibtex`, or `tectonic`
executable. Local PDF compilation is therefore blocked by the missing TeX
toolchain; this verification does not claim an Overleaf build result.

### Non-gating whitespace check

The staging check
`git diff --cached --check` reported existing trailing whitespace in imported
LaTeX/CSV content and binary-PDF noise. The scoped check for the new hook and
configuration files passed. No paper wording or data files were rewritten to
silence that diagnostic.
