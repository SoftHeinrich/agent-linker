# Local paper synchronization setup verification

**Date:** 2026-09-15
**Paper commit:** `15af9fac53389ad47792c3cb7bc5f61438b05007`

The `paper/` submodule was initialized from its configured GitHub remote and
switched to its local `main` branch. The package and paper hook paths were
installed with:

```bash
git submodule update --init paper
git -C paper switch main
./scripts/install-hooks.sh
```

The paper repository has an `origin` remote for GitHub and an `overleaf`
remote for the Overleaf Git endpoint. Authentication uses the locally exported
`overleaf_token`; the token is not stored in the repository or remote URL.

The hook and synchronization scripts passed their shell syntax checks. The
readiness and dry-run checks completed with:

```text
origin/main already contains 15af9fac53389ad47792c3cb7bc5f61438b05007.
overleaf/main already contains 15af9fac53389ad47792c3cb7bc5f61438b05007.
```

The paper worktree was clean on `main`, tracking `origin/main`, after the
checks.
