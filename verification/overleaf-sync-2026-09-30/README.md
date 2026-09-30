# Overleaf sync repair, 2026-09-30

## Observed failure

Both repositories had `core.hooksPath=.githooks` and executable hooks.
The paper HEAD was `287371bbb5aa94dda71bc4e134d73d56b7727b66`.
Running `./scripts/sync-paper-overleaf.sh --check` exited 1 with:

```text
Paper worktree is not clean; commit the paper before syncing.
```

Only untracked files under `paper/verification/bibliography-independent-2026-09-30/`
were present; there were no tracked edits. Fetching each paper remote's `main`
branch succeeded. `git -C paper rev-list --left-right --count HEAD...origin/main`
returned `0 0`; the same command for `overleaf/main` returned `4 1`.
`git -C paper show overleaf/main` identified commit
`026e6ebe2bc4dc71150d8271009b8a7e2d08d64c`: it only changed
`.githooks/post-commit` and `scripts/sync-paper.sh` from mode 100755 to 100644.
The divergent history was a second blocker even after the untracked-file guard.

## Repair

Merged `overleaf/main` without discarding either history and restored both
executable bits before committing. Changed the sync guard to accept untracked
files while continuing to reject staged and unstaged tracked edits. No paper
prose or bibliography content changed in this repair. Existing untracked files
were left in place and were not committed or published.

The paper merge commit is `9f8b44dd220e48c97ee489038b110a7db3b6421d`.
Its commit hook was disabled for that commit so the explicit sync command could
be recorded before the parent verification commit. Installed hooks remain enabled.
Remote-ahead and divergence checks remain in place; future divergent edits still
require reconciliation. No force push was used.

## Verification commands and configuration

Run from the parent repository root:

```bash
bash scripts/test-submodule-pointer-hook.sh
./scripts/sync-paper-overleaf.sh
./scripts/sync-paper-overleaf.sh --check
```

- `fixture.txt`: successful hook integration test using temporary local bare
  GitHub, Overleaf, and parent remotes on branch `main`. Confirms the child
  commit reaches both remotes, the parent pointer is updated and pushed,
  an untracked file remains local, and both staged and unstaged tracked edits
  are rejected. The script exited 0.
- `live-sync.txt`: successful non-forced pushes of the repaired paper commit to
  the configured `origin/main` and `overleaf/main`; command exited 0.
- `live-check.txt`: follow-up fetch and comparison against both live remote
  tracking heads; command exited 0.
- `checks.txt`: exact syntax, whitespace, content-diff, and executable-mode
  verification commands with their results and exit statuses.

This verification covers Git synchronization. It does not verify PDF compilation.
