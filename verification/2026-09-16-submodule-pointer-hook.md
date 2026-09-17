# Submodule pointer hook verification

Date: 2026-09-16

This record describes the original local-commit behavior. The parent auto-push
extension and its verification are recorded in
`verification/2026-09-17-auto-push-paper-pointer.md`.

## Configuration under test

- `paper/.githooks/post-commit` synchronizes the paper remotes, locates the
  superproject, clears the child repository's inherited Git environment, and
  invokes the parent helper from the superproject root.
- `scripts/update-submodule-pointer.sh` stages only the requested submodule
  gitlink and creates `chore: update <submodule> submodule` when the parent
  pointer is stale. It refuses unresolved conflicts and a different staged
  pointer.
- `.githooks/post-commit` continues the existing paper remote synchronization
  after the parent pointer commit.

The hook does not push the parent repository. The resulting parent commit must
be pushed explicitly.

## Commands and results

Syntax validation:

```text
bash -n scripts/update-submodule-pointer.sh \
  paper/.githooks/post-commit .githooks/post-commit \
  scripts/test-submodule-pointer-hook.sh
PASS
```

End-to-end disposable-fixture test:

```text
scripts/test-submodule-pointer-hook.sh
Pushed ab16ef3d6b328e6529300f42afaa58a86bc99c41 to origin/main.
Pushed ab16ef3d6b328e6529300f42afaa58a86bc99c41 to overleaf/main.
origin/main already contains ab16ef3d6b328e6529300f42afaa58a86bc99c41.
overleaf/main already contains ab16ef3d6b328e6529300f42afaa58a86bc99c41.
[main 67173d6] chore: update paper submodule
 1 file changed, 1 insertion(+), 1 deletion(-)
submodule pointer hook fixture passed: child=ab16ef3d6b328e6529300f42afaa58a86bc99c41 parent=ab16ef3d6b328e6529300f42afaa58a86bc99c41 origin=ab16ef3d6b328e6529300f42afaa58a86bc99c41 overleaf=ab16ef3d6b328e6529300f42afaa58a86bc99c41
```

The fixture verified that one child commit updates both child remotes, creates
one parent pointer commit, leaves the parent clean, and leaves the child clean.

## Failure evidence retained during implementation

The first fixture run stopped before testing the hook because Git rejected the
fixture's local submodule transport:

```text
fatal: transport 'file' not allowed
fatal: clone of '/tmp/.../origin.git' into submodule path '/tmp/.../parent/paper' failed
```

After allowing the disposable fixture's local transport, the next run exposed
the hook boundary bug:

```text
Pushed ... to origin/main.
Pushed ... to overleaf/main.
error: pathspec 'paper' did not match any file(s) known to git
Did you forget to 'git add'?
```

The child hook was inheriting the child repository's Git environment while the
parent helper queried the parent index. The hook now clears that environment
and changes to the superproject before invoking the helper; the final fixture
run above passes.

The unrelated existing edit in `paper/sections/approach.tex` was not staged or
committed by this change.
