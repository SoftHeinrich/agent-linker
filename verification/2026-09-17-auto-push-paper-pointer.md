# Automatic parent-pointer push verification

Date: 2026-09-17

## Configuration under test

- A paper commit synchronizes the paper's `origin` and `overleaf` branches.
- The child hook invokes `scripts/update-submodule-pointer.sh` in the parent.
- The helper commits only the changed paper gitlink and performs a non-force
  push from the attached parent branch to its same-named branch on `origin`.
- `PARENT_REMOTE` can select a different parent remote. A detached parent,
  missing remote, conflict, staged competing pointer, or rejected push stops
  the hook without forcing remote history.

## Commands and results

Syntax validation and the disposable end-to-end fixture:

```text
$ bash -n scripts/update-submodule-pointer.sh paper/.githooks/post-commit .githooks/post-commit scripts/test-submodule-pointer-hook.sh
$ scripts/test-submodule-pointer-hook.sh
Pushed 745a06d6d55c796bc81f799d45057e1ecff2275c to origin/main.
Pushed 745a06d6d55c796bc81f799d45057e1ecff2275c to overleaf/main.
origin/main already contains 745a06d6d55c796bc81f799d45057e1ecff2275c.
overleaf/main already contains 745a06d6d55c796bc81f799d45057e1ecff2275c.
[main 2f1fc7c] chore: update paper submodule
 Author: SoftHeinrich <SoftHeinrich@users.noreply.github.com>
 1 file changed, 1 insertion(+), 1 deletion(-)
To /tmp/submodule-pointer-hook.9G2psx/parent-origin.git
   f90c540..2f1fc7c  HEAD -> main
submodule pointer hook fixture passed: child=745a06d6d55c796bc81f799d45057e1ecff2275c local_parent=745a06d6d55c796bc81f799d45057e1ecff2275c child_origin=745a06d6d55c796bc81f799d45057e1ecff2275c child_overleaf=745a06d6d55c796bc81f799d45057e1ecff2275c remote_parent=745a06d6d55c796bc81f799d45057e1ecff2275c
```

The fixture uses separate bare repositories for the paper's two remotes and
the parent's `origin`. Its final assertions compare the paper commit with both
paper remote tips, the local parent gitlink, and the gitlink read from the
remote parent branch. It also checks that the local and remote parent tips are
identical and that both working trees are clean.
