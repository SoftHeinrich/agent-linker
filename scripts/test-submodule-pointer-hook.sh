#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_root="$(mktemp -d "${TMPDIR:-/tmp}/submodule-pointer-hook.XXXXXX")"
trap 'rm -rf -- "$test_root"' EXIT

parent="$test_root/parent"
child_seed="$test_root/child-seed"
child_origin="$test_root/child-origin.git"
child_overleaf="$test_root/child-overleaf.git"
parent_origin="$test_root/parent-origin.git"

git init -q -b main "$child_seed"
git -C "$child_seed" config user.name "Submodule Hook Test"
git -C "$child_seed" config user.email "submodule-hook-test@example.invalid"
git -C "$child_seed" config protocol.file.allow always
mkdir -p "$child_seed/.githooks" "$child_seed/scripts"
cp "$project_root/paper/.githooks/post-commit" "$child_seed/.githooks/post-commit"
cp "$project_root/paper/scripts/sync-paper.sh" "$child_seed/scripts/sync-paper.sh"
chmod +x "$child_seed/.githooks/post-commit" "$child_seed/scripts/sync-paper.sh"
git -C "$child_seed" add .githooks/post-commit scripts/sync-paper.sh
git -C "$child_seed" commit -q -m "child initial"

git init -q --bare "$child_origin"
git init -q --bare "$child_overleaf"
git init -q --bare "$parent_origin"
git -C "$child_seed" remote add origin "$child_origin"
git -C "$child_seed" remote add overleaf "$child_overleaf"
git -C "$child_seed" push -q origin HEAD:main
git -C "$child_seed" push -q overleaf HEAD:main
git --git-dir="$child_origin" symbolic-ref HEAD refs/heads/main
git --git-dir="$child_overleaf" symbolic-ref HEAD refs/heads/main

git init -q -b main "$parent"
git -C "$parent" config user.name "Submodule Hook Test"
git -C "$parent" config user.email "submodule-hook-test@example.invalid"
git -c protocol.file.allow=always -C "$parent" submodule add -q "$child_origin" paper
mkdir -p "$parent/.githooks" "$parent/scripts"
cp "$project_root/.githooks/post-commit" "$parent/.githooks/post-commit"
cp "$project_root/scripts/sync-paper-overleaf.sh" "$parent/scripts/sync-paper-overleaf.sh"
cp "$project_root/scripts/update-submodule-pointer.sh" "$parent/scripts/update-submodule-pointer.sh"
chmod +x "$parent/.githooks/post-commit" "$parent/scripts/sync-paper-overleaf.sh" \
  "$parent/scripts/update-submodule-pointer.sh"
git -C "$parent" config core.hooksPath .githooks
git -C "$parent/paper" remote add overleaf "$child_overleaf"
git -C "$parent/paper" config core.hooksPath .githooks
git -C "$parent/paper" config protocol.file.allow always
git -C "$parent" add .gitmodules paper .githooks/post-commit scripts/sync-paper-overleaf.sh \
  scripts/update-submodule-pointer.sh
git -C "$parent" commit -q -m "parent initial"
git -C "$parent" remote add origin "$parent_origin"
git -C "$parent" push -q origin HEAD:main
git --git-dir="$parent_origin" symbolic-ref HEAD refs/heads/main

git -C "$parent/paper" commit --allow-empty -q -m "child update"

child_head="$(git -C "$parent/paper" rev-parse HEAD)"
parent_pointer="$(git -C "$parent" rev-parse HEAD:paper)"
child_origin_head="$(git --git-dir="$child_origin" rev-parse refs/heads/main)"
child_overleaf_head="$(git --git-dir="$child_overleaf" rev-parse refs/heads/main)"
parent_origin_head="$(git --git-dir="$parent_origin" rev-parse refs/heads/main)"
parent_origin_pointer="$(git --git-dir="$parent_origin" rev-parse refs/heads/main:paper)"

test "$child_head" = "$parent_pointer"
test "$child_head" = "$child_origin_head"
test "$child_head" = "$child_overleaf_head"
test "$child_head" = "$parent_origin_pointer"
test "$(git -C "$parent" rev-parse HEAD)" = "$parent_origin_head"
test -z "$(git -C "$parent" status --porcelain)"
test -z "$(git -C "$parent/paper" status --porcelain)"

printf 'submodule pointer hook fixture passed: child=%s local_parent=%s child_origin=%s child_overleaf=%s remote_parent=%s\n' \
  "$child_head" "$parent_pointer" "$child_origin_head" "$child_overleaf_head" \
  "$parent_origin_pointer"
