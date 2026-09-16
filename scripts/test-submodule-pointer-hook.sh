#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_root="$(mktemp -d "${TMPDIR:-/tmp}/submodule-pointer-hook.XXXXXX")"
trap 'rm -rf -- "$test_root"' EXIT

parent="$test_root/parent"
child_seed="$test_root/child-seed"
origin="$test_root/origin.git"
overleaf="$test_root/overleaf.git"

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

git init -q --bare "$origin"
git init -q --bare "$overleaf"
git -C "$child_seed" remote add origin "$origin"
git -C "$child_seed" remote add overleaf "$overleaf"
git -C "$child_seed" push -q origin HEAD:main
git -C "$child_seed" push -q overleaf HEAD:main
git --git-dir="$origin" symbolic-ref HEAD refs/heads/main
git --git-dir="$overleaf" symbolic-ref HEAD refs/heads/main

git init -q -b main "$parent"
git -C "$parent" config user.name "Submodule Hook Test"
git -C "$parent" config user.email "submodule-hook-test@example.invalid"
git -c protocol.file.allow=always -C "$parent" submodule add -q "$origin" paper
mkdir -p "$parent/.githooks" "$parent/scripts"
cp "$project_root/.githooks/post-commit" "$parent/.githooks/post-commit"
cp "$project_root/scripts/sync-paper-overleaf.sh" "$parent/scripts/sync-paper-overleaf.sh"
cp "$project_root/scripts/update-submodule-pointer.sh" "$parent/scripts/update-submodule-pointer.sh"
chmod +x "$parent/.githooks/post-commit" "$parent/scripts/sync-paper-overleaf.sh" \
  "$parent/scripts/update-submodule-pointer.sh"
git -C "$parent" config core.hooksPath .githooks
git -C "$parent/paper" remote add overleaf "$overleaf"
git -C "$parent/paper" config core.hooksPath .githooks
git -C "$parent/paper" config protocol.file.allow always
git -C "$parent" add .gitmodules paper .githooks/post-commit scripts/sync-paper-overleaf.sh \
  scripts/update-submodule-pointer.sh
git -C "$parent" commit -q -m "parent initial"

git -C "$parent/paper" commit --allow-empty -q -m "child update"

child_head="$(git -C "$parent/paper" rev-parse HEAD)"
parent_pointer="$(git -C "$parent" rev-parse HEAD:paper)"
origin_head="$(git --git-dir="$origin" rev-parse refs/heads/main)"
overleaf_head="$(git --git-dir="$overleaf" rev-parse refs/heads/main)"

test "$child_head" = "$parent_pointer"
test "$child_head" = "$origin_head"
test "$child_head" = "$overleaf_head"
test -z "$(git -C "$parent" status --porcelain)"
test -z "$(git -C "$parent/paper" status --porcelain)"

printf 'submodule pointer hook fixture passed: child=%s parent=%s origin=%s overleaf=%s\n' \
  "$child_head" "$parent_pointer" "$origin_head" "$overleaf_head"
