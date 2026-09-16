#!/usr/bin/env bash
set -euo pipefail

submodule_path="${1:-paper}"
case "$submodule_path" in
  /*|.|..|*/../*|../*|*/..)
    echo "Refusing unsafe submodule path: $submodule_path" >&2
    exit 2
    ;;
  esac

unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_PREFIX

parent_root="$(git rev-parse --show-toplevel)"
submodule_root="$parent_root/$submodule_path"

git -C "$parent_root" ls-files --error-unmatch -- "$submodule_path" >/dev/null
git -C "$submodule_root" rev-parse --show-toplevel >/dev/null
submodule_head="$(git -C "$submodule_root" rev-parse HEAD)"
parent_head="$(git -C "$parent_root" rev-parse --verify HEAD)"
parent_gitlink="$(git -C "$parent_root" rev-parse --verify ":$submodule_path" 2>/dev/null || true)"
recorded_gitlink="$(git -C "$parent_root" rev-parse --verify "$parent_head:$submodule_path" 2>/dev/null || true)"

if [[ "$parent_gitlink" == "$submodule_head" ]]; then
  echo "$submodule_path already records $submodule_head."
  exit 0
fi

if [[ -n "$(git -C "$parent_root" diff --name-only --diff-filter=U)" ||
      -n "$(git -C "$parent_root" diff --cached --name-only --diff-filter=U)" ]]; then
  echo "Parent repository has unresolved conflicts; refusing to update $submodule_path." >&2
  exit 1
fi

if [[ -n "$parent_gitlink" && "$parent_gitlink" != "$recorded_gitlink" &&
      "$parent_gitlink" != "$submodule_head" ]]; then
  echo "Parent has a different staged $submodule_path pointer; refusing to overwrite it." >&2
  exit 1
fi

git -C "$parent_root" add -- "$submodule_path"
git -C "$parent_root" commit --only --no-verify \
  -m "chore: update $submodule_path submodule" -- "$submodule_path"
