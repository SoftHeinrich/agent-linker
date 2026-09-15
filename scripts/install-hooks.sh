#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
git -C "$root" config core.hooksPath .githooks
echo "Installed package hooks via core.hooksPath=.githooks."

if [[ -e "$root/paper/.git" ]]; then
  git -C "$root/paper" config core.hooksPath .githooks
  echo "Installed paper hook via paper/core.hooksPath=.githooks."
else
  echo "Paper hook skipped: initialize the paper submodule, then rerun this script." >&2
fi
