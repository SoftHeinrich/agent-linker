#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$root/paper/scripts/sync-paper.sh" ]]; then
  echo "Paper submodule is not initialized; run git submodule update --init paper." >&2
  exit 1
fi

exec "$root/paper/scripts/sync-paper.sh" "$@"
