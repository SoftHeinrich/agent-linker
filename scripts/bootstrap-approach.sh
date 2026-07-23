#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command -v python3 >/dev/null || { echo "Python 3.11+ is required." >&2; exit 1; }
test -d "$root/benchmark" || { echo "Missing vendored benchmark." >&2; exit 1; }
python3 -m venv "$root/.venv"
"$root/.venv/bin/python" -m pip install --upgrade pip
"$root/.venv/bin/python" -m pip install -e "$root/approach[openai]"
if [ ! -e "$root/approach/.env" ]; then
  {
    echo "# Created by scripts/bootstrap-approach.sh; add OPENAI_API_KEY separately."
    printf 'ALINKER_BENCHMARK=%s\n' "$root/benchmark"
  } >"$root/approach/.env"
  echo "Configured vendored benchmark in approach/.env"
else
  printf 'Existing approach/.env left unchanged. Use ALINKER_BENCHMARK=%s if needed.\n' "$root/benchmark"
fi
echo "Environment ready: $root/.venv"
