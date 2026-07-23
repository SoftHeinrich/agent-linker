#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TRANSARC_BENCHMARK="$root/benchmark"
export TRANSARC_RESULTS_DIR="$root/evaluation/mini-data"

command -v python3 >/dev/null || { echo "Python 3 is required." >&2; exit 1; }
test -d "$TRANSARC_BENCHMARK" || { echo "Missing vendored benchmark." >&2; exit 1; }
test -d "$TRANSARC_RESULTS_DIR" || { echo "Missing vendored evaluation results." >&2; exit 1; }

python3 "$root/evaluation/mini-src/check.py"
python3 "$root/evaluation/mini-src/metrics.py" --task sad-code
python3 "$root/evaluation/mini-src/metrics.py" --task sad-sam
