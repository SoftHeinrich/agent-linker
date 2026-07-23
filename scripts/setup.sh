#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$root/scripts/bootstrap-approach.sh"
echo "Setup complete. The approach runner will use: $root/benchmark"
