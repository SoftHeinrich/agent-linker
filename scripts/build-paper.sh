#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test -f "$root/paper/main.tex" || {
  echo "The paper submodule is not initialized; run git submodule update --init paper." >&2
  exit 1
}
command -v latexmk >/dev/null || {
  echo "latexmk is required to build the paper (install TeX Live with latexmk)." >&2
  exit 1
}
cd "$root/paper"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
