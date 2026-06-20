#!/usr/bin/env bash
# Run s_linker20_union_aliasb on ONE backend, N runs, all 5 datasets, isolated dirs.
# Usage: run_union_aliasb_backend.sh <gpt|sonnet> [N]
set -uo pipefail
cd "$(dirname "$0")"

TAG="${1:?usage: $0 <gpt|sonnet> [N]}"
N="${2:-3}"
VARIANT=s_linker20_union_aliasb
DATASETS="mediastore teastore teammates bigbluebutton jabref"
OUTROOT=results/v2.6.5_union_aliasb
LOGROOT=logs/v2.6.5_union_aliasb
mkdir -p "$LOGROOT"

case "$TAG" in
  gpt)    ENVV=(LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.4 OPENAI_SERVICE_TIER=flex) ;;
  sonnet) ENVV=(LLM_BACKEND=claude CLAUDE_MODEL=sonnet) ;;
  *) echo "unknown backend: $TAG" >&2; exit 2 ;;
esac

for n in $(seq 1 "$N"); do
  outdir="$OUTROOT/${TAG}_run${n}"
  log="$LOGROOT/${TAG}_run${n}.log"
  echo "[$(printf '%(%Y-%m-%d %H:%M:%S)T')] START ${TAG} run${n} -> $outdir"
  env "${ENVV[@]}" python run_ablation.py \
      --variants "$VARIANT" --datasets $DATASETS \
      --results-dir "$outdir" > "$log" 2>&1
  echo "[$(printf '%(%Y-%m-%d %H:%M:%S)T')] DONE  ${TAG} run${n} rc=$? (log: $log)"
done
echo "[$(printf '%(%Y-%m-%d %H:%M:%S)T')] ===== ${TAG} ALL ${N} RUNS COMPLETE ====="
