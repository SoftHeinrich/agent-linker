#!/usr/bin/env bash
# Driver: run s_linker20_union_aliasb on BOTH backends (gpt-5.4 + Sonnet), N=3 each,
# across all 5 datasets, into isolated result dirs. Mirrors the v2.6.5 run layout.
set -uo pipefail
cd "$(dirname "$0")"

VARIANT=s_linker20_union_aliasb
DATASETS="mediastore teastore teammates bigbluebutton jabref"
N=3
OUTROOT=results/v2.6.5_union_aliasb
LOGROOT=logs/v2.6.5_union_aliasb
mkdir -p "$LOGROOT"

run_one () {
  local tag="$1"; shift          # gpt | sonnet
  local n="$1"; shift            # run index
  local outdir="$OUTROOT/${tag}_run${n}"
  local log="$LOGROOT/${tag}_run${n}.log"
  echo "[$(printf '%(%H:%M:%S)T')] START ${tag} run${n} -> $outdir"
  env "$@" python run_ablation.py \
      --variants "$VARIANT" --datasets $DATASETS \
      --results-dir "$outdir" > "$log" 2>&1
  local rc=$?
  echo "[$(printf '%(%H:%M:%S)T')] DONE  ${tag} run${n} rc=$rc (log: $log)"
}

echo "===== gpt-5.4 (openai, flex) ====="
for n in $(seq 1 $N); do
  run_one gpt "$n" LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.4 OPENAI_SERVICE_TIER=flex
done

echo "===== Sonnet (claude CLI) ====="
for n in $(seq 1 $N); do
  run_one sonnet "$n" LLM_BACKEND=claude CLAUDE_MODEL=sonnet
done

echo "===== ALL RUNS COMPLETE ====="
