#!/usr/bin/env bash
# The typed round's end-to-end confirmation: the composed variant against s_linker85,
# three paired runs per model, both arms in every invocation.
#
#   OAI_KEY=... bash pilot/run_typed_e2e.sh <variant> [terra|luna ...]
set -uo pipefail
cd "$(dirname "$0")/.."
VARIANT="${1:?variant, e.g. s_linker86}"
shift
MODELS=("${@:-terra luna}")
STAMP=$(date '+%Y%m%d')
for tag in ${MODELS[@]}; do
  case $tag in terra) M=gpt-5.6-terra;; luna) M=gpt-5.6-luna;; esac
  for R in 1 2 3; do
    OUT="../results/typed_e2e_${tag}_r${R}_${STAMP}"
    mkdir -p "$OUT"
    OPENAI_API_KEY="$OAI_KEY" LLM_BACKEND=openai OPENAI_MODEL_NAME=$M \
    OPENAI_SERVICE_TIER=flex OPENAI_REASONING_EFFORT=none \
    PHASE_CACHE_DIR="$OUT/phase_states" LLM_LOG_DIR="$OUT/llm_logs" \
      ../.venv/bin/python run_ablation.py \
      --variants "$VARIANT" s_linker85 \
      --datasets mediastore teammates teastore bigbluebutton jabref \
      --results-dir "$OUT" > "../results/typed_e2e_${tag}_r${R}_${STAMP}.log" 2>&1
    echo "$tag run $R rc=$? $(date '+%H:%M:%S')"
  done
done
echo TYPED_E2E_DONE
