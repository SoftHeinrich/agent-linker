#!/usr/bin/env bash
# The judge round's level-2 pilots: one gate at a time, fixed candidates, every arm
# of that gate in the SAME invocation.
#
# Both gates measured here sit behind deterministic scans, so the candidate set is
# identical across arms by construction and the only calls spent are the gate's.
#
#     pilot/run_judge_round.sh terra
#     pilot/run_judge_round.sh luna
set -u
MODEL=${1:?usage: run_judge_round.sh <terra|luna>}
SAMPLES=${2:-3}
OUT=../results/judge_round
mkdir -p "${OUT}"
for GATE in sortal lenient; do
  echo "=== ${GATE} gate on ${MODEL}"
  OPENAI_API_KEY="$OAI_KEY" \
  LLM_BACKEND=openai \
  OPENAI_MODEL_NAME=gpt-5.6-${MODEL} \
  OPENAI_REASONING_EFFORT=none \
  OPENAI_SERVICE_TIER=flex \
  OPENAI_ENFORCE_FLEX=1 \
  LLM_LOG_DIR="${OUT}/llm_logs_${GATE}_${MODEL}" \
    ../.venv/bin/python pilot/nextgen_pilots.py \
      --gate "${GATE}" --samples "${SAMPLES}" \
      --dump "${OUT}/dump_${GATE}_${MODEL}.json" \
      > "${OUT}/stage_${MODEL}_${GATE}.log" 2>&1
  echo "    exit $?"
done
echo "JUDGE ROUND DONE (${MODEL})"
