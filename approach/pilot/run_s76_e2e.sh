#!/usr/bin/env bash
# The finetune round, second batch: s_linker76 (two batch constants) against s_linker75.
#
# Two arms, three runs, paired in one invocation. The change is one number --
# COREFERENCE_BATCH set to JUDGE_BATCH -- and `s_linker45` already measured that
# unification at parity over six paired runs on the s25 base. This batch confirms it on
# the s70-s75 line and prices the call saving (s75 sends 91.7 calls per five-project run,
# 40.0 of them coreference resolution).
#
# Deltas are read against the null arm's delta, never against zero, and never across
# invocation sets.
set -u
STAMP=$(date +%Y%m%d)
for i in 1 2 3; do
  RUN="../results/s76_e2e_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker75_jabref_links.csv" ]; then
    echo "run ${i} already complete — skipping"; continue
  fi
  mkdir -p "${RUN}"
  echo "=== run ${i} -> ${RUN}"
  OPENAI_API_KEY="$OAI_KEY" \
  LLM_BACKEND=openai \
  OPENAI_MODEL_NAME=gpt-5.6-terra \
  OPENAI_REASONING_EFFORT=none \
  OPENAI_SERVICE_TIER=flex \
  OPENAI_ENFORCE_FLEX=1 \
  PHASE_CACHE_DIR="${RUN}/phase_states" \
  LLM_LOG_DIR="${RUN}/llm_logs" \
    ../.venv/bin/python run_ablation.py \
    --variants s_linker76 s_linker75 \
    --datasets mediastore teammates teastore bigbluebutton jabref \
    --results-dir "${RUN}" > "${RUN}.log" 2>&1
  echo "    exit $? — $(grep -c 'Final:' "${RUN}.log" 2>/dev/null) linker runs logged"
done
echo "ALL RUNS DONE"
