#!/usr/bin/env bash
# s_linker69 alone, three five-project runs — overall performance, no paired arms.
# Single-arm by policy (see CLAUDE.md "Measurement Policy"): this batch answers
# "what does s69 score", not "does s69 differ from s66", so it carries no control and
# no null. Deltas against another variant must NOT be read off it — absolute levels
# drift between invocation sets.
set -u
STAMP=$(date +%Y%m%d)
for i in 1 2 3; do
  RUN="../results/s69_solo_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker69_jabref_links.csv" ]; then
    echo "run ${i} already complete — skipping"; continue
  fi
  echo "=== run ${i} -> ${RUN}"
  OPENAI_API_KEY="$OAI_KEY" \
  LLM_BACKEND=openai \
  OPENAI_MODEL_NAME=gpt-5.6-terra \
  OPENAI_REASONING_EFFORT=none \
  PHASE_CACHE_DIR="${RUN}/phase_states" \
  LLM_LOG_DIR="${RUN}/llm_logs" \
    ../.venv/bin/python run_ablation.py \
    --variants s_linker69 \
    --datasets mediastore teammates teastore bigbluebutton jabref \
    --results-dir "${RUN}" > "${RUN}.log" 2>&1
  echo "    exit $? — $(grep -c 'Final:' "${RUN}.log" 2>/dev/null) linker runs logged"
done
echo "ALL RUNS DONE"
