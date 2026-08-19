#!/usr/bin/env bash
# Six paired five-project runs carrying s_linker49 (control), s_linker50 and
# s_linker51 in the same invocation, so every arm sees the same model, the same
# day and the same ordering. Run from the approach/ directory.
#
#   OAI_KEY=... bash pilot/run_s5455_e2e.sh
#
# The credential is mapped into the process environment only and never written.
set -u
STAMP=20260813
for i in 1 2 3 4 5 6; do
  RUN="../results/s5455_e2e_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker55_jabref_links.csv" ]; then
    echo "run ${i} already complete — skipping"
    continue
  fi
  echo "=== run ${i} -> ${RUN}"
  OPENAI_API_KEY="$OAI_KEY" \
  LLM_BACKEND=openai \
  OPENAI_MODEL_NAME=gpt-5.6-terra \
  OPENAI_REASONING_EFFORT=none \
  PHASE_CACHE_DIR="${RUN}/phase_states" \
  LLM_LOG_DIR="${RUN}/llm_logs" \
    ../.venv/bin/python run_ablation.py \
    --variants s_linker49 s_linker54 s_linker55 \
    --datasets mediastore teammates teastore bigbluebutton jabref \
    --results-dir "${RUN}" > "${RUN}.log" 2>&1
  echo "    exit $? — $(grep -c 'Final:' "${RUN}.log" 2>/dev/null) linker runs logged"
done
echo "ALL RUNS DONE"
