#!/usr/bin/env bash
# Six paired five-project runs carrying s_linker59 (control), s_linker59_null (the
# in-set harness null), s_linker62 (the confirmed proposer repair) and s_linker64
# (s62 plus the stated-name net) in the same invocation. Run from approach/.
#
#   OAI_KEY=... bash pilot/run_s64_e2e.sh
#
# s62 rides along a second time on purpose: it makes s64's comparison against the same
# control in the same invocation, and it doubles s62's own evidence to twelve runs.
set -u
STAMP=20260814
for i in 1 2 3 4 5 6; do
  RUN="../results/s64_e2e_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker64_jabref_links.csv" ]; then
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
    --variants s_linker59 s_linker59_null s_linker62 s_linker64 \
    --datasets mediastore teammates teastore bigbluebutton jabref \
    --results-dir "${RUN}" > "${RUN}.log" 2>&1
  echo "    exit $? — $(grep -c 'Final:' "${RUN}.log" 2>/dev/null) linker runs logged"
done
echo "ALL RUNS DONE"
