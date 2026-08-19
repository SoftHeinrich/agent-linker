#!/usr/bin/env bash
# Six paired five-project runs carrying s_linker59 (control), s_linker59_null (the
# in-set harness null), s_linker62 (inflection-bounded partial-name proposer) and
# s_linker63 (s62 plus the sentence-boundary repair) in the same invocation, so every
# arm sees the same model, the same day and the same ordering. Run from approach/.
#
#   OAI_KEY=... bash pilot/run_s6263_e2e.sh
#
# The credential is mapped into the process environment only and never written.
set -u
STAMP=20260814
for i in 1 2 3 4 5 6; do
  RUN="../results/s6263_e2e_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker63_jabref_links.csv" ]; then
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
    --variants s_linker59 s_linker59_null s_linker62 s_linker63 \
    --datasets mediastore teammates teastore bigbluebutton jabref \
    --results-dir "${RUN}" > "${RUN}.log" 2>&1
  echo "    exit $? — $(grep -c 'Final:' "${RUN}.log" 2>/dev/null) linker runs logged"
done
echo "ALL RUNS DONE"
