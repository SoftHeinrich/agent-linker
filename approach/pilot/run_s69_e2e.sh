#!/usr/bin/env bash
# Six paired five-project runs of the fold round. Run from approach/.
#
#   OAI_KEY=... bash pilot/run_s69_e2e.sh
#
#   s_linker66       the control (the bind round's confirmed endpoint)
#   s_linker66_null  the in-set harness null
#   s_linker69       the four gate changes composed
#
# Stage evidence that bought these runs (pilot/fold_pilots.py, five samples a side):
#   foldqualified       gate deleted FP +7.0 (p=0.01); folded TP -0.4 (0.44), FP -0.2 (1.00)
#   foldantecedent_net  gate deleted TP +0.0 (1.00), FP +0.0 (1.00)
#   fixunlinked         repairing the inert predicate TP -0.6 (0.17), FP +0.4 (0.71)
#   (foldowner          folded TP -8.4 (0.01) -- NOT carried; the gate stays in code)
set -u
STAMP=$(date +%Y%m%d)
for i in 1 2 3 4 5 6; do
  RUN="../results/s69_e2e_r${i}_${STAMP}"
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
    --variants s_linker66 s_linker66_null s_linker69 \
    --datasets mediastore teammates teastore bigbluebutton jabref \
    --results-dir "${RUN}" > "${RUN}.log" 2>&1
  echo "    exit $? — $(grep -c 'Final:' "${RUN}.log" 2>/dev/null) linker runs logged"
done
echo "ALL RUNS DONE"
