#!/usr/bin/env bash
# Six paired five-project runs, batch 2 of the bind round. Run from approach/.
#
#   OAI_KEY=... bash pilot/run_s68_e2e.sh
#
#   s_linker65       the round's control
#   s_linker65_null  the in-set harness null
#   s_linker66       the confirmed relocation, riding along a second time so it
#                    reaches twelve runs
#   s_linker68       s66 minus the mention label's qualified-path value
#
# Why an E2E for a change whose stage arm is TP +/-0.0 (p = 1.00), FP -0.2 (p = 1.00),
# composition p = 1.00: this is a mention-label change, and the label is the one field
# on which this branch has twice been wrong in exactly this direction --
# `s_linker43` (trace-screen neutral, E2E macro F1 -1.3) and `s_linker44` (n=3 neutral,
# n=6 macro F1 -0.9). The full-name judge's verdicts feed `_unlinked`, so a verdict
# delta of any size reaches two later linkers.
set -u
STAMP=$(date +%Y%m%d)
for i in 1 2 3 4 5 6; do
  RUN="../results/s68_e2e_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker68_jabref_links.csv" ]; then
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
    --variants s_linker65 s_linker65_null s_linker66 s_linker68 \
    --datasets mediastore teammates teastore bigbluebutton jabref \
    --results-dir "${RUN}" > "${RUN}.log" 2>&1
  echo "    exit $? — $(grep -c 'Final:' "${RUN}.log" 2>/dev/null) linker runs logged"
done
echo "ALL RUNS DONE"
