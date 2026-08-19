#!/usr/bin/env bash
# The finetune round's E2E: s_linker75 against s_linker74, with an in-set null.
#
# Three arms in one invocation, three runs. s74 is the control (the adopted head), the
# null measures what this harness reports as a difference when there is none, and s75 is
# the arm. Composition risk is why this batch is paid for at all: two of the four changed
# spans are in the alias and extraction prompts, which feed every later stage, so
# `composition_check.py`'s precondition is non-zero and the stage pilots cannot decide it
# (see CLAUDE.md, Measurement Policy step 4).
#
# Deltas are read against the null arm's delta, never against zero, and never across
# invocation sets.
set -u
STAMP=$(date +%Y%m%d)
for i in 1 2 3; do
  RUN="../results/s75_e2e_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker74_jabref_links.csv" ]; then
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
    --variants s_linker75 s_linker75_null s_linker74 \
    --datasets mediastore teammates teastore bigbluebutton jabref \
    --results-dir "${RUN}" > "${RUN}.log" 2>&1
  echo "    exit $? — $(grep -c 'Final:' "${RUN}.log" 2>/dev/null) linker runs logged"
done
echo "ALL RUNS DONE"
