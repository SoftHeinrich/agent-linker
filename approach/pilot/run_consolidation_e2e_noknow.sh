#!/usr/bin/env bash
# The RQ4 knowledge A/B for the arm the paper now reports: `s_linker110_noknow`,
# three five-project runs per model, same invocation shape as
# pilot/run_regex_e2e_noknow.sh one lineage over.
#
# WHY THIS BATCH EXISTS. Promoting s110 to the reported arm dropped tab:rq4's
# "No knowledge" row: the only no-knowledge runs on disk belong to s92a
# (`results/regex_noknow_e2e_*_20260826`), and rq_tables.py omits the row rather
# than borrow another arm's. This batch fills it on the arm actually reported.
#
# The comparison is IN-SET: the control is
# `results/consolidation_e2e_{model}_r{1,2,3}_20260825`, the same variant at the
# same model with the alias table on, run on the same benchmark.
#
# LANDMINE: s_linker110_noknow's _VARIANT_NAME is "s_linker110" (same as Full), so
# its phase states nest under <run>/phase_states/s_linker110/. PHASE_CACHE_DIR is
# therefore per-run and MUST NOT point at a Full arm's directory -- pointing it at
# a consolidation run would silently overwrite the states RQ3/RQ4 read.
set -u
STAMP=$(date +%Y%m%d)
MODEL=${1:?usage: run_consolidation_e2e_noknow.sh <terra|luna>}
: "${OAI_KEY:?OAI_KEY must be set}"
for i in 1 2 3; do
  RUN="../results/consolidation_noknow_e2e_${MODEL}_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker110_noknow_jabref_links.csv" ]; then
    echo "run ${i} already complete -- skipping"; continue
  fi
  mkdir -p "${RUN}"
  echo "=== ${MODEL} noknow run ${i} -> ${RUN}"
  OPENAI_API_KEY="$OAI_KEY" \
  LLM_BACKEND=openai \
  OPENAI_MODEL_NAME=gpt-5.6-${MODEL} \
  OPENAI_REASONING_EFFORT=none \
  OPENAI_SERVICE_TIER=${OPENAI_SERVICE_TIER:-flex} \
  OPENAI_ENFORCE_FLEX=${OPENAI_ENFORCE_FLEX:-1} \
  PHASE_CACHE_DIR="${RUN}/phase_states" \
  LLM_LOG_DIR="${RUN}/llm_logs" \
    ../.venv/bin/python run_ablation.py \
    --variants s_linker110_noknow \
    --datasets mediastore teammates teastore bigbluebutton jabref \
    --results-dir "${RUN}" > "${RUN}.log" 2>&1
  echo "    exit $? -- $(grep -c 'Final:' "${RUN}.log" 2>/dev/null) linker runs logged"
done
echo "ALL RUNS DONE (${MODEL} noknow)"
