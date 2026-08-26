#!/usr/bin/env bash
# The RQ4 knowledge A/B for the arm the paper reports: `s_linker92a_noknow`, three
# five-project runs per model, same invocation shape as pilot/run_regex_e2e.sh.
#
# WHY THIS BATCH EXISTS. tab:rq4 ablates every module of the reported arm except the
# knowledge module, because the only no-knowledge runs on disk belong to the retired s21
# lineage (`results/v2.6.6_s21_noknow_*`) and mixing arms inside one table is worse than
# an absent row. This batch fills that row on the arm actually being reported.
#
# The comparison is IN-SET: the control is `results/regex_e2e_{model}_r{1,2,3}_20260822`,
# the same variant at the same model with the alias table on, run on the same benchmark.
#
# LANDMINE: s_linker92a_noknow's _VARIANT_NAME is "s_linker92a" (same as Full), so its
# phase states nest under <run>/phase_states/s_linker92a/. PHASE_CACHE_DIR is therefore
# per-run and MUST NOT point at a Full arm's directory. The links CSV and the ablation
# JSON key, by contrast, carry the registry name `s_linker92a_noknow`.
set -u
STAMP=$(date +%Y%m%d)
MODEL=${1:?usage: run_regex_e2e_noknow.sh <terra|luna>}
: "${OAI_KEY:?OAI_KEY must be set}"
for i in 1 2 3; do
  RUN="../results/regex_noknow_e2e_${MODEL}_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker92a_noknow_jabref_links.csv" ]; then
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
    --variants s_linker92a_noknow \
    --datasets mediastore teammates teastore bigbluebutton jabref \
    --results-dir "${RUN}" > "${RUN}.log" 2>&1
  echo "    exit $? -- $(grep -c 'Final:' "${RUN}.log" 2>/dev/null) linker runs logged"
done
echo "ALL RUNS DONE (${MODEL} noknow)"
