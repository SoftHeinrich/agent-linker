#!/usr/bin/env bash
# The RQ4 knowledge A/B for the arm the paper now reports: `s_linker120_noknow`,
# three five-project runs per model, same invocation shape as
# pilot/run_consolidation_e2e_noknow.sh one variant over.
#
# WHY THIS BATCH EXISTS. tab:rq4's "No knowledge" row is an in-set A/B, and the
# no-knowledge runs on disk belong to s110 (`results/consolidation_noknow_e2e_*`).
# rq_tables.py omits the row rather than borrow another arm's, so promoting s120 to
# the reported arm drops the row unless this batch fills it on the arm reported.
# The union reads the alias table in one place more than s110 does: `writes=alias`
# is one of the three values of its `writes` evidence, so with the table off that
# value never appears and the rule reads two forms instead of three. That is what
# the A/B measures on this arm, and it is why the row cannot be borrowed from s110.
#
# The comparison is IN-SET: the control is
# `results/union_e2e_{model}_r{1,2,3}_20260911`, the same variant at the
# same model with the alias table on, run on the same benchmark.
#
# LANDMINE: s_linker120_noknow's _VARIANT_NAME is "s_linker120" (same as Full), so
# its phase states nest under <run>/phase_states/s_linker120/. PHASE_CACHE_DIR is
# therefore per-run and MUST NOT point at a Full arm's directory -- pointing it at
# a consolidation run would silently overwrite the states RQ3/RQ4 read.
set -u
STAMP=$(date +%Y%m%d)
MODEL=${1:?usage: run_union_e2e_noknow.sh <terra|luna>}
set -a; . ../.env; set +a
for i in 1 2 3; do
  RUN="../results/union_noknow_e2e_${MODEL}_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker120_noknow_jabref_links.csv" ]; then
    echo "run ${i} already complete -- skipping"; continue
  fi
  mkdir -p "${RUN}"
  echo "=== ${MODEL} noknow run ${i} -> ${RUN}"
  LLM_BACKEND=openai \
  OPENAI_MODEL_NAME=gpt-5.6-${MODEL} \
  OPENAI_REASONING_EFFORT=none \
  OPENAI_SERVICE_TIER=${OPENAI_SERVICE_TIER:-default} \
  PHASE_CACHE_DIR="${RUN}/phase_states" \
  LLM_LOG_DIR="${RUN}/llm_logs" \
    ../.venv/bin/python run_ablation.py \
    --variants s_linker120_noknow \
    --datasets mediastore teammates teastore bigbluebutton jabref \
    --results-dir "${RUN}" > "${RUN}.log" 2>&1
  echo "    exit $? -- $(grep -c 'Final:' "${RUN}.log" 2>/dev/null) linker runs logged"
done
echo "ALL RUNS DONE (${MODEL} noknow)"
