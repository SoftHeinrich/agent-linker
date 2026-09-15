#!/usr/bin/env bash
# The union round's E2E: `s_linker120` against the head it replaces one stage of.
#
# The stage pilot measured the union at fixed candidates and level 3 read the
# composition risk as 0 distinct gold pairs on terra and 2 on luna -- below the
# recorded TP floor of 4.8, so a batch cannot resolve luna's two pairs. What the
# batch is for is the other half: `s_linker120` merges two judging stages, so the
# pairs it keeps reach the coreference linker through a different door, and only a
# composed run shows what that stage does with them.
#
# TWO arms, both in every invocation, per "never compare across invocation sets".
# The arm ORDER alternates by run -- control first on odd runs, arm first on even --
# because the finetune round's batch could not separate the arm from its slot
# (`../results/finetune_round/README.md`: "arm order is s75, null, s74 and s74 leads
# in all three runs; this batch did not pay for the order reversal").
#
# No in-set null: the floor is measured (`CLAUDE.md`, measurement policy).
#
#     pilot/run_union_e2e.sh terra
#     pilot/run_union_e2e.sh luna 3
set -u
STAMP=$(date +%Y%m%d)
MODEL=${1:?usage: run_union_e2e.sh <terra|luna> [runs]}
RUNS=${2:-3}
set -a; . ../.env; set +a
for i in $(seq 1 "${RUNS}"); do
  RUN="../results/union_e2e_${MODEL}_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker120_jabref_links.csv" ]; then
    echo "run ${i} already complete -- skipping"; continue
  fi
  if [ $((i % 2)) -eq 1 ]; then ARMS="s_linker110 s_linker120";
  else ARMS="s_linker120 s_linker110"; fi
  mkdir -p "${RUN}"
  echo "=== ${MODEL} run ${i} (${ARMS}) -> ${RUN}"
  LLM_BACKEND=openai \
  OPENAI_MODEL_NAME=gpt-5.6-${MODEL} \
  OPENAI_REASONING_EFFORT=none \
  OPENAI_SERVICE_TIER=${OPENAI_SERVICE_TIER:-default} \
  PHASE_CACHE_DIR="${RUN}/phase_states" \
  LLM_LOG_DIR="${RUN}/llm_logs" \
    ../.venv/bin/python run_ablation.py \
    --variants ${ARMS} \
    --datasets mediastore teammates teastore bigbluebutton jabref \
    --results-dir "${RUN}" > "${RUN}.log" 2>&1
  echo "    exit $? -- $(grep -c 'Final:' "${RUN}.log" 2>/dev/null) linker runs logged"
done
echo "ALL RUNS DONE (${MODEL})"
