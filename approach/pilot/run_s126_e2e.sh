#!/usr/bin/env bash
# `s_linker126` against `s_linker123`, end to end — for the PAPER'S NUMBERS, not for the
# adoption.
#
# WHY THIS BATCH IS NOT WHAT DECIDED THE ARM. s124's delta changes the LAST linker, so the
# measurement policy's level 3 makes its composition risk structurally zero -- nothing
# downstream can be starved. The round went further and proved the identity rather than
# citing it: `pinned name links | kept coreference` reproduces a recorded run's own final
# CSV with symmetric difference 0, so the stage read IS the pipeline answer
# (`../results/coref_annot_round/README.md`).
#
# WHY IT IS RUN ANYWAY. The RQ engines key an arm to its E2E run directories (`rq34.py`'s
# arm map), so the paper's s124 row has to be generated from an s124 run set. Until this
# batch exists the paper reports s120.
#
# TWO arms, both in every invocation, per "never compare across invocation sets". The arm
# ORDER alternates by run -- control first on odd runs, arm first on even -- per the
# finetune round's warning that a batch could not separate its arm from its slot.
#
# No in-set null: the floor is measured (`CLAUDE.md`, measurement policy).
#
#     pilot/run_s126_e2e.sh terra
#     pilot/run_s126_e2e.sh luna 3
set -u
set -o pipefail
STAMP=${STAMP:-$(date +%Y%m%d)}
MODEL=${1:?usage: run_s126_e2e.sh <terra|luna> [runs]}
RUNS=${2:-3}

PY=${PY:-}
if [ -z "${PY}" ]; then
  for candidate in ../.venv/bin/python \
                   "$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)/../.venv/bin/python"; do
    if [ -x "${candidate}" ]; then PY="${candidate}"; break; fi
  done
fi
if [ -z "${PY}" ] || [ ! -x "${PY}" ]; then
  echo "no project interpreter found; set PY=/path/to/.venv/bin/python" >&2; exit 2
fi
echo "interpreter: ${PY}"

for env_file in ../.env "$(dirname "${PY}")/../../.env"; do
  if [ -f "${env_file}" ]; then set -a; . "${env_file}"; set +a; break; fi
done

for i in $(seq 1 "${RUNS}"); do
  RUN="../results/antecedentrule_e2e_${MODEL}_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker126_jabref_links.csv" ]; then
    echo "run ${i} already complete -- skipping"; continue
  fi
  if [ $((i % 2)) -eq 1 ]; then ARMS="s_linker123 s_linker126";
  else ARMS="s_linker126 s_linker123"; fi
  mkdir -p "${RUN}"
  echo "=== ${MODEL} run ${i} (${ARMS}) -> ${RUN}"
  LLM_BACKEND=openai \
  OPENAI_MODEL_NAME=gpt-5.6-${MODEL} \
  OPENAI_REASONING_EFFORT=none \
  OPENAI_SERVICE_TIER=${OPENAI_SERVICE_TIER:-default} \
  PHASE_CACHE_DIR="${RUN}/phase_states" \
  LLM_LOG_DIR="${RUN}/llm_logs" \
    "${PY}" run_ablation.py \
    --variants ${ARMS} \
    --datasets mediastore teammates teastore bigbluebutton jabref \
    --results-dir "${RUN}" 2>&1 | tee "${RUN}.log"
done
echo "score with: ${PY} pilot/score_runs.py \\"
echo "  --arm s_linker123 ../results/antecedentrule_e2e_${MODEL}_r{1,2,3}_${STAMP} \\"
echo "  --arm s_linker126 ../results/antecedentrule_e2e_${MODEL}_r{1,2,3}_${STAMP}"
