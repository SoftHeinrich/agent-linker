#!/usr/bin/env bash
# `s_linker126` against `s_linker123`, paired end to end on flex tier.
#
# TWO arms are in every invocation because the name-stage change can starve coreference.
# The arm
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
  RUN="../results/greedymerge_e2e_${MODEL}_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker126_jabref_links.csv" ]; then
    echo "run ${i} already complete -- skipping"; continue
  fi
  if [ $((i % 2)) -eq 1 ]; then
    ARMS="s_linker123 s_linker126"
  else
    ARMS="s_linker126 s_linker123"
  fi
  mkdir -p "${RUN}"
  echo "=== ${MODEL} run ${i} (${ARMS}) -> ${RUN}"
  LLM_BACKEND=openai \
  OPENAI_MODEL_NAME=gpt-5.6-${MODEL} \
  OPENAI_REASONING_EFFORT=none \
  OPENAI_SERVICE_TIER=flex \
  PHASE_CACHE_DIR="${RUN}/phase_states" \
  LLM_LOG_DIR="${RUN}/llm_logs" \
    "${PY}" run_ablation.py \
    --variants ${ARMS} \
    --datasets mediastore teammates teastore bigbluebutton jabref \
    --results-dir "${RUN}" 2>&1 | tee "${RUN}.log"
done
echo "score with: ${PY} pilot/score_runs.py \\"
echo "  --arm s_linker123 ../results/greedymerge_e2e_${MODEL}_r{1,2,3}_${STAMP} \\"
echo "  --arm s_linker126 ../results/greedymerge_e2e_${MODEL}_r{1,2,3}_${STAMP}"
