#!/usr/bin/env bash
# `s_linker126`, paired end to end on flex tier.
#
# NO LIVE CONTROL ARM: `s_linker123` (the in-invocation control this script used to
# run alongside s126) was archived along with the rest of the ancestor chain in the
# s126-only consolidation, so this batch runs s126 alone. If you need a paired
# control again, restore `s_linker123.py` and its `run_ablation.py` registry entry
# from `origin/archive/master-pre-s126-consolidation` first -- this departs from the
# measurement policy's own no-in-set-null rule (`CLAUDE.md`) until that is done.
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
  mkdir -p "${RUN}"
  echo "=== ${MODEL} run ${i} (s_linker126) -> ${RUN}"
  LLM_BACKEND=openai \
  OPENAI_MODEL_NAME=gpt-5.6-${MODEL} \
  OPENAI_REASONING_EFFORT=none \
  OPENAI_SERVICE_TIER=flex \
  PHASE_CACHE_DIR="${RUN}/phase_states" \
  LLM_LOG_DIR="${RUN}/llm_logs" \
    "${PY}" run_ablation.py \
    --variants s_linker126 \
    --datasets mediastore teammates teastore bigbluebutton jabref \
    --results-dir "${RUN}" 2>&1 | tee "${RUN}.log"
done
echo "score with: ${PY} pilot/score_runs.py \\"
echo "  --arm s_linker126 ../results/greedymerge_e2e_${MODEL}_r{1,2,3}_${STAMP}"
