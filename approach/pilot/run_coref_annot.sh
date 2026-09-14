#!/usr/bin/env bash
# The coreference-shortlist annotation round's level-2 stage pilot, one model per call.
#
# Every arm runs in the SAME invocation on the same pinned inputs -- the recorded run's
# alias table and its name-linker link set -- so nothing here is compared across
# invocation sets and nothing about the name stage is resampled into the comparison.
#
#     pilot/run_coref_annot.sh terra
#     pilot/run_coref_annot.sh luna 3
set -u
MODEL=${1:?usage: run_coref_annot.sh <terra|luna> [samples]}
SAMPLES=${2:-3}
STAMP=${STAMP:-$(date +%Y%m%d)}
RUN=${RUN:-../results/noanchor_e2e_${MODEL}_r1_20260914}

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

OUT=../results/coref_annot_${MODEL}_${STAMP}
mkdir -p "${OUT}"
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-${MODEL} \
OPENAI_REASONING_EFFORT=none \
OPENAI_SERVICE_TIER=${OPENAI_SERVICE_TIER:-default} \
LLM_LOG_DIR="${OUT}/llm_logs" \
  "${PY}" pilot/coref_annot_pilots.py \
  --samples "${SAMPLES}" --run "${RUN}" \
  --dump "${OUT}/dump.json" 2>&1 | tee "${OUT}/pilot.log"
echo "dump: ${OUT}/dump.json"
