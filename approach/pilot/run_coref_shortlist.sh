#!/usr/bin/env bash
# The shortlist-prior round's level-2 stage pilot, one model per call.
#
# Three arms -- the head's shortlist, the shortlist without its endorsing sentence, and
# no shortlist at all -- in the SAME invocation on the same pinned inputs: the recorded
# run's alias table and its name-linker link set. Nothing is compared across invocation
# sets, and the name stage is NOT resampled into the comparison, which is the defect the
# s124 E2E had (its `full_name` noise exceeded the effect under test).
#
#     pilot/run_coref_shortlist.sh terra
#     pilot/run_coref_shortlist.sh luna 3
# `tee` would otherwise mask a crashed pilot as exit 0 -- which it did, once.
set -u
set -o pipefail
MODEL=${1:?usage: run_coref_shortlist.sh <terra|luna> [samples]}
SAMPLES=${2:-3}
STAMP=${STAMP:-$(date +%Y%m%d)}
RUN=${RUN:-../results/shortlistmark_e2e_${MODEL}_r1_20260914}

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

OUT=../results/coref_shortlist_${MODEL}_${STAMP}
mkdir -p "${OUT}"
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-${MODEL} \
OPENAI_REASONING_EFFORT=none \
OPENAI_SERVICE_TIER=${OPENAI_SERVICE_TIER:-default} \
LLM_LOG_DIR="${OUT}/llm_logs" \
  "${PY}" pilot/coref_shortlist_pilots.py \
  --samples "${SAMPLES}" --run "${RUN}" \
  --dump "${OUT}/dump.json" 2>&1 | tee "${OUT}/pilot.log"
# `pipefail` alone is not enough: a trailing command would overwrite the script's
# exit status, which is how a crashed pilot reported success twice.
status=$?
if [ "${status}" -ne 0 ]; then
  echo "PILOT FAILED (exit ${status}) -- see ${OUT}/pilot.log" >&2
  exit "${status}"
fi
echo "dump: ${OUT}/dump.json"
