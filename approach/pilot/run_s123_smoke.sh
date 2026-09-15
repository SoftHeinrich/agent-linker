#!/usr/bin/env bash
# One project, one arm: does `s_linker123` run end to end and produce a link set?
# Not a measurement -- the round's numbers come from `pilot/run_coref_annot.sh`.
set -u
MODEL=${1:-terra}
DATASET=${2:-mediastore}
PY=${PY:-}
if [ -z "${PY}" ]; then
  for candidate in ../.venv/bin/python \
                   "$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)/../.venv/bin/python"; do
    if [ -x "${candidate}" ]; then PY="${candidate}"; break; fi
  done
fi
[ -x "${PY}" ] || { echo "set PY=/path/to/.venv/bin/python" >&2; exit 2; }

for env_file in ../.env "$(dirname "${PY}")/../../.env"; do
  if [ -f "${env_file}" ]; then set -a; . "${env_file}"; set +a; break; fi
done

OUT=../results/s123_smoke_${MODEL}
mkdir -p "${OUT}"
LLM_BACKEND=openai \
OPENAI_MODEL_NAME=gpt-5.6-${MODEL} \
OPENAI_REASONING_EFFORT=none \
OPENAI_SERVICE_TIER=${OPENAI_SERVICE_TIER:-default} \
PHASE_CACHE_DIR="${OUT}/phase_states" \
LLM_LOG_DIR="${OUT}/llm_logs" \
  "${PY}" run_ablation.py --variants s_linker123 --datasets "${DATASET}" \
  --results-dir "${OUT}" 2>&1 | tail -25
