#!/usr/bin/env bash
# The RQ4 knowledge A/B for the arm the paper now reports: `s_linker126_noknow`,
# three five-project runs per model, same invocation shape as
# pilot/run_union_e2e_noknow.sh one arm over.
#
# WHY THIS BATCH EXISTS. Promoting s126 to the reported arm dropped tab:rq4's
# "No knowledge" row: the only no-knowledge runs on disk belong to s120
# (`results/union_noknow_e2e_*_20260911`), and rq_tables.py omits the row rather
# than borrow another arm's. This batch fills it on the arm actually reported.
#
# The comparison is IN-SET: the control is
# `results/greedymerge_e2e_{model}_r{1,2,3}_20260916v2`, the same variant at the
# same model with the alias table on, run on the same benchmark.
#
# LANDMINE: s_linker126_noknow's _VARIANT_NAME is "s_linker126" (same as Full), so
# its phase states nest under <run>/phase_states/s_linker126/. PHASE_CACHE_DIR is
# therefore per-run and MUST NOT point at a Full arm's directory -- pointing it at
# a greedymerge run would silently overwrite the states RQ3/RQ4 read.
#
#     pilot/run_s126_e2e_noknow.sh terra
#     pilot/run_s126_e2e_noknow.sh luna 3
set -u
set -o pipefail
STAMP=${STAMP:-$(date +%Y%m%d)}
MODEL=${1:?usage: run_s126_e2e_noknow.sh <terra|luna> [runs]}
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
  RUN="../results/greedymerge_noknow_e2e_${MODEL}_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker126_noknow_jabref_links.csv" ]; then
    echo "run ${i} already complete -- skipping"; continue
  fi
  mkdir -p "${RUN}"
  echo "=== ${MODEL} noknow run ${i} -> ${RUN}"
  LLM_BACKEND=openai \
  OPENAI_MODEL_NAME=gpt-5.6-${MODEL} \
  OPENAI_REASONING_EFFORT=none \
  OPENAI_SERVICE_TIER=flex \
  PHASE_CACHE_DIR="${RUN}/phase_states" \
  LLM_LOG_DIR="${RUN}/llm_logs" \
    "${PY}" run_ablation.py \
    --variants s_linker126_noknow \
    --datasets mediastore teammates teastore bigbluebutton jabref \
    --results-dir "${RUN}" > "${RUN}.log" 2>&1
  echo "    exit $? -- $(grep -c 'Final:' "${RUN}.log" 2>/dev/null) linker runs logged"
done
echo "ALL RUNS DONE (${MODEL} noknow)"
