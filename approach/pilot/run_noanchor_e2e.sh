#!/usr/bin/env bash
# The anchors round's E2E: `s_linker122` (no anchor block, one 73-byte clause in its
# place) against `s_linker121`, the head it cuts 27% of a judging call out of.
#
# WHY A BATCH IS OWED HERE. The stage arms measured the name stage on fixed candidates,
# and the name stage is not the pipeline: the coreference linker runs behind it and
# re-proposes some of what it drops. `anchors` is also the only evidence field whose
# removal moves cases in BOTH directions at the stage (terra spurious +7.0 a run at gold
# -1.3), so the pairs the name stage stops keeping reach the resolver through a different
# door and only a composed run shows what it does with them.
#
# AND BECAUSE THE STAGE COULD NOT SETTLE IT. Two invocation sets disagree on the laxer
# model: `noanchor` read net -8.7 in the first and +0.7 in the second, and `anchor_count`
# read +8.7 gold (p = 0.008) in the first and +/-0.0 (p = 1.000) in the second. Terra
# reproduced in both (spurious +11.7, then +7.0). Per the branch's standing finding,
# absolute levels drift between invocation sets; what a batch buys here is three paired
# runs of the composed pipeline instead of three samples of one stage.
#
# TWO arms, both in every invocation, per "never compare across invocation sets".
# The arm ORDER alternates by run -- control first on odd runs, arm first on even --
# per the finetune round's warning that a batch could not separate its arm from its slot.
#
# No in-set null: the floor is measured (`CLAUDE.md`, measurement policy).
#
#     pilot/run_noanchor_e2e.sh terra
#     pilot/run_noanchor_e2e.sh luna 3
set -u
STAMP=$(date +%Y%m%d)
MODEL=${1:?usage: run_noanchor_e2e.sh <terra|luna> [runs]}
RUNS=${2:-3}

# The interpreter, resolved rather than assumed: `../.venv` is right in a normal
# checkout and wrong in a git worktree, where the venv stays in the main one. Override
# with PY=... . Resolving it late and loudly beats six runs that fail in a second each.
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
  RUN="../results/noanchor_e2e_${MODEL}_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker122_jabref_links.csv" ]; then
    echo "run ${i} already complete -- skipping"; continue
  fi
  if [ $((i % 2)) -eq 1 ]; then ARMS="s_linker121 s_linker122";
  else ARMS="s_linker122 s_linker121"; fi
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
echo "score with: ${PY} pilot/score_runs.py ../results/noanchor_e2e_${MODEL}_r{1,2,3}_${STAMP}"
