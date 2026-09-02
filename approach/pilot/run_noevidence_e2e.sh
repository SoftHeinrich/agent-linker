#!/usr/bin/env bash
# RETIRED 2026-09-02 -- KEPT AS PROVENANCE, NOT RUNNABLE.
#
# RQ4's floor is `s_linker110_onecall` alone. `s_linker110_noevidence` and
# `s_linker110_nocoderef` were killed with it: both module files, both invariant tests
# and both registrations are gone, so `--variants s_linker110_noevidence` below can no
# longer resolve. Restore them from git history if the arm is ever wanted again.
#
# This file stays because its output directories are still read: the runs it produced,
# `../results/noevidence_e2e_{terra,luna}_r{1,2,3}_20260902`, are where the RQ4 floor
# table gets its `s_linker110` control from (`evaluation/mini-rq34/rq4_floor.py`,
# RQ4_FLOOR_HEAD_TMPL). The header below is the record of how those runs were made --
# the noevidence arm rode in the same invocations as the control, which is why the
# directories carry its name.
#
# What follows describes the batch as it ran.
#
# RQ4's evidence A/B: the head against the head with no code-computed context.
#
# The paper's claim is that each judge rules on a case code assembled. This batch is
# what makes that claim a number rather than a design note. `s_linker110_noevidence`
# removes exactly the assembled context and nothing else -- the full-name Evidence
# block, the `[prev:]` in every case, the denotation step's +/-5 window (narrowed to
# the candidate's own sentence), and the resolver's NAMED BEFORE THIS CASE shortlist.
# The coreference SENTENCES window is KEPT: without it a refer-back is unresolvable in
# principle, so removing it would price the task's impossibility instead.
#
# TWO arms, both in every invocation, per "never compare across invocation sets":
# `s_linker110` is the control (the head) and `s_linker110_noevidence` the arm. They
# carry different `_VARIANT_NAME`s, so their phase states do not collide under the one
# PHASE_CACHE_DIR this batch gives them -- the LANDMINE that applies to the `_noknow`
# variants does not apply here.
#
# Invariants: pilot/test_s110_noevidence.py (21 checks, no calls) -- deleted with the arm.
# Scoring:    pilot/score_runs.py
#
#     pilot/run_noevidence_e2e.sh terra
#     pilot/run_noevidence_e2e.sh luna
#
# Smoke first, one project, one run:
#     pilot/run_noevidence_e2e.sh terra 1 mediastore
set -u
STAMP=$(date +%Y%m%d)
MODEL=${1:?usage: run_noevidence_e2e.sh <terra|luna> [runs] [datasets...]}
RUNS=${2:-3}
shift 2 2>/dev/null || shift $#
DATASETS=${*:-"mediastore teammates teastore bigbluebutton jabref"}
: "${OAI_KEY:?OAI_KEY must be set}"

for i in $(seq 1 "${RUNS}"); do
  RUN="../results/noevidence_e2e_${MODEL}_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker110_noevidence_jabref_links.csv" ]; then
    echo "run ${i} already complete -- skipping"; continue
  fi
  mkdir -p "${RUN}"
  echo "=== ${MODEL} run ${i} -> ${RUN}"
  OPENAI_API_KEY="$OAI_KEY" \
  LLM_BACKEND=openai \
  OPENAI_MODEL_NAME=gpt-5.6-${MODEL} \
  OPENAI_REASONING_EFFORT=none \
  OPENAI_SERVICE_TIER=${OPENAI_SERVICE_TIER:-flex} \
  OPENAI_ENFORCE_FLEX=${OPENAI_ENFORCE_FLEX:-1} \
  PHASE_CACHE_DIR="${RUN}/phase_states" \
  LLM_LOG_DIR="${RUN}/llm_logs" \
    ../.venv/bin/python run_ablation.py \
    --variants s_linker110 s_linker110_noevidence \
    --datasets ${DATASETS} \
    --results-dir "${RUN}" > "${RUN}.log" 2>&1
  echo "    exit $? -- $(grep -c 'Final:' "${RUN}.log" 2>/dev/null) linker runs logged"
done
echo "ALL RUNS DONE (${MODEL} noevidence)"
