#!/usr/bin/env bash
# RQ4's total floor (D3): the head against one linking call.
#
# `s_linker110_onecall` is given the document, the component list and the discovered
# alias table, and returns the final link set. No scan, no window, no evidence bundle,
# no antecedent shortlist, no judge, no union. The head's four rubrics are rendered
# verbatim, so the arm removes the arrangement and not the guidance.
#
# ONE arm, by decision. The control is `s_linker110` as run TODAY in
# `../results/noevidence_e2e_{terra,luna}_r{1,2,3}_20260902` (3 runs a backend, five
# projects, same code and same backends), so this comparison is **cross-set** -- the
# thing the branch normally forbids, taken here to save ~30 of 40 minutes because the
# head arm is ~9 calls a project against this arm's ~3.
#
# WHAT THAT COSTS, measured on this exact arm: `s_linker110` on terra read macro F1
# 93.85 (consolidation set, 0825) and 92.90 (noevidence set, today) -- 0.95 F1 of drift
# from the invocation set alone. Any delta this batch reports carries that band on top
# of its own. State it wherever the number is used; do not quote it as an in-set result.
#
# READ PER-PROJECT BEFORE THE MACRO. s_linker27 measured a whole-document call and
# found accuracy tracks document length (jabref 13 sentences 100.0, teammates 198
# 84.1). A macro loss here is partly that effect, and the smoke run already shows the
# floor AHEAD on mediastore (37 sentences).
#
# The arm has no linker_* phases, so mini-rq34 cannot read it; score with score_runs.py.
# Invariants: pilot/test_s110_onecall.py (27 checks, no calls) -- run it first.
#
#     pilot/run_onecall_e2e.sh terra
#     pilot/run_onecall_e2e.sh luna
#
set -u
STAMP=$(date +%Y%m%d)
MODEL=${1:?usage: run_onecall_e2e.sh <terra|luna> [runs] [datasets...]}
RUNS=${2:-3}
shift 2 2>/dev/null || shift $#
DATASETS=${*:-"mediastore teammates teastore bigbluebutton jabref"}
: "${OAI_KEY:?OAI_KEY must be set}"

for i in $(seq 1 "${RUNS}"); do
  RUN="../results/onecall_e2e_${MODEL}_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker110_onecall_jabref_links.csv" ]; then
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
    --variants s_linker110_onecall \
    --datasets ${DATASETS} \
    --results-dir "${RUN}" > "${RUN}.log" 2>&1
  echo "    exit $? -- $(grep -c 'Final:' "${RUN}.log" 2>/dev/null) linker runs logged"
done
echo "ALL RUNS DONE (${MODEL} onecall)"
