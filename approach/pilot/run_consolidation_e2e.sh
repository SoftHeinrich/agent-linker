#!/usr/bin/env bash
# The consolidation round's E2E: the composed head against the arm it composes onto.
#
# `s_linker109` is decided at level 1 and owes nothing -- the refusal it adds is
# deterministic (exactly 12 candidates a run on both models), costs 0.0 gold in six
# recorded runs, and 0.0 of the links it removes are proposed by the coreference
# linker, so `_unlinked` frees nothing re-proposable. What this batch buys is the
# *composed* claim: `s_linker110` adds the resolver's antecedent shortlist on top of
# it, and a resolver change has the branch's usual composition risk -- a refer-back the
# shortlist withholds is a pair the strict judge never sees, and s76 is the standing
# warning that touching this stage has been refused once already on a neighbouring base.
#
# TWO arms, both in every invocation, per "never compare across invocation sets":
# `s_linker92a` is the control (the regex round's head) and `s_linker110` the arm.
# `s_linker109` is deliberately NOT a third arm -- the measurement policy says not to
# pair-run an arm a checkpoint replay already separates, and a third arm would cost a
# third of the batch to re-measure a settled question.
#
#     pilot/run_consolidation_e2e.sh terra
#     pilot/run_consolidation_e2e.sh luna
set -u
STAMP=$(date +%Y%m%d)
MODEL=${1:?usage: run_consolidation_e2e.sh <terra|luna>}
RUNS=${2:-3}
for i in $(seq 1 "${RUNS}"); do
  RUN="../results/consolidation_e2e_${MODEL}_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker110_jabref_links.csv" ]; then
    echo "run ${i} already complete -- skipping"; continue
  fi
  mkdir -p "${RUN}"
  echo "=== ${MODEL} run ${i} -> ${RUN}"
  OPENAI_API_KEY="$OAI_KEY" \
  LLM_BACKEND=openai \
  OPENAI_MODEL_NAME=gpt-5.6-${MODEL} \
  OPENAI_REASONING_EFFORT=none \
  OPENAI_SERVICE_TIER=flex \
  OPENAI_ENFORCE_FLEX=1 \
  PHASE_CACHE_DIR="${RUN}/phase_states" \
  LLM_LOG_DIR="${RUN}/llm_logs" \
    ../.venv/bin/python run_ablation.py \
    --variants s_linker92a s_linker110 \
    --datasets mediastore teammates teastore bigbluebutton jabref \
    --results-dir "${RUN}" > "${RUN}.log" 2>&1
  echo "    exit $? -- $(grep -c 'Final:' "${RUN}.log" 2>/dev/null) linker runs logged"
done
echo "ALL RUNS DONE (${MODEL})"
