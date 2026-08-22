#!/usr/bin/env bash
# The regex round's end-to-end batch, one invocation set per model.
#
# ONE arm: `s_linker92a`, the extraction call deleted and its own contract run as a
# scan. The control is not re-run. `s_linker92` is byte-unchanged by this round and
# has three recorded five-project runs per model of its own from the same week
# (`../results/solo_e2e_{terra,luna}_r{1,2,3}_20260821`), so paying for it again would
# buy sampling noise, not a comparison.
#
# **This makes the control cross-set, which this branch normally forbids.** The
# stage arm (same-invocation, both arms) is what carries the in-set claim; this batch
# answers the one question a stage arm structurally cannot -- composition -- and the
# read to trust is the composition statistic and the per-source decomposition, not the
# absolute level.
#
# `s_linker92f`'s judge template is a separate question and stays out of this batch.
#
# The stage arm (`pilot/regex_proposer_pilots.py`) read s92a at macro F2 +2.0 terra /
# +1.8 luna with F1 neutral on both. What a stage arm cannot see is composition: the
# scan adds pairs the coreference linker also proposes, so a link admitted early is
# locked into the union and stolen from a later, stricter linker. That is what this
# batch is for.
set -u
STAMP=$(date +%Y%m%d)
MODEL=${1:?usage: run_regex_e2e.sh <terra|luna>}
for i in 1 2 3; do
  RUN="../results/regex_e2e_${MODEL}_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker92a_jabref_links.csv" ]; then
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
    --variants s_linker92a \
    --datasets mediastore teammates teastore bigbluebutton jabref \
    --results-dir "${RUN}" > "${RUN}.log" 2>&1
  echo "    exit $? -- $(grep -c 'Final:' "${RUN}.log" 2>/dev/null) linker runs logged"
done
echo "ALL RUNS DONE (${MODEL})"
