#!/usr/bin/env bash
# s110 on the GitLab architecture document, N runs, paper backend (terra/flex).
#     studies/oss-scale/gitlab/run_s110.sh [runs] [model]
# Must be run from approach/ (run_ablation.py's cwd convention).
set -u
RUNS=${1:-3}
MODEL=${2:-terra}
STAMP=$(date +%Y%m%d)
HERE=$(cd "$(dirname "$0")" && pwd)
for i in $(seq 1 "${RUNS}"); do
  RUN="../results/oss_scale_gitlab_${MODEL}_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker110_gitlab_arch_links.csv" ]; then echo "run ${i} done -- skip"; continue; fi
  mkdir -p "${RUN}"
  echo "=== ${MODEL} run ${i} -> ${RUN}"
  OPENAI_API_KEY="$OAI_KEY" LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.6-${MODEL} \
  OPENAI_REASONING_EFFORT=none OPENAI_SERVICE_TIER=flex OPENAI_ENFORCE_FLEX=1 \
  PHASE_CACHE_DIR="${RUN}/phase_states" LLM_LOG_DIR="${RUN}/llm_logs" \
  ALINKER_EXTRA_DATASETS="${HERE}/data/datasets.json" \
    ../.venv/bin/python run_ablation.py --variants s_linker110 --datasets gitlab_arch \
    --results-dir "${RUN}" > "${RUN}.log" 2>&1
  echo "    exit $? -- $(grep -c 'Final:' "${RUN}.log" 2>/dev/null) linker runs logged"
done
echo "ALL RUNS DONE"
