#!/usr/bin/env bash
# The elegance round, priced on F2 with a 3 pp budget. ONE invocation set, six arms, so
# every comparison below is in-set and none is read across sets.
#
#   s_linker75       the finetune round's head (control): prompts fully general
#   s_linker75_null  the in-set harness null -- this branch's null is not zero
#   s_linker77       + the two tight `SCANS` rows relocated into the extraction prompt
#                      (deterministic layer: three rows -> one)
#   s_linker78       + the judging rubric's four numbered reject-conditions -> one
#                      principle (the last enumeration in the workflow)
#   s_linker79       + the one row's last two options gone (`unique_owner`,
#                      `skip_when_named`) -- no gate anywhere in the deterministic layer
#   s_linker80       + the computed mention label gone -- the code decides nothing at all
#
# Each arm is the previous one plus one cut, so the batch prices the cuts individually and
# cumulatively in a single set. Expected order of cost, from the rounds that measured each
# on an older base: s77 F2 -1.1, s78 F2 -0.8, s79 F2 ~-1.0, s80 (-10.7 TP) likely over
# budget -- it is carried to price the design law's hardest case rather than to adopt it.
set -u
STAMP=$(date +%Y%m%d)
for i in 1 2 3; do
  RUN="../results/elegance_e2e_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker80_jabref_links.csv" ]; then
    echo "run ${i} already complete -- skipping"; continue
  fi
  mkdir -p "${RUN}"
  echo "=== run ${i} -> ${RUN}"
  OPENAI_API_KEY="$OAI_KEY" \
  LLM_BACKEND=openai \
  OPENAI_MODEL_NAME=gpt-5.6-terra \
  OPENAI_REASONING_EFFORT=none \
  OPENAI_SERVICE_TIER=flex \
  OPENAI_ENFORCE_FLEX=1 \
  PHASE_CACHE_DIR="${RUN}/phase_states" \
  LLM_LOG_DIR="${RUN}/llm_logs" \
    ../.venv/bin/python run_ablation.py \
    --variants s_linker75 s_linker75_null s_linker77 s_linker78 s_linker79 s_linker80 \
    --datasets mediastore teammates teastore bigbluebutton jabref \
    --results-dir "${RUN}" > "${RUN}.log" 2>&1
  echo "    exit $? -- $(grep -c 'Final:' "${RUN}.log" 2>/dev/null) linker runs logged"
done
echo "ALL RUNS DONE"
