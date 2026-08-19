#!/usr/bin/env bash
# Six paired five-project runs of the bind round, all four arms in the same
# invocation. Run from approach/.
#
#   OAI_KEY=... bash pilot/run_s6667_e2e.sh
#
#   s_linker65       the control: the deterministic layer as one relation
#   s_linker65_null  the in-set harness null — byte-identical to s65 apart from the
#                    checkpoint namespace. Mandatory: this pipeline's null has read
#                    0.7 macro F1 in one set and +0.4 in another, so a delta is only
#                    readable against the null of its own invocation set.
#   s_linker66       the admission contract relocated into the extraction prompt
#   s_linker67       s66 plus the two tight scans relocated into the same prompt
#
# Stage evidence that bought these runs (pilot/bind_pilots.py, five samples a side):
#   bindcontract  filter deleted TP +4.8 / FP +10.6 (p = 0.01 both);
#                 contract in the prompt TP -1.4 (p = 0.21), FP -1.8 (p = 0.47)
#   bindboth      TP -1.2 (p = 0.14), FP -1.2 (p = 0.37)
# and the composition risk that makes an E2E owed rather than optional
# (pilot/bind_audit.py --only B6): 19.5 pairs per run for the uncompensated filter
# deletion, of which 6.5 are also proposed by the partial-name linker and 5.0 by the
# coreference linker.
set -u
STAMP=$(date +%Y%m%d)
for i in 1 2 3 4 5 6; do
  RUN="../results/s6667_e2e_r${i}_${STAMP}"
  if [ -f "${RUN}/s_linker67_jabref_links.csv" ]; then
    echo "run ${i} already complete — skipping"
    continue
  fi
  echo "=== run ${i} -> ${RUN}"
  OPENAI_API_KEY="$OAI_KEY" \
  LLM_BACKEND=openai \
  OPENAI_MODEL_NAME=gpt-5.6-terra \
  OPENAI_REASONING_EFFORT=none \
  PHASE_CACHE_DIR="${RUN}/phase_states" \
  LLM_LOG_DIR="${RUN}/llm_logs" \
    ../.venv/bin/python run_ablation.py \
    --variants s_linker65 s_linker65_null s_linker66 s_linker67 \
    --datasets mediastore teammates teastore bigbluebutton jabref \
    --results-dir "${RUN}" > "${RUN}.log" 2>&1
  echo "    exit $? — $(grep -c 'Final:' "${RUN}.log" 2>/dev/null) linker runs logged"
done
echo "ALL RUNS DONE"
