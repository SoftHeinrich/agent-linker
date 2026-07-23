#!/usr/bin/env bash
# s_linker20_union gpt-5.4 A/B: reasoning_effort=medium, N=3.
# Identical methodology to run_s20union_gpt_n3.sh (per-run isolated phase_cache, cooldowns,
# retry-once, .done resume) EXCEPT it engages gpt-5.4 reasoning at medium effort. Diff the
# resulting macro-F1 against the no-reasoning baseline (results/v2.6.5_s20union/gpt/, 0.8939).
# Reasoning tokens count against max_completion_tokens, so the cap is raised to 8192.
# Writes into: results/v2.6.5_s20union_gpt_re_medium/run{1,2,3}/<ds>/.
set -uo pipefail

VARIANT="s_linker20_union"
DATASETS=(mediastore teastore jabref bigbluebutton teammates)
N=3
BASE="results/v2.6.5_s20union_gpt_re_medium"
LOGBASE="logs/v2.6.5_s20union_gpt_re_medium"
COOLDOWN_DS=90
COOLDOWN_RUN=240
COOLDOWN_RETRY=300

export LLM_BACKEND=openai
export OPENAI_MODEL_NAME=gpt-5.4
export OPENAI_REASONING_EFFORT=medium      # none|low|medium|high|xhigh ('minimal' rejected by gpt-5.4)
export OPENAI_MAX_COMPLETION_TOKENS=8192   # headroom: reasoning tokens count against this cap
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set for the openai backend}"

mkdir -p "$LOGBASE" "$BASE"
PROG="$LOGBASE/PROGRESS.log"
ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$PROG"; }

log "SWEEP START variant=$VARIANT backend=$LLM_BACKEND model=$OPENAI_MODEL_NAME reasoning_effort=$OPENAI_REASONING_EFFORT max_completion=$OPENAI_MAX_COMPLETION_TOKENS N=$N datasets='${DATASETS[*]}'"

run_one() {  # args: run_idx dataset ; 0=ok 1=fail
  local i="$1" ds="$2"
  local rdir="$BASE/run$i"
  local rds="$rdir/$ds"
  local done_marker="$rds/.done"
  local csv="$rds/${VARIANT}_${ds}_links.csv"
  if [ -f "$done_marker" ]; then log "SKIP run$i/$ds (resume: already done)"; return 0; fi
  mkdir -p "$rds" "$rdir/phase_cache" "$rdir/llm_logs" "$rdir/llm_checkpoint"
  export PHASE_CACHE_DIR="$rdir/phase_cache"
  export LLM_LOG_DIR="$rdir/llm_logs"
  export CHECKPOINT_DIR="$rdir/llm_checkpoint"
  local logf="$LOGBASE/run${i}_${ds}.log"
  log "START run$i/$ds -> $logf"
  python run_ablation.py --variants "$VARIANT" --datasets "$ds" --results-dir "$rds" > "$logf" 2>&1
  local rc=$?
  local nlines=0
  [ -f "$csv" ] && nlines=$(wc -l < "$csv" | tr -d ' ')
  if [ "$rc" -eq 0 ] && [ "${nlines:-0}" -gt 1 ]; then
    touch "$done_marker"
    log "OK   run$i/$ds rc=$rc links_csv_lines=$nlines"
    return 0
  fi
  log "FAIL run$i/$ds rc=$rc links_csv_lines=${nlines:-0} (empty/error -> will retry once)"
  return 1
}

for i in $(seq 1 "$N"); do
  log "===== RUN $i / $N ====="
  for ds in "${DATASETS[@]}"; do
    if ! run_one "$i" "$ds"; then
      log "COOLDOWN ${COOLDOWN_RETRY}s before retry run$i/$ds"
      sleep "$COOLDOWN_RETRY"
      if ! run_one "$i" "$ds"; then
        log "GIVEUP run$i/$ds after one retry — continuing sweep"
      fi
    fi
    log "COOLDOWN ${COOLDOWN_DS}s (between datasets)"
    sleep "$COOLDOWN_DS"
  done
  if [ "$i" -lt "$N" ]; then
    log "COOLDOWN ${COOLDOWN_RUN}s (between runs)"
    sleep "$COOLDOWN_RUN"
  fi
done

log "SWEEP COMPLETE"
touch "$LOGBASE/.ALL_DONE"
