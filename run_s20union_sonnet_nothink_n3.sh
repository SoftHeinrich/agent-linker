#!/usr/bin/env bash
# s_linker20_union Sonnet baseline with extended thinking DISABLED, N=3.
# Twin of run_s20union_sonnet_n3.sh; the only behavioral difference is
# CLAUDE_DISABLE_THINKING=1 (sets MAX_THINKING_TOKENS=0 + adaptive-off on the
# `claude -p` child, verified to drop thinking blocks 1->0 with no quality loss).
# Separate BASE/LOGBASE so its .done markers and results never collide with the
# thinking-on sweep. Strictly sequential, cooldowns, retry-once, resume via
# per-(run,dataset) .done markers. All intermediate results saved.
set -uo pipefail

VARIANT="s_linker20_union"
# light -> heavy so failures surface early and the slow ones run last
DATASETS=(mediastore teastore jabref bigbluebutton teammates)
N=3
BASE="results/v2.6.5_s20union_sonnet_nothink"
LOGBASE="logs/v2.6.5_s20union_sonnet_nothink"
COOLDOWN_DS=90        # between datasets
COOLDOWN_RUN=240      # between runs
COOLDOWN_RETRY=300    # before a retry after an empty/failed dataset

export LLM_BACKEND=claude
export CLAUDE_MODEL=sonnet
export CLAUDE_DISABLE_THINKING=1   # tokens=0: full off (MAX_THINKING_TOKENS=0 + adaptive off)

mkdir -p "$LOGBASE"
PROG="$LOGBASE/PROGRESS.log"
ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$PROG"; }

log "SWEEP START variant=$VARIANT backend=$LLM_BACKEND model=$CLAUDE_MODEL thinking=disabled N=$N datasets='${DATASETS[*]}'"

run_one() {  # args: run_idx dataset ; 0=ok 1=fail
  local i="$1" ds="$2"
  local rdir="$BASE/run$i"
  local rds="$rdir/$ds"
  local done_marker="$rds/.done"
  local csv="$rds/${VARIANT}_${ds}_links.csv"
  if [ -f "$done_marker" ]; then log "SKIP run$i/$ds (resume: already done)"; return 0; fi
  mkdir -p "$rds" "$rdir/phase_cache" "$rdir/llm_logs" "$rdir/llm_checkpoint"
  export PHASE_CACHE_DIR="$rdir/phase_cache"   # per-run -> genuine N=3 independence
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
