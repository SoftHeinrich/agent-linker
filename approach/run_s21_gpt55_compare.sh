#!/usr/bin/env bash
# s_linker21 (CANONICAL) gpt-5.5 sweep, N=3 — fresh live calls, for the gpt-5.5
# head-to-head vs Artemis. Mirrors run_s21_gpt_n3.sh (no-reasoning, per-run cache
# isolation for genuine N=3 independence) but:
#   - model pinned to gpt-5.5-2026-04-23 (matches Artemis GPT_5_5 snapshot)
#   - writes into a caller-provided timestamped results root (never overwrites)
# Usage: run_s21_gpt55_compare.sh <OURS_ROOT>   (e.g. .../gpt55_compare_<TS>/ours)
set -uo pipefail

OURS_ROOT="${1:?pass the timestamped ours/ results root}"
VARIANT="s_linker21"
DATASETS=(mediastore teastore jabref bigbluebutton teammates)
N=3
LOGBASE="$OURS_ROOT/_logs"
COOLDOWN_DS=15
COOLDOWN_RUN=30
COOLDOWN_RETRY=60

export LLM_BACKEND=openai
export OPENAI_MODEL_NAME=gpt-5.5-2026-04-23
# gpt-5.5 REJECTS temperature != 1 (HTTP 400), so the gpt-5.4 "temperature path" for
# no-reasoning does not work. The faithful reasoning-OFF equivalent on gpt-5.5 is
# reasoning_effort=none (0 reasoning tokens, default temperature), which the layered
# validator needs and which also matches how Artemis GPT_5_5 runs (temp 1, flex).
export OPENAI_REASONING_EFFORT=none
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set for the openai backend}"

mkdir -p "$LOGBASE"
PROG="$LOGBASE/PROGRESS.log"
ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$PROG"; }

log "SWEEP START variant=$VARIANT backend=openai model=$OPENAI_MODEL_NAME reasoning=NONE N=$N datasets='${DATASETS[*]}' root=$OURS_ROOT"

run_one() {  # args: run_idx dataset ; 0=ok 1=fail
  local i="$1" ds="$2"
  local rdir="$OURS_ROOT/run$i"
  local rds="$rdir/$ds"
  local done_marker="$rds/.done"
  local csv="$rds/${VARIANT}_${ds}_links.csv"
  if [ -f "$done_marker" ]; then log "SKIP run$i/$ds (resume)"; return 0; fi
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
    touch "$done_marker"; log "OK   run$i/$ds rc=$rc links_csv_lines=$nlines"; return 0
  fi
  log "FAIL run$i/$ds rc=$rc links_csv_lines=${nlines:-0} (will retry once)"; return 1
}

for i in $(seq 1 "$N"); do
  log "===== RUN $i / $N ====="
  for ds in "${DATASETS[@]}"; do
    if ! run_one "$i" "$ds"; then
      log "COOLDOWN ${COOLDOWN_RETRY}s before retry run$i/$ds"; sleep "$COOLDOWN_RETRY"
      if ! run_one "$i" "$ds"; then log "GIVEUP run$i/$ds after one retry"; fi
    fi
    log "COOLDOWN ${COOLDOWN_DS}s (between datasets)"; sleep "$COOLDOWN_DS"
  done
  if [ "$i" -lt "$N" ]; then log "COOLDOWN ${COOLDOWN_RUN}s (between runs)"; sleep "$COOLDOWN_RUN"; fi
done

log "SWEEP COMPLETE"
touch "$LOGBASE/.ALL_DONE"
