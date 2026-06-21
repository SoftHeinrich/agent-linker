#!/usr/bin/env bash
# s_linker20_union_noknow gpt-5.4 No-Knowledge sweep, N=3 — fresh live calls; companion to run_s20union_noknow_sonnet_n3.sh.
# Strictly sequential (no parallelism beyond the variant's own <=2-way DAG),
# cooldowns between datasets/runs, retry-once on empty/failed datasets, and
# resume via per-(run,dataset) .done markers. All intermediate results saved.
# Writes into the annotated _noknow folder: results/v2.6.6_s20union_noknow/gpt/run{1,2,3}/<ds>/.
set -uo pipefail

VARIANT="s_linker20_union_noknow"
# light -> heavy so failures surface early and the slow ones run last
DATASETS=(mediastore teastore jabref bigbluebutton teammates)
N=3
BASE="results/v2.6.6_s20union_noknow/gpt"
LOGBASE="logs/v2.6.6_s20union_noknow_gpt"
COOLDOWN_DS=90        # between datasets
COOLDOWN_RUN=240      # between runs
COOLDOWN_RETRY=300    # before a retry after an empty/failed dataset

export LLM_BACKEND=openai
export OPENAI_MODEL_NAME=gpt-5.4
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set for the openai backend}"

mkdir -p "$LOGBASE" "$BASE"
PROG="$LOGBASE/PROGRESS.log"
ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$PROG"; }

log "SWEEP START variant=$VARIANT backend=$LLM_BACKEND model=$OPENAI_MODEL_NAME N=$N datasets='${DATASETS[*]}'"

CUM_CALLS=0

run_one() {  # args: run_idx dataset ; 0=ok 1=fail
  local i="$1" ds="$2"
  local rdir="$BASE/run$i"
  local rds="$rdir/$ds"
  local done_marker="$rds/.done"
  local csv="$rds/${VARIANT}_${ds}_links.csv"
  if [ -f "$done_marker" ]; then log "SKIP run$i/$ds (resume: already done)"; return 0; fi
  mkdir -p "$rds" "$rdir/phase_cache" "$rdir/llm_logs" "$rdir/llm_checkpoint"
  # LANDMINE 3: PHASE_CACHE_DIR MUST be per-run under the _noknow BASE.
  # NEVER point this at a shared or Full cache path — the linker's _VARIANT_NAME is
  # "s_linker20_union" so a shared PHASE_CACHE_DIR would silently clobber Full caches.
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
    # D-08: cumulative call-count cost visibility (log-and-continue; no hard abort).
    # _calls.json is a JSON list; len() = call count. Actual $ via API dashboard.
    local_calls=0
    rdir_for_cost="$BASE/run$i"
    for f in "$rdir_for_cost/llm_logs/"*"_${ds}_"*"_calls.json"; do
      [ -f "$f" ] || continue
      n=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(len(d))" "$f" 2>/dev/null || echo 0)
      local_calls=$((local_calls + n))
    done
    CUM_CALLS=$((CUM_CALLS + local_calls))
    log "COST run$i/$ds calls=$local_calls cum_calls=$CUM_CALLS (actual \$ via API dashboard; ~\$60 soft cap, log-and-continue)"
    log "COOLDOWN ${COOLDOWN_DS}s (between datasets)"
    sleep "$COOLDOWN_DS"
  done
  if [ "$i" -lt "$N" ]; then
    log "COOLDOWN ${COOLDOWN_RUN}s (between runs)"
    sleep "$COOLDOWN_RUN"
  fi
done

log "SWEEP COMPLETE cum_calls=$CUM_CALLS"
touch "$LOGBASE/.ALL_DONE"
