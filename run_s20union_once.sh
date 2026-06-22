#!/usr/bin/env bash
# run_s20union_once.sh — one hand-run of s_linker20_union with NO overwrite and
# self-describing, config-distinguishable output directories.
#
# Problem this solves: calling `python run_ablation.py --variants s_linker20_union`
# directly uses FIXED default paths — the per-dataset link CSV is overwritten and
# the phase cache (results/phase_cache) is SHARED, so a "fresh" hand-run can silently
# reuse a previous run's cached LLM outputs. This wrapper gives every invocation its
# own stamped directory and a per-run phase cache (genuinely fresh, nothing clobbered).
#
# Every invocation writes ONLY under one fresh dir:
#   results/manual/<slug>/                  --results-dir (link CSVs + ablation_*.json)
#   results/manual/<slug>/phase_cache/      per-run  -> fresh, no cache reuse
#   results/manual/<slug>/llm_logs/
#   results/manual/<slug>/llm_checkpoint/
#   results/manual/<slug>/run.log           full tee'd console log
#
# <slug> encodes the config so runs are distinguishable at a glance, e.g.:
#   s20union_openai_gpt-5.4_eff-medium__260622-204512
#   s20union-noknow_claude_sonnet__260622-204530__mytag
#
# Usage:
#   ./run_s20union_once.sh                         # all 5 datasets, backend from env (-> claude default)
#   ./run_s20union_once.sh -b openai -d mediastore # openai, one dataset
#   ./run_s20union_once.sh -b openai -e medium -t hi-effort
#   ./run_s20union_once.sh -k -b openai            # the s_linker20_union_noknow variant
#   ./run_s20union_once.sh -n -b openai -e medium  # dry run: print resolved config + paths, run nothing
#
# Flags:
#   -b BACKEND   openai | claude         (default: $LLM_BACKEND, else claude)
#   -m MODEL     model name             (default: gpt-5.4 for openai, sonnet for claude)
#   -e EFFORT    none|low|medium|high|xhigh   (openai reasoning effort; optional)
#   -k           use the s_linker20_union_noknow variant (no_knowledge=True)
#   -v VARIANT   explicit variant name  (overrides -k; default s_linker20_union)
#   -d "DS ..."  datasets               (default: all 5)
#   -t TAG       free-text tag appended to the slug
#   -n           dry run: resolve + print config/paths/command, execute nothing
#   --           pass any following args straight through to run_ablation.py
set -uo pipefail
cd "$(dirname "$0")"

VARIANT=""; BACKEND=""; MODEL=""; EFFORT=""; DATASETS=""; TAG=""; NOKNOW=0; DRY=0
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    -b) BACKEND="$2"; shift 2;;
    -m) MODEL="$2"; shift 2;;
    -e) EFFORT="$2"; shift 2;;
    -k) NOKNOW=1; shift;;
    -v) VARIANT="$2"; shift 2;;
    -d) DATASETS="$2"; shift 2;;
    -t) TAG="$2"; shift 2;;
    -n) DRY=1; shift;;
    --) shift; EXTRA=("$@"); break;;
    -h|--help) sed -n '2,40p' "$0"; exit 0;;
    *) echo "unknown arg: $1 (try -h)" >&2; exit 2;;
  esac
done

# Load .env defaults WITHOUT clobbering anything already exported (mirrors run_ablation).
if [ -f .env ]; then
  while IFS= read -r line; do
    line="${line#"${line%%[![:space:]]*}"}"          # ltrim
    [ -z "$line" ] && continue
    case "$line" in \#*) continue;; esac
    [ "$line" = "${line#*=}" ] && continue            # no '='
    k="${line%%=*}"; v="${line#*=}"
    k="${k//[[:space:]]/}"
    [ -z "${!k:-}" ] && export "$k=$v"
  done < .env
fi

# Resolve config (flags > env > sane defaults).
[ -z "$VARIANT" ] && { [ "$NOKNOW" = 1 ] && VARIANT="s_linker20_union_noknow" || VARIANT="s_linker20_union"; }
BACKEND="${BACKEND:-${LLM_BACKEND:-claude}}"
if [ "$BACKEND" = "openai" ]; then
  MODEL="${MODEL:-${OPENAI_MODEL_NAME:-gpt-5.4}}"
else
  MODEL="${MODEL:-${CLAUDE_MODEL:-sonnet}}"
fi

# Export the config so run_ablation/llm_client actually use what the slug claims.
export LLM_BACKEND="$BACKEND"
if [ "$BACKEND" = "openai" ]; then
  export OPENAI_MODEL_NAME="$MODEL"
  : "${OPENAI_API_KEY:?OPENAI_API_KEY must be set (export it or put it in .env) for the openai backend}"
  if [ -n "$EFFORT" ] && [ "$EFFORT" != "none" ]; then
    export OPENAI_REASONING_EFFORT="$EFFORT"
    export OPENAI_MAX_COMPLETION_TOKENS="${OPENAI_MAX_COMPLETION_TOKENS:-8192}"  # headroom for reasoning tokens
  fi
else
  export CLAUDE_MODEL="$MODEL"
fi

# Build a self-describing, filesystem-safe slug.
slug_variant="${VARIANT/s_linker20_union/s20union}"
slug_variant="${slug_variant/_noknow/-noknow}"
slug="${slug_variant}_${BACKEND}_${MODEL}"
[ -n "${OPENAI_REASONING_EFFORT:-}" ] && slug="${slug}_eff-${OPENAI_REASONING_EFFORT}"
slug="${slug}__$(date +%y%m%d-%H%M%S)-$$"   # -$$ (pid) guards against same-second collisions
[ -n "$TAG" ] && slug="${slug}__${TAG//[^A-Za-z0-9._-]/-}"

RUNDIR="results/manual/${slug}"

# Route ALL artifacts into the fresh per-run dir => no overwrite, no stale cache.
export PHASE_CACHE_DIR="$RUNDIR/phase_cache"
export LLM_LOG_DIR="$RUNDIR/llm_logs"
export CHECKPOINT_DIR="$RUNDIR/llm_checkpoint"

DS_ARGS=(); [ -n "$DATASETS" ] && DS_ARGS=(--datasets $DATASETS)
CMD=(python run_ablation.py --variants "$VARIANT" "${DS_ARGS[@]}" --results-dir "$RUNDIR" "${EXTRA[@]}")

print_header() {
  echo "RUN_ID   : $slug"
  echo "variant  : $VARIANT"
  echo "backend  : $BACKEND    model: $MODEL    effort: ${OPENAI_REASONING_EFFORT:-<none>}"
  echo "datasets : ${DATASETS:-<all 5>}"
  echo "results  : $RUNDIR"
  echo "command  : ${CMD[*]}"
}

if [ "$DRY" = 1 ]; then
  echo "=== DRY RUN (nothing executed) ==="
  print_header
  exit 0
fi

mkdir -p "$RUNDIR/phase_cache" "$RUNDIR/llm_logs" "$RUNDIR/llm_checkpoint"
LOGF="$RUNDIR/run.log"
{ print_header; echo "started  : $(date '+%F %T')"; echo "----------------------------------------------------------------------"; } | tee "$LOGF"

"${CMD[@]}" 2>&1 | tee -a "$LOGF"
rc=${PIPESTATUS[0]}

{ echo "----------------------------------------------------------------------"
  echo "finished : $(date '+%F %T')   rc=$rc"
  echo "artifacts: $RUNDIR"; } | tee -a "$LOGF"
exit "$rc"
