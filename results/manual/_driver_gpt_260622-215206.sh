#!/usr/bin/env bash
set -uo pipefail
cd /mnt/hostshare/ardoco-home/agent-linker
L=/mnt/hostshare/ardoco-home/agent-linker/run_s20union_once.sh
COOLDOWN=45
log(){ echo "[$(date '+%F %T')] $*"; }
run(){ local label="$1"; shift; log ">>> START $label :: $*"; "$L" "$@"; log "<<< END   $label rc=$?"; }
log "GPT TRACK START — gpt x3 + gpt-noknow x3, all 5 datasets, sequential (parallel to sonnet track)"
for rep in rep1 rep2 rep3; do run "gpt/$rep"          -b openai -m gpt-5.4 -t "$rep"; sleep $COOLDOWN; done
for rep in rep1 rep2 rep3; do run "gpt-noknow/$rep" -k -b openai -m gpt-5.4 -t "$rep"; sleep $COOLDOWN; done
log "GPT TRACK COMPLETE"
