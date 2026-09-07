#!/usr/bin/env bash
set -uo pipefail
cd /mnt/hostshare/ardoco-home/agent-linker
L=/mnt/hostshare/ardoco-home/agent-linker/run_s20union_once.sh
COOLDOWN=45
WAIT_PID=1334213
log(){ echo "[$(date '+%F %T')] $*"; }
run(){ local label="$1"; shift; log ">>> START $label :: $*"; "$L" "$@"; log "<<< END   $label rc=$?"; }
log "SONNET TRACK START — waiting for in-flight sonnet/rep1 (pid $WAIT_PID) before rep2/rep3 (keeps sonnet sequential)"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
log "sonnet/rep1 finished — starting rep2, rep3"
for rep in rep2 rep3; do run "sonnet/$rep" -b claude -m sonnet -t "$rep"; sleep $COOLDOWN; done
log "SONNET TRACK COMPLETE (rep1 was completed by the original run)"
