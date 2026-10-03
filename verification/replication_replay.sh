#!/usr/bin/env bash
# Replays all 60 recorded AgentLinker dataset runs of replication/agentlinker and compares link CSVs.
# usage: verification/replication_replay.sh <package-agentlinker-dir> <python> <outdir> <hashseed>
P=$1; PY=$2; OUT=$3; SEED=$4; ok=0; bad=0
cd "$P"
for m in terra luna; do for a in full no-aliases; do for r in 1 2 3; do
  d=recorded/$m/$a/run$r; o=$OUT/$m/$a/run$r; mkdir -p $o; flag=""; [ $a = no-aliases ] && flag=--no-aliases
  for ds in mediastore teammates teastore bigbluebutton jabref; do
    if PYTHONHASHSEED=$SEED $PY run.py --replay $d $flag --datasets $ds --output $o > $o/$ds.log 2>&1 && cmp -s $d/${ds}_links.csv $o/${ds}_links.csv; then ok=$((ok+1)); else bad=$((bad+1)); echo "FAIL $m/$a/run$r $ds: $(tail -1 $o/$ds.log)"; fi
  done; done; done; done
echo "replay seed $SEED: $ok/60 byte-identical, $bad failed"
