#!/usr/bin/env bash
# Build the self-contained ICSE24 TransArC package (ardoco+arcotl, rev 0.18.0-ICSE24-SNAPSHOT,
# no NER) and run RawLinkDumpIT to dump deterministic baseline links on all 5 benchmark projects:
#   SWATTR  -> SAD-SAM  (model-doc): target/raw-links/SWATTR-<proj>-sad-sam.csv
#   TransArC-> SAD-Code (doc-code) : target/raw-links/TRANSARC-<proj>-sad-code.csv
set -o pipefail
TA="/mnt/hostshare/ardoco-home/sota/baseline-repos/transarc-icse24/ardoco+arcotl"
MVN="mvn -B -ntp -Dmaven.javadoc.skip=true"

echo "### [$(date -u +%H:%M:%S)] STEP 1/2: build+install reactor (skip test exec)"
( cd "$TA" && $MVN -DskipTests install ) || { echo "### REACTOR BUILD FAILED"; exit 10; }

echo "### [$(date -u +%H:%M:%S)] STEP 2/2: run RawLinkDumpIT (SWATTR + TransArC, 5 projects)"
( cd "$TA/tests/tests-tlr" && $MVN -Dtest=_none_ -Dit.test=RawLinkDumpIT -DfailIfNoTests=false verify ) || { echo "### IT RUN FAILED"; exit 30; }

echo "### [$(date -u +%H:%M:%S)] DONE. Outputs:"
ls -la "$TA/tests/tests-tlr/target/raw-links/" 2>/dev/null || echo "(no raw-links dir)"
