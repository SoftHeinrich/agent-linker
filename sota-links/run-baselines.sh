#!/usr/bin/env bash
# Builds the ArDoCo reactor (core+tlr together) and runs BaselineRawTraceLinksIT to
# dump SWATTR (SAD-SAM / model-doc) and TransArC (SAD-Code / doc-code) recovered links.
# Deterministic baselines -> no API key required.
# Notes:
#  - core+tlr must build in ONE reactor: core:pipeline-core depends on tlr:model-provider.
#  - use -DskipTests (NOT -Dmaven.test.skip): a downstream module needs pipeline-core's
#    test-jar, which is only built when test sources are compiled.
set -o pipefail
TAAS="/mnt/hostshare/ardoco-home/sota/Replication-Package-TAAS25_LLM-assisted-Software-Traceability-with-Architecture-Entity-Recognition/Replication-Package-TAAS25"
# metrics:0.2.0-SNAPSHOT is unpublished; override to the released 0.2.0 on Central
# (closest to the snapshot -> lowest API-drift risk). tests-base/evaluations need it.
MVN="mvn -B -ntp -Dmaven.javadoc.skip=true -Dmetrics.version=0.2.0"

echo "### [$(date -u +%H:%M:%S)] STEP 1/2: build+install core+tlr reactor (skip test exec, keep test-jars)"
( cd "$TAAS" && $MVN -f aggregator-pom.xml -DskipTests install ) || { echo "### REACTOR BUILD FAILED"; exit 10; }

echo "### [$(date -u +%H:%M:%S)] STEP 2/2: run BaselineRawTraceLinksIT (SWATTR + TransArC)"
( cd "$TAAS/tlr/tests-tlr" && $MVN verify -Dtest=_none_ -Dit.test=BaselineRawTraceLinksIT -DfailIfNoTests=false ) || { echo "### IT RUN FAILED"; exit 30; }

echo "### [$(date -u +%H:%M:%S)] DONE. Outputs:"
ls -la "$TAAS/tlr/tests-tlr/target/raw-tracelinks/" | grep -iE 'SWATTR|TRANSARC' || echo "(no SWATTR/TRANSARC files produced)"
