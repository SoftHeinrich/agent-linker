#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?usage: run.sh <terra|luna> [run numbers...]}
shift
RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(1 2 3)
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/run-output/$MODEL"
MVN="mvn -B -ntp -Dmaven.javadoc.skip=true -Dmetrics.version=0.2.0"
export ARTEMIS_LLM=GPT_5_6
export OPENAI_MODEL_NAME_5_6="gpt-5.6-$MODEL"

(cd "$HERE/ner" && mvn -B -ntp -Dflatten.skip=true -DskipTests install)
(cd "$HERE/taas25" && $MVN -N -DskipTests install)
(cd "$HERE/taas25" && $MVN -f aggregator-pom.xml -DskipTests install)

RAW="$HERE/taas25/tlr/tests-tlr/target/raw-tracelinks/$ARTEMIS_LLM"
for run in "${RUNS[@]}"; do
  rm -rf "$RAW"
  (cd "$HERE/taas25/tlr/tests-tlr" \
    && LLM_CACHE_DIR="$HERE/run-output/cache-$MODEL-run$run/" \
       $MVN verify -Dtest=_none_ -Dsurefire.failIfNoSpecifiedTests=false \
            -Dit.test=RawTraceLinksIT -DfailIfNoTests=false)
  python3 - "$RAW" "$OUT/run$run" <<'PY'
import csv, os, sys

raw, out = sys.argv[1:]
for task, suffix in (("doc-model", "sad-sam"), ("doc-code", "sad-code")):
    os.makedirs(os.path.join(out, task), exist_ok=True)
    for project in ("mediastore", "teastore", "teammates", "bigbluebutton", "jabref"):
        rows = set()
        with open(os.path.join(raw, f"{project.upper()}-{suffix}.tsv"), encoding="utf-8") as handle:
            for line in handle:
                if line.strip() and not line.startswith("#"):
                    sentence, target = line.rstrip("\n").split("\t", 1)
                    rows.add((int(sentence), target.strip()))
        with open(os.path.join(out, task, f"{project}.csv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["sentence_id", "target_id"])
            writer.writerows(sorted(rows))
        print(f"run{os.path.basename(out)[3:]} {task} {project}: {len(rows)} links")
PY
done
