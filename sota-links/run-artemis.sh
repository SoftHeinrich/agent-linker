#!/usr/bin/env bash
# Re-run the ArTEMiS baseline on a chosen backend, N times, and refresh its slot here.
#
# WHY THIS EXISTS. The released ArTEMiS links are a single gpt-5.4 run, while \approach
# reports GPT-5.6-terra/luna as the mean of three. Comparing across backends confounds the
# workflow with the model, and comparing a mean-of-3 against an n=1 confounds it with run
# variance. This re-runs the baseline on the same backend, the same number of times.
# Everything except the model id is the baseline's own released configuration
# (ArtemisInTransArC, seed 422413373, temperature 1.0, the authors' NER stage) -- see
# patches/artemis-gpt56.patch for the diff this needs against the TAAS25 package.
#
#   ./run-artemis.sh terra            # runs 1 2 3 -> artemis/terra_5.6/run{1,2,3}/
#   ./run-artemis.sh terra 2 3        # only those runs (run1 already on disk)
#   ./run-artemis.sh luna
#
# RUN INDEPENDENCE -- do not collapse the cache dirs. LargeLanguageModel.create() wraps
# every backend in CachedChatLanguageModel, which memoizes each prompt to
# <LLM_CACHE_DIR>/<ENUM>-cache.json and replays it forever after. Sharing one cache dir
# would make runs 2 and 3 replay run 1 verbatim, and the "mean of three" would be a
# fiction with zero variance. Each run therefore gets its own cache dir.
#
# PREREQUISITE. ArTEMiS's NER stage depends on
# io.github.ardoco:named-architecture-entity-recognition:1.0.0-SNAPSHOT, which was never
# published to any Sonatype repository. Rebuild and install it once from source:
#   git clone https://github.com/ArDoCo/named-architecture-entity-recognition.git
#   git -C named-architecture-entity-recognition worktree add ../ner-1.0.0-SNAPSHOT 09cf428^
#   # 09cf428 ("Ardoco v2 Release", 2026-02-06) bumped 1.0.0-SNAPSHOT -> 2.0.0, so its
#   # parent commit is the last state carrying the version TAAS25 asks for.
#   # Its own parent (io.github.ardoco:parent:2.0.0-SNAPSHOT) is likewise unpublished;
#   # repoint it to the released 2.0.1 -- plugin/dependency management only, and every
#   # dependency this module compiles against is pinned in its own dependencyManagement.
#   cd ../ner-1.0.0-SNAPSHOT && mvn -Dflatten.skip=true -DskipTests install
set -o pipefail
set -u

BACKEND=${1:?usage: run-artemis.sh <terra|luna> [run indices...]}
shift
RUN_IDS=("$@")
[ ${#RUN_IDS[@]} -eq 0 ] && RUN_IDS=(1 2 3)

HERE="$(cd "$(dirname "$0")" && pwd)"
TAAS="/mnt/hostshare/ardoco-home/sota/Replication-Package-TAAS25_LLM-assisted-Software-Traceability-with-Architecture-Entity-Recognition/Replication-Package-TAAS25"
LOGS="$HERE/_build-logs"
mkdir -p "$LOGS"

export ARTEMIS_LLM=${ARTEMIS_LLM:-GPT_5_6}
export OPENAI_MODEL_NAME_5_6="gpt-5.6-${BACKEND}"
SLOT="${BACKEND}_5.6"
TAG="gpt-5.6-${BACKEND}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set for live calls}"

# metrics:0.2.0-SNAPSHOT is unpublished; pin the released 0.2.0 (same override the
# deterministic baselines use in run-baselines.sh).
MVN="mvn -B -ntp -Dmaven.javadoc.skip=true -Dmetrics.version=0.2.0"

# The root pom (io.github.ardoco:parent:2.0.0-taas25) is NOT a module of
# aggregator-pom.xml, so the reactor build never installs it. The reactor gets away with
# that because it resolves the parent from itself; the tests-tlr build below resolves
# pipeline-core's descriptor from ~/.m2, where the parent must exist.
echo "### [$(date -u +%H:%M:%S)] BUILD 1/2: install the root parent pom (non-recursive)"
( cd "$TAAS" && $MVN -N -DskipTests install ) > "$LOGS/build-artemis-parent-$TAG.log" 2>&1 \
  || { echo "### PARENT POM INSTALL FAILED -- see $LOGS/build-artemis-parent-$TAG.log"; tail -20 "$LOGS/build-artemis-parent-$TAG.log"; exit 5; }

echo "### [$(date -u +%H:%M:%S)] BUILD 2/2: build+install core+tlr reactor"
( cd "$TAAS" && $MVN -f aggregator-pom.xml -DskipTests install ) > "$LOGS/build-artemis-$TAG.log" 2>&1 \
  || { echo "### REACTOR BUILD FAILED -- see $LOGS/build-artemis-$TAG.log"; tail -30 "$LOGS/build-artemis-$TAG.log"; exit 10; }

RAW="$TAAS/tlr/tests-tlr/target/raw-tracelinks/$ARTEMIS_LLM"

for i in "${RUN_IDS[@]}"; do
  RUN="run${i}"
  echo "### [$(date -u +%H:%M:%S)] RUN $RUN ($ARTEMIS_LLM = $OPENAI_MODEL_NAME_5_6)"
  rm -rf "$RAW"
  # -Dtest=_none_ skips the unit tests so only the IT runs; surefire >= 3.x errors on a
  # pattern that matches nothing, hence failIfNoSpecifiedTests=false.
  ( cd "$TAAS/tlr/tests-tlr" \
    && LLM_CACHE_DIR="$TAAS/.cache-llm-artemis-$TAG-$RUN/" \
       $MVN verify -Dtest=_none_ -Dsurefire.failIfNoSpecifiedTests=false \
            -Dit.test=RawTraceLinksIT -DfailIfNoTests=false ) \
    > "$LOGS/run-artemis-$TAG-$RUN.log" 2>&1 \
    || { echo "### RUN $RUN FAILED -- see $LOGS/run-artemis-$TAG-$RUN.log"; tail -40 "$LOGS/run-artemis-$TAG-$RUN.log"; exit 30; }

  python3 - "$RAW" "$HERE" "$SLOT" "$RUN" <<'PY'
import csv, sys, os
raw, root, slot, run = sys.argv[1:5]
PROJECTS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]
# The IT dumps <PROJECT>-sad-sam.tsv (the NER doc-model links) and <PROJECT>-sad-code.tsv
# (ArTEMiS composed through TransArC's own file mapping -- the baseline's native route,
# not a re-composition through the standalone ArCoTL bridge).
for task, suffix in (("model-doc", "sad-sam"), ("doc-code", "sad-code")):
    for proj in PROJECTS:
        src = os.path.join(raw, f"{proj.upper()}-{suffix}.tsv")
        if not os.path.exists(src):
            raise SystemExit(f"missing dump: {src}")
        rows = set()
        with open(src, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                s, t = line.rstrip("\n").split("\t", 1)
                rows.add((int(s), t.strip()))
        dst_dir = os.path.join(root, task, "artemis", slot, run)
        os.makedirs(dst_dir, exist_ok=True)
        with open(os.path.join(dst_dir, f"{proj}.csv"), "w", newline="\n", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["sentence_id", "target_id"])
            for s, t in sorted(rows):
                w.writerow([s, t])
        print(f"  {task}/artemis/{slot}/{run}/{proj}.csv  {len(rows):6d} links")
PY
done

echo "### [$(date -u +%H:%M:%S)] DONE. Regenerate: see evaluation/HOWTO-REGENERATE-RQ.md."
