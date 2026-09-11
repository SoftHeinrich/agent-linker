#!/usr/bin/env bash
# Deterministic baseline: ArDoCo's SAD-SAM stage (SWATTR) through ardoco-cli on an
# oss-scale dataset, then the CoreNLP re-split map and the links.csv conversion.
#     tools/run_swattr.sh <dataset dir> <dataset name>     e.g. gitlab gitlab_arch
set -eu
DIR=$(cd "$(dirname "$0")/../$1" && pwd); NAME=$2
JAR=$(ls /mnt/hostshare/ardoco-home/ardoco-cli-dev/target/ardoco-cli-*-jar-with-dependencies.jar | head -1)
OUT="$DIR/out/swattr"; mkdir -p "$OUT"
cd "$OUT"
java -jar "$JAR" -t sad-sam -n "$NAME" -d "$DIR/data/sentences.txt" -m "$DIR/data/$NAME.repository" -o "$OUT" > cli.log 2>&1
java -cp "$JAR" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit -file "$DIR/data/sentences.txt" -outputFormat json -outputDirectory "$OUT" > corenlp.log 2>&1
SRC=$(ls "$OUT"/sadSamTlr_*.csv "$OUT"/*/sadSamTlr_*.csv 2>/dev/null | head -1)
PY=$(cd "$DIR/../../.." && pwd)/.venv/bin/python
(cd "$DIR/../rustc/semgold" && "$PY" from_ardoco.py "$SRC" "$OUT/swattr_links.csv" --corenlp-json "$OUT/sentences.txt.json" --sentences "$DIR/data/sentences.txt")
echo "SWATTR DONE: $(wc -l < "$OUT/swattr_links.csv") lines"
