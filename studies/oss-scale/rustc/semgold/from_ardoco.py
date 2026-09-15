#!/usr/bin/env python3
"""Convert an ArDoCo CLI SAD-SAM output into the links.csv shape `rescore.py` reads.

ArDoCo re-splits the input text with CoreNLP, so its sentence ids do not match the dataset's
line numbers (1,915 CoreNLP sentences for 1,762 lines on the rustc core chapters).  Produce the
splitter's own JSON with the CLI's bundled CoreNLP and pass it with --corenlp-json; each ArDoCo
sentence is then mapped to the dataset line its first token falls in.

  java -cp ardoco-cli-*-jar-with-dependencies.jar edu.stanford.nlp.pipeline.StanfordCoreNLP \
    -annotators tokenize,ssplit -file sentences.txt -outputFormat json -outputDirectory <dir>

usage: from_ardoco.py sadSamTlr_<project>.csv out_links.csv
                      [--corenlp-json sentences.txt.json] [--sentences sentences.txt]
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json

from common import DATA


def line_map(corenlp_json: str, sentences: str):
    sents = json.load(open(corenlp_json))["sentences"]
    raw = open(sentences).read()
    starts, off = [], 0
    for line in raw.split("\n"):
        starts.append(off)
        off += len(line) + 1
    # ArDoCo numbers the CoreNLP sentences from 1
    return {i: bisect.bisect_right(starts, s["tokens"][0]["characterOffsetBegin"])
            for i, s in enumerate(sents, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--corenlp-json", default="")
    ap.add_argument("--sentences", default=str(DATA / "sentences.txt"))
    ap.add_argument("--source", default="swattr")
    args = ap.parse_args()
    remap = line_map(args.corenlp_json, args.sentences) if args.corenlp_json else None
    rows = list(csv.DictReader(open(args.src)))
    links = set()
    dropped = 0
    for r in rows:
        n = int(r["sentence"])
        if remap is not None:
            if n not in remap:
                dropped += 1
                continue
            n = remap[n]
        links.add((n, r["modelElementID"]))
    with open(args.dst, "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["sentence", "component_id", "component_name", "confidence", "source"])
        for n, c in sorted(links):
            w.writerow([n, c, c, "1.00", args.source])
    print(f"{len(rows)} rows -> {len(links)} unique links"
          + (f" (dropped {dropped} unmappable ids)" if dropped else ""))


if __name__ == "__main__":
    main()
