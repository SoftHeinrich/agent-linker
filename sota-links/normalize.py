#!/usr/bin/env python3
"""Canonicalize every recovered-links CSV in place.

Idempotent. Produces uniform files: header `sentence_id,target_id`, integer
sentence id, rows deduplicated and sorted by (sentence_id, target_id), unix
newlines. Verified convention: sentence ids are 1-based and aligned to the ArDoCo
gold standard for both tasks (see README); no index shifting is applied.
"""
import csv, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
HEADER = ["sentence_id", "target_id"]


def normalize(path):
    rows = set()
    with open(path, newline="") as f:
        r = csv.reader(f)
        first = next(r, None)
        # Skip a header row if present; otherwise treat it as data.
        if first and not (first[0].strip().lower() in ("sentence_id", "sentenceid", "sentence")):
            if len(first) >= 2 and first[0].strip().lstrip("-").isdigit():
                rows.add((int(first[0]), first[1].strip()))
        for row in r:
            if len(row) >= 2 and row[0].strip().lstrip("-").isdigit():
                rows.add((int(row[0]), row[1].strip()))
    ordered = sorted(rows, key=lambda t: (t[0], t[1]))
    with open(path, "w", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(HEADER)
        for s, t in ordered:
            w.writerow([s, t])
    return len(ordered)


def main():
    total = 0
    for task in ("doc-code", "model-doc"):
        d = os.path.join(ROOT, task)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".csv"):
                n = normalize(os.path.join(d, fn))
                total += 1
                print(f"  {task}/{fn:46s} {n:5d} unique links")
    print(f"normalized {total} files")


if __name__ == "__main__":
    main()
