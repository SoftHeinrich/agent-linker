#!/usr/bin/env python3
"""Convert an ArDoCo CLI SAD-SAM output (modelElementID,sentence) into the links.csv shape
`rescore.py` and `surface.py` read, so the deterministic baseline is scored on exactly the
same gold as the linker.

usage: from_ardoco.py sadSamTlr_<project>.csv out_links.csv [--source swattr]
"""
from __future__ import annotations

import argparse
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--source", default="swattr")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.src)))
    with open(args.dst, "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["sentence", "component_id", "component_name", "confidence", "source"])
        seen = set()
        for r in rows:
            key = (r["sentence"], r["modelElementID"])
            if key in seen:
                continue
            seen.add(key)
            w.writerow([r["sentence"], r["modelElementID"], r["modelElementID"], "1.00", args.source])
    print(f"{len(rows)} rows -> {len(seen)} unique links")


if __name__ == "__main__":
    main()
