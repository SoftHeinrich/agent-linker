#!/usr/bin/env python3
"""Neutral extracts for an agent-linker ablation arm, from its per-project link CSVs.

``build_dump.py`` reads one JSON per (backend, run, project) cell and uses exactly
one field of it — ``final.links`` — to write the sota ``model-doc`` slot and compose the
``doc-code`` slot through the ArCoTL bridge.  ``approach/scripts/extract_s20union_caches.py``
produces those JSONs for the s20U/s21 sweep by walking that pipeline's phase pickles, and
its own faithfulness oracle is the run's ``*_links.csv``: the extracted ``final.links`` set
must equal the CSV on ``(sentence, component_id, source)``.

The s25-lineage arms (``s_linker25`` … ``s_linker92*``) write different phase names, so that
extractor does not read them — but they write the same link CSVs.  This builds the cell JSONs
from the oracle directly, so any arm with recorded ``run_ablation.py`` output can be scored by
``mini-src/`` and ``rq12.py`` without touching either.

Only ``final`` and ``meta`` are emitted.  The ``entity`` / ``coref`` / ``audit`` sections of the
s21 extracts are the per-validator decisions ``mini-rq34`` needs for RQ3/RQ4; they are not
recoverable from a link CSV, and RQ1/RQ2 never read them.

Stdlib only.  No LLM calls.  Deterministic: re-running produces byte-identical output.

    python3 mini-src/build_alinker_extracts.py --variant s_linker92a \
        --out ../results/s92a_extracts \
        --model terra ../results/regex_e2e_terra_r1_20260822 \
                      ../results/regex_e2e_terra_r2_20260822 \
                      ../results/regex_e2e_terra_r3_20260822
"""
import argparse
import csv
import json
import os
from pathlib import Path

import metrics as m   # same directory: the shared core (the project list)

PROJECTS = m.PROJECTS


def load_links(run_dir: Path, variant: str, project: str):
    """The arm's recovered doc->model links for one cell, in extracts dialect."""
    path = run_dir / f"{variant}_{project}_links.csv"
    if not path.exists():
        return None
    links = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            links.append({
                "s": int(row["sentence"]),
                "c": row["component_id"],
                "component_name": row.get("component_name", ""),
                "confidence": float(row["confidence"]) if row.get("confidence") else "",
                "source": row.get("source", ""),
            })
    # Deterministic order; the dump dedupes and re-sorts on (sentence, target) anyway.
    links.sort(key=lambda link: (link["s"], link["c"], link["source"]))
    return links, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        help="the arm's _VARIANT_NAME, e.g. s_linker92a")
    parser.add_argument("--out", required=True,
                        help="extracts root; cells land at <out>/<model>/run<i>/<project>.json")
    parser.add_argument("--model", nargs="+", action="append", required=True,
                        metavar=("NAME", "RUN"),
                        help="a backend tag followed by its run directories, in run order")
    args = parser.parse_args()

    total = 0
    for spec in args.model:
        model, run_dirs = spec[0], spec[1:]
        for index, run_dir in enumerate(run_dirs, start=1):
            run = f"run{index}"
            run_dir = Path(run_dir)
            for project in PROJECTS:
                loaded = load_links(run_dir, args.variant, project)
                if loaded is None:
                    print(f"  MISSING {model}/{run}/{project}: "
                          f"{run_dir / f'{args.variant}_{project}_links.csv'}")
                    continue
                links, src = loaded
                cell = {
                    "final": {"links": links, "provenance": {"source_csv": str(src)}},
                    "meta": {"variant": args.variant, "backend_tag": model,
                             "project": project, "run": run,
                             "run_dir": run_dir.name, "n_links": len(links)},
                }
                out = Path(args.out) / model / run / f"{project}.json"
                os.makedirs(out.parent, exist_ok=True)
                with out.open("w", newline="\n") as handle:
                    json.dump(cell, handle, indent=1, sort_keys=True)
                    handle.write("\n")
                total += 1
                print(f"  {model}/{run}/{project}: {len(links)} links -> {out}")
    print(f"wrote {total} cells for {args.variant} into {args.out}")


if __name__ == "__main__":
    main()
