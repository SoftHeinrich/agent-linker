#!/usr/bin/env python3
"""Turn (sentences.txt, components.json, gold.csv) into a run_ablation dataset.

Writes <out>/<name>.repository (minimal PCM XML the approach's parser accepts) and
<out>/datasets.json to pass via ALINKER_EXTRA_DATASETS.  components.json is a list of
{"id": ..., "name": ...}; gold.csv has columns modelElementID,sentence (sentence is
1-based line number in sentences.txt).  Stdlib only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import quoteattr


def write_repository(components: list[dict], path: Path) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<repository:Repository xmi:version="2.0" xmlns:xmi="http://www.omg.org/XMI" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xmlns:repository="http://palladiosimulator.org/PalladioComponentModel/Repository/5.2" '
        'id="oss-scale-repository" entityName="oss-scale">',  # ArDoCo's PCM parser needs a repository id
    ]
    for comp in components:
        lines.append(
            f'  <components__Repository xsi:type="repository:BasicComponent" '
            f'id={quoteattr(comp["id"])} entityName={quoteattr(comp["name"])}/>'
        )
    lines.append("</repository:Repository>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name")
    ap.add_argument("--sentences", required=True, type=Path)
    ap.add_argument("--components", required=True, type=Path)
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    components = json.loads(args.components.read_text())
    repo = args.out / f"{args.name}.repository"
    write_repository(components, repo)
    spec_path = args.out / "datasets.json"
    spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
    spec[args.name] = {
        "text": str(args.sentences.resolve().relative_to(args.out.resolve()))
        if args.sentences.resolve().is_relative_to(args.out.resolve()) else str(args.sentences.resolve()),
        "model": repo.name,
        "gold_sam": str(args.gold.resolve().relative_to(args.out.resolve()))
        if args.gold.resolve().is_relative_to(args.out.resolve()) else str(args.gold.resolve()),
    }
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"wrote {repo} and {spec_path} ({len(components)} components)")


if __name__ == "__main__":
    main()
