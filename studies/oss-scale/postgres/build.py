#!/usr/bin/env python3
"""Second in-the-wild pattern: a design document filed inside the directory it describes.

PostgreSQL keeps 91 `README` files under `src/`, each written by the authors of the module it
sits in (`src/backend/access/nbtree/README` is the B-tree design document).  The directory is
the component and the placement is the developers' own assignment — no registry, no labels.

Writes the same shape the Linux audit uses, so `../linux/annotate.py --dir ../postgres` and
`../linux/score.py --dir ../postgres` run on it unchanged.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RAW = "https://raw.githubusercontent.com/postgres/postgres/master/"
sys.path.insert(0, str(HERE.parent / "linux"))
from build import split_sentences  # noqa: E402


def fetch(path: str) -> str | None:
    r = subprocess.run(["curl", "-s", "-m", "30", RAW + path], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 and r.stdout and "404: Not Found" not in r.stdout[:40] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="/tmp/pg.json", help="GitHub recursive tree JSON")
    ap.add_argument("--docs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    tree = json.load(open(args.tree))["tree"]
    files = [e["path"] for e in tree if e["type"] == "blob"]

    # components: every source directory that holds C files, named by its path
    dirs: dict[str, list[str]] = {}
    for f in files:
        if f.endswith((".c", ".h")) and f.startswith("src/"):
            dirs.setdefault(f.rsplit("/", 1)[0], []).append(f)
    comps = {d for d, v in dirs.items() if len(v) >= 2}
    subs = [{"name": d, "files": sorted(dirs[d])[:40], "docs": [], "maintainers": []}
            for d in sorted(comps)]
    json.dump(subs, open(OUT / "subsystems.json", "w"), indent=1)
    print(f"source directories with >=2 C files (components): {len(subs)}")

    readmes = [f for f in files
               if f.rsplit("/", 1)[-1] in ("README", "README.md") and f.startswith("src/")
               and f.rsplit("/", 1)[0] in comps]
    print(f"READMEs sitting in such a directory: {len(readmes)}")
    rng = random.Random(args.seed)
    rng.shuffle(readmes)
    docs = []
    for path in readmes:
        if len(docs) >= args.docs:
            break
        text = fetch(path)
        if not text:
            continue
        sents = split_sentences(re.sub(r"\n\n", "\n\n", text))
        if not (15 <= len(sents) <= 60):
            continue
        docs.append({"path": path, "owner": path.rsplit("/", 1)[0], "sentences": sents})
        print(f"  {path}  [{docs[-1]['owner']}]  {len(sents)} sentences")
    json.dump(docs, open(OUT / "dataset.json", "w"), indent=1)
    print(f"docs {len(docs)}, sentences {sum(len(d['sentences']) for d in docs)}")


if __name__ == "__main__":
    main()
