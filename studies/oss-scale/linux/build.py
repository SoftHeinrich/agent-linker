#!/usr/bin/env python3
"""Build a doc-sentence -> subsystem check set from the Linux kernel MAINTAINERS file.

MAINTAINERS is a human-maintained mapping: every subsystem entry lists its code paths and,
for half of them, its `Documentation/` files.  That makes each documentation file a
*human-assigned* trace link to a component, on a system nobody has tuned a linker against.
We use it to audit the semantic-gold recipe from ../rustc/semgold on foreign ground.

out/subsystems.json  every entry with code paths: {name, files, docs, maintainers}
out/dataset.json     the sampled documentation files, sentence-split, with their owner
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RAW = "https://raw.githubusercontent.com/torvalds/linux/master/"
TAG = re.compile(r"^([A-Z]):\t(.*)$")


def parse_maintainers(path: Path) -> list[dict]:
    entries, cur = [], None
    for line in path.read_text(errors="ignore").splitlines():
        m = TAG.match(line)
        if m and cur is not None:
            cur.setdefault(m.group(1), []).append(m.group(2).strip())
        elif not line.strip():
            cur = None
        elif cur is None or "title" in cur:
            cur = {"title": line.strip()}
            entries.append(cur)
    out = []
    for e in entries:
        files = e.get("F", [])
        docs = [f for f in files if f.startswith("Documentation/")]
        code = [f for f in files if not f.startswith("Documentation/")]
        if not code:
            continue
        out.append({"name": e["title"], "files": code, "docs": docs,
                    "maintainers": e.get("M", []), "desc": " ".join(e.get("W", []))})
    return out


def split_sentences(txt: str) -> list[str]:
    txt = re.sub(r"::\n\n(?:\s{2,}.*\n|\n)+", " ", txt)          # drop literal blocks
    txt = re.sub(r"^[=~^\-`'\"#*+]{3,}$", "", txt, flags=re.M)   # rst underlines
    txt = re.sub(r"^\s*\.\..*$", "", txt, flags=re.M)            # rst directives/comments
    txt = re.sub(r"\s+", " ", txt)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z`\"'(\[])", txt)
    return [p.strip() for p in parts if 5 <= len(p.strip().split()) <= 80]


def fetch(path: str) -> str | None:
    r = subprocess.run(["curl", "-s", "-m", "30", RAW + path], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 and r.stdout and "404: Not Found" not in r.stdout[:40] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maintainers", default="/tmp/MAINTAINERS")
    ap.add_argument("--docs", type=int, default=12, help="documentation files to sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    subs = parse_maintainers(Path(args.maintainers))
    json.dump(subs, open(OUT / "subsystems.json", "w"), indent=1)
    print(f"subsystems with code paths: {len(subs)}; with docs: {sum(1 for s in subs if s['docs'])}")

    # a documentation file is usable when exactly one subsystem claims it (no glob), so the
    # human assignment is unambiguous
    owner: dict[str, list[str]] = {}
    for s in subs:
        for d in s["docs"]:
            if any(ch in d for ch in "*?[") or not d.endswith((".rst", ".txt", ".md")):
                continue
            owner.setdefault(d, []).append(s["name"])
    single = sorted(d for d, names in owner.items() if len(names) == 1)
    print(f"documentation files claimed by exactly one subsystem: {len(single)}")

    rng = random.Random(args.seed)
    rng.shuffle(single)
    docs = []
    for path in single:
        if len(docs) >= args.docs:
            break
        text = fetch(path)
        if not text:
            continue
        sents = split_sentences(text)
        if not (15 <= len(sents) <= 60):
            continue
        docs.append({"path": path, "owner": owner[path][0], "sentences": sents})
        print(f"  {path}  [{owner[path][0]}]  {len(sents)} sentences")
    json.dump(docs, open(OUT / "dataset.json", "w"), indent=1)
    print(f"docs {len(docs)}, sentences {sum(len(d['sentences']) for d in docs)}")


if __name__ == "__main__":
    main()
