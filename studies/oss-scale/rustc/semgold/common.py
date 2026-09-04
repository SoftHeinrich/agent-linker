"""Shared paths and loaders for the semantic-gold pipeline (rustc core chapters)."""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "core"
OUT = HERE / "out"
CACHE = HERE / "cache"
RUST = Path(os.environ.get("RUST_REPO", "/tmp/oss-case/rustc/rust"))
TREE = Path(os.environ.get("RUST_TREE", "/tmp/oss-case/rustc/tree"))  # git-archive of compiler/
GUIDE_PREFIX = "src/doc/rustc-dev-guide/src/"


def load_sentences() -> list[dict]:
    """[{number, text, chapter, link, verbatim, link_kind, link_text}] in document order."""
    texts = DATA.joinpath("sentences.txt").read_text().splitlines()
    meta = json.loads(DATA.joinpath("meta.json").read_text())
    assert len(texts) == len(meta), (len(texts), len(meta))
    rows = []
    for text, m in zip(texts, meta):
        row = dict(m)
        row["text"] = text
        rows.append(row)
    return rows


def load_components() -> list[str]:
    return [c["id"] for c in json.loads(DATA.joinpath("components.json").read_text())]


def load_anchor_gold() -> set[tuple[int, str]]:
    with open(DATA / "gold.csv") as handle:
        return {(int(r["sentence"]), r["modelElementID"]) for r in csv.DictReader(handle)}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


_NORM = re.compile(r"[^a-z0-9]+")


def norm(text: str) -> str:
    return _NORM.sub(" ", text.lower()).strip()


def tokens(text: str) -> list[str]:
    return norm(text).split()
