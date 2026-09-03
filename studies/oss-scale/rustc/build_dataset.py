#!/usr/bin/env python3
"""Build the rustc-dev-guide -> compiler crates dataset for the s110 linker.

Inputs: a checkout of rust-lang/rustc-dev-guide (src/), crates.txt (the
`compiler/` listing of rust-lang/rust).  Outputs, under --out:
  sentences.txt      one sentence per line, link markup removed (what a reader sees)
  components.json    [{"id": crate, "name": crate}] for every compiler crate
  gold.csv           modelElementID,sentence -- crate anchors per sentence
  meta.json          per sentence: number, chapter, anchors by kind
  datasets.json / <name>.repository via tools/make_dataset.py

Gold anchors, two kinds, both project-authored:
  link     the sentence carries a markdown link (inline or reference-style) whose
           target is doc.rust-lang.org/nightly/nightly-rustc/<crate>/...; 97% are
           reference-style, so the crate is NOT visible in the sentence text
  verbatim the sentence names a crate literally (`rustc_borrowck`)
Stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

NR = re.compile(r"doc\.rust-lang\.org/nightly/nightly-rustc/(rustc_[a-z_]+|rustc)\b")
INLINE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
REF = re.compile(r"\[([^\]]+)\](?:\[([^\]]*)\])?")
CRATE_TOKEN = re.compile(r"\brustc_[a-z_]+\b")

CORE_CHAPTERS = [
    "overview.md", "compiler-src.md", "query.md", "memory.md", "serialization.md",
    "parallel-rustc.md", "the-parser.md", "macro-expansion.md", "name-resolution.md",
    "attributes.md", "hir.md", "hir/lowering.md", "thir.md", "mir/index.md",
    "mir/construction.md", "mir/passes.md", "mir/dataflow.md", "mir/optimizations.md",
    "rustc-driver/intro.md", "diagnostics.md", "ty.md", "type-inference.md",
    "traits/resolution.md", "solve/trait-solving.md", "hir-typeck/summary.md",
    "coherence.md", "borrow-check.md", "const-eval.md", "backend/monomorph.md",
    "backend/lowering-mir.md", "backend/codegen.md", "backend/libs-and-metadata.md",
]

PARTS = {
    "# High-level compiler architecture", "# Source code representation",
    "# Supporting infrastructure", "# Analysis", "# MIR to binaries",
}


def summary_chapters(src: Path) -> list[str]:
    part = None
    out = []
    for line in (src / "SUMMARY.md").read_text().splitlines():
        if line.startswith("# "):
            part = line.strip() if line.strip() in PARTS else None
            continue
        m = re.search(r"\]\(\.?/?([^)]+\.md)\)", line)
        if part and m and not m.group(1).startswith("appendix/"):
            out.append(m.group(1))
    return out


def body_lines(raw: str) -> str:
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    raw = re.sub(r"```.*?```", "", raw, flags=re.S)
    raw = re.sub(r"^\s{4,}\S.*$", "", raw, flags=re.M)
    out = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("|") or s.startswith("<"):
            continue
        if re.match(r"^\[[^\]]+\]:\s*\S+", s):
            continue
        s = re.sub(r"^[-*+]\s+|^\d+\.\s+", "", s)
        out.append(s)
    return re.sub(r"\s+", " ", " ".join(out))


def split_sentences(txt: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z`\"'(\[])", txt)
    return [p.strip() for p in parts if len(p.strip().split()) >= 4]


def anchors(sentence: str, defs: dict[str, str], crates: set[str]) -> tuple[set[str], set[str], dict[str, list[str]]]:
    link, verbatim, texts = set(), set(), {}
    for m in INLINE.finditer(sentence):
        hit = NR.search(m.group(2))
        if hit and hit.group(1) in crates:
            link.add(hit.group(1))
            texts.setdefault(hit.group(1), []).append(m.group(1))
    stripped = INLINE.sub(r"\1", sentence)
    for m in REF.finditer(stripped):
        key = (m.group(2) or m.group(1)).strip().lower()
        hit = NR.search(defs.get(key, ""))
        if hit and hit.group(1) in crates:
            link.add(hit.group(1))
            texts.setdefault(hit.group(1), []).append(m.group(1))
    for tok in CRATE_TOKEN.findall(sentence):
        if tok in crates:
            verbatim.add(tok)
    return link, verbatim, texts


ITEM = re.compile(r"^(`[^`]+`|[A-Z][A-Za-z0-9_]*(::[A-Za-z0-9_]+)*)$")


def link_kind(texts: list[str]) -> str:
    """concept if any link text is a prose expression ("the borrow checker"),
    item if every link text is a code identifier (`TyKind`, `ty::Ty`, `foo()`)."""
    return "item" if all(ITEM.match(t.strip()) for t in texts) else "concept"


def visible(sentence: str) -> str:
    s = INLINE.sub(r"\1", sentence)
    s = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--guide-src", required=True, type=Path)
    ap.add_argument("--crates", default=Path(__file__).with_name("crates.txt"), type=Path)
    ap.add_argument("--chapters", choices=["core", "all"], default="core")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    crates = {line.strip().split("/")[-1] for line in args.crates.read_text().splitlines() if line.strip()}
    chapters = summary_chapters(args.guide_src)
    if args.chapters == "core":
        chapters = [c for c in chapters if c in CORE_CHAPTERS]
        missing = set(CORE_CHAPTERS) - set(chapters)
        if missing:
            raise SystemExit(f"core chapters not in SUMMARY.md: {sorted(missing)}")
    args.out.mkdir(parents=True, exist_ok=True)
    lines, meta, gold = [], [], []
    n = 0
    for ch in chapters:
        raw = (args.guide_src / ch).read_text()
        defs = {k.strip().lower(): v for k, v in re.findall(r"^\[([^\]]+)\]:\s*(\S+)", raw, flags=re.M)}
        for sent in split_sentences(body_lines(raw)):
            link, verb, texts = anchors(sent, defs, crates)
            n += 1
            lines.append(visible(sent))
            meta.append({"number": n, "chapter": ch, "link": sorted(link), "verbatim": sorted(verb),
                         "link_kind": {c: link_kind(t) for c, t in texts.items()},
                         "link_text": texts})
            for crate in link | verb:
                gold.append((crate, n))
    (args.out / "sentences.txt").write_text("\n".join(lines) + "\n")
    (args.out / "meta.json").write_text(json.dumps(meta, indent=0) + "\n")
    (args.out / "components.json").write_text(json.dumps([{"id": c, "name": c} for c in sorted(crates)], indent=0) + "\n")
    with open(args.out / "gold.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["modelElementID", "sentence"])
        w.writerows(gold)
    anchored = sum(1 for m in meta if m["link"] or m["verbatim"])
    link_only = sum(1 for m in meta if m["link"] and not m["verbatim"])
    print(f"chapters {len(chapters)} sentences {n} anchored {anchored} (link-only {link_only}) gold pairs {len(gold)} crates {len(crates)}")


if __name__ == "__main__":
    main()
