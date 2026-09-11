#!/usr/bin/env python3
"""Build the GitLab architecture dataset in the benchmark's own shape.

Source: gitlab-org/gitlab `doc/development/architecture.md` — the project's single
architecture-overview document: a component diagram with connectors, a component list
(name, one-line description), one short section per component, and narrative sections
(request cycles, system layout).  This is the same document genre the five benchmark
texts were taken from (TeaStore wiki "Services", BigBlueButton "Architecture" page,
TEAMMATES "Design"), not a developer guide.

Outputs under data/:
  architecture.md        raw document at the pinned commit
  sentences.txt          one sentence per line (benchmark adaptation: headings/captions
                         dropped, links replaced by their text, code spans unwrapped,
                         tables/diagrams/link-only bullets removed)
  sentence_meta.json     per sentence: heading path, whether it sits in a component's
                         own "Component details" section, source line
  components.json        [{id, name, description, layer, process}] from the doc itself
  gold_structural.csv    modelElementID,sentence — sentences inside a component's own
                         section (doc-inside-the-unit prior; a vote, not the gold)
  meta.json              commit, counts

Stdlib only.  `--offline` reuses data/architecture.md.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PROJECT = "gitlab-org%2Fgitlab"
DOC = "doc%2Fdevelopment%2Farchitecture.md"
COMMIT = "7eb01fc436a2"  # 2026-08-10, last commit touching the file when built (2026-09-04)

META_BULLET = re.compile(r"^\s*-\s*(\[Project page\]|Configuration:|Layer:|Process:|GitLab\.com:|\[Omnibus\]|\[Charts\]|\[Source\]|\[GitLab\.com\])")
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
SHORTCODE = re.compile(r"\{\{<[^>]*>\}\}")


def fetch(ref: str) -> str:
    url = f"https://gitlab.com/api/v4/projects/{PROJECT}/repository/files/{DOC}/raw?ref={ref}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8")


def clean_inline(s: str) -> str:
    s = IMG.sub("", s)
    s = LINK.sub(r"\1", s)
    s = SHORTCODE.sub("", s)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_sentences(par: str) -> list[str]:
    out = []
    par = re.sub(r"\b(vs|e\.g|i\.e|etc)\.\s", lambda m: m.group(1) + "\u2024 ", par)
    for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", par):
        s = s.replace("\u2024", ".").strip()
        if len(s.split()) >= 3 and (s[0].isascii() or s[0].isalnum()):  # drops the table legend lines
            out.append(s)
    return out


def parse(md: str):
    lines = md.splitlines()
    in_fence = False
    in_front = False
    heads: list[str] = []
    paras: list[tuple[str, list[str], int]] = []  # (text, heading path, first line)
    buf: list[str] = []
    buf_line = 0

    def flush():
        nonlocal buf
        if buf:
            paras.append((" ".join(buf), list(heads), buf_line))
        buf = []

    for n, raw in enumerate(lines, 1):
        if n == 1 and raw.strip() == "---":
            in_front = True
            continue
        if in_front:
            if raw.strip() == "---":
                in_front = False
            continue
        if raw.strip().startswith("```"):
            flush()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if raw.startswith("#"):
            flush()
            level = len(raw) - len(raw.lstrip("#"))
            title = clean_inline(raw.lstrip("#").strip())
            heads = heads[: level - 1] + [""] * max(0, level - 1 - len(heads)) + [title]
            continue
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue
        if line.lstrip().startswith("|") or line.strip().startswith("<!--") or line.strip().startswith("{{<"):
            flush()
            continue
        if META_BULLET.match(line):
            flush()
            continue
        if re.match(r"^\s*([-*]|\d+\.)\s", line):
            # a list item starts its own paragraph; indented continuation lines join it
            flush()
            item = clean_inline(re.sub(r"^\s*([-*]|\d+\.)\s+", "", line))
            if item and not re.fullmatch(r"[A-Za-z0-9_./-]+", item):  # drop bare-path bullets
                buf = [item]
                buf_line = n
            continue
        if not buf:
            buf_line = n
        buf.append(clean_inline(line))
    flush()
    return paras


def components_from(md: str) -> list[dict]:
    """Component list = the document's own '#### <name>' sections under 'Component details',
    with the description from the 'Component list' table row that anchors to the section."""
    lines = md.splitlines()
    table_desc: dict[str, str] = {}
    for l in lines:
        m = re.match(r"\|\s*\[([^\]]+)\]\(#([^)]+)\)\s*\|\s*([^|]*)\|", l)
        if m:
            table_desc[m.group(2)] = m.group(3).strip()
    comps: list[dict] = []
    in_details = False
    cur: dict | None = None
    for l in lines:
        if l.startswith("### "):
            in_details = l.strip() == "### Component details"
            cur = None
            continue
        if in_details and l.startswith("#### "):
            name = l[5:].strip()
            anchor = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
            cur = {"id": anchor, "name": name, "description": table_desc.get(anchor, ""),
                   "layer": "", "process": ""}
            comps.append(cur)
            continue
        if cur is not None:
            m = re.match(r"^\s*-\s*Layer:\s*(.+)$", l)
            if m:
                cur["layer"] = m.group(1).strip()
            m = re.match(r"^\s*-\s*Process:\s*(.+)$", l)
            if m:
                cur["process"] = clean_inline(m.group(1))
    return comps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default=COMMIT)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    raw_path = DATA / "architecture.md"
    if args.offline and raw_path.exists():
        md = raw_path.read_text()
    else:
        md = fetch(args.ref)
        raw_path.write_text(md)

    comps = components_from(md)
    by_name = {c["name"]: c for c in comps}
    paras = parse(md)
    sentences: list[str] = []
    meta: list[dict] = []
    for text, heads, line in paras:
        own = None
        if len(heads) >= 4 and heads[2] == "Component details" and heads[3] in by_name:
            own = by_name[heads[3]]["id"]
        for s in split_sentences(text):
            sentences.append(s)
            meta.append({"heading": " > ".join(h for h in heads if h), "own_component": own, "line": line})

    (DATA / "sentences.txt").write_text("\n".join(sentences) + "\n")
    json.dump(meta, open(DATA / "sentence_meta.json", "w"), indent=1)
    json.dump(comps, open(DATA / "components.json", "w"), indent=1)
    with open(DATA / "gold_structural.csv", "w") as f:
        f.write("modelElementID,sentence\n")
        n_struct = 0
        for i, m in enumerate(meta, 1):
            if m["own_component"]:
                f.write(f"{m['own_component']},{i}\n")
                n_struct += 1
    # no-transarc placeholder the runner expects
    (DATA / "no-transarc.csv").write_text("modelElementID,sentence\n")
    info = {"source": "gitlab-org/gitlab doc/development/architecture.md", "commit": args.ref,
            "sentences": len(sentences), "components": len(comps), "structural_pairs": n_struct,
            "sections": sorted({m["heading"].split(" > ")[0] for m in meta})}
    json.dump(info, open(DATA / "meta.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
