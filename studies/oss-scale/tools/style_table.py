#!/usr/bin/env python3
"""Is a document in the benchmark's style?  One table, measured, for the five benchmark
projects and the oss-scale datasets.

Columns (all from sentences + component names + the gold used for scoring):
  sents, comps                  size
  gold pairs, sent w/ gold      gold density (share of sentences carrying a link)
  verbatim                      share of gold pairs whose sentence contains the component
                                name verbatim (case-insensitive) — the "explicit" part
  name-echo                     not verbatim, but a name word (>3 chars, not the system's
                                own name) occurs in the sentence
  caps                          share of component names written as capitalised / CamelCase
                                proper nouns (first character upper-case)
  snake                         share of names containing '_' or '-' (implementation ids)
  shared-word names             share of names that share a word (>2 chars) with another name
  top shared word               the most-shared word across names and how many names carry it

usage: style_table.py   (paths are fixed; run from anywhere)
"""
from __future__ import annotations

import collections
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmark"
OSS = ROOT / "studies" / "oss-scale"

BENCHMARK = {
    "mediastore": ("mediastore/text_2016/mediastore.txt", "mediastore/model_2016/pcm/ms.repository", "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv"),
    "teastore": ("teastore/text_2020/teastore.txt", "teastore/model_2020/pcm/teastore.repository", "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv"),
    "teammates": ("teammates/text_2021/teammates.txt", "teammates/model_2021/pcm/teammates.repository", "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "bigbluebutton": ("bigbluebutton/text_2021/bigbluebutton.txt", "bigbluebutton/model_2021/pcm/bbb.repository", "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "jabref": ("jabref/text_2021/jabref.txt", "jabref/model_2021/pcm/jabref.repository", "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
}
OSS_SETS = {
    "gitlab (semantic gold)": ("gitlab/data/sentences.txt", "gitlab/data/components.json", "gitlab/out/gold_semantic.csv"),
    "kubernetes (semantic gold)": ("kubernetes/data/sentences.txt", "kubernetes/data/components.json", "kubernetes/out/gold_semantic.csv"),
    "rustc core (semantic gold)": ("rustc/data/core/sentences.txt", "rustc/data/core/components.json", "rustc/semgold/out/gold_semantic.csv"),
}


def pcm_components(path: Path) -> dict[str, str]:
    txt = path.read_text()
    out = {}
    for m in re.finditer(r'<components__Repository[^>]*?xsi:type="repository:(?:Basic|Composite)Component"[^>]*?>', txt):
        tag = m.group(0)
        i = re.search(r'id="([^"]+)"', tag)
        n = re.search(r'entityName="([^"]+)"', tag)
        if i and n:
            out[i.group(1)] = n.group(1)
    return out


def words(name: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", name)) if len(w) > 2]


def row(label: str, sents: list[str], comps: dict[str, str], gold: set[tuple[int, str]], system: str):
    low = [s.lower() for s in sents]
    gold = {(s, c) for s, c in gold if c in comps and 1 <= s <= len(sents)}
    verb = {(s, c) for s, c in gold if comps[c].lower() in low[s - 1]}
    sysw = set(words(system))
    echo = {(s, c) for s, c in gold - verb
            if any(re.search(r"\b" + re.escape(w) + r"\b", low[s - 1]) for w in words(comps[c]) if len(w) > 3 and w not in sysw)}
    names = list(comps.values())
    caps = sum(1 for n in names if n[:1].isupper()) / len(names)
    snake = sum(1 for n in names if "_" in n or "-" in n) / len(names)
    wc = collections.Counter(w for n in names for w in set(words(n)))
    shared = sum(1 for n in names if any(wc[w] > 1 for w in words(n))) / len(names)
    top = wc.most_common(1)[0] if wc else ("", 0)
    return [label, len(sents), len(comps), len(gold), f"{len({s for s, _ in gold}) / len(sents):.2f}",
            f"{len(verb) / len(gold):.2f}" if gold else "-", f"{len(echo) / len(gold):.2f}" if gold else "-",
            f"{caps:.2f}", f"{snake:.2f}", f"{shared:.2f}", f"{top[0]} ({top[1]})"]


def main() -> None:
    rows = []
    for name, (t, m, g) in BENCHMARK.items():
        sents = [l.strip() for l in (BENCH / t).read_text().splitlines() if l.strip()]
        comps = pcm_components(BENCH / m)
        gold = {(int(r["sentence"]), r["modelElementID"]) for r in csv.DictReader(open(BENCH / g))}
        rows.append(row(name, sents, comps, gold, name))
    for label, (t, c, g) in OSS_SETS.items():
        if not (OSS / g).exists():
            continue
        sents = (OSS / t).read_text().splitlines()
        comps = {x["id"]: x["name"] for x in json.load(open(OSS / c))}
        gold = {(int(r["sentence"]), r["modelElementID"]) for r in csv.DictReader(open(OSS / g))}
        rows.append(row(label, sents, comps, gold, label.split()[0]))
    head = ["dataset", "sents", "comps", "gold", "sent w/ gold", "verbatim", "name-echo", "caps", "snake", "shared-word names", "top shared word"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")


if __name__ == "__main__":
    main()
