"""Per-sentence deterministic evidence: code identifiers in the sentence resolved through
the public-symbol index, and BM25 retrieval of candidate components over their
self-descriptions. No LLM. Output feeds the annotator and the label model."""
from __future__ import annotations

import collections
import json
import math
import re

from common import OUT, load_components, load_sentences, tokens

BACKTICK = re.compile(r"`([^`]+)`")
CAMEL = re.compile(r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b")
PATH = re.compile(r"\b([a-z_]+::[A-Za-z_][A-Za-z0-9_:]*)\b")
SNAKE_FN = re.compile(r"\b([a-z_][a-z0-9_]*_[a-z0-9_]+)\b")
STOP = set("the a an of to in and or is are be for on with as by that this it its from at into which".split())


def identifiers(text: str) -> list[str]:
    found = []
    for span in BACKTICK.findall(text):
        span = span.strip().rstrip("()").rstrip("!")
        last = span.split("::")[-1]
        last = re.sub(r"<.*", "", last).strip(".")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", last):
            found.append(last)
    plain = BACKTICK.sub(" ", text)
    found += CAMEL.findall(plain)
    found += [p.split("::")[-1] for p in PATH.findall(plain)]
    seen, out = set(), []
    for f in found:
        if f not in seen and len(f) > 2:
            seen.add(f)
            out.append(f)
    return out


def split_ident(name: str) -> list[str]:
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ").lower().split()
    return parts


class BM25:
    def __init__(self, docs: dict[str, list[str]], k1: float = 1.2, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.avg = sum(len(d) for d in docs.values()) / max(1, len(docs))
        self.tf = {k: collections.Counter(v) for k, v in docs.items()}
        df = collections.Counter()
        for c in self.tf.values():
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: list[str]) -> dict[str, float]:
        out = {}
        for k, tf in self.tf.items():
            dl = len(self.docs[k])
            s = 0.0
            for t in query:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            out[k] = s
        return out


def profile_doc(crate: str, p: dict) -> list[str]:
    parts = [" ".join(split_ident(crate.removeprefix("rustc_")))] * 3  # the name itself, weighted
    parts += [p["doc"], p["readme"], p["description"], " ".join(p["module_docs"].values())]
    parts += [" ".join(split_ident(m)) for m in p["modules"]]
    parts += [" ".join(split_ident(i)) for i in p["items"][:40]]
    parts += [" ".join(split_ident(f.split("/")[-1])) for f in p["files"]]
    return [t for t in tokens(" ".join(parts)) if t not in STOP and len(t) > 1]


def main(top_k: int = 8) -> None:
    crates = load_components()
    profiles = json.loads((OUT / "profiles.json").read_text())
    symbols = json.loads((OUT / "symbols.json").read_text())
    bm25 = BM25({c: profile_doc(c, profiles[c]) for c in crates})
    rows = load_sentences()
    out = []
    n_ident = n_resolved = 0
    for i, r in enumerate(rows):
        idents = identifiers(r["text"])
        resolved = {}
        for ident in idents:
            if ident in crates:
                continue  # verbatim component names are their own evidence class
            hit = symbols.get(ident)
            if hit and len(hit["crates"]) <= 3:
                resolved[ident] = list(hit["crates"].keys())
        # query = sentence + light context (neighbours in same chapter)
        ctx = [rows[j]["text"] for j in (i - 1, i + 1) if 0 <= j < len(rows) and rows[j]["chapter"] == r["chapter"]]
        q = [t for t in tokens(r["text"]) if t not in STOP] * 2 + [t for t in tokens(" ".join(ctx)) if t not in STOP]
        q += [w for ident in idents for w in split_ident(ident)]
        scores = bm25.score(q)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top = [c for c, s in ranked[:top_k] if s > 0]
        verbatim = [c for c in crates if re.search(rf"\b{re.escape(c)}\b", r["text"])]
        n_ident += bool(idents)
        n_resolved += bool(resolved)
        out.append({"number": r["number"], "chapter": r["chapter"], "identifiers": idents, "symbol_crates": resolved,
                    "bm25_top": top, "verbatim": verbatim})
    (OUT / "evidence.json").write_text(json.dumps(out, indent=0))
    print(f"sentences {len(out)}  with identifiers {n_ident}  with symbol-resolved identifiers {n_resolved}")
    # how often does BM25 top-k contain the anchor gold crate? (retrieval recall for candidate lists)
    from common import load_anchor_gold
    gold = load_anchor_gold()
    by_s = collections.defaultdict(set)
    for s, c in gold:
        by_s[s].add(c)
    hit = sum(1 for e in out if e["number"] in by_s and by_s[e["number"]] & set(e["bm25_top"]))
    hit_sym = sum(1 for e in out if e["number"] in by_s and by_s[e["number"]] & {c for v in e["symbol_crates"].values() for c in v})
    print(f"anchored sentences {len(by_s)}  anchor crate in bm25 top-{top_k}: {hit}  via symbol index: {hit_sym}  "
          f"either: {sum(1 for e in out if e['number'] in by_s and by_s[e['number']] & (set(e['bm25_top']) | {c for v in e['symbol_crates'].values() for c in v} | set(e['verbatim'])))}")


if __name__ == "__main__":
    main()
