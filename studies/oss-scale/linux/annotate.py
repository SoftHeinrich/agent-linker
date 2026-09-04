#!/usr/bin/env python3
"""Run the semantic-gold annotation recipe on Linux documentation and check it against the
human assignment in MAINTAINERS.

The annotator never sees which subsystem MAINTAINERS gives the file.  Candidates come from
BM25 over subsystem profiles (name + code paths); the owner is forced into the candidate
list so that the check measures *annotation*, not retrieval — retrieval is reported
separately.

out/annotations_<tag>.json   {"labels": {"<doc>#<i>": {"about": [...], "refers": [...]}}}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rustc" / "semgold"))
from evidence import BM25            # noqa: E402
from llm import call_many, extract_json, extract_entries  # noqa: E402

OUT = HERE / "out"
SPLIT = re.compile(r"[^a-z0-9]+")

INSTRUCTIONS = """You label whether a documentation sentence is ABOUT a component of a large software system.

ABOUT  the sentence states something a maintainer of that component would have to keep
       true: what it does, how it behaves, what it is responsible for.  If the component
       changed, this sentence might need to change.
REFERS the sentence only names or mentions something the component owns, while being about
       something else.

Rules:
1. A sentence can be ABOUT more than one component, or about none.
2. Judge the sentence in the context shown, not in isolation; a sentence continuing the
   previous topic is ABOUT the same component even if it names nothing.
3. Only use components from the candidate list.
4. Generic statements about the system as a whole are about no component.
5. Prefer the most specific component that owns the behaviour described.
"""


def tokens(s: str) -> list[str]:
    return [t for t in SPLIT.split(s.lower()) if t and len(t) > 1]


def profile(sub: dict) -> list[str]:
    toks = tokens(sub["name"]) * 3
    for f in sub["files"][:12]:
        toks += tokens(f)
    return toks


def excerpt(sub: dict) -> str:
    return f"{sub['name']} — code: {', '.join(sub['files'][:6])}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="openai")
    ap.add_argument("--model", default="gpt-5.6-terra")
    ap.add_argument("--candidates", type=int, default=12)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dir", default="", help="audit directory holding out/ (default: this one)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    global OUT
    if args.dir:
        OUT = Path(args.dir).resolve() / "out"

    subs = json.load(open(OUT / "subsystems.json"))
    docs = json.load(open(OUT / "dataset.json"))
    by_name = {s["name"]: s for s in subs}
    bm = BM25({s["name"]: profile(s) for s in subs})

    prompts, keys, retrieval, doc_retrieval = [], [], [], []
    for doc in docs:
        sents = doc["sentences"]
        ranked_doc = sorted(bm.score(tokens(" ".join(sents))).items(), key=lambda kv: -kv[1])
        doc_retrieval.append({"doc": doc["path"], "owner_rank": next(
            (i for i, (n, _) in enumerate(ranked_doc) if n == doc["owner"]), None)})
        for start in range(0, len(sents), args.batch):
            batch = list(range(start, min(start + args.batch, len(sents))))
            q = tokens(" ".join(sents[i] for i in batch))
            ranked = sorted(bm.score(q).items(), key=lambda kv: -kv[1])
            cands = [n for n, _ in ranked[:args.candidates]]
            rank = next((i for i, (n, _) in enumerate(ranked) if n == doc["owner"]), None)
            retrieval.append({"doc": doc["path"], "batch": start, "owner_rank": rank})
            if doc["owner"] not in cands:
                cands = cands[:-1] + [doc["owner"]]
            cand_block = "\n".join(f"- {excerpt(by_name[c])}" for c in sorted(cands))
            ctx = "\n".join(
                f"{'>>' if i in batch else '  '} S{i}: {s}"
                for i, s in enumerate(sents)
                if batch[0] - 2 <= i <= batch[-1] + 2)
            prompts.append(
                f"{INSTRUCTIONS}\nDOCUMENTATION FILE: {doc['path']}\n\n"
                f"CANDIDATE COMPONENTS\n{cand_block}\n\n"
                f"SENTENCES (label the ones marked >>)\n{ctx}\n\n"
                'Return JSON only: {"<sentence number>": {"about": ["Component"], '
                '"refers": ["Component"], "why": "one clause"}}\n')
            keys.append((doc["path"], batch))

    print(f"{len(prompts)} prompts, {sum(len(d['sentences']) for d in docs)} sentences")
    if args.dry:
        print(prompts[0][:2500])
        return
    res = call_many(args.backend, args.model, prompts, workers=args.workers, progress="annotate")
    labels, unparsed = {}, 0
    for (path, batch), r in zip(keys, res):
        data = extract_json(r.get("text", "")) or extract_entries(r.get("text", ""))
        if not data:
            unparsed += 1
            continue
        for k, v in data.items():
            m = re.search(r"\d+", str(k))
            if not m:
                continue
            i = int(m.group())
            if i in batch:
                labels[f"{path}#{i}"] = {"about": v.get("about", []) or [],
                                         "refers": v.get("refers", []) or [],
                                         "why": v.get("why", "")}
    tag = args.tag or args.model
    json.dump({"model": args.model, "backend": args.backend, "unparsed": unparsed,
               "retrieval": retrieval, "doc_retrieval": doc_retrieval,
               "labels": labels},
              open(OUT / f"annotations_{tag}.json", "w"), indent=1)
    print(f"labelled {len(labels)} sentences, unparsed batches {unparsed}")


if __name__ == "__main__":
    main()
