"""Second, independent view: per (chapter, component), given the component's
self-description and the whole chapter, list the sentences that are ABOUT it. Different
failure mode from the sentence view (recall-oriented, sees the component first). Used as
a third vote in the label model.
usage: annotate_crateview.py --backend openai --model gpt-5.6-terra
"""
from __future__ import annotations

import argparse
import collections
import json
import re

from annotate import excerpt
from common import OUT, load_components, load_sentences
from evidence import BM25, STOP, profile_doc
from common import tokens
from llm import call_many, extract_json

INSTR = """You are building a traceability dataset for a large software system split into COMPONENTS (Rust crates). Below is ONE component's self-description written by the project's developers, followed by one chapter of the system's developer guide, sentence by sentence.

List every sentence that is ABOUT this component: it describes the component's behaviour, responsibility, data structures, algorithm, or design, or a process this component implements. Test: if this component's code changed so that the sentence became false, the component's maintainers would need to update that sentence. Do NOT list sentences that merely use a type or function the component defines while talking about something else, and do not list sentences because they share a word with the component's name. Sentences about the system as a whole or about other components are not ABOUT this one.

Answer only with JSON: {"about": [sentence numbers], "why": "<one short sentence>"}.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--per-chapter", type=int, default=10)
    args = ap.parse_args()
    rows = load_sentences()
    crates = load_components()
    profiles = json.loads((OUT / "profiles.json").read_text())
    by_chapter = collections.defaultdict(list)
    for r in rows:
        by_chapter[r["chapter"]].append(r)
    # candidates per chapter: crates any sentence-view annotator said ABOUT, plus BM25 top-5 over the chapter text
    sent_about = collections.defaultdict(collections.Counter)
    for p in OUT.glob("annotations_*.json"):
        if "crateview" in p.name:
            continue
        for n, v in json.loads(p.read_text())["labels"].items():
            ch = rows[int(n) - 1]["chapter"]
            for c in v["about"]:
                sent_about[ch][c] += 1
    bm25 = BM25({c: profile_doc(c, profiles[c]) for c in crates})
    prompts, keys = [], []
    for ch, sents in by_chapter.items():
        q = [t for t in tokens(" ".join(s["text"] for s in sents)) if t not in STOP]
        top = [c for c, s in sorted(bm25.score(q).items(), key=lambda kv: -kv[1])[:5] if s > 0]
        cands = [c for c, _ in sent_about[ch].most_common(args.per_chapter)]
        for c in top:
            if c not in cands:
                cands.append(c)
        body = "\n".join(f"{s['number']}: {s['text']}" for s in sents)
        for c in cands:
            prompts.append(f"{INSTR}\n## Component\n{excerpt(c, profiles[c])}\n\n## Chapter: {ch}\n{body}\n\nReturn the JSON now.")
            keys.append((ch, c))
    print(f"(chapter, crate) prompts {len(prompts)}  avg chars {sum(map(len, prompts)) // len(prompts)}", flush=True)
    results = call_many(args.backend, args.model, prompts, workers=args.workers, progress="crateview")
    valid = {s["number"] for s in rows}
    labels = collections.defaultdict(list)
    usage = collections.Counter()
    bad = 0
    for res, (ch, c) in zip(results, keys):
        for k, v in res.get("usage", {}).items():
            usage[k] += v
        d = extract_json(res["text"])
        if not isinstance(d, dict):
            bad += 1
            continue
        for n in d.get("about", []):
            try:
                n = int(n)
            except (TypeError, ValueError):
                continue
            if n in valid and rows[n - 1]["chapter"] == ch:
                labels[str(n)].append(c)
    tag = f"{args.backend}_{args.model}".replace("/", "_")
    (OUT / f"annotations_crateview_{tag}.json").write_text(json.dumps({"model": args.model, "backend": args.backend, "usage": usage,
        "unparsed": bad, "labels": {n: {"about": cs} for n, cs in labels.items()}}, indent=0))
    print(f"sentences with ABOUT {len(labels)}  pairs {sum(len(v) for v in labels.values())}  unparsed {bad}  usage {dict(usage)}")


if __name__ == "__main__":
    main()
