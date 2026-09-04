"""Grounded LLM annotation, sentence view. The annotator sees what the linker never sees:
each candidate component's project-authored self-description, the code symbols the
sentence names and where they are defined, and the surrounding prose. It labels every
sentence (not only anchored ones) with ABOUT / REFERS per component.

usage: annotate.py --backend openai|claude_cli --model <id> [--limit N] [--workers K]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys

from common import OUT, load_components, load_sentences
from llm import call_many, extract_entries, extract_json

BATCH = 5
MAX_CANDIDATES = 14

INSTRUCTIONS = """You are building a traceability dataset for a large software system. The system is split into COMPONENTS (Rust crates). Each component has a self-description written by the project's own developers (crate-level doc comment, module docs, public items, source files). You will see a passage from the system's developer guide and must decide, for each TARGET sentence, which components it is ABOUT.

Labels (per sentence, per component):
- ABOUT: the sentence describes the behaviour, responsibility, data structures, algorithm, or design of that component, or of a process that this component implements. Test: if that component's code changed so that the sentence became false, its maintainers would have to update this sentence.
- REFERS: the sentence only names or uses an item (type, function, module) that the component defines, but what the sentence says is about something else. Example: a sentence about how borrow checking reports an error that mentions a source-span type defined elsewhere -> REFERS to the span component, not ABOUT it.
- Otherwise the component is not listed.

Rules:
1. Prefer the most specific component(s) that implement what the sentence says. A sentence can be ABOUT several components (a hand-off, a shared responsibility). Many sentences are ABOUT none: general background, remarks about the guide itself, statements about the system as a whole, or about external tools.
2. The component that merely DEFINES a type the sentence talks about is REFERS unless the sentence is about that definition itself (e.g. "the type context holds the interners" is ABOUT the crate that defines the type context; "the borrow checker takes a type context" is REFERS).
3. Never assign a component because its name shares a word with the sentence. Decide from the self-descriptions and the defined-symbol evidence, plus your knowledge of the system.
4. You may name a component outside the candidate list; it must be one of the listed system components.
5. Answer only with JSON: {"<sentence number>": {"about": [component ids], "refers": [component ids], "why": "<one short sentence>"}, ...} covering every TARGET sentence number.
"""


def excerpt(crate: str, p: dict) -> str:
    lines = [f"### {crate}"]
    if p["doc"]:
        lines.append("crate doc: " + re.sub(r"\s+", " ", p["doc"])[:600])
    if p["readme"]:
        lines.append("readme: " + re.sub(r"\s+", " ", p["readme"])[:300])
    if p["description"]:
        lines.append("description: " + p["description"])
    for m, d in list(p["module_docs"].items())[:6]:
        lines.append(f"module {m}: {d[:140]}")
    if p["modules"]:
        lines.append("modules: " + ", ".join(p["modules"][:20]))
    if p["items"]:
        lines.append("public items: " + ", ".join(p["items"][:15]))
    if not p["doc"] and not p["module_docs"] and p["files"]:
        lines.append("source files: " + ", ".join(p["files"][:15]))
    if p.get("source_excerpt"):
        lines.append("source excerpt: " + p["source_excerpt"])
    return "\n".join(lines)


def build_prompts(rows, evidence, profiles, crates):
    ev = {e["number"]: e for e in evidence}
    by_chapter = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_chapter[r["chapter"]].append(i)
    prompts, batches = [], []
    for chapter, idxs in by_chapter.items():
        for start in range(0, len(idxs), BATCH):
            batch = idxs[start:start + BATCH]
            before = idxs[max(0, start - 2):start]
            after = idxs[start + BATCH:start + BATCH + 2]
            score = collections.Counter()
            for i in batch:
                e = ev[rows[i]["number"]]
                for rank, c in enumerate(e["bm25_top"][:6]):
                    score[c] += 6 - rank
                for cs in e["symbol_crates"].values():
                    for c in cs:
                        score[c] += 8
                for c in e["verbatim"]:
                    score[c] += 10
            cands = [c for c, _ in score.most_common(MAX_CANDIDATES)]
            parts = [INSTRUCTIONS, f"## Chapter: {chapter}", "## Passage (TARGET sentences are numbered; [ctx] lines are context only)"]
            for i in before:
                parts.append(f"[ctx] {rows[i]['text']}")
            for i in batch:
                parts.append(f"TARGET {rows[i]['number']}: {rows[i]['text']}")
            for i in after:
                parts.append(f"[ctx] {rows[i]['text']}")
            parts.append("## Code symbols named in the TARGET sentences and the components that define them")
            any_sym = False
            for i in batch:
                e = ev[rows[i]["number"]]
                if e["symbol_crates"]:
                    any_sym = True
                    parts.append(f"{rows[i]['number']}: " + "; ".join(f"`{k}` defined in {', '.join(v)}" for k, v in e["symbol_crates"].items()))
            if not any_sym:
                parts.append("(none)")
            parts.append("## Candidate components (retrieved by similarity; may be wrong or incomplete)")
            for c in cands:
                parts.append(excerpt(c, profiles[c]))
            parts.append("## All system components: " + ", ".join(crates))
            parts.append("Return the JSON now.")
            prompts.append("\n".join(parts))
            batches.append([rows[i]["number"] for i in batch])
    return prompts, batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--salt", default="", help="re-run id; same prompt, fresh sample (3-run consistency)")
    args = ap.parse_args()
    rows = load_sentences()
    crates = load_components()
    evidence = json.loads((OUT / "evidence.json").read_text())
    profiles = json.loads((OUT / "profiles.json").read_text())
    prompts, batches = build_prompts(rows, evidence, profiles, crates)
    if args.limit:
        prompts, batches = prompts[:args.limit], batches[:args.limit]
    print(f"batches {len(prompts)}  avg prompt chars {sum(map(len, prompts)) // len(prompts)}", flush=True)
    if args.dry:
        print(prompts[3])
        return
    results = call_many(args.backend, args.model, prompts, workers=args.workers, progress=args.model + args.salt, salt=args.salt)
    tag = f"{args.backend}_{args.model}{args.salt}".replace("/", "_")
    ann, bad, usage = {}, 0, collections.Counter()
    crate_set = set(crates)
    for res, nums, prompt in zip(results, batches, prompts):
        for k, v in res.get("usage", {}).items():
            usage[k] += v
        data = extract_json(res["text"])
        if not isinstance(data, dict):
            data = extract_entries(res["text"])
            bad += 1  # counted as truncated; entries recovered where complete
            if not data:
                continue
        for n in nums:
            item = data.get(str(n)) or data.get(n) or {}
            about = [c for c in item.get("about", []) if c in crate_set] if isinstance(item, dict) else []
            refers = [c for c in item.get("refers", []) if c in crate_set] if isinstance(item, dict) else []
            ann[str(n)] = {"about": about, "refers": refers, "why": (item.get("why", "") if isinstance(item, dict) else "")}
    (OUT / f"annotations_{tag}.json").write_text(json.dumps({"model": args.model, "backend": args.backend, "usage": usage,
                                                            "unparsed_batches": bad, "labels": ann}, indent=0))
    n_about = sum(1 for v in ann.values() if v["about"])
    print(f"labelled {len(ann)} sentences; ABOUT non-empty {n_about}; unparsed batches {bad}; usage {dict(usage)}")


if __name__ == "__main__":
    main()
