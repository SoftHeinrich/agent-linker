#!/usr/bin/env python3
"""Grounded two-family annotation of the GitLab architecture document (the semantic-gold
recipe of ../rustc/semgold, without retrieval: 37 components fit in every prompt).

The annotator sees what the linker never sees: each component's own description from the
document's component table, its layer and process name, and the text of its own section.
It labels every sentence ABOUT / REFERS per component, five sentences at a time with ±2
sentences of context and the heading path.

  annotate.py --view sentence  --backend openai     --model gpt-5.6-terra [--salt r2]
  annotate.py --view sentence  --backend claude_cli --model sonnet
  annotate.py --view component --backend openai     --model gpt-5.6-terra
      (component first: one prompt per component over the whole document)

Writes out/annotations_<view>_<tag>.json: {"labels": {"<sentence idx 0-based>":
{"about": [ids], "refers": [ids], "why": str}}}.  LLM responses are cached under
../rustc/semgold/cache (keyed by model+salt+prompt), so re-runs are free.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import os as _os
DIR = Path(_os.environ.get("OSS_DIR", HERE)).resolve()  # dataset dir holding data/ and out/
DATA = DIR / "data"
OUT = DIR / "out"
sys.path.insert(0, str(HERE.parent / "rustc" / "semgold"))
from llm import call_many, extract_json, extract_entries  # noqa: E402

INSTRUCTIONS = """You label whether a sentence of a software architecture document is ABOUT a component of the system.

ABOUT  the sentence states something a maintainer of that component would have to keep
       true: what it does, how it behaves, what it talks to, what it is responsible for.
       If the component changed, this sentence might need to change.
REFERS the sentence only names or mentions the component (or something it owns) while
       being about something else — e.g. a log path, a port number in a list, an example.

Rules:
1. A sentence can be ABOUT more than one component, or about none.
2. Judge the sentence in the context shown, not in isolation; a sentence continuing the
   previous topic is ABOUT the same component even if it names nothing.
3. Only use component ids from the list.
4. Generic statements about the system as a whole, about installation methods, or about
   the document itself are about no component.
5. Prefer the most specific component that owns the behaviour described.
"""


def notes() -> str:
    """Dataset-specific reading notes for the annotator (project-authored facts such as
    which process an application runs in), kept in data/annotator_notes.txt."""
    p = DATA / "annotator_notes.txt"
    return ("\nNotes on this system:\n" + p.read_text().strip() + "\n") if p.exists() else ""


def load():
    sents = DATA.joinpath("sentences.txt").read_text().splitlines()
    meta = json.load(open(DATA / "sentence_meta.json"))
    comps = json.load(open(DATA / "components.json"))
    assert len(sents) == len(meta)
    own_text: dict[str, list[str]] = {}
    for s, m in zip(sents, meta):
        if m["own_component"]:
            own_text.setdefault(m["own_component"], []).append(s)
    return sents, meta, comps, own_text


def component_block(comps, own_text) -> str:
    lines = []
    for c in comps:
        desc = c["description"] or "(no description)"
        extra = []
        if c["layer"]:
            extra.append(f"layer: {c['layer']}")
        if c["process"]:
            extra.append(f"process: {c['process']}")
        own = " ".join(own_text.get(c["id"], []))[:400]
        lines.append(f"- {c['id']}  ({c['name']}): {desc}" + (f" [{'; '.join(extra)}]" if extra else "")
                     + (f"\n    own section: {own}" if own else ""))
    return "\n".join(lines)


def sentence_prompts(sents, meta, comps, own_text, batch: int):
    block = component_block(comps, own_text)
    prompts, keys = [], []
    for start in range(0, len(sents), batch):
        idx = list(range(start, min(start + batch, len(sents))))
        ctx = []
        for i in range(max(0, idx[0] - 2), min(len(sents), idx[-1] + 3)):
            mark = ">>" if i in idx else "  "
            ctx.append(f"{mark} S{i} [{meta[i]['heading']}]: {sents[i]}")
        prompts.append(
            f"{INSTRUCTIONS}{notes()}\nCOMPONENTS (id (name): description)\n{block}\n\n"
            f"SENTENCES (label the ones marked >>; the bracket is the heading path)\n" + "\n".join(ctx) + "\n\n"
            'Return JSON only: {"<sentence number>": {"about": ["component-id"], '
            '"refers": ["component-id"], "why": "one clause"}}\n')
        keys.append(idx)
    return prompts, keys


def component_prompts(sents, meta, comps, own_text):
    doc = "\n".join(f"S{i} [{meta[i]['heading']}]: {s}" for i, s in enumerate(sents))
    prompts, keys = [], []
    for c in comps:
        desc = c["description"] or "(no description)"
        own = " ".join(own_text.get(c["id"], []))
        prompts.append(
            f"{INSTRUCTIONS}{notes()}\nTHE COMPONENT: {c['id']} ({c['name']}): {desc}"
            + (f"\nLayer: {c['layer']}" if c["layer"] else "") + (f"\nProcess: {c['process']}" if c["process"] else "")
            + (f"\nIts own section of the document: {own}" if own else "")
            + f"\n\nTHE DOCUMENT ({len(sents)} sentences)\n{doc}\n\n"
            f"List every sentence that is ABOUT this component and every sentence that only REFERS to it.\n"
            'Return JSON only: {"about": [sentence numbers], "refers": [sentence numbers]}\n')
        keys.append(c["id"])
    return prompts, keys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--view", choices=["sentence", "component"], default="sentence")
    ap.add_argument("--backend", default="openai")
    ap.add_argument("--model", default="gpt-5.6-terra")
    ap.add_argument("--salt", default="")
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    sents, meta, comps, own_text = load()
    ids = {c["id"] for c in comps}
    if args.view == "sentence":
        prompts, keys = sentence_prompts(sents, meta, comps, own_text, args.batch)
    else:
        prompts, keys = component_prompts(sents, meta, comps, own_text)
    print(f"{len(prompts)} prompts, {len(sents)} sentences, {len(comps)} components")
    if args.dry:
        print(prompts[0][:4000])
        return
    res = call_many(args.backend, args.model, prompts, workers=args.workers, progress="annotate", salt=args.salt)
    labels: dict[str, dict] = {}
    unparsed = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for key, r in zip(keys, res):
        for k in usage:
            usage[k] += (r.get("usage") or {}).get(k, 0)
        data = extract_json(r.get("text", ""))
        if args.view == "sentence":
            data = data or extract_entries(r.get("text", ""))
            if not isinstance(data, dict):
                unparsed += 1
                continue
            for k, v in data.items():
                m = re.search(r"\d+", str(k))
                if not m or not isinstance(v, dict):
                    continue
                i = int(m.group())
                if i in key:
                    labels[str(i)] = {"about": [c for c in (v.get("about") or []) if c in ids],
                                      "refers": [c for c in (v.get("refers") or []) if c in ids],
                                      "why": str(v.get("why", ""))[:200]}
        else:
            if not isinstance(data, dict):
                unparsed += 1
                continue
            for field in ("about", "refers"):
                for n in data.get(field) or []:
                    m = re.search(r"\d+", str(n))
                    if not m:
                        continue
                    i = int(m.group())
                    if 0 <= i < len(sents):
                        labels.setdefault(str(i), {"about": [], "refers": [], "why": ""})[field].append(key)
    tag = args.tag or (args.model + (f"_{args.salt}" if args.salt else ""))
    path = OUT / f"annotations_{args.view}_{tag}.json"
    json.dump({"view": args.view, "model": args.model, "backend": args.backend, "salt": args.salt,
               "unparsed": unparsed, "usage": usage, "labels": labels}, open(path, "w"), indent=1)
    n_about = sum(len(v["about"]) for v in labels.values())
    print(f"wrote {path}: {len(labels)} sentences labelled, {n_about} ABOUT pairs, unparsed {unparsed}, usage {usage}")


if __name__ == "__main__":
    main()
