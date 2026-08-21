"""Where the judging prompt's bytes go, rebuilt offline from recorded runs.

No LLM calls. The validation family is the largest of the module (228 kB per
five-project run on terra, 289 kB on luna) and its bytes are not rules: a judging
call is 25 cases, and each case carries its sentence, the sentence before it, an
evidence line and up to `ANCHOR_LIMIT` whole sentences that name the component.
This script rebuilds those prompts from the checkpoints' own candidates and
reports the split, so a compaction arm is aimed at the part that is actually big
and the rules are priced against it honestly.

The one structural question it answers: within a single judging call, how many
anchor-sentence bytes are *repeats* -- the same sentence shown again because two
cases in the batch concern the same component. A repeat carries no information the
call does not already hold, so its bytes are the only ones a lossless compaction
can remove.

Usage (from approach/):

    ../.venv/bin/python pilot/judge_prompt_bytes.py \
        --variant s_linker87 "dedup_e2e_terra_r*_20260821"
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import pickle
import sys

BASE = "/mnt/hostshare/ardoco-home/alinker-replication-package"
sys.path.insert(0, BASE + "/approach/src")

from llm_sad_sam.core.document_loader_v2 import (                      # noqa: E402
    build_sent_map, load_sentences,
)
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository             # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker87 as L87         # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker87 import SLinker87      # noqa: E402

PROJECTS = {
    "mediastore": ("mediastore/text_2016/mediastore.txt",
                   "mediastore/model_2016/pcm/ms.repository"),
    "teastore": ("teastore/text_2020/teastore.txt",
                 "teastore/model_2020/pcm/teastore.repository"),
    "teammates": ("teammates/text_2021/teammates.txt",
                  "teammates/model_2021/pcm/teammates.repository"),
    "bigbluebutton": ("bigbluebutton/text_2021/bigbluebutton.txt",
                      "bigbluebutton/model_2021/pcm/bbb.repository"),
    "jabref": ("jabref/text_2021/jabref.txt",
               "jabref/model_2021/pcm/jabref.repository"),
}


def state(run, variant, proj, phase):
    fn = os.path.join(run, "phase_states", variant, "openai", proj, f"{phase}.pkl")
    return pickle.load(open(fn, "rb")) if os.path.exists(fn) else None


def main(globs, variant):
    runs = sorted(d for g in globs for d in glob.glob(os.path.join(BASE, "results", g))
                  if os.path.isdir(d))
    if not runs:
        sys.exit(f"no runs matched {globs}")
    n_runs = len(runs)
    b = collections.Counter()
    calls = 0
    cases_total = 0

    for run in runs:
        for proj, (text, model_path) in PROJECTS.items():
            full = state(run, variant, proj, "linker_full_name")
            if not full:
                continue
            comps = parse_pcm_repository(os.path.join(BASE, "benchmark", model_path))
            by_name = {c.name: c for c in comps}
            sents = load_sentences(os.path.join(BASE, "benchmark", text))
            sent_map = build_sent_map(sents)
            lk = SLinker87.__new__(SLinker87)
            kstate = state(run, variant, proj, "knowledge")
            lk.doc_knowledge = kstate["doc_knowledge"] if kstate else None

            cands = full["feedback"]["candidates"]
            batch_size = SLinker87.JUDGE_BATCH
            for start in range(0, len(cands), batch_size):
                batch = cands[start:start + batch_size]
                calls += 1
                cases_total += len(batch)
                seen_anchor = set()
                for cand in batch:
                    comp = by_name.get(cand["component"])
                    if comp is None:
                        continue
                    snum = cand["sentence"]
                    sent = sent_map.get(snum)
                    if sent is None:
                        continue
                    prev = sent_map.get(snum - 1)
                    b["case sentence"] += len(sent.text)
                    b["preceding sentence"] += len(prev.text) if prev else 0
                    b["case header"] += len(
                        f'Case 1: "{cand.get("text", "")}" -> {comp.name}\n')
                    label = lk._retained_mention_label(comp.name, sent.text)
                    b["evidence line"] += len(
                        f'  Evidence: source=full_name, span="{cand.get("text","")}"'
                        + (f", mention={label}" if label else "") + "\n")
                    anchors = []
                    for s in sorted(sent_map.values(), key=lambda x: x.number):
                        if s.number == snum:
                            continue
                        if lk._find_exact_form(s.text, comp.name):
                            anchors.append(s)
                            if len(anchors) >= SLinker87.ANCHOR_LIMIT:
                                break
                    for a in anchors:
                        size = len(f"    S{a.number}: {a.text}\n")
                        key = (comp.name, a.number)
                        if key in seen_anchor:
                            b["anchor sentences (repeat in same call)"] += size
                        else:
                            seen_anchor.add(key)
                            b["anchor sentences (first in call)"] += size

    # The rule text every judging call carries, once per call.
    rules = (len(L87.LAYERED_ENTITY_RULES) + len(L87.QUALIFIED_CLAUSE)
             + len(L87.STRICTER_CLAUSE))
    b["rule constants (lenient side)"] = rules * calls
    b["claim-first instruction"] = 196 * calls

    total = sum(b.values())
    print(f"runs read: {n_runs} ({variant});  full-name judging calls "
          f"{calls / n_runs:.1f} per five-project run, "
          f"{cases_total / max(1, calls):.1f} cases per call\n")
    print(f"{'part of the prompt':>42} {'bytes/run':>10} {'share':>7} "
          f"{'bytes/call':>11}")
    for part, size in b.most_common():
        print(f"{part:>42} {size / n_runs:>10.0f} "
              f"{100 * size / max(1, total):>6.1f}% {size / max(1, calls):>11.0f}")
    print(f"{'TOTAL (rebuilt)':>42} {total / n_runs:>10.0f}")
    rep = b["anchor sentences (repeat in same call)"]
    print(f"\nlossless headroom: {rep / n_runs:.0f} B per five-project run are anchor "
          f"sentences repeated inside the call that already shows them "
          f"({100 * rep / max(1, total):.1f}% of the judging prompt).")
    resolver_bytes()
    other_prompts(runs, variant)


def other_prompts(runs, variant):
    """The three smaller families, so the inventory covers every prompt the module
    sends: extraction, denotation and the two alias calls. Rebuilt the same way --
    the batching is deterministic and the recorded checkpoints supply the rest."""
    fam = collections.Counter()
    calls = collections.Counter()
    n_runs = max(1, len(runs))
    for run in runs:
        for proj, (text, model_path) in PROJECTS.items():
            comps = parse_pcm_repository(os.path.join(BASE, "benchmark", model_path))
            sents = load_sentences(os.path.join(BASE, "benchmark", text))
            sent_map = build_sent_map(sents)
            kstate = state(run, variant, proj, "knowledge")
            knowledge = kstate["doc_knowledge"] if kstate else None
            names = [c.name for c in comps]
            aliases = sorted(knowledge.aliases) if knowledge else []
            lk = SLinker87.__new__(SLinker87)
            lk.doc_knowledge = knowledge

            # extraction: one call per EXTRACTION_BATCH of sentences
            size = SLinker87.EXTRACTION_BATCH
            for start in range(0, len(sents), size):
                batch = sents[start:start + size]
                prompt = SLinker87._prompt_extraction(names, aliases, batch)
                calls["extraction"] += 1
                fam["extraction: rules (ENTITY_EXTRACTION_RULES + QUALIFIED_CLAUSE)"] \
                    += len(L87.ENTITY_EXTRACTION_RULES) + len(L87.QUALIFIED_CLAUSE)
                fam["extraction: the document batch"] += sum(
                    len(f"S{s.number}: {s.text}\n") for s in batch)
                fam["extraction: header, COMPONENTS, ALIASES, JSON"] += (
                    len(prompt) - len(L87.ENTITY_EXTRACTION_RULES)
                    - len(L87.QUALIFIED_CLAUSE)
                    - sum(len(f"S{s.number}: {s.text}\n") for s in batch))

            # alias proposal and alias judge: one call each per project
            prompt = SLinker87._prompt_doc_knowledge_extract(
                names, [s.text for s in sents])
            calls["alias proposal"] += 1
            fam["alias proposal: rules (extraction + exclusion)"] += (
                len(L87.DOC_KNOWLEDGE_EXTRACTION_RULES)
                + len(L87.ALIAS_EXCLUSION_RULES))
            fam["alias proposal: the whole document"] += sum(
                len(s.text) + 1 for s in sents)
            fam["alias proposal: header, COMPONENTS, JSON"] += (
                len(prompt) - len(L87.DOC_KNOWLEDGE_EXTRACTION_RULES)
                - len(L87.ALIAS_EXCLUSION_RULES) - sum(len(s.text) + 1 for s in sents))
            calls["alias judge"] += 1
            fam["alias judge: rules"] += len(L87.DOC_KNOWLEDGE_JUDGE_RULES)
            fam["alias judge: the proposed mappings + header + JSON"] += max(
                0, 400 + 40 * len(aliases))

            # denotation: one call per JUDGE_BATCH of scanned candidates
            cands = lk._scan(sents, comps)
            for start in range(0, len(cands), SLinker87.JUDGE_BATCH):
                batch = cands[start:start + SLinker87.JUDGE_BATCH]
                calls["denotation"] += 1
                ids = {s.number for c in batch
                       for s in lk._window(c.sentence_number, sents)}
                fam["denotation: the sentence table"] += sum(
                    len(f'{{"sentence": {n}, "text": "{sent_map[n].text}"}}, ')
                    for n in ids if n in sent_map)
                fam["denotation: the cases"] += sum(
                    len(f'{{"case": 1, "source": {c.sentence_number}, '
                        f'"expression": "{c.matched_text}"}}, ') for c in batch)
                fam["denotation: QUALIFIED_CLAUSE"] += len(L87.QUALIFIED_CLAUSE)
                fam["denotation: question, quote contract, JSON"] += 330

    print(f"\n=== the other three families, rebuilt ===")
    print(f"{'part of the prompt':>56} {'bytes/run':>10} {'calls/run':>10}")
    for part, size in fam.most_common():
        family = part.split(":")[0]
        print(f"{part:>56} {size / n_runs:>10.0f} "
              f"{calls[family] / n_runs:>10.1f}")


def resolver_bytes():
    """The resolver prompt's split, rebuilt from the documents alone.

    The batching is deterministic -- `COREFERENCE_BATCH` sentences a call, each
    target's +/-5 window unioned into one table -- so no checkpoint is needed: the
    same call the runs made is the one rebuilt here.
    """
    c = collections.Counter()
    calls = 0
    for proj, (text, _model) in PROJECTS.items():
        sents = load_sentences(os.path.join(BASE, "benchmark", text))
        sent_map = build_sent_map(sents)
        for start in range(0, len(sents), SLinker87.COREFERENCE_BATCH):
            batch = sents[start:start + SLinker87.COREFERENCE_BATCH]
            calls += 1
            window_ids = set()
            for sent in batch:
                window = [s.number for s in sents
                          if abs(s.number - sent.number) <= SLinker87.CONTEXT_SENTENCES]
                window_ids.update(window)
                c["case: target text"] += len(f"TARGET S{sent.number}: {sent.text}\n")
                c["case: header"] += len(f"--- Case 1 ---\n")
                c["case: CONTEXT range line"] += len(
                    f"CONTEXT: sentences S{min(window)}-S{max(window)} above.\n")
            for n in sorted(window_ids):
                if n in sent_map:
                    size = len(f'{{"sentence": {n}, "text": "{sent_map[n].text}"}}, ')
                    if any(s.number == n for s in batch):
                        c["SENTENCES table: rows that are also a target below"] += size
                    else:
                        c["SENTENCES table: context-only rows"] += size
        # the fixed parts, once per call
    c["preamble (question + input contract + conservatism)"] = 331 * calls
    c["COREF_RULES"] = len(L87.COREF_RULES) * calls
    c["header + COMPONENTS + JSON schema"] = 330 * calls
    total = sum(c.values())
    per_run = lambda v: v  # one pass over all five projects == one five-project run
    print(f"\n=== the resolver prompt, rebuilt: {calls} calls per five-project run, "
          f"{SLinker87.COREFERENCE_BATCH} cases a call ===")
    print(f"{'part of the prompt':>52} {'bytes/run':>10} {'share':>7} "
          f"{'bytes/call':>11}")
    for part, size in c.most_common():
        print(f"{part:>52} {per_run(size):>10.0f} "
              f"{100 * size / max(1, total):>6.1f}% {size / max(1, calls):>11.0f}")
    print(f"{'TOTAL (rebuilt)':>52} {total:>10.0f}")
    dup = c["SENTENCES table: rows that are also a target below"]
    ctx = c["case: CONTEXT range line"]
    print(f"\nlossless headroom: {dup:.0f} B per run are table rows for sentences the "
          f"same call also prints as a TARGET ({100 * dup / max(1, total):.1f}%), and "
          f"{ctx:.0f} B are the per-case range line ({100 * ctx / max(1, total):.1f}%).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="*")
    ap.add_argument("--variant", default="s_linker87")
    a = ap.parse_args()
    main(a.globs or ["dedup_e2e_terra_r*_20260821"], a.variant)
