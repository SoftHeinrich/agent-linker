"""The populations a paraphrase of the static text would be talking to.

No LLM calls. `pilot/static_audit.py` names which authored clauses legislate a
SURFACE (spacing, hyphenation, capitalization, dotted paths) or ENUMERATE instances
where a concept would name the class. Those are the paper liabilities. This script
asks the branch's step-1 question of each of them: **how many recorded cases does
the clause actually speak about, and how many of those are gold?** A clause whose
population is empty can be generalized on the text alone; a clause with gold in its
population needs a paired arm on both models before a word of it moves.

Read off the head's own end-to-end batch, so the populations are the ones the head
produces, not an older variant's.

    ../.venv/bin/python pilot/static_screen.py --variant s_linker89 \\
        "compact_e2e_terra_r*_20260821"
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import pickle
import re
import sys

BASE = "/mnt/hostshare/ardoco-home/alinker-replication-package"
sys.path.insert(0, BASE + "/approach/src")

from llm_sad_sam.core.document_loader_v2 import (                      # noqa: E402
    build_sent_map, load_sentences,
)
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository             # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker89 import SLinker89      # noqa: E402

PROJECTS = {
    "mediastore": ("mediastore/text_2016/mediastore.txt",
                   "mediastore/model_2016/pcm/ms.repository",
                   "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv"),
    "teastore": ("teastore/text_2020/teastore.txt",
                 "teastore/model_2020/pcm/teastore.repository",
                 "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv"),
    "teammates": ("teammates/text_2021/teammates.txt",
                  "teammates/model_2021/pcm/teammates.repository",
                  "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "bigbluebutton": ("bigbluebutton/text_2021/bigbluebutton.txt",
                      "bigbluebutton/model_2021/pcm/bbb.repository",
                      "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "jabref": ("jabref/text_2021/jabref.txt",
               "jabref/model_2021/pcm/jabref.repository",
               "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
}

#: The four nouns `LAYERED_COREF_RULES` enumerates. The arm keeps the principle and
#: drops the list, so the question is whether the recorded objections lean on the
#: listed words or on the principle they instantiate.
ENUMERATED = ("data", "artifact", "request", "result")
#: Wording that states the same ground WITHOUT any listed word.
PRINCIPLE = re.compile(
    r"\b(acts on|produces|operates on|output|input|what the component|"
    r"the thing|object of|produced by|processed by|refers to (the|that) "
    r"(thing|content|item|message|file|page|record))\b", re.I)


def gold(path):
    out = set()
    with open(os.path.join(BASE, "benchmark", path)) as fh:
        for row in csv.DictReader(fh):
            out.add((int(row["sentence"]), row["modelElementID"]))
    return out


def state(run, variant, proj, phase):
    fn = os.path.join(run, "phase_states", variant, "openai", proj, f"{phase}.pkl")
    return pickle.load(open(fn, "rb")) if os.path.exists(fn) else None


def _squash(s: str) -> str:
    return re.sub(r"[\s_-]+", "", s).casefold()


def separator_variant(text: str, name: str) -> str:
    """A writing of ``name`` in ``text`` that differs only in separators, or "".

    `_find_exact_form` will not return one -- it matches the name's own characters
    -- so a candidate whose sentence writes "AB" for "A B" would otherwise be
    counted as reaching the extractor through an alias. This is the direct test of
    the clause `ENTITY_EXTRACTION_RULES` states: same words, different joining.
    """
    target = _squash(name)
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9 _-]{0,%d}" % (len(name) + 4), text):
        window = m.group(0)
        for end in range(len(window), 0, -1):
            if _squash(window[:end]) == target:
                return window[:end]
    return ""


def surface_class(name: str, surface: str) -> str:
    """How this writing of the name differs from the COMPONENTS spelling."""
    if surface == name:
        return "exact"
    if surface.casefold() == name.casefold():
        return "case only"
    if _squash(surface) == _squash(name):
        return "separators (spacing/hyphen/compound)"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="s_linker89")
    ap.add_argument("globs", nargs="+")
    args = ap.parse_args()
    runs = sorted(d for g in args.globs
                  for d in glob.glob(os.path.join(BASE, "results", g))
                  if os.path.isdir(d))
    if not runs:
        sys.exit(f"no runs matched {args.globs}")
    n = len(runs)

    surf = collections.Counter()
    surf_gold = collections.Counter()
    surf_appr = collections.Counter()
    ident = collections.Counter()
    ident_gold = collections.Counter()
    artifact = collections.Counter()
    artifact_gold = collections.Counter()
    alias_shape = collections.Counter()

    for run in runs:
        for proj, (text, model_path, gold_path) in PROJECTS.items():
            full = state(run, args.variant, proj, "linker_full_name")
            if not full:
                continue
            comps = parse_pcm_repository(os.path.join(BASE, "benchmark", model_path))
            by_name = {c.name: c for c in comps}
            g = gold(gold_path)
            sents = load_sentences(os.path.join(BASE, "benchmark", text))
            sent_map = build_sent_map(sents)
            approved = {(d["sentence"], d["component_id"])
                        for d in full["feedback"]["judge_decisions"] if d["approved"]}

            # --- S1: how the sentence writes the name (STRICTER_CLAUSE clause 3,
            #         ENTITY_EXTRACTION_RULES clause 1) -------------------------
            for cand in full["feedback"]["candidates"]:
                comp = by_name.get(cand["component"])
                sent = sent_map.get(cand["sentence"])
                if comp is None or sent is None:
                    continue
                form = SLinker89._find_exact_form(sent.text, comp.name)
                if not form:
                    form = separator_variant(sent.text, comp.name)
                cls = surface_class(comp.name, form) if form else "via alias only"
                key = (cand["sentence"], comp.id)
                surf[cls] += 1
                surf_gold[cls] += key in g
                surf_appr[cls] += key in approved

                # --- S2: the span inside a longer identifier (QUALIFIED_CLAUSE,
                #         ALIAS_EXCLUSION_RULES) ---------------------------------
                # The clause says "occurs ONLY as part of", so a candidate counts
                # only when EVERY writing of the name in the sentence is embedded.
                if form:
                    kinds = []
                    for m in re.finditer(re.escape(form), sent.text):
                        before, after = sent.text[:m.start()], sent.text[m.end():]
                        left, right = before[-1:], after[:1]
                        if ((left == "." and before[-2:-1].isalnum())
                                or (right == "." and after[1:2].isalnum())):
                            kinds.append("dotted")
                        elif (left in "_" or right in "_"
                              or (right.isupper() and form[-1:].islower())
                              or (left.isupper() and before[-2:-1].islower())):
                            kinds.append("joined")
                        elif left.isalnum() or right.isalnum():
                            kinds.append("in-word")
                        else:
                            kinds.append("free")
                    if kinds and "free" not in kinds:
                        k = ("only inside a dotted path" if "dotted" in kinds
                             else "only inside a joined identifier"
                             if "joined" in kinds
                             else "only inside a longer word")
                        ident[k] += 1
                        ident_gold[k] += key in g
                    elif "dotted" in kinds or "joined" in kinds:
                        ident["embedded somewhere, but also written free"] += 1
                        ident_gold["embedded somewhere, but also written free"] += key in g

            # --- S3: does the strict judge's objection use the listed words? ---
            coref = state(run, args.variant, proj, "linker_coreference")
            if coref:
                for d in coref["feedback"]["judge_decisions"]:
                    obj = (d.get("objection") or "").strip()
                    if not obj or obj.lower() == "none":
                        continue
                    words = set(re.findall(r"[a-z]+", obj.lower()))
                    listed = words & set(ENUMERATED)
                    if listed and PRINCIPLE.search(obj):
                        k = "ground cited, listed word AND principle"
                    elif listed:
                        k = "ground cited via a LISTED word only"
                    elif PRINCIPLE.search(obj):
                        k = "ground cited via the PRINCIPLE only"
                    else:
                        k = "some other objection"
                    artifact[k] += 1
                    if (d["sentence"], d["component_id"]) in g:
                        artifact_gold[k] += 1

            # --- S4: what shapes the alias extractor actually returns ----------
            kstate = state(run, args.variant, proj, "knowledge")
            dk = kstate["doc_knowledge"] if kstate else None
            for bucket in ("aliases", "abbreviations", "synonyms"):
                for t, comp_name in (getattr(dk, bucket, {}) or {}).items():
                    if True:
                        tl, cl = str(t).casefold(), str(comp_name).casefold()
                        cw = re.findall(r"[a-z]+", cl)
                        if tl in cl or cl in tl:
                            alias_shape["a short form of the name"] += 1
                        elif len(cw) > 1 and tl in cw:
                            alias_shape["a word of a multi-word name"] += 1
                        elif (len(t) <= 5 and t.isupper()) or "".join(
                                w[0] for w in cw) == tl:
                            alias_shape["an initialism"] += 1
                        else:
                            alias_shape["an alternate name sharing no word"] += 1

    def table(title, counts, extra=None, extra_name=""):
        print(f"\n{title}")
        tot = sum(counts.values()) or 1
        for k, v in counts.most_common():
            line = f"  {k:>44}: {v / n:7.1f}/run  {100 * v / tot:5.1f}%"
            if extra is not None:
                line += f"   {extra_name} {extra[k] / n:5.1f}/run"
            print(line)

    print(f"{n} runs, {args.variant}\n" + "=" * 74)
    table("S1  How the sentence writes the name  (STRICTER_CLAUSE#3 'Capitalization "
          "is evidence',\n    ENTITY_EXTRACTION_RULES#1 'spacing, hyphenation or "
          "compound joining')", surf, surf_gold, "gold")
    print("      approved by the judge: " + ", ".join(
        f"{k} {surf_appr[k] / n:.1f}" for k in surf))
    table("S2  The span inside a longer identifier  (QUALIFIED_CLAUSE 'joined or "
          "dotted',\n    ALIAS_EXCLUSION_RULES 'compound or qualified name')",
          ident, ident_gold, "gold")
    table("S3  What the strict judge's objections lean on  (LAYERED_COREF_RULES#4\n"
          "    '-- the data, the artifact, the request, the result --')", artifact,
          artifact_gold, "gold")
    table("S4  What the alias extractor returns  (DOC_KNOWLEDGE_EXTRACTION_RULES#1\n"
          "    '(introduced short forms, alternate names, or words of multi-word "
          "names)')", alias_shape)


if __name__ == "__main__":
    main()
