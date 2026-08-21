"""Every long prompt of the head, clause by clause, priced off recorded runs.

No LLM calls. Reads the checkpoints and call logs of runs already on disk, and
answers one question per clause the compaction round proposes to delete, so that
an arm is designed against evidence instead of against a reading of the prose.

The clauses, and the checkpoint that prices each:

  guard    The strict judge's leniency guard -- "approve unless that ground is one
           the rules above make decisive. An objection you could raise against most
           sentences is not a ground for rejecting this one." The judge records its
           `objection` next to its verdict, so the guard's whole visible effect is
           the cases with a non-none objection that are approved anyway. If that
           cell is empty the guard decides nothing.
  ground   The reject grounds the strict rubric enumerates. `LAYERED_COREF_RULES`
           spends its longest clause on one of them (an expression denoting what a
           component acts on or produces). Classifying the recorded objections says
           which grounds the judge actually reaches for, and how often the
           enumerated one is among them.
  named    What the resolver proposes for sentences that write the component's name.
           The resolver prompt never says the component must be absent; the judge's
           rubric opens by saying it is. Every such resolution is a call the judge
           spends to reject something the resolver should not have proposed.
  window   Whether resolutions ever cite an antecedent outside the +/-5 window each
           case declares ("CONTEXT: sentences Sx-Sy above."). If none do, that line
           is 10 restatements per resolver call of what the table already shows.
  claim    The full-name claim-first instruction against the lenient rubric, on the
           new head's runs -- the typed round's Q1, re-asked where the focus line is
           already gone.
  alias    Which of the three shapes `DOC_KNOWLEDGE_EXTRACTION_RULES` enumerates the
           approved aliases actually take. A shape with no instances is an
           enumeration item that costs bytes in every alias call and admits nothing.
  denot    What `QUALIFIED_CLAUSE` can be about in the denotation prompt. The
           partial-name scan has no qualified-identifier skip, so the clause guards
           a population that does reach that judge -- unless the population is
           empty, which this counts by rebuilding the scan the run made.
  bytes    The prompt-byte inventory per family, per five-project run, from the
           recorded call log: what a clause deletion is worth is its size times the
           calls that carry it.

Usage (from approach/):

    ../.venv/bin/python pilot/clause_audit.py \
        --variant s_linker87 "dedup_e2e_terra_r*_20260821"
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import pickle
import re
import sys

BASE = "/mnt/hostshare/ardoco-home/alinker-replication-package"
sys.path.insert(0, BASE + "/approach/src")

from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository            # noqa: E402
from llm_sad_sam.core.document_loader_v2 import (                     # noqa: E402
    build_sent_map, load_sentences,
)
from llm_sad_sam.linkers.experimental.s_linker87 import (             # noqa: E402
    SLinker87, NameForm,
)

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

CONTEXT_SENTENCES = 5
COREFERENCE_BATCH = 10

#: How a recorded objection is classified. Each pattern is the ground as the rubric
#: states it, not a term taken from any document: the classifier reads the judge's
#: own English, and the residue is reported rather than hidden.
GROUNDS = (
    ("named-in-sentence", r"\b(explicitly |directly )?nam(ed|es)\b|\bnamed in the "
                          r"sentence\b|\bnot (a )?corefer"),
    ("artifact/acts-on", r"\b(data|artifact|request|result|response|message|file|"
                         r"stream|content|item)\b.{0,60}\b(refers?|denot|not the "
                         r"component)\b|\bwhat the component (acts on|produces)\b"),
    ("ambiguous-antecedent", r"\bambig|\bcould equally\b|\bmore than one\b|\btwo or "
                             r"more\b|\bunclear which\b|\bequally plausible\b"),
    ("no-referring-expression", r"\bno (genuine |such )?referring expression\b|"
                                r"\bno (pronoun|reference)\b|\bnot a referring\b"),
    ("no-architectural-claim", r"\bno architectural claim\b|\bmakes no claim\b|"
                               r"\bstates nothing\b|\bdoes not (state|assert)\b"),
)


def gold(path):
    with open(os.path.join(BASE, "benchmark", path)) as fh:
        return {(int(r["sentence"]), r["modelElementID"]) for r in csv.DictReader(fh)}


def names_of(component, knowledge):
    """The component's names: the model name plus the run's approved aliases."""
    out = {component.name}
    if knowledge is not None:
        out |= {term for term, comp in knowledge.aliases.items()
                if comp == component.name}
    return out


def written(text, names):
    """Does the sentence write any of these names, ignoring case?"""
    return any(SLinker87._name_spans(text, n, NameForm.ANY_CASE) for n in names)


def classify(objection: str) -> str:
    low = (objection or "").strip().lower()
    if low in ("", "none", "none."):
        return "none"
    for label, pattern in GROUNDS:
        if re.search(pattern, low):
            return label
    return "other"


def state(run, variant, proj, phase):
    fn = os.path.join(run, "phase_states", variant, "openai", proj, f"{phase}.pkl")
    return pickle.load(open(fn, "rb")) if os.path.exists(fn) else None


#: Which prompt family a call belongs to, read off the first line the log records.
FAMILIES = (
    ("alias proposal", "Find all alternative names"),
    ("alias judge", "JUDGE: Review these component"),
    ("extraction", "Extract ALL references"),
    ("validation", "Validate components in a document"),
    ("coreference", "Resolve references (pronouns"),
    ("denotation", "Classify what each expression"),
)


def byte_inventory(runs, variant):
    """Prompt bytes and calls per family per five-project run, from the call log.

    The log does not record which variant sent a call, so a run directory holding
    two arms is halved: both arms run the same five projects and the same stages.
    """
    per_family = collections.Counter()
    calls = collections.Counter()
    variants = 0
    for run in runs:
        variants = max(variants, len(glob.glob(
            os.path.join(run, "phase_states", "*"))) or 1)
        for log in glob.glob(os.path.join(run, "llm_logs", "*.jsonl")):
            for line in open(log):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                preview = rec.get("prompt_preview", "")
                name = next((f for f, head in FAMILIES if preview.startswith(head)),
                            "other")
                per_family[name] += int(rec.get("prompt_length", 0))
                calls[name] += 1
    scale = max(1, len(runs) * max(1, variants))
    print(f"\n=== prompt bytes per five-project run ({variant}, {len(runs)} runs, "
          f"{variants} arms per run) ===")
    print(f"{'family':>16} {'calls/run':>10} {'bytes/run':>12} {'bytes/call':>11}")
    for name, total in sorted(per_family.items(), key=lambda kv: -kv[1]):
        n = calls[name]
        print(f"{name:>16} {n / scale:>10.1f} {total / scale:>12.0f} "
              f"{total / max(1, n):>11.0f}")


def main(globs, variant):
    runs = sorted(d for g in globs for d in glob.glob(os.path.join(BASE, "results", g))
                  if os.path.isdir(d))
    if not runs:
        sys.exit(f"no runs matched {globs}")
    n_runs = len(runs)

    guard = collections.Counter()      # (objection none?, approved, gold)
    ground = collections.Counter()     # (classified ground, approved, gold)
    named = collections.Counter()      # (target writes the name?, approved, gold)
    window = collections.Counter()     # (antecedent inside the declared window?,)
    claim = collections.Counter()      # (claim none?, approved, gold)
    alias_shape = collections.Counter()
    extracted = collections.Counter()  # (span sits in a dotted path?, gold)
    denot = collections.Counter()      # (span sits in a dotted path?, verdict, gold)
    named_examples, other_objections = [], []

    for run in runs:
        for proj, (text, model_path, gold_path) in PROJECTS.items():
            comps = parse_pcm_repository(os.path.join(BASE, "benchmark", model_path))
            by_id = {c.id: c for c in comps}
            by_name = {c.name: c for c in comps}
            sents = load_sentences(os.path.join(BASE, "benchmark", text))
            sent_map = build_sent_map(sents)
            g = gold(gold_path)

            kstate = state(run, variant, proj, "knowledge")
            knowledge = kstate["doc_knowledge"] if kstate else None
            if knowledge is not None:
                for term, comp in knowledge.aliases.items():
                    c = by_name.get(comp)
                    if c is None:
                        alias_shape["component not in model"] += 1
                        continue
                    words = re.findall(r"[A-Za-z]+[A-Za-z0-9]*", c.name)
                    if len(words) > 1 and term.casefold() in {w.casefold()
                                                              for w in words}:
                        alias_shape["word of a multi-word name"] += 1
                    elif term.casefold() in c.name.casefold():
                        alias_shape["shortening of the name"] += 1
                    else:
                        alias_shape["alternate name"] += 1

            coref = state(run, variant, proj, "linker_coreference")
            if coref:
                meta = {(m["sentence"], m["component_id"]): m
                        for m in coref["feedback"]["metadata"]}
                for d in coref["feedback"]["judge_decisions"]:
                    key = (d["sentence"], d["component_id"])
                    c = by_id.get(d["component_id"])
                    if c is None:
                        continue
                    obj = str(d.get("objection", ""))
                    label = classify(obj)
                    approved = bool(d.get("approved"))
                    isg = key in g
                    guard[(label == "none", approved, isg)] += 1
                    ground[(label, approved, isg)] += 1
                    if label == "other" and len(other_objections) < 25:
                        other_objections.append((proj, key[0], c.name, obj[:120]))

                    sent = sent_map.get(key[0])
                    writes = bool(sent) and written(sent.text, names_of(c, knowledge))
                    named[(writes, approved, isg)] += 1
                    if writes and len(named_examples) < 20:
                        m = meta.get(key, {})
                        named_examples.append(
                            (proj, key[0], c.name, isg, approved,
                             str(m.get("reference", ""))[:40],
                             str(m.get("antecedent_text", ""))[:40]))

                # Which batch a sentence went to is deterministic: the resolver
                # walks the document in blocks of COREFERENCE_BATCH, so the batch
                # of a target is its position in the document, and a row of the
                # SENTENCES table is a "target row" when it is one of that block's
                # own sentences -- the rows an arm could drop as already inline.
                order = {s.number: i for i, s in enumerate(sents)}
                for m in coref["feedback"]["metadata"]:
                    ant = m.get("antecedent_sentence")
                    if ant is None:
                        continue
                    ant, tgt = int(ant), int(m["sentence"])
                    inside = abs(ant - tgt) <= CONTEXT_SENTENCES
                    window[inside] += 1
                    if ant in order and tgt in order:
                        same_batch = (order[ant] // COREFERENCE_BATCH
                                      == order[tgt] // COREFERENCE_BATCH)
                        window[("antecedent is a target row", same_batch)] += 1

            # The denotation prompt's one rule clause, against the population the
            # scan actually hands it. The scan is deterministic given the run's own
            # alias table, so it is rebuilt here rather than guessed at.
            partial = state(run, variant, proj, "linker_partial_name")
            if partial:
                lk = SLinker87.__new__(SLinker87)
                lk.doc_knowledge = knowledge
                verdicts = {(d["sentence"], d["component_id"]): d
                            for d in partial["feedback"]["judge_decisions"]}
                for cand in lk._scan(sents, comps):
                    key = (cand.sentence_number, cand.component_id)
                    text = cand.sentence_text
                    span = cand.matched_text
                    dotted = any(
                        SLinker87._in_dotted_path(text, m.start(), m.end())
                        for m in re.finditer(re.escape(span), text))
                    d = verdicts.get(key, {})
                    denot[(dotted, d.get("denotation", "not judged"),
                           key in g)] += 1

            full = state(run, variant, proj, "linker_full_name")
            if full:
                # The same clause in the extraction prompt, against the spans the
                # extractor actually reported. Absence here is ambiguous -- the clause
                # may be what suppresses them -- so this is a size, not a verdict.
                for cand in full["feedback"]["candidates"]:
                    text, span = cand.get("text", ""), str(cand.get("matched", "")) \
                        or str(cand.get("component", ""))
                    dotted = any(
                        SLinker87._in_dotted_path(text, m.start(), m.end())
                        for m in re.finditer(re.escape(span), text)) if span else False
                    extracted[(dotted, (cand["sentence"],
                                        by_name[cand["component"]].id) in g
                               if cand["component"] in by_name else False)] += 1
                for d in full["feedback"]["judge_decisions"]:
                    if d.get("stage") != "full_name_judge":
                        continue
                    key = (d["sentence"], d["component_id"])
                    if key[1] not in by_id:
                        continue
                    none = (d.get("claim") or "").strip().lower() in ("", "none")
                    claim[(none, bool(d["approved"]), key in g)] += 1

    print(f"runs read: {n_runs}  (variant {variant})")

    print("\n=== guard: the strict judge's objection against its verdict ===")
    print(f"{'objection':>12} {'verdict':>8} {'gold':>5} {'n':>6} {'per run':>8}")
    for none in (True, False):
        for appr in (True, False):
            for isg in (True, False):
                n = guard[(none, appr, isg)]
                if n:
                    print(f"{'none' if none else 'stated':>12} "
                          f"{'approve' if appr else 'reject':>8} "
                          f"{'gold' if isg else '-':>5} {n:>6} {n / n_runs:>8.1f}")
    stated_appr = sum(guard[(False, True, x)] for x in (0, 1))
    stated = sum(guard[(False, a, x)] for a in (0, 1) for x in (0, 1))
    none_rej = sum(guard[(True, False, x)] for x in (0, 1))
    none_tot = sum(guard[(True, a, x)] for a in (0, 1) for x in (0, 1))
    if stated:
        print(f"  objection stated yet approved: {stated_appr}/{stated} "
              f"({100 * stated_appr / stated:.1f}%) -- the guard's whole effect")
    if none_tot:
        print(f"  objection none yet rejected:   {none_rej}/{none_tot} "
              f"({100 * none_rej / none_tot:.1f}%)")

    print("\n=== ground: which reject ground the strict judge reaches for ===")
    print(f"{'ground':>24} {'n':>6} {'per run':>8} {'approved':>9} {'gold':>5}")
    for label in [g_ for g_, _ in GROUNDS] + ["other", "none"]:
        n = sum(ground[(label, a, x)] for a in (0, 1) for x in (0, 1))
        if not n:
            continue
        appr = sum(ground[(label, True, x)] for x in (0, 1))
        isg = sum(ground[(label, a, True)] for a in (0, 1))
        print(f"{label:>24} {n:>6} {n / n_runs:>8.1f} {appr:>9} {isg:>5}")
    if other_objections:
        print("  unclassified objections (the residue this screen does not name):")
        for proj, s, comp, obj in other_objections[:10]:
            print(f"    {proj:<14} S{s:<4} {comp:<20} {obj}")

    print("\n=== named: resolutions whose target sentence writes the name ===")
    print(f"{'target':>14} {'verdict':>8} {'gold':>5} {'n':>6} {'per run':>8}")
    for writes in (True, False):
        for appr in (True, False):
            for isg in (True, False):
                n = named[(writes, appr, isg)]
                if n:
                    print(f"{'writes name' if writes else 'no name':>14} "
                          f"{'approve' if appr else 'reject':>8} "
                          f"{'gold' if isg else '-':>5} {n:>6} {n / n_runs:>8.1f}")
    wr = sum(named[(True, a, x)] for a in (0, 1) for x in (0, 1))
    wr_appr = sum(named[(True, True, x)] for x in (0, 1))
    wr_gold = sum(named[(True, True, True)] for _ in (0,))
    tot = sum(named.values())
    if tot:
        print(f"  {wr}/{tot} judged cases ({100 * wr / tot:.1f}%, {wr / n_runs:.1f} "
              f"per run) name the component in the target sentence; "
              f"{wr_appr} approved, {wr_gold} of those gold")
    if named_examples:
        print("  examples:")
        for proj, s, comp, isg, appr, ref, ant in named_examples[:10]:
            print(f"    {proj:<14} S{s:<4} {comp:<18} "
                  f"{'GOLD' if isg else '    '} {'appr' if appr else 'rej ':>4} "
                  f'ref="{ref}" ant="{ant}"')

    print("\n=== window: antecedents against the declared context range ===")
    for inside in (True, False):
        n = window[inside]
        if n:
            print(f"  antecedent {'inside' if inside else 'OUTSIDE':>7} "
                  f"+/-{CONTEXT_SENTENCES}: {n:>5} ({n / n_runs:.1f} per run)")
    for same in (True, False):
        n = window[("antecedent is a target row", same)]
        if n:
            where = ("a TARGET of the same call" if same
                     else "a context-only row of the table")
            print(f"  antecedent sits in {where}: {n:>5} ({n / n_runs:.1f} per run)")

    print("\n=== claim: the full-name claim-first instruction on this head ===")
    for none in (True, False):
        for appr in (True, False):
            for isg in (True, False):
                n = claim[(none, appr, isg)]
                if n:
                    print(f"{'none' if none else 'quoted':>10} "
                          f"{'approve' if appr else 'reject':>8} "
                          f"{'gold' if isg else '-':>5} {n:>6} {n / n_runs:>8.1f}")
    nq_tot = sum(claim[(True, a, x)] for a in (0, 1) for x in (0, 1))
    nq_rej = sum(claim[(True, False, x)] for x in (0, 1))
    if nq_tot:
        print(f"  claim=none: {nq_tot} cases, rejected {nq_rej} "
              f"({100 * nq_rej / nq_tot:.1f}%), gold among them "
              f"{claim[(True, False, True)]} ({claim[(True, False, True)] / n_runs:.1f}"
              f" per run)")

    print("\n=== denot: what QUALIFIED_CLAUSE can be about at the denotation judge ===")
    print(f"{'span':>22} {'verdict':>12} {'gold':>5} {'n':>6} {'per run':>8}")
    for dotted in (True, False):
        for verdict in ("participant", "associated", "not judged"):
            for isg in (True, False):
                n = denot[(dotted, verdict, isg)]
                if n:
                    print(f"{'in a dotted path' if dotted else 'plain':>22} "
                          f"{verdict:>12} {'gold' if isg else '-':>5} {n:>6} "
                          f"{n / n_runs:>8.1f}")
    dot = sum(n for k, n in denot.items() if k[0])
    tot = sum(denot.values())
    print(f"  {dot} of {tot} scanned candidates ({dot / n_runs:.1f} per run) carry a "
          f"span inside a dotted path -- the whole population the clause speaks about")

    print("\n=== extract: what QUALIFIED_CLAUSE can be about at the extractor ===")
    for dotted in (True, False):
        for isg in (True, False):
            n = extracted[(dotted, isg)]
            if n:
                print(f"  {'span in a dotted path' if dotted else 'plain span':>22} "
                      f"{'gold' if isg else '-':>5} {n:>5} ({n / n_runs:.1f} per run)")

    print("\n=== alias: the shapes the extraction enumeration lists ===")
    total_alias = sum(alias_shape.values())
    for shape, n in alias_shape.most_common():
        print(f"{shape:>28} {n:>6} {n / n_runs:>8.1f} per run "
              f"({100 * n / max(1, total_alias):.1f}%)")

    byte_inventory(runs, variant)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="*", default=["dedup_e2e_terra_r*_20260821"])
    ap.add_argument("--variant", default="s_linker87")
    a = ap.parse_args()
    main(a.globs or ["dedup_e2e_terra_r*_20260821"], a.variant)
