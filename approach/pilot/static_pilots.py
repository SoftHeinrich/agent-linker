"""Paraphrase arms over the module's static text, priced one stage at a time.

The compaction round removed repetition and left the authored text alone. This one
asks the other question: does each static clause state a **concept** -- something a
paper can defend as general -- or a recipe fitted to the surfaces these documents
happen to show? `pilot/static_audit.py` names the clauses that legislate a surface
or enumerate instances; `pilot/static_screen.py` prices the population each one
speaks about. Every arm below is a **paraphrase**: the clause keeps its job and
loses its recipe. None deletes a rule, so a neutral verdict is the outcome that
adopts the arm, and a negative one says the recipe was doing work the concept does
not.

Each arm swaps module constants on the head and re-runs the one stage that reads
them, over the head's own recorded checkpoints, with every other stage held at what
that run recorded. Same policy as every round on this branch: stage first, both
models, compose after.

    ../.venv/bin/python pilot/static_pilots.py --group qual1 --model terra --runs 3
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import pickle
import statistics as st
import sys
from pathlib import Path

BASE = "/mnt/hostshare/ardoco-home/alinker-replication-package"
sys.path.insert(0, BASE + "/approach/src")

from llm_sad_sam.core.document_loader_v2 import (                      # noqa: E402
    build_sent_map, load_sentences,
)
from llm_sad_sam.llm_client import LLMBackend                          # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository             # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker89 as L           # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker89 import SLinker89      # noqa: E402
from llm_sad_sam.core.data_types_v2 import SadSamLink                  # noqa: E402

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

#: The head's own end-to-end batch. Reading the arms against the variant they patch
#: keeps the populations the ones the head produces.
RECORDED = {
    "terra": ("compact_e2e_terra_r*_20260821", "s_linker89"),
    "luna": ("compact_e2e_luna_r*_20260821", "s_linker89"),
}

# ─── the paraphrases ──────────────────────────────────────────────────────────
# Every one of these is authored from the concept, not from any document. No
# benchmark word appears in any of them (GATE-06) and none names a shape these
# five documents happen to use (GATE-07).

#: A1. `QUALIFIED_CLAUSE` without "joined or dotted". The screen shows the wording
#: over-reaches: of the candidates whose every writing is embedded, the ones inside
#: a *joined* identifier are 13.3 a run and 13.0 of them are gold, because a name
#: written as one word is the whole name, not a piece of a longer one. The clause
#: survives only because a reader disambiguates "joined" correctly. The paraphrase
#: says what it means: a FRAGMENT of a longer identifier.
GEN_QUALIFIED = ("An expression that appears only as a fragment of a longer "
                 "identifier is naming a piece of that identifier, not a "
                 "participant in what the sentence describes.")

#: A4. `STRICTER_CLAUSE` with its orthographic sentence generalized. The clause's
#: point is that surface evidence never settles a use/mention question; naming
#: capitalization is one instance of surface evidence, not the concept. Population:
#: 28.7 candidates a run are written in a case the COMPONENTS list does not use,
#: 24.7 of them gold, so this arm is a real risk and is paired on both models.
GEN_STRICTER = (
    "Some sentences use an ordinary English word that happens to coincide with a "
    "component's name. Approve only when the sentence uses that word as the name of "
    "the component; if it is used in its ordinary sense and the component is not what "
    "the sentence is talking about, reject. How the word is written is evidence "
    "either way and never settles it on its own."
)

#: A7. One use/mention sentence, shared. `DOC_KNOWLEDGE_EXTRACTION_RULES` clause 2
#: ("Reject terms whose ordinary English use dominates") and `STRICTER_CLAUSE`
#: clause 1 state the same principle for two stages -- the audit measures their
#: content-word overlap at J = 0.50, the highest non-trivial pair in the module.
#: This is the shared statement; the two constants below reference it instead of
#: each restating it.
USE_MENTION = ("A word can occur in a document either as the name of a component "
               "or in its ordinary English sense.")
MERGED_STRICTER = (
    f"{USE_MENTION} Approve only when the sentence uses the word as the name of the "
    "component; if the ordinary sense is the one at work and the component is not "
    "what the sentence is talking about, reject. How the word is written is evidence "
    "either way and never settles it on its own."
)
MERGED_DOC_EXTRACTION = (
    "Find surface forms the document uses to refer to a single named component "
    "(introduced short forms, alternate names, or words of multi-word names when "
    f"they alone clearly mean the full name). {USE_MENTION} Reject a term whose "
    "ordinary sense is the one the document is using."
)

#: A5. `LAYERED_COREF_RULES` without its four-item list. The screen: 10.7 recorded
#: objections a run reach for one of the listed words and 0.7 state the ground
#: without one, so the list is what the judge leans on -- and 3.0 of those
#: rejections are gold, so the clause as a whole is not free either way. Deleting
#: the whole ground cost luna 6.7 gold resolutions a run last round; this arm keeps
#: the ground and drops only the instances.
GEN_LAYERED_COREF = (
    "These are coreference links: a pronoun or noun phrase in the sentence is claimed to "
    "refer back to the component, which is NOT named in the sentence itself. Approve only "
    "when the sentence contains a genuine referring expression that unambiguously points "
    "to THIS component and makes an architectural claim about it. Reject when there is no "
    "such referring expression or when the antecedent could equally be a different "
    "component. An expression denoting what a component acts on or produces refers to "
    "that thing and not to the component, however clearly the component is the one "
    "acting on it. When uncertain, reject."
)

#: A3. `ENTITY_EXTRACTION_RULES` without the spelling recipe. The licence it grants
#: is real but tiny -- 1.0 candidate a run is written with the name's words joined
#: differently, and it is gold -- so the paraphrase keeps the licence and states it
#: as a property of names rather than a list of three typographic operations.
GEN_ENTITY_EXTRACTION = """Report a reference only when the sentence itself writes the component's name, as the COMPONENTS list gives it or as one of the KNOWN ALIASES. A writing that differs only in how the name's own words are separated is still that name.

Do not report a component that the sentence only implies as a participant in a described interaction without naming it. Among the sentences that do name it, report every one, however incidental the mention: whether the mention carries an architectural claim is decided later."""

#: A6. `DOC_KNOWLEDGE_EXTRACTION_RULES` without its three-item parenthetical. The
#: screen: of the forms the extractor returns, 17.3 a run are short forms of the
#: name and 11.7 share no word with it, while the third listed shape -- a word of a
#: multi-word name -- is returned 0.0 times a run. The paraphrase names the
#: condition all three instances satisfy instead of listing them.
GEN_DOC_EXTRACTION = ("Find surface forms the document uses to refer to a single "
                      "named component, whatever shape the form takes, when the "
                      "form alone clearly means that component. Reject terms whose "
                      "ordinary English use dominates.")

#: A2. One clause for identifier fragments, shared. `ALIAS_EXCLUSION_RULES` and
#: `QUALIFIED_CLAUSE` state the same principle in two wordings (J = 0.19 on content
#: words, the shared words being exactly "identifier", "longer", "part"). The merge
#: gives the alias stage the general clause plus the imperative its stage needs.
MERGED_ALIAS_EXCLUSION = f"{GEN_QUALIFIED} Do not offer such a fragment as an alias."

#: A6 + A7 in one constant. `genalias` and `mergeord` both rewrite
#: `DOC_KNOWLEDGE_EXTRACTION_RULES` and cannot compose, so the head needs a form
#: that does both jobs: the three-shape enumeration replaced by the condition all
#: three satisfy (`genalias`), and the use/mention principle stated once and shared
#: with the judging prompt (`mergeord`). Measured as its own arm, because a
#: composition of two measured texts is not a measured text.
MERGED_GEN_DOC_EXTRACTION = (
    "Find surface forms the document uses to refer to a single named component, "
    f"whatever shape the form takes, when the form alone clearly means that "
    f"component. {USE_MENTION} Reject a term whose ordinary sense is the one the "
    "document is using."
)

ORIG = {name: getattr(L, name) for name in (
    "QUALIFIED_CLAUSE", "STRICTER_CLAUSE", "LAYERED_COREF_RULES",
    "ENTITY_EXTRACTION_RULES", "DOC_KNOWLEDGE_EXTRACTION_RULES",
    "ALIAS_EXCLUSION_RULES",
)}

ARMS = {
    # lenient judging: 8.7 calls a five-project run
    "qual1": {
        "ctl": {},
        "genqual": {"QUALIFIED_CLAUSE": GEN_QUALIFIED},
        "genform": {"STRICTER_CLAUSE": GEN_STRICTER},
        "mergeord": {"STRICTER_CLAUSE": MERGED_STRICTER},
    },
    # strict judging: 3.6 calls a run
    "strict1": {
        "ctl": {},
        "genartifact": {"LAYERED_COREF_RULES": GEN_LAYERED_COREF},
    },
    # the alias family: one knowledge call a project, then the stages that
    # consume what it returns. An alias arm cannot be read at its own stage --
    # its output is a vocabulary, not links -- so it is read at its consumer.
    "alias1": {
        "ctl": {},
        "genalias": {"DOC_KNOWLEDGE_EXTRACTION_RULES": GEN_DOC_EXTRACTION},
        "mergefrag": {"ALIAS_EXCLUSION_RULES": MERGED_ALIAS_EXCLUSION,
                      "QUALIFIED_CLAUSE": GEN_QUALIFIED},
        "mergeord": {"DOC_KNOWLEDGE_EXTRACTION_RULES": MERGED_DOC_EXTRACTION,
                     "STRICTER_CLAUSE": MERGED_STRICTER},
    },
    # the merged-and-generalized alias rule, the form the head needs
    "alias2": {
        "ctl": {},
        "mergealias": {"DOC_KNOWLEDGE_EXTRACTION_RULES": MERGED_GEN_DOC_EXTRACTION,
                       "STRICTER_CLAUSE": MERGED_STRICTER},
    },
    # full-name extraction: 9.0 calls a run
    "extract1": {
        "ctl": {},
        "genextract": {"ENTITY_EXTRACTION_RULES": GEN_ENTITY_EXTRACTION},
        "genqual": {"QUALIFIED_CLAUSE": GEN_QUALIFIED},
    },
}

#: Which recorded stages an arm's own stage is composed with.
OTHER_STAGES = {
    "qual1": ("partial_name", "coreference"),
    "strict1": ("full_name", "partial_name"),
    "extract1": ("partial_name", "coreference"),
    "alias1": ("partial_name", "coreference"),
    "alias2": ("partial_name", "coreference"),
}


class Arm:
    """The constants an arm swaps, in force for the duration of its stage."""

    def __init__(self, name, spec):
        self.name, self.spec = name, spec

    def __enter__(self):
        for key, value in ORIG.items():
            setattr(L, key, self.spec.get(key, value))
        return self

    def __exit__(self, *exc):
        for key, value in ORIG.items():
            setattr(L, key, value)


def check_parity():
    """The control must render the head's own bytes, and every arm must differ.

    An arm prices its paraphrase only if the control is the head; and a spec whose
    key was misspelled would silently measure nothing, which this catches.
    """
    with Arm("ctl", {}):
        for key, value in ORIG.items():
            assert getattr(L, key) == value, f"ctl drifted on {key}"
    for group, arms in ARMS.items():
        for arm, spec in arms.items():
            if arm == "ctl":
                continue
            assert spec, f"{group}/{arm} is empty"
            for key, value in spec.items():
                assert key in ORIG, f"{group}/{arm} patches unknown {key}"
                assert value != ORIG[key], f"{group}/{arm} is a no-op on {key}"
                with Arm(arm, spec):
                    assert getattr(L, key) == value
    print("parity: control renders the head's constants; every arm differs\n")


def gold(path):
    with open(os.path.join(BASE, "benchmark", path)) as fh:
        return {(int(r["sentence"]), r["modelElementID"]) for r in csv.DictReader(fh)}


def recorded_runs(model):
    pattern, variant = RECORDED[model]
    return sorted(glob.glob(os.path.join(BASE, "results", pattern,
                                         "phase_states"))), variant


def state(run_dir, variant, proj, phase):
    fn = os.path.join(run_dir, variant, "openai", proj, f"{phase}.pkl")
    return pickle.load(open(fn, "rb")) if os.path.exists(fn) else None


def scores(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    f2 = 5 * p * r / (4 * p + r) if 4 * p + r else 0.0
    return f1 * 100, f2 * 100


def linker_for(recorded_knowledge, sink):
    lk = SLinker89(backend=LLMBackend.OPENAI)
    lk.doc_knowledge = (recorded_knowledge["doc_knowledge"]
                        if recorded_knowledge else None)
    inner_ask = lk._ask

    def ask(prompt, **kw):
        out = inner_ask(prompt, **kw)
        sink.append({"chars": len(prompt), "label": kw.get("label", ""),
                     "prompt": prompt, "response": out})
        return out

    lk._ask = ask
    return lk


def judge_fullname(lk, cands, comps, sent_map):
    bundles = {(c.sentence_number, c.component_id): lk._build_evidence_bundle(c, sent_map)
               for c in cands}
    approved, _ = lk._validate_with_evidence(
        cands, bundles, comps, sent_map,
        phase_tag="pilot_full_name_judge", stage_label="full_name")
    return {(c.sentence_number, c.component_id) for c in approved}


def judge_coref(lk, run_dir, variant, proj, comps, sent_map):
    rec = state(run_dir, variant, proj, "linker_coreference")
    if not rec:
        return set()
    meta = {(m["sentence"], m["component_id"]): m
            for m in rec["feedback"]["metadata"]}
    i2n = {c.id: c.name for c in comps}
    raw = [SadSamLink(s, c, i2n[c], source="coreference")
           for (s, c) in meta if c in i2n]
    approved, _ = lk._validate_coref_links(raw, sent_map, comps, meta)
    return {(l.sentence_number, l.component_id) for l in approved}


def run_group(group, model, runs, out_dir):
    run_dirs, variant = recorded_runs(model)
    if not run_dirs:
        sys.exit(f"no recorded runs for {model}: {RECORDED[model][0]}")
    arms = ARMS[group]
    kept = {arm: collections.defaultdict(dict) for arm in arms}
    stage_tot = {arm: collections.Counter() for arm in arms}
    prompt_chars = {arm: [] for arm in arms}
    calls = collections.Counter()

    for r in range(runs):
        run_dir = run_dirs[r % len(run_dirs)]
        for proj, (text, model_path, gold_path) in PROJECTS.items():
            comps = parse_pcm_repository(os.path.join(BASE, "benchmark", model_path))
            sents = load_sentences(os.path.join(BASE, "benchmark", text))
            sent_map = build_sent_map(sents)
            name_to_id = {c.name: c.id for c in comps}
            g = gold(gold_path)
            knowledge = state(run_dir, variant, proj, "knowledge")

            # The judging groups hold the extraction pass fixed, so every arm
            # judges the same cases and the arm is the only difference.
            shared = None
            if group == "qual1":
                sink = []
                lk = linker_for(knowledge, sink)
                with Arm("ctl", {}):
                    shared = lk._extract_named_mentions(
                        sents, comps, name_to_id, sent_map)
                calls["extract"] += len(sink)

            for arm, spec in arms.items():
                sink = []
                lk = linker_for(knowledge, sink)
                with Arm(arm, spec):
                    if group == "qual1":
                        pairs = judge_fullname(lk, list(shared.values()),
                                               comps, sent_map)
                    elif group in ("alias1", "alias2"):
                        # The arm's own knowledge, not the recorded one: that is
                        # what it changes. Its consumer is the full-name pair.
                        lk.doc_knowledge = lk._learn_document_knowledge(
                            sents, comps)
                        cands = lk._extract_named_mentions(
                            sents, comps, name_to_id, sent_map)
                        pairs = judge_fullname(lk, list(cands.values()),
                                               comps, sent_map)
                    elif group == "extract1":
                        cands = lk._extract_named_mentions(
                            sents, comps, name_to_id, sent_map)
                        pairs = judge_fullname(lk, list(cands.values()),
                                               comps, sent_map)
                    else:
                        pairs = judge_coref(lk, run_dir, variant, proj,
                                            comps, sent_map)
                prompt_chars[arm] += [c["chars"] for c in sink]
                calls[arm] += len(sink)
                tg = len(pairs & g)
                stage_tot[arm]["g"] += tg
                stage_tot[arm]["n"] += len(pairs) - tg
                kept[arm][f"run{r + 1}"][proj] = sorted(list(x) for x in pairs)
                print(f"  run{r + 1} {proj:<14} {arm:<12} "
                      f"{tg:3d}g/{len(pairs) - tg:3d}n", flush=True)
                exemplar = out_dir / "prompts" / f"{group}_{model}_{arm}.txt"
                if sink and not exemplar.exists():
                    exemplar.parent.mkdir(parents=True, exist_ok=True)
                    longest = max(sink, key=lambda c: c["chars"])
                    exemplar.write_text(
                        f"# one prompt this arm sent: {group} / {model} / {arm} / "
                        f"run {r + 1} / {proj} / {longest['chars']} chars\n\n"
                        + longest["prompt"])

    for arm in arms:
        json.dump(kept[arm], open(out_dir / f"kept_{group}_{model}_{arm}.json", "w"))

    print(f"\n{group} on {model}, {runs} runs, per five-project run:")
    for arm in arms:
        chars = st.mean(prompt_chars[arm]) if prompt_chars[arm] else 0
        print(f"  {arm:<12} gold {stage_tot[arm]['g'] / runs:6.1f}  "
              f"spurious {stage_tot[arm]['n'] / runs:6.1f}  "
              f"calls {calls[arm] / runs:5.1f}  mean prompt {chars:7.0f} chars")
    compose(group, model, runs, arms, kept, variant, run_dirs)


def compose(group, model, runs, arms, kept, variant, run_dirs):
    """The pipeline score: the arm's kept pairs unioned with the same recorded
    run's other two stages, which no constant here touches."""
    print(f"\n{group} on {model}, composed with the recorded "
          f"{' + '.join(OTHER_STAGES[group])} of the same run:")
    for arm in arms:
        per = []
        for r in range(runs):
            run_dir = run_dirs[r % len(run_dirs)]
            f1s, f2s, TP, FP = [], [], 0, 0
            for proj, (_t, _m, gold_path) in PROJECTS.items():
                g = gold(gold_path)
                links = {tuple(x) for x in kept[arm][f"run{r + 1}"].get(proj, [])}
                for stage in OTHER_STAGES[group]:
                    rec = state(run_dir, variant, proj, f"linker_{stage}")
                    if rec:
                        links |= {(l.sentence_number, l.component_id)
                                  for l in rec["links"]}
                tp = len(links & g)
                TP += tp
                FP += len(links) - tp
                a, b = scores(tp, len(links) - tp, len(g) - tp)
                f1s.append(a)
                f2s.append(b)
            per.append((st.mean(f1s), st.mean(f2s), TP, FP))
        print(f"  {arm:<12} macroF1 {st.mean(x[0] for x in per):6.2f}  "
              f"macroF2 {st.mean(x[1] for x in per):6.2f}  "
              f"TP {st.mean(x[2] for x in per):6.1f}  "
              f"FP {st.mean(x[3] for x in per):6.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=sorted(ARMS), required=True)
    ap.add_argument("--model", choices=sorted(RECORDED), required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default=os.environ.get(
        "AB_OUT", "../results/static_round"))
    args = ap.parse_args()
    check_parity()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_group(args.group, args.model, args.runs, out_dir)


if __name__ == "__main__":
    main()
