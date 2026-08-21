"""Typed verdicts, and the two single points the entity prompts turn on.

Replays ONE stage of `s_linker85` against the checkpoints of a recorded run, so an
arm costs that stage's calls and nothing else, and every arm in an invocation sees
the same recorded aliases (`knowledge.pkl`). Two groups:

  fullname  the full-name judge, over ONE extraction pass shared by every arm in
            the run, so the arms differ in the judging prompt and in nothing else:

              ctl           s_linker85 as it stands
              nodead        minus "A mention that says nothing further about the
                            component still counts as a valid link." The offline
                            audit (`pilot/entity_prompt_audit.py`) reads that
                            sentence as inert: over three recorded runs a side,
                            claim="none" is rejected 45/45 on terra and 23/23 on
                            luna, so the decide-clause already settles every case
                            the sentence speaks about. If it is inert, deleting it
                            is free compaction; this arm is what says so.
              typed         the rubric as a closed set of verdicts, one of which
                            approves. Replaces LAYERED_ENTITY_RULES,
                            QUALIFIED_CLAUSE and STRICTER_CLAUSE with four named
                            types, and resolves the contradiction the way the
                            judge already resolves it: NO_CLAIM is a reject type.
              typedlenient  the same closed set, resolved the other way: NO_CLAIM
                            approves, as LAYERED_ENTITY_RULES says it should. The
                            audit prices that direction at +2.0 gold and +10.3
                            spurious per five-project run on terra; this arm is
                            the measurement, not the projection.

  extract   the extraction prompt, judged by the control judge in every arm:

              ctl           s_linker85 as it stands
              nomorph       minus "count a name written with different spacing,
                            hyphenation or compound joining as that name." The
                            clause's whole visible yield is the candidates whose
                            sentence writes no name at ANY_CASE: 2.3 gold and 1.0
                            spurious per run on terra, 2.3 gold and 9.7 spurious
                            on luna. The two models disagree about whether it pays,
                            which is why it is measured on both.

  coref     the coreference judge, over the resolutions recorded in the same run:

              ctl           s_linker85 as it stands (free-text objection first)
              typedcoref    the objection paragraph and its field replaced by a
                            closed set of four verdicts. s83 bought 2.4 macro F1
                            with "state the strongest ground for rejecting before
                            deciding"; naming the grounds is that mechanism made
                            discrete, and this arm asks whether it survives it.

Every arm reports the size of the prompt it sends, because compaction that costs
quality is not compaction and compaction that is not measured is not a claim.

Usage (from approach/), one model per invocation:

    OPENAI_API_KEY="$OAI_KEY" LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.6-terra \
    OPENAI_SERVICE_TIER=flex OPENAI_REASONING_EFFORT=none \
      ../.venv/bin/python pilot/typed_prompt_pilots.py \
        --group fullname --model terra --runs 3
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
os.environ.setdefault("PHASE_CACHE_DIR", "/tmp/typed_prompt_pilots_cache")

from llm_sad_sam.core.data_types_v2 import SadSamLink                  # noqa: E402
from llm_sad_sam.core.document_loader_v2 import (                      # noqa: E402
    build_sent_map, load_sentences,
)
from llm_sad_sam.llm_client import LLMBackend                          # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository             # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker85 as L85         # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker85 import SLinker85      # noqa: E402

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

#: The recorded runs each model's stages are replayed against. Their `s_linker85`
#: checkpoints supply the aliases every arm starts from and the other two stages
#: the arm's own stage is composed with.
RECORDED = {
    "terra": ("s85_e2e_terra_r*_20260820", "s_linker85"),
    "luna": ("s85_e2e_luna_r*_20260820", "s_linker85"),
}

ORIG_VALIDATION = SLinker85._prompt_validation
ORIG_ALIAS_JUDGE_RULES = L85.DOC_KNOWLEDGE_JUDGE_RULES
ORIG_FOCUS = L85.VALIDATION_FOCUS
ORIG_COREF_RULES = L85.COREF_RULES

#: The resolver prompt is the module's largest instruction item -- 40 of the ~82 calls
#: a five-project run makes, 190 kB of prompt. Its preamble already states the question
#: ("identify any pronoun or noun phrase in THAT sentence that refers back to a
#: component listed above", "if a target sentence has no such reference ... return no
#: resolution", "be conservative -- only include resolutions you are CERTAIN about"),
#: and `COREF_RULES` then restates it. s56 measured deleting the whole preamble at
#: TP -16.2 because it is also the input-format contract; this arm deletes the
#: RESTATEMENT instead and keeps the contract, which is the untried half.
COREF_RULES_DEDUP = (
    "Resolve when the surrounding sentences make one component the clear antecedent, "
    "under any form the document uses for it. Avoid resolving when two or more equally "
    "plausible antecedents exist."
)
ORIG_ENTITY_RULES = L85.ENTITY_EXTRACTION_RULES
ORIG_LAYERED = L85.LAYERED_ENTITY_RULES

# ── the single points ────────────────────────────────────────────────────────

#: The sentence the audit reads as inert.
DEAD_SENTENCE = ("A mention that says nothing further about the component "
                 "still counts as a valid link. ")

#: The morphology clause, and the extraction rules without it.
MORPH_CLAUSE = ("; count a name written with different spacing, hyphenation or "
                "compound joining as that name")

NODEAD_LAYERED = ORIG_LAYERED.replace(DEAD_SENTENCE, "")
NOMORPH_ENTITY_RULES = ORIG_ENTITY_RULES.replace(MORPH_CLAUSE, "")

# ── the typed rubrics ────────────────────────────────────────────────────────

#: The four ways a case can go, for a sentence that writes the name. The first
#: approves; the other three are the reject-grounds LAYERED_ENTITY_RULES states as
#: prose, QUALIFIED_CLAUSE states as a sentence about identifiers, and
#: STRICTER_CLAUSE states as a sentence about ordinary English use.
TYPED_ENTITY_RUBRIC = """Give each case exactly one verdict:

  NAMES      the sentence writes this component's name and states something of the
             component as part of the system. Capitalization is evidence for this
             reading and its absence is evidence against, but neither settles it.
  NO_CLAIM   the sentence writes the name but asserts nothing of the component.
  OTHER_JOB  the word is used in its ordinary English sense and the component is not
             what the sentence is talking about, or it occurs only as part of a
             longer joined or dotted identifier, where it names a piece of that
             identifier rather than a participant in what the sentence describes.
  DENIED     the sentence denies of this component what it would otherwise say of it."""

#: The same closed set, resolved the other way at NO_CLAIM.
TYPED_COREF_RUBRIC = """These are coreference links: a pronoun or noun phrase in the sentence is claimed to
refer back to the component, which is NOT named in the sentence itself.

Give each case exactly one verdict:

  REFERS     the sentence contains a genuine referring expression that unambiguously
             points to THIS component and makes an architectural claim about it.
  AMBIGUOUS  there is a referring expression, but its antecedent could equally be a
             different component.
  ARTIFACT   the expression denotes what a component acts on or produces -- the data,
             the artifact, the request, the result -- and so refers to that thing and
             not to the component, however clearly the component is the one acting.
  NONE       the sentence contains no referring expression pointing to this component."""


#: The lenient default, restated for a typed rubric. `LAYERED_ENTITY_RULES` opens with
#: it ("Approve the link by default ... Reject only on a positive ground"), and the
#: first typed arm dropped it: naming three reject types with no default invites the
#: judge to reach for one. This line puts the default back without the prose.
ENTITY_DEFAULT = ("The default verdict is NAMES. Reach for one of the other three only "
                  "when the sentence positively shows it; an objection you could raise "
                  "against most sentences is not one of them.")


#: The strict default, restated for a typed rubric. `LAYERED_COREF_RULES` ends with
#: "When uncertain, reject", and the first typed coreference arm dropped it: naming
#: one approving type and three rejecting ones with no default made the judge far more
#: permissive (gold 33.3 -> 62.7 per five-project run on terra, spurious 3.7 -> 12.0).
#: The alias judge, typed. `DOC_KNOWLEDGE_JUDGE_RULES` states two invalidity grounds
#: and a lenient default in prose; the types name the same three outcomes. The judge's
#: leniency is load-bearing (removing the judge entirely reads F1 94.57 against 96.42,
#: and the general round found the table collapses without "prefer APPROVE"), so the
#: default is restated inside the typed rubric from the start rather than dropped and
#: measured -- the fullname arms already priced dropping a default.
TYPED_ALIAS_RULES = """Give each proposed mapping exactly one verdict:

  ALIAS    the document establishes an equivalence between the phrase and that one
           named component.
  GENERIC  the phrase is ordinary vocabulary whose general use dominates.
  OTHER    the phrase identifies something other than that one component.

Approve exactly ALIAS. When uncertain between ALIAS and another verdict, choose ALIAS.
Return the ALIAS mappings only, each carrying its verdict."""

COREF_DEFAULT = ("When uncertain, the verdict is not REFERS: the name is absent from "
                 "the sentence, so a resolution that is merely plausible is one of the "
                 "other three.")


def typed_prompt(approve_types, rubric, strict_side, default=""):
    """A `_prompt_validation` that asks for a verdict from a closed set.

    ``strict_side`` says which of the two rubrics this replaces; the other side is
    left to `s_linker85`, so an arm changes one judge and not both.
    """
    approving = ", ".join(approve_types)

    def build(comp_names, cases, focus, strict: bool = False) -> str:
        if strict != strict_side:
            return ORIG_VALIDATION(comp_names, cases, focus, strict=strict)
        return f"""Validate components in a document. {focus}

COMPONENTS: {', '.join(comp_names)}

{rubric}
{default}
For each case, first quote the EXACT words from the sentence that decide the verdict
(or "none" if the sentence states nothing of the component), then give the verdict,
then approve true for {approving} and false for every other verdict.

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "claim": "<exact quote or none>", "verdict": "<one of the verdicts above>", "approve": true}}]}}
JSON only:"""

    return build


ARMS = {
    # group: {arm: {what it patches}}
    "fullname": {
        "ctl": {},
        "nodead": {"layered": NODEAD_LAYERED},
        "typed": {"validation": typed_prompt(
            ["NAMES"], TYPED_ENTITY_RUBRIC, strict_side=False)},
        "typedlenient": {"validation": typed_prompt(
            ["NAMES", "NO_CLAIM"], TYPED_ENTITY_RUBRIC, strict_side=False)},
    },
    # The three ways a typed rubric can resolve the contradiction, in one invocation.
    "fullname2": {
        "ctl": {},
        "typeddefault": {"validation": typed_prompt(
            ["NAMES"], TYPED_ENTITY_RUBRIC, strict_side=False,
            default="\n" + ENTITY_DEFAULT)},
        "typedlenient": {"validation": typed_prompt(
            ["NAMES", "NO_CLAIM"], TYPED_ENTITY_RUBRIC, strict_side=False)},
    },
    "coref2": {
        "ctl": {},
        "typedcorefstrict": {"validation": typed_prompt(
            ["REFERS"], TYPED_COREF_RUBRIC, strict_side=True,
            default="\n" + COREF_DEFAULT)},
    },
    # The compaction the typed arms did not buy: drop what the rubric already says.
    # `VALIDATION_FOCUS` asks for architectural participation and referential
    # specificity; `LAYERED_ENTITY_RULES` states the first as its approve-condition
    # and `STRICTER_CLAUSE` states the second as its whole subject. The finetune
    # round measured exactly this shape once before -- dropping the full-name focus
    # tail with nothing added read TP +2.3 / FP +/-0.0.
    "fullname3": {
        "ctl": {},
        "nofocus": {"focus": ""},
        "compact": {"focus": "", "layered": NODEAD_LAYERED},
    },
    "resolve": {
        "ctl": {},
        "dedup": {"coref_rules": COREF_RULES_DEDUP},
    },
    "alias": {
        "ctl": {},
        "typedalias": {"alias_rules": TYPED_ALIAS_RULES},
    },
    "extract": {
        "ctl": {},
        "nomorph": {"entity_rules": NOMORPH_ENTITY_RULES},
    },
    "coref": {
        "ctl": {},
        "typedcoref": {"validation": typed_prompt(
            ["REFERS"], TYPED_COREF_RUBRIC, strict_side=True)},
    },
}


class Arm:
    """The patches an arm applies, applied for the duration of its stage."""

    def __init__(self, name, spec):
        self.name, self.spec = name, spec

    def __enter__(self):
        L85.DOC_KNOWLEDGE_JUDGE_RULES = self.spec.get(
            "alias_rules") or ORIG_ALIAS_JUDGE_RULES
        L85.LAYERED_ENTITY_RULES = self.spec.get("layered", ORIG_LAYERED)
        L85.VALIDATION_FOCUS = self.spec.get("focus", ORIG_FOCUS)
        L85.COREF_RULES = self.spec.get("coref_rules", ORIG_COREF_RULES)
        L85.ENTITY_EXTRACTION_RULES = self.spec.get(
            "entity_rules", ORIG_ENTITY_RULES)
        SLinker85._prompt_validation = staticmethod(
            self.spec.get("validation", ORIG_VALIDATION))
        return self

    def __exit__(self, *exc):
        L85.DOC_KNOWLEDGE_JUDGE_RULES = ORIG_ALIAS_JUDGE_RULES
        L85.LAYERED_ENTITY_RULES = ORIG_LAYERED
        L85.VALIDATION_FOCUS = ORIG_FOCUS
        L85.COREF_RULES = ORIG_COREF_RULES
        L85.ENTITY_EXTRACTION_RULES = ORIG_ENTITY_RULES
        SLinker85._prompt_validation = staticmethod(ORIG_VALIDATION)


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


def linker_for(proj_model, recorded_knowledge, sink):
    lk = SLinker85(backend=LLMBackend.OPENAI)
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


def run_group(group, model, runs, out_dir):
    run_dirs, variant = recorded_runs(model)
    if not run_dirs:
        sys.exit(f"no recorded runs for {model}: {RECORDED[model][0]}")
    arms = ARMS[group]
    kept = {arm: collections.defaultdict(dict) for arm in arms}
    stage_tot = {arm: collections.Counter() for arm in arms}
    prompt_chars = {arm: [] for arm in arms}
    aliases = {arm: [] for arm in arms}
    calls_csv: list[dict] = []
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

            shared_candidates = None
            if group.startswith("fullname"):
                sink = []
                lk = linker_for(proj, knowledge, sink)
                with Arm("ctl", {}):
                    shared_candidates = lk._extract_named_mentions(
                        sents, comps, name_to_id, sent_map)
                calls["extract"] += len(sink)

            for arm, spec in arms.items():
                sink = []
                lk = linker_for(proj, knowledge, sink)
                with Arm(arm, spec):
                    if group.startswith("fullname"):
                        cands = list(shared_candidates.values())
                        pairs = judge_fullname(lk, cands, comps, sent_map)
                    elif group == "resolve":
                        # the resolver AND the judge behind it: what a resolver arm
                        # proposes is only a link if the strict gate keeps it
                        links, _ = lk._run_coreference_linker(
                            sents, comps, name_to_id, sent_map)
                        pairs = {(l.sentence_number, l.component_id) for l in links}
                    elif group == "alias":
                        lk.doc_knowledge = lk._learn_document_knowledge(
                            sents, comps)
                        links, _ = lk._run_full_name_linker(
                            sents, comps, name_to_id, sent_map)
                        pairs = {(l.sentence_number, l.component_id) for l in links}
                        aliases[arm].append(len(lk.doc_knowledge.aliases))
                    elif group == "extract":
                        links, _ = lk._run_full_name_linker(
                            sents, comps, name_to_id, sent_map)
                        pairs = {(l.sentence_number, l.component_id) for l in links}
                    else:
                        pairs = judge_coref(lk, run_dir, variant, proj,
                                            comps, sent_map)
                prompt_chars[arm] += [c["chars"] for c in sink]
                calls[arm] += len(sink)
                tg = len(pairs & g)
                stage_tot[arm]["g"] += tg
                stage_tot[arm]["n"] += len(pairs) - tg
                kept[arm][f"run{r + 1}"][proj] = sorted(list(x) for x in pairs)
                print(f"  run{r + 1} {proj:<14} {arm:<13} "
                      f"{tg:3d}g/{len(pairs) - tg:3d}n", flush=True)
                # One exemplar prompt per (group, model, arm) and a row per call:
                # the full 540-file trace is 17 MB of repeated document text, and what
                # the round rests on is which prompt each arm sent and how big it was.
                calls_csv.extend(
                    {"group": group, "model": model, "arm": arm, "run": r + 1,
                     "project": proj, "label": c.get("label", ""),
                     "chars": c["chars"]} for c in sink)
                exemplar = out_dir / "prompts" / f"{group}_{model}_{arm}.txt"
                if sink and not exemplar.exists():
                    exemplar.parent.mkdir(exist_ok=True)
                    longest = max(sink, key=lambda c: c["chars"])
                    exemplar.write_text(
                        f"# one prompt this arm sent: {group} / {model} / {arm} / "
                        f"run {r + 1} / {proj} / {longest['chars']} chars\n\n"
                        + longest["prompt"])

    for arm in arms:
        json.dump(kept[arm], open(out_dir / f"kept_{group}_{model}_{arm}.json", "w"))
    if calls_csv:
        summary = out_dir / "calls_summary.csv"
        with open(summary, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(calls_csv[0]))
            if summary.stat().st_size == 0:
                writer.writeheader()
            writer.writerows(calls_csv)

    print(f"\n{group} on {model}, {runs} runs, per five-project run:")
    for arm in arms:
        chars = st.mean(prompt_chars[arm]) if prompt_chars[arm] else 0
        if aliases[arm]:
            print(f"  {arm:<13} alias terms per five-project run "
                  f"{sum(aliases[arm]) / runs:6.1f}")
        print(f"  {arm:<13} gold {stage_tot[arm]['g'] / runs:6.1f}  "
              f"spurious {stage_tot[arm]['n'] / runs:6.1f}  "
              f"calls {calls[arm] / runs:5.1f}  mean prompt {chars:7.0f} chars")
    compose(group, model, runs, arms, kept, variant, run_dirs)


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


#: Which recorded stages an arm's own stage is composed with.
OTHER_STAGES = {
    "fullname": ("partial_name", "coreference"),
    "fullname2": ("partial_name", "coreference"),
    "extract": ("partial_name", "coreference"),
    "coref": ("full_name", "partial_name"),
    "coref2": ("full_name", "partial_name"),
    "alias": ("partial_name", "coreference"),
    "resolve": ("full_name", "partial_name"),
    "fullname3": ("partial_name", "coreference"),
}


def compose(group, model, runs, arms, kept, variant, run_dirs):
    """The exact pipeline score: the arm's kept pairs, unioned with the same
    recorded run's other two stages. Not a projection -- the other stages are
    untouched by any of these prompts."""
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
        print(f"  {arm:<13} macroF1 {st.mean(x[0] for x in per):6.2f}  "
              f"macroF2 {st.mean(x[1] for x in per):6.2f}  "
              f"TP {st.mean(x[2] for x in per):6.1f}  "
              f"FP {st.mean(x[3] for x in per):6.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=sorted(ARMS), required=True)
    ap.add_argument("--model", choices=sorted(RECORDED), required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--projects", nargs="*", default=None,
                    help="restrict to these projects (smoke tests)")
    ap.add_argument("--out", default=os.environ.get(
        "AB_OUT", "../results/typed_round"))
    args = ap.parse_args()
    if args.projects:
        for name in list(PROJECTS):
            if name not in args.projects:
                del PROJECTS[name]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_group(args.group, args.model, args.runs, out_dir)


if __name__ == "__main__":
    main()
