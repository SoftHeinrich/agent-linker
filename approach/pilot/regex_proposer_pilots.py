"""Level 2: put the scan in front of the head's own full-name gate and see.

`pilot/regex_extract_audit.py` settled the proposer question off the recorded runs --
the scan reaches 7.8 net gold pairs a run more than the LLM extraction pass -- and
left exactly one thing open: what the lenient gate does with the 53.3 pairs a run the
scan adds and the extractor never proposed. The audit could only bracket it (every
such pair rejected, then every one approved). This arm asks the gate.

Four arms in one invocation, so every comparison below is in-set:

    ctl     the extraction pass the recorded run actually made, replayed as a fixed
            input. Costs no extraction call.
    scan    `s_linker92a` -- every pair whose sentence writes a name of the component,
            at `ANY_CASE`, over the catalog and that same run's recorded aliases.
    scan+e  the same candidates, judged by `s_linker92e`: the reply names the case's
            own surface before the claim and the verdict.
    scan+f  the same candidates, judged by `s_linker92f`: the reply lists the readings
            that surface could have here and names the one it has, then decides.

The two judge arms add no rule text and no code gate -- only the order the reply is
written in, which is `s_linker106`'s mechanism at a different question. Their strict
branch renders `s_linker92`'s bytes exactly, so the coreference judge is untouched.

and one question decided inside the `scan` arm rather than by a second variant:
**does the judge do `QUALIFIED_CLAUSE`'s job?**  The clause tells it that an
expression occurring only inside a longer dotted identifier names a piece of that
identifier, not a participant. 20.8 of the scan's added pairs a run are exactly that
case. `s_linker92b` exists to not propose them; if the gate rejects them anyway, that
variant is a code gate restating a clause that works, and the simplest arm wins.
So the arm reports its verdicts split by subpopulation:

    shared      the pair was also in the recorded extraction
    added/qual  only the scan proposes it, and every writing of the name in that
                sentence is inside a dotted identifier
    added/plain the rest

Usage (from approach/), one model per invocation:

    OPENAI_API_KEY="$OAI_KEY" LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.6-terra \
    OPENAI_SERVICE_TIER=flex OPENAI_ENFORCE_FLEX=1 OPENAI_REASONING_EFFORT=none \
      ../.venv/bin/python pilot/regex_proposer_pilots.py --model terra --runs 3
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PHASE_CACHE_DIR", "/tmp/regex_proposer_pilots_cache")

from llm_sad_sam.core.document_loader_v2 import (                     # noqa: E402
    build_sent_map, load_sentences)
from llm_sad_sam.llm_client import LLMBackend                         # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository            # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker92 import SLinker92     # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker92a import SLinker92a   # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker92e import SLinker92e   # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker92f import SLinker92f   # noqa: E402

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

#: The recorded runs the arms replay. Both variants that wrote these directories share
#: the extraction pass byte for byte, so the recorded candidates are one extractor's.
RECORDED = {
    "terra": ("solo_e2e_terra_r*_20260821", "s_linker89"),
    "luna": ("solo_e2e_luna_r*_20260821", "s_linker89"),
}

#: The stages this arm does not touch, taken from the same recorded run.
OTHER_STAGES = ("partial_name", "coreference")

#: arm -> (where its candidates come from, whose judge reads them). Every arm judges
#: in the same invocation, so the comparison is in-set.
ARM_SPEC = {
    "ctl": ("recorded", SLinker92),
    "scan": ("scan", SLinker92a),
    "scan+e": ("scan", SLinker92e),
    "scan+f": ("scan", SLinker92f),
}
ARMS = tuple(ARM_SPEC)


def gold(path):
    with open(os.path.join(BASE, "benchmark", path)) as handle:
        return {(int(r["sentence"]), r["modelElementID"])
                for r in csv.DictReader(handle)}


def recorded_runs(model):
    pattern, variant = RECORDED[model]
    return sorted(glob.glob(os.path.join(BASE, "results", pattern,
                                         "phase_states"))), variant


def state(run_dir, variant, project, phase):
    path = os.path.join(run_dir, variant, "openai", project, f"{phase}.pkl")
    return pickle.load(open(path, "rb")) if os.path.exists(path) else None


def scores(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = (5 * precision * recall / (4 * precision + recall)
          if 4 * precision + recall else 0.0)
    return f1 * 100, f2 * 100


def linker_for(cls, knowledge, sink):
    linker = cls(backend=LLMBackend.OPENAI)
    linker.doc_knowledge = knowledge["doc_knowledge"] if knowledge else None
    inner = linker._ask

    def ask(prompt, **kwargs):
        out = inner(prompt, **kwargs)
        sink.append({"chars": len(prompt), "label": kwargs.get("label", "")})
        return out
    linker._ask = ask
    return linker


def judge(linker, candidates, components, sent_map):
    """The head's full-name gate over these candidates. Returns (kept, decisions)."""
    bundles = {(c.sentence_number, c.component_id):
               linker._build_evidence_bundle(c, sent_map) for c in candidates}
    approved, decisions = linker._validate_with_evidence(
        candidates, bundles, components, sent_map,
        phase_tag="pilot_regex_full_name_judge", stage_label="full_name")
    return ({(c.sentence_number, c.component_id) for c in approved}, decisions)


def only_qualified(linker, text, comp_name, aliases):
    """True when every writing of any name of this component here is inside a path.

    The population `QUALIFIED_CLAUSE` speaks about, and the one `s_linker92b` would
    not have proposed. Uses the head's own predicates, not a second copy of them.
    """
    spans = []
    for name in (comp_name, *aliases.get(comp_name, ())):
        spans += linker._named_spans(text, name)
    return bool(spans) and all(linker._in_dotted_path(text, s, e) for s, e in spans)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(RECORDED), required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--projects", nargs="*", default=None)
    parser.add_argument("--arms", nargs="*", default=list(ARM_SPEC))
    parser.add_argument("--out", default="../results/regex_round")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs, variant = recorded_runs(args.model)
    if not run_dirs:
        sys.exit(f"no recorded runs for {args.model}: {RECORDED[args.model][0]}")
    projects = {k: v for k, v in PROJECTS.items()
                if not args.projects or k in args.projects}

    arms = [a for a in ARM_SPEC if a in args.arms]
    kept = {arm: collections.defaultdict(dict) for arm in arms}
    stage = {arm: collections.Counter() for arm in arms}
    calls = collections.Counter()
    proposed = collections.Counter()
    # the question s_linker92b exists to answer
    split = collections.defaultdict(collections.Counter)

    for run in range(args.runs):
        run_dir = run_dirs[run % len(run_dirs)]
        for project, (text, model_path, gold_path) in projects.items():
            components = parse_pcm_repository(
                os.path.join(BASE, "benchmark", model_path))
            sentences = load_sentences(os.path.join(BASE, "benchmark", text))
            sent_map = build_sent_map(sentences)
            name_to_id = {c.name: c.id for c in components}
            truth = gold(gold_path)
            knowledge = state(run_dir, variant, project, "knowledge")
            recorded = state(run_dir, variant, project, "linker_full_name")

            # ── ctl: the recorded extraction, rebuilt as candidates ──────────
            from llm_sad_sam.core.data_types_v2 import CandidateLink
            control = {}
            for item in recorded["feedback"]["candidates"]:
                if item["component"] not in name_to_id:
                    continue
                control[(item["sentence"], name_to_id[item["component"]])] = (
                    CandidateLink(item["sentence"], item["text"], item["component"],
                                  name_to_id[item["component"]], "",
                                  source="full_name"))

            # ── scan: s_linker92a's proposer, same aliases ───────────────────
            sink = []
            scanner = linker_for(SLinker92a, knowledge, sink)
            scanned = scanner._extract_named_mentions(
                sentences, components, name_to_id, sent_map)
            aliases = scanner._names_by_component()

            for arm in arms:
                source, judge_cls = ARM_SPEC[arm]
                candidates = control if source == "recorded" else scanned
                sink = []
                linker = linker_for(judge_cls, knowledge, sink)
                pairs, decisions = judge(linker, list(candidates.values()),
                                         components, sent_map)
                calls[arm] += len(sink)
                proposed[arm] += len(candidates)
                hit = len(pairs & truth)
                stage[arm]["g"] += hit
                stage[arm]["n"] += len(pairs) - hit
                kept[arm][f"run{run + 1}"][project] = sorted(
                    list(x) for x in pairs)
                print(f"  run{run + 1} {project:<14} {arm:<5} "
                      f"{len(candidates):3d} cand -> {hit:3d}g/"
                      f"{len(pairs) - hit:3d}n", flush=True)

                if arm != "scan":
                    continue
                for key, candidate in candidates.items():
                    if key in control:
                        bucket = "shared"
                    elif only_qualified(scanner, candidate.sentence_text,
                                        candidate.component_name, aliases):
                        bucket = "added/qual"
                    else:
                        bucket = "added/plain"
                    split[bucket]["n"] += 1
                    split[bucket]["gold"] += key in truth
                    if key in pairs:
                        split[bucket]["approved"] += 1
                        split[bucket]["approved_gold"] += key in truth

    for arm in arms:
        (out_dir / f"kept_{args.model}_{arm}.json").write_text(json.dumps(kept[arm]))

    print(f"\nstage, {args.model}, {args.runs} runs, per five-project run:")
    for arm in arms:
        print(f"  {arm:<5} proposed {proposed[arm] / args.runs:6.1f}  "
              f"gold {stage[arm]['g'] / args.runs:6.1f}  "
              f"spurious {stage[arm]['n'] / args.runs:6.1f}  "
              f"judge calls {calls[arm] / args.runs:5.1f}")

    print(f"\nthe scan's candidates by subpopulation, per five-project run "
          f"(does the gate do QUALIFIED_CLAUSE's job?):")
    print(f"  {'bucket':<12}{'cand':>7}{'gold':>7}{'appr':>7}{'appr gold':>11}"
          f"{'appr rate':>11}")
    for bucket in ("shared", "added/plain", "added/qual"):
        row = split[bucket]
        n = row["n"] / args.runs
        print(f"  {bucket:<12}{n:>7.1f}{row['gold'] / args.runs:>7.1f}"
              f"{row['approved'] / args.runs:>7.1f}"
              f"{row['approved_gold'] / args.runs:>11.1f}"
              f"{(row['approved'] / row['n'] if row['n'] else 0):>11.3f}")

    print(f"\ncomposed with the recorded {' + '.join(OTHER_STAGES)} of the same run:")
    for arm in arms:
        per = []
        for run in range(args.runs):
            run_dir = run_dirs[run % len(run_dirs)]
            f1s, f2s, tp_total, fp_total = [], [], 0, 0
            for project, (_t, _m, gold_path) in projects.items():
                truth = gold(gold_path)
                links = {tuple(x) for x in kept[arm][f"run{run + 1}"].get(project, [])}
                for other in OTHER_STAGES:
                    recorded = state(run_dir, variant, project, f"linker_{other}")
                    if recorded:
                        links |= {(l.sentence_number, l.component_id)
                                  for l in recorded["links"]}
                tp = len(links & truth)
                tp_total += tp
                fp_total += len(links) - tp
                f1, f2 = scores(tp, len(links) - tp, len(truth) - tp)
                f1s.append(f1)
                f2s.append(f2)
            per.append((st.mean(f1s), st.mean(f2s), tp_total, fp_total))
        print(f"  {arm:<5} macroF1 {st.mean(x[0] for x in per):6.2f}  "
              f"macroF2 {st.mean(x[1] for x in per):6.2f}  "
              f"TP {st.mean(x[2] for x in per):6.1f}  "
              f"FP {st.mean(x[3] for x in per):6.1f}")


if __name__ == "__main__":
    main()
