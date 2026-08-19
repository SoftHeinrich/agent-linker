"""Stage A/B of the partial-name proposer's two boundary repairs, judged for real.

`pilot/partial_screen.py` prices both repairs deterministically over all five documents
(base 60.3 candidates / 18.7 gold per run):

    guard  the `"" in "-_"` bug in `_inside_qualified_identifier`   +0.0 gold, +1.0 spurious
    exact  an exact word match outranks a prefix-only one          +2.0 gold, +1.0 spurious
    infl   a prefix match must be an English inflection            +2.0 gold, +0.0 spurious

`infl` dominates `exact` -- same two gold candidates, and it also drops the one spurious
candidate `webcams -> BBB web`, because `rtc` and `cams` are not inflections of `web`
while `webrtc` is an exact name word of `WebRTC-SFU`.

A candidate is not a link, though: the denotation judge sees it next, and on the current
mix it approves 95% of gold candidates and 17% of the rest. So this runs the real judge,
unchanged, over each arm's candidate set, `AB_RUNS` samples a side, and reports the
approved (sentence, component) pairs through the same permutation test every arm in this
branch uses.

    AB_RUNS=5 ../.venv/bin/python pilot/partial_pilots.py --pilot proposer
    AB_RUNS=5 ../.venv/bin/python pilot/partial_pilots.py --pilot guard
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

import design_pilots                                                # noqa: E402
from ab_stats import permutation_report                             # noqa: E402
from design_audit import PROJECTS, load_gold, load_project          # noqa: E402
from design_pilots import RUNS, collect, report                     # noqa: E402
from partial_screen import Probe, project_cache                     # noqa: E402
from llm_sad_sam.llm_client import LLMBackend                       # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker59 import (           # noqa: E402
    CandidateLink, SLinker59)

MODEL = os.environ.get("OPENAI_MODEL_NAME", "gpt-5.6-terra")
OUT = Path(os.environ.get("AB_OUT", "../results/partial_name_pilots"))
SOURCE_RUN = Path(os.environ.get(
    "AB_SOURCE_RUN", "../results/s5960_e2e_r1_20260813"))
SOURCE_VARIANT = os.environ.get("AB_SOURCE_VARIANT", "s_linker49")

design_pilots.OUT = OUT


def phase(project, name):
    path = (SOURCE_RUN / "phase_states" / SOURCE_VARIANT / "openai" / project
            / f"{name}.pkl")
    with path.open("rb") as handle:
        return pickle.load(handle)


_INPUTS: dict = {}


def inputs(project):
    """The upstream state this stage consumes, frozen at one recorded realization."""
    if project not in _INPUTS:
        info = load_project(project)
        info["gold"] = load_gold(project)
        info["knowledge"] = phase(project, "knowledge")["doc_knowledge"]
        info["linked"] = {(l.sentence_number, l.component_id)
                          for l in phase(project, "linker_full_name")["links"]}
        _INPUTS[project] = info
    return _INPUTS[project]


def judge(project, variant):
    """One sample: build this arm's candidates, run the real denotation judge."""
    info = inputs(project)
    sentences, components = project_cache(project)
    probe = Probe(info["knowledge"].aliases)
    keys = {k: v for k, v in probe.candidates(sentences, components, variant).items()
            if k not in info["linked"]}
    if not keys:
        return set()
    by_number = {s.number: s for s in sentences}
    by_id = {c.id: c for c in components}
    candidates = [
        CandidateLink(snum, by_number[snum].text, by_id[cid].name, cid, text,
                      source="partial_name_candidate")
        for (snum, cid), text in sorted(keys.items())
    ]
    linker = SLinker59(backend=LLMBackend.OPENAI, model=MODEL)
    linker.doc_knowledge = info["knowledge"]
    linker.model_knowledge = None
    approved, _ = linker._judge_partial_names(candidates, sentences)
    return {(project, (c.sentence_number, c.component_id)) for c in approved}


def scorers():
    gold = {p: load_gold(p) for p in PROJECTS}

    def tp(pairs):
        return sum(1 for project, key in pairs if key in gold[project])

    def fp(pairs):
        return len(pairs) - tp(pairs)

    return {"TP": tp, "FP": fp}


def arms_pilot(name, arms, projects):
    sets = {}
    for label, variant in arms.items():
        sets[label] = collect(label, lambda run, project, v=variant: judge(project, v),
                              projects)
        print(f"  {label:28s} approved pairs per sample: "
              f"{[len(s) for s in sets[label]]}")
    stats = permutation_report(sets, quality=scorers(),
                               title=f"{name}: partial-name stage")
    report(name, stats)
    return stats


def stated_name_net(project, case_sensitive=True):
    """(sentence, component) pairs stating the component's model name that the
    extraction call did not propose and the full-name linker did not link.

    `pilot/statednet_screen.py` prices the three readings of "states the name" over
    all six runs: with the discovered aliases 41.2 new pairs/run at 0.07 gold each,
    with the model name case-insensitively 31.3 at 0.06, and **case-sensitively 1.2 at
    0.86** -- level with the extractor's own 0.87. Case is what separates a proper
    noun from ordinary English, which is why the lenient primitive that serves every
    other site is the wrong one here.
    """
    info = inputs(project)
    sentences, components = project_cache(project)
    fn = phase(project, "linker_full_name")
    by_name = {c.name: c.id for c in components}
    proposed = {(int(r["sentence"]), by_name[r["component"]])
                for r in fn["feedback"]["candidates"] if r["component"] in by_name}
    flags = 0 if case_sensitive else re.IGNORECASE
    net = {(s.number, c.id) for s in sentences for c in components
           if re.search(rf"(?<!\w){re.escape(c.name)}(?!\w)", s.text, flags)}
    return net - proposed - info["linked"]


def full_name_judge(project, with_net):
    """One sample of the full-name two-pass judge, with and without the net."""
    info = inputs(project)
    sentences, components = project_cache(project)
    sent_map = {s.number: s.text for s in sentences}
    linker = SLinker59(backend=LLMBackend.OPENAI, model=MODEL)
    linker.doc_knowledge = info["knowledge"]
    linker.model_knowledge = None
    recorded = phase(project, "linker_full_name")["feedback"]["candidates"]
    by_name = {c.name: c.id for c in components}
    by_number = {s.number: s for s in sentences}
    by_id = {c.id: c.name for c in components}
    cands = []
    for item in recorded:
        cid, sentence = by_name.get(item["component"]), by_number.get(item["sentence"])
        if cid is None or sentence is None:
            continue
        cands.append(CandidateLink(
            item["sentence"], sentence.text, item["component"], cid,
            item.get("matched_text") or item["component"],
            source=item.get("source", "full_name_candidate"),
            mention_type=linker._classify_mention_typed(item["component"],
                                                       sentence.text)))
    if with_net:
        have = {(c.sentence_number, c.component_id) for c in cands}
        for snum, cid in sorted(stated_name_net(project)):
            if (snum, cid) in have:
                continue
            cands.append(CandidateLink(
                snum, by_number[snum].text, by_id[cid], cid, by_id[cid],
                source="stated_name_candidate",
                mention_type=linker._classify_mention_typed(by_id[cid],
                                                           by_number[snum].text)))
    real_sent_map = {s.number: s for s in sentences}
    bundles = {(c.sentence_number, c.component_id):
               linker._build_evidence_bundle(c, real_sent_map) for c in cands}
    approved, _ = linker._validate_with_evidence(
        cands, bundles, components, real_sent_map,
        "pilot_p1", "pilot_p2", "full_name_twopass")
    del sent_map
    return {(project, (c.sentence_number, c.component_id)) for c in approved}


PILOTS = {
    # Only the two projects whose partial-name linker fires can differ; the other
    # three propose nothing under any arm (`partial_screen.py`), so including them
    # would add samples that are identical by construction and shrink every p-value.
    "proposer": lambda: arms_pilot(
        "proposer", {"s59 (bare prefix)": "base",
                     "inflection-bounded": "infl"},
        ["teammates", "bigbluebutton"]),
    "guard": lambda: arms_pilot(
        "guard", {"inflection-bounded": "infl",
                  "+ boundary guard": "inflguard"},
        ["teammates", "bigbluebutton"]),
    "exact": lambda: arms_pilot(
        "exact", {"inflection-bounded": "infl",
                  "exact-beats-prefix": "exact"},
        ["teammates", "bigbluebutton"]),
    # The bucket the partial-name proposer defers to the full-name linker, screened at
    # the stage that owns it. Only the two projects whose net is non-empty are run.
    "statednet": lambda: _net_pilot(),
}


def _net_pilot():
    sets = {}
    projects = ["teammates", "bigbluebutton"]
    for label, with_net in (("s59 candidates", False), ("+ stated-name net", True)):
        sets[label] = collect(
            label, lambda run, project, w=with_net: full_name_judge(project, w),
            projects)
        print(f"  {label:24s} approved pairs per sample: "
              f"{[len(s) for s in sets[label]]}")
    stats = permutation_report(sets, quality=scorers(),
                               title="statednet: full-name stage")
    report("statednet", stats)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True, choices=sorted(PILOTS))
    args = ap.parse_args()
    print(f"\n{args.pilot}: {RUNS} samples a side, model {MODEL}, "
          f"upstream frozen at {SOURCE_RUN.name}/{SOURCE_VARIANT}\n")
    PILOTS[args.pilot]()


if __name__ == "__main__":
    main()
