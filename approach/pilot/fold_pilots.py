"""Fold a *gate* into the prompt of the judge that follows it.

The bind round relocated a rule that only proposed (`_keep_stated_names`, s66). What is
left in the deterministic layer that still *decides* is a handful of one-directional
gates: predicates that remove a case before any judge sees it. `pilot/bind_pilots.py`
measured two of them as straight deletions and one of those (`--pilot cutqualified`) cost
5.8 false positives. **Deletion is not the question this file asks.** Each gate here is
deleted *and* its content stated as a general instruction in the prompt of the judge that
would now have to make the call — the same three-arm shape as the bind round, so "the
judge can do this" is separable from "the gate was worth nothing".

The clauses are written as general guidance, not as descriptions of the predicate: no
catalog term, no benchmark-derived wording, nothing that names a shape observed in one
document (GATE-06).

Pilots:

    foldowner       `SCANS[name_word].unique_owner` (s_linker66:1490) into the
                    denotation prompt. The gate drops a surface owned by more than one
                    component. Deterministic screen: it frees 12.0 pairs per run and
                    **0.0 gold**, so deletion can only cost precision; the question is
                    whether a judge told to answer for one particular participant does
                    the same work. Note the constraint that shapes the clause: the
                    denotation judge is deliberately target-blind (it never sees the
                    component catalog, worth 12 FP), so the clause cannot mention
                    ownership of a name — only whether the expression picks out one
                    participant.
    foldqualified   `SCANS[name_word].skip_qualified` (:1485) into the same prompt.
                    Deleting it alone is TP +2.0 / **FP +5.8** (`cutqualified`); this is
                    the untried half.
    foldantecedent  the coreference antecedent gate (:1950) into the coreference judging
                    prompt. The gate accepts a resolution only when the antecedent
                    sentence itself states a name of the component; priced at 12 FP by
                    `pilot/gate_pilots.py`. Replayed on the resolutions recorded in the
                    run's call log, so the resolution calls are not paid for again.

Usage (from approach/):
    AB_RUNS=5 ../.venv/bin/python pilot/fold_pilots.py --pilot all
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from ab_stats import permutation_report                              # noqa: E402
import design_pilots                                                 # noqa: E402
from design_pilots import MODEL, RUNS, collect, load_gold, report     # noqa: E402
from bind_audit import PROJECTS, phase_state, project                 # noqa: E402
from bind_pilots import against_first, linker, pair_scorers           # noqa: E402
from llm_sad_sam.core.data_types_v2 import SadSamLink                 # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker66 as L66        # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker66 import (             # noqa: E402
    SCANS, SLinker66,
)

OUT = Path(os.environ.get("AB_OUT", "../results/fold_round"))
SOURCE_RUN = Path(os.environ.get(
    "AB_SOURCE_RUN", "../results/s6667_e2e_r1_20260817"))
SOURCE_VARIANT = os.environ.get("AB_SOURCE_VARIANT", "s_linker66")

design_pilots.OUT = OUT


# ── the clauses the gates become ─────────────────────────────────────────────

#: `unique_owner`, for a judge that must not be told which component is meant.
OWNER_CLAUSE = (
    "Answer participant only when the expression, as used in this sentence, picks out "
    "one particular participant. If the same wording would serve equally well for "
    "several different ones, answer associated."
)

#: `skip_qualified`, stated as what the expression is rather than where its characters sit.
QUALIFIED_CLAUSE = (
    "An expression that occurs only as part of a longer joined or dotted identifier is "
    "naming a piece of that identifier, not a participant in what the sentence "
    "describes."
)

#: the coreference antecedent gate.
ANTECEDENT_CLAUSE = (
    "Approve only when one of the sentences shown states the name of the component the "
    "reference is said to point at. A resolution no sentence names is not an "
    "architectural claim."
)


# ── arms ─────────────────────────────────────────────────────────────────────

def denotation_arm(name, *, extra="", scan_override=None):
    """An `SLinker66` whose denotation prompt carries `extra` and whose scan is patched.

    `_classify_denotations` builds its prompt inline, so the arm re-declares the method.
    `assert_builder_parity` renders both against real data with the LLM stubbed out and
    requires the un-swapped copy to be byte-identical to s66's.
    """
    clause = f"\n{extra}\n" if extra else ""

    def _classify_denotations(self, candidates, sentences):
        sent_map = {s.number: s for s in sentences}
        decisions = {}
        for _, batch in self._iter_batches(candidates, self.JUDGE_BATCH):
            evidence_ids = {
                sentence.number
                for candidate in batch
                for sentence in self._window(candidate.sentence_number, sentences)
            }
            sentence_table = [{"sentence": n, "text": sent_map[n].text}
                              for n in sorted(evidence_ids)]
            cases = [{"case": n, "source": c.sentence_number,
                      "expression": c.matched_text}
                     for n, c in enumerate(batch, 1)]
            prompt = f"""Classify what each expression itself denotes in its
local context: participant for a software participant, or associated for
something merely associated with software.
{clause}
SENTENCES
{json.dumps(sentence_table)}

CASES
{json.dumps(cases)}

Claim must be a contiguous exact substring of the source sentence.

JSON only:
{{"judgments":[{{"case":1,"denotation":"participant",
"claim":"exact source quote"}}]}}
"""
            data = self._ask(
                prompt, phase="phase_25_partial_denotation",
                require_present="judgments", label="Denotation", timeout=240,
            )
            for item in data.get("judgments", []):
                case_value = str(item.get("case", ""))
                if not case_value.isdigit():
                    continue
                number = int(case_value)
                if not 1 <= number <= len(batch):
                    continue
                candidate = batch[number - 1]
                claim = str(item.get("claim", "")).strip().strip("\"'“”‘’")
                denotation = str(item.get("denotation", "")).strip()
                valid = (denotation in {"participant", "associated"}
                         and bool(claim)
                         and claim.casefold() in candidate.sentence_text.casefold())
                decisions[(candidate.sentence_number, candidate.component_id)] = {
                    "approved": False, "requested_keep": False,
                    "evidence_valid": valid, "claim": claim,
                    "denotation": denotation, "alternative": "not reviewed",
                    "path": "denotation", "stage": "partial_name",
                }
        participants = [
            c for c in candidates
            if decisions.get((c.sentence_number, c.component_id), {}).get(
                "denotation") == "participant"
            and decisions[(c.sentence_number, c.component_id)]["evidence_valid"]
        ]
        return participants, decisions

    attrs = {"_VARIANT_NAME": f"s_linker66_{name}",
             "_classify_denotations": _classify_denotations}
    if scan_override is not None:
        attrs["SCAN"] = scan_override
    return type(f"Arm_{name}", (SLinker66,), attrs)


def coref_judge_arm(name, *, extra="", gate=True):
    """An `SLinker66` whose coreference judging prompt carries `extra`."""
    def _prompt_validation(comp_names, cases, focus, strict: bool = False) -> str:
        block = L66.LAYERED_COREF_RULES if strict else L66.LAYERED_ENTITY_RULES
        tail = f"\n{extra}\n" if (extra and strict) else ""
        return f"""Validate components in a document. {focus}

COMPONENTS: {', '.join(comp_names)}

{block}
{tail}
For each case, first quote the EXACT words from the sentence that state the
architectural claim about the component (or write "none" if the sentence makes no
such claim), then decide approve true/false based on that claim.

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "claim": "<exact quote or none>", "approve": true}}]}}
JSON only:"""

    return type(f"Arm_{name}", (SLinker66,),
                {"_VARIANT_NAME": f"s_linker66_{name}",
                 "_prompt_validation": staticmethod(_prompt_validation),
                 "GATE": gate})


# ── fixed inputs ─────────────────────────────────────────────────────────────

_INPUTS: dict = {}


def inputs(name):
    if name not in _INPUTS:
        info = dict(project(name))
        knowledge = phase_state(SOURCE_RUN, SOURCE_VARIANT, name, "knowledge")
        info["knowledge"] = knowledge["doc_knowledge"]
        _INPUTS[name] = info
    return _INPUTS[name]


def recorded_resolutions(name):
    """Every resolution the coreference call reported, BEFORE the antecedent gate.

    The checkpoint's candidate view is post-gate, so the dropped resolutions only exist
    in the call log — the same route `bind_audit.extractor_pairs` takes.
    """
    info = inputs(name)
    out = {}
    for path in (SOURCE_RUN / "llm_logs").glob(
            f"{SOURCE_VARIANT}_openai_{name}_*_calls.json"):
        with path.open() as handle:
            calls = json.load(handle)
        for call in calls:
            if call.get("phase") != "phase_25_coreference":
                continue
            body = (call.get("response_text") or "").strip()
            fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
            if fence:
                body = fence.group(1).strip()
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                continue
            for res in data.get("resolutions", []):
                comp = res.get("component")
                snum = res.get("sentence")
                ant = res.get("antecedent_sentence")
                try:
                    snum = int(str(snum).lstrip("Ss"))
                    ant = int(str(ant).lstrip("Ss"))
                except (TypeError, ValueError):
                    continue
                if comp not in info["name_to_id"] or snum not in info["sent_map"]:
                    continue
                if ant not in info["sent_map"]:
                    continue
                out[(snum, info["name_to_id"][comp])] = (comp, ant)
    return out


# ── pilots ───────────────────────────────────────────────────────────────────

def _denotation_pilot(name, arms):
    """Every arm proposes with its own scan and judges with its own prompt."""
    sets = {}

    def unit(cls, run, project_name):
        info = inputs(project_name)
        obj = linker(cls, knowledge=info["knowledge"])
        scan = getattr(cls, "SCAN", SCANS["name_word"])
        candidates = obj._scan(info["sentences"], info["components"], scan)
        approved, _ = obj._judge_partial_names(candidates, info["sentences"])
        return {(project_name, (c.sentence_number, c.component_id))
                for c in approved}

    for label, cls in arms.items():
        sets[label] = collect(label, lambda run, p, c=cls: unit(c, run, p), PROJECTS)
        print(f"  {label:38s} approved per run: {[len(s) for s in sets[label]]}")
    return against_first(sets, name)


def pilot_foldowner():
    return _denotation_pilot("foldowner", {
        "s66 (gate in code)": denotation_arm("owner_base"),
        "gate deleted": denotation_arm(
            "owner_off", scan_override=replace(SCANS["name_word"],
                                               unique_owner=False)),
        "gate folded into the prompt": denotation_arm(
            "owner_fold", extra=OWNER_CLAUSE,
            scan_override=replace(SCANS["name_word"], unique_owner=False)),
    })


def pilot_foldqualified():
    return _denotation_pilot("foldqualified", {
        "s66 (gate in code)": denotation_arm("qual_base"),
        "gate deleted": denotation_arm(
            "qual_off", scan_override=replace(SCANS["name_word"],
                                              skip_qualified=False)),
        "gate folded into the prompt": denotation_arm(
            "qual_fold", extra=QUALIFIED_CLAUSE,
            scan_override=replace(SCANS["name_word"], skip_qualified=False)),
    })


def pilot_foldantecedent():
    """The antecedent gate, on the resolutions the source run actually reported."""
    sets = {}
    arms = {
        "s66 (gate in code)": coref_judge_arm("ant_base", gate=True),
        "gate deleted": coref_judge_arm("ant_off", gate=False),
        "gate folded into the prompt": coref_judge_arm(
            "ant_fold", extra=ANTECEDENT_CLAUSE, gate=False),
    }

    def unit(cls, run, project_name):
        info = inputs(project_name)
        obj = linker(cls, knowledge=info["knowledge"])
        links = []
        for (snum, cid), (comp, ant) in recorded_resolutions(project_name).items():
            if cls.GATE and not obj._states_a_name(
                    info["sent_map"][ant].text, comp):
                continue
            links.append(SadSamLink(snum, cid, comp, source="coreference"))
        approved, _ = obj._validate_coref_links(
            links, info["sent_map"], info["components"])
        return {(project_name, (l.sentence_number, l.component_id))
                for l in approved}

    for label, cls in arms.items():
        sets[label] = collect(label, lambda run, p, c=cls: unit(c, run, p), PROJECTS)
        print(f"  {label:38s} approved per run: {[len(s) for s in sets[label]]}")
    return against_first(sets, "foldantecedent")


def stale_pairs(name):
    """Pairs the earlier linkers already produced in the source run.

    A working `_unlinked` would remove these from the coreference judge's input. They
    are in the union either way, so approving them changes no final link — but they do
    change which cases share a judging batch, and they inflate any TP count taken at
    this stage. `pilot/unlinked_audit.py` measures 65.0 of them per five-project run.
    """
    out = set()
    for phase in ("linker_full_name", "linker_partial_name"):
        state = phase_state(SOURCE_RUN, SOURCE_VARIANT, name, phase)
        if state:
            out |= {(l.sentence_number, l.component_id) for l in state["links"]}
    return out


def _coref_pilot(name, arms, drop_stale):
    """Replay the coreference gate and judge on the run's recorded resolutions."""
    sets = {}

    def unit(cls, run, project_name):
        info = inputs(project_name)
        obj = linker(cls, knowledge=info["knowledge"])
        stale = stale_pairs(project_name)
        links = []
        for (snum, cid), (comp, ant) in recorded_resolutions(project_name).items():
            if drop_stale and (snum, cid) in stale:
                continue
            if cls.GATE and not obj._states_a_name(info["sent_map"][ant].text, comp):
                continue
            links.append(SadSamLink(snum, cid, comp, source="coreference"))
        approved, _ = obj._validate_coref_links(
            links, info["sent_map"], info["components"])
        # score only what the coreference linker actually contributes: a pair an
        # earlier linker already produced is in the union regardless of this verdict
        return {(project_name, (l.sentence_number, l.component_id))
                for l in approved if (l.sentence_number, l.component_id) not in stale}

    for label, cls in arms.items():
        sets[label] = collect(label, lambda run, p, c=cls: unit(c, run, p), PROJECTS)
        print(f"  {label:38s} approved per run: {[len(s) for s in sets[label]]}")
    return against_first(sets, name)


def pilot_foldantecedent_net():
    """`foldantecedent`, scored on what coreference actually contributes.

    The first version of this pilot judged every recorded resolution and scored every
    approval, which counts pairs an earlier linker had already linked. Because
    `_unlinked` is inert (`pilot/unlinked_audit.py`), 65.0 such pairs per run reach this
    stage and 62.7 of them are gold, so a gate that admits more of them reads as a large
    true-positive gain that changes no final link. This version excludes them on both
    sides.
    """
    return _coref_pilot("foldantecedent_net", {
        "s66 (gate in code)": coref_judge_arm("antn_base", gate=True),
        "gate deleted": coref_judge_arm("antn_off", gate=False),
        "gate folded into the prompt": coref_judge_arm(
            "antn_fold", extra=ANTECEDENT_CLAUSE, gate=False),
    }, drop_stale=False)


def pilot_fixunlinked():
    """Repairing `_unlinked`: does removing the stale cases change the verdicts?

    Both arms run the same gate, the same prompt and the same judge; they differ only in
    whether the 65.0 already-linked pairs per run share the judging batches. Scored on
    the coreference linker's own contribution, so the stale approvals themselves are out
    of both sets and what is left is the batch-composition effect the `_unlinked`
    docstring claims (-6.8 FP, +0.8 TP).
    """
    sets = {}
    arms = {"as it runs (inert _unlinked)": False, "repaired _unlinked": True}

    def unit(drop, run, project_name):
        info = inputs(project_name)
        cls = coref_judge_arm("fixunlinked", gate=True)
        obj = linker(cls, knowledge=info["knowledge"])
        stale = stale_pairs(project_name)
        links = []
        for (snum, cid), (comp, ant) in recorded_resolutions(project_name).items():
            if drop and (snum, cid) in stale:
                continue
            if not obj._states_a_name(info["sent_map"][ant].text, comp):
                continue
            links.append(SadSamLink(snum, cid, comp, source="coreference"))
        approved, _ = obj._validate_coref_links(
            links, info["sent_map"], info["components"])
        return {(project_name, (l.sentence_number, l.component_id))
                for l in approved if (l.sentence_number, l.component_id) not in stale}

    for label, drop in arms.items():
        sets[label] = collect(label, lambda run, p, d=drop: unit(d, run, p), PROJECTS)
        print(f"  {label:38s} approved per run: {[len(s) for s in sets[label]]}")
    return against_first(sets, "fixunlinked")


#: `skip_stricter`, as what the judge should actually decide. The gate blocks the
#: ANY_CASE whole-name cell -- a sentence that writes the component's name in a
#: different case, which on one document is overwhelmingly the ordinary English word
#: (0.13 gold per pair against 0.96 for the as-spelled cell). The clause states the
#: distinction rather than the character rule: case is evidence, use as a name is the
#: question. No catalog term, no observed shape (GATE-06).
STRICTER_CLAUSE = (
    "Some sentences use an ordinary English word that happens to coincide with a "
    "component's name. Approve only when the sentence uses that word as the name of "
    "the component; if it is used in its ordinary sense and the component is not what "
    "the sentence is talking about, reject. Capitalization is evidence for a name and "
    "its absence is evidence against, but neither settles it on its own."
)


def fullname_judge_arm(name, *, extra=""):
    """An `SLinker69` whose FULL-NAME judging prompt carries `extra`."""
    from llm_sad_sam.linkers.experimental import s_linker69 as L69v
    from llm_sad_sam.linkers.experimental.s_linker69 import SLinker69
    def _prompt_validation(comp_names, cases, focus, strict: bool = False) -> str:
        block = L69v.LAYERED_COREF_RULES if strict else L69v.LAYERED_ENTITY_RULES
        tail = f"\n{extra}\n" if (extra and not strict) else ""
        return f"""Validate components in a document. {focus}

COMPONENTS: {', '.join(comp_names)}

{block}
{tail}
For each case, first quote the EXACT words from the sentence that state the
architectural claim about the component (or write "none" if the sentence makes no
such claim), then decide approve true/false based on that claim.

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "claim": "<exact quote or none>", "approve": true}}]}}
JSON only:"""

    return type(f"Arm_{name}", (SLinker69,),
                {"_VARIANT_NAME": f"s_linker69_{name}",
                 "_prompt_validation": staticmethod(_prompt_validation)})


def pilot_foldstricter():
    """`skip_stricter` into the full-name judging prompt.

    The gate keeps the ANY_CASE whole-name cell out of the candidate set: removing it
    adds 37.2 candidates per run at 0.13 gold per pair, almost all on one document. The
    open question is whether that is a property of the candidates or of the prompt --
    the judge's 98.7% approval rate on lowercase mentions is measured on the mix this
    gate already filtered, so it does not predict behaviour on candidates it has never
    been shown. Three arms: the gate in code, the gate deleted, the gate stated as the
    distinction the judge should be drawing.
    """
    from llm_sad_sam.core.data_types_v2 import CandidateLink
    from bind_audit import extractor_pairs
    import llm_sad_sam.linkers.experimental.s_linker69 as L

    sets = {}
    arms = {
        "s69 (gate in code)": (fullname_judge_arm("st_base"), True),
        "gate deleted": (fullname_judge_arm("st_off"), False),
        "gate folded into the prompt": (
            fullname_judge_arm("st_fold", extra=STRICTER_CLAUSE), False),
    }

    def unit(spec, run, project_name):
        cls, strict = spec
        info = inputs(project_name)
        obj = linker(cls, knowledge=info["knowledge"])
        names = {c.id: c.name for c in info["components"]}
        cands = [CandidateLink(s, info["sent_map"][s].text, names[cid], cid, "",
                               source="full_name")
                 for s, cid in extractor_pairs(SOURCE_RUN, SOURCE_VARIANT,
                                               project_name)]
        spelling = (L.SCANS["spelling"] if strict
                    else replace(L.SCANS["spelling"], skip_stricter=False))
        merged = {(c.sentence_number, c.component_id): c for c in cands}
        for scan in (spelling, L.SCANS["stated_name"]):
            for cand in obj._scan(info["sentences"], info["components"], scan):
                merged.setdefault((cand.sentence_number, cand.component_id), cand)
        cands = list(merged.values())
        bundles = {(c.sentence_number, c.component_id):
                   obj._build_evidence_bundle(c, info["sent_map"]) for c in cands}
        approved, _ = obj._validate_with_evidence(
            cands, bundles, info["components"], info["sent_map"],
            "pilot_p1", "pilot_p2", "full_name_twopass")
        return {(project_name, (c.sentence_number, c.component_id))
                for c in approved}

    for label, spec in arms.items():
        sets[label] = collect(label, lambda run, p, s=spec: unit(s, run, p), PROJECTS)
        print(f"  {label:38s} approved per run: {[len(s) for s in sets[label]]}")
    return against_first(sets, "foldstricter")


PILOTS = {
    "foldstricter": pilot_foldstricter,
    "foldantecedent_net": pilot_foldantecedent_net,
    "fixunlinked": pilot_fixunlinked,
    "foldowner": pilot_foldowner,
    "foldqualified": pilot_foldqualified,
    "foldantecedent": pilot_foldantecedent,
}


def assert_builder_parity():
    """Both re-declared prompts must be s66's, byte for byte, on real project data."""
    captured: list[str] = []

    class Stub:
        def __init__(self, sink):
            self.sink = sink

        def __call__(self, prompt, **_kwargs):
            self.sink.append(prompt)
            return {}

    info = inputs("bigbluebutton")
    for cls, tag in ((SLinker66, "s66"), (denotation_arm("parity"), "arm")):
        obj = linker(cls, knowledge=info["knowledge"])
        obj._ask = Stub(captured)
        candidates = obj._scan(info["sentences"], info["components"],
                               SCANS["name_word"])
        obj._classify_denotations(candidates[:30], info["sentences"])
    half = len(captured) // 2
    same = captured[:half] == captured[half:]
    assert same, "the re-declared denotation prompt differs from s66's"
    print(f"  builder parity: {half} denotation prompts byte-identical to s66's")

    comp_names = [c.name for c in info["components"]]
    a = coref_judge_arm("parity2")._prompt_validation(
        comp_names, ["Case 1"], L66.COREF_VALIDATION_FOCUS, strict=True)
    b = SLinker66._prompt_validation(
        comp_names, ["Case 1"], L66.COREF_VALIDATION_FOCUS, strict=True)
    assert a == b, "the re-declared coreference judging prompt differs from s66's"
    print("  builder parity: the coreference judging prompt is byte-identical")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", nargs="+", required=True,
                        choices=sorted(PILOTS) + ["all"])
    args = parser.parse_args()
    chosen = sorted(PILOTS) if "all" in args.pilot else args.pilot
    print(f"model {MODEL}   samples/arm {RUNS}   source run {SOURCE_RUN.name}")
    assert_builder_parity()
    for name in chosen:
        print(f"\n{'=' * 78}\n{name}")
        PILOTS[name]()


if __name__ == "__main__":
    main()
