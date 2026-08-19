"""A/B pilots for the s25 workflow points a reviewer would have to read about.

Every arm changes exactly one thing and holds every upstream stage fixed, so the
comparison carries one stochastic stage instead of a pipeline's worth of drift.
Upstream comes from a promoted run's checkpoints (`AB_SOURCE_RUN`), never from a
fresh five-project end-to-end run. Arms run N times per side and go through the
permutation test in `ab_stats`. `design_audit.py` sizes each question off the
same checkpoints first, so an arm is only paid for where a decision can move.

Outcomes are recorded in `../results/s25_design_pilots/README.md`. The four
adopted ones are already in `s_linker25.py`; the arms that target surfaces since
removed (`claim`, `noclaim`, `ambiguity`) need the preserved
`s_linker25_pre_pilot_baseline.py` to run again.

    --pilot sequence   ADOPTED. Subtract already-linked pairs before the
                       coreference judge, so "each linker sees only what the
                       earlier ones left unlinked" holds for all three linkers.
                       -6.8 FP (p=0.01), +0.8 TP (p=0.05), 57% fewer judge cases.
    --pilot union      ADOPTED. One extraction sample, not two unioned.
                       TP -1.2 (p=0.30), FP -1.2 (p=0.42).
    --pilot alias      ADOPTED. Offer every alias to extraction instead of only
                       the "global" ones. +3.0 TP (p=0.01), +1.0 FP (p=0.59).
    --pilot ambiguity  ADOPTED. Remove the ambiguity map, its call and its
                       bundle field. TP -0.2 (p=1.00), FP +0.8 (p=0.40).

    --pilot noclaim    REJECTED. Dropping the judges' quote request costs 35.2
                       TP (p=0.01): unread output, but load-bearing.
    --pilot claim      REJECTED. Instructing contiguity and enforcing the quote
                       as a substring voided 0 verdicts in 25 project-runs; the
                       instruction alone cost +1.6 FP (p=0.02).
    --pilot corefpass  REJECTED. A second coreference judging pass moves neither
                       score (TP -0.6, p=0.40; FP -0.8, p=0.17).
    --pilot batch      REJECTED, and run to answer an objection: one candidate
                       per judging call rather than 25 is neutral (TP +0.7,
                       p=0.60; FP +0.3, p=1.00), so batching does not decide
                       links.
    --pilot window     REJECTED, same purpose: halving CONTEXT_SENTENCES and
                       ANCHOR_LIMIT costs 2.0 TP (p=0.20) for no precision gain.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

import design_audit as audit
from ab_stats import permutation_report
from design_audit import (
    PROJECTS, load_project, load_phase, load_gold, DEFAULT_RUN,
)

from llm_sad_sam.core.data_types_v2 import (
    SadSamLink, ModelKnowledge, DocumentKnowledge,
)
from llm_sad_sam.llm_client import LLMBackend
from llm_sad_sam.linkers.experimental.s_linker25 import (
    SLinker25, LAYERED_ENTITY_RULES, LAYERED_COREF_RULES, P2_FOCUS,
)

RUNS = int(os.environ.get("AB_RUNS", "3"))
WORKERS = int(os.environ.get("AB_WORKERS", "5"))
MODEL = os.environ.get("OPENAI_MODEL_NAME", "gpt-5.6-terra")
OUT = Path(os.environ.get("AB_OUT", "../results/s25_design_pilots"))
SOURCE_RUN = Path(os.environ.get("AB_SOURCE_RUN", str(DEFAULT_RUN)))


# ── variants ─────────────────────────────────────────────────────────────────

CONTIGUITY_LINE = (
    "The quote must be one contiguous exact substring of the sentence; do not "
    "join separate parts of it and do not use ellipses."
)


class ClaimChecked(SLinker25):
    """Instruct contiguity, then void a verdict whose quote is not a substring.

    The rule is the fabrication rule, not the partial-name linker's stricter
    presence rule: "none" stays a legitimate answer, because the full-name
    rubric approves a bare mention that makes no architectural claim.

    The sentence a case ruled on is recovered from the case string rather than
    threaded through the call. That is a pilot convenience -- the production
    form passes the sentences alongside the cases, which both call sites
    already hold -- and it reads exactly the text the judge saw.
    """

    _SENTENCE_LINE = re.compile(r'^  (?:\[prev: .*?\]\s*)?"(.*)"\s*$', re.MULTILINE)

    @staticmethod
    def _prompt_validation(comp_names, cases, focus, strict: bool = False) -> str:
        rules = LAYERED_COREF_RULES if strict else LAYERED_ENTITY_RULES
        return f"""Validate components in a document. {focus}

COMPONENTS: {', '.join(comp_names)}

{rules}

For each case, first quote the EXACT words from the sentence that state the
architectural claim about the component (or write "none" if the sentence makes no
such claim), then decide approve true/false based on that claim.
{CONTIGUITY_LINE}

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "claim": "<exact quote or none>", "approve": true}}]}}
JSON only:"""

    @classmethod
    def _case_sentence(cls, case_text):
        match = cls._SENTENCE_LINE.search(case_text)
        return match.group(1) if match else ""

    def _run_validation_pass(self, comp_names, cases, focus, phase_tag=None,
                             strict=False):
        results = super()._run_validation_pass(
            comp_names, cases, focus, phase_tag=phase_tag, strict=strict)
        # super() dropped the claims, so re-read the response it just recorded.
        claims = self._last_claims(len(cases))
        voided = []
        for index, approved in list(results.items()):
            if not approved:
                continue
            claim = claims.get(index, "")
            if not claim or claim.casefold() in ("none", "n/a"):
                continue
            sentence = self._case_sentence(cases[index])
            if sentence and claim.casefold() not in sentence.casefold():
                results[index] = False
                voided.append(index)
        if voided:
            print(f"    claim check voided {len(voided)} approval(s)")
        self.voided_total = getattr(self, "voided_total", 0) + len(voided)
        return results

    def _last_claims(self, n_cases):
        """{case index: claim} from the most recent recorded validation call."""
        for call in reversed(self._llm_calls):
            text = call.get("response_text") or ""
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                continue
            try:
                data = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
            if "validations" not in data:
                continue
            out = {}
            for item in data["validations"]:
                index = item.get("case", 0) - 1
                if 0 <= index < n_cases:
                    out[index] = str(item.get("claim", "")).strip().strip("\"'“”‘’")
            return out
        return {}


class NoClaimRequest(SLinker25):
    """Stop asking the two unverified judges for a quote at all.

    The `claim` arm showed the check is inert once contiguity is instructed: it
    voided nothing in 25 project-runs, and the added instruction alone moved the
    verdicts. That leaves two honest designs for a quote nobody reads --
    describe it as a commit-to-text device and keep it, or stop asking. This arm
    prices the second: same rubric, same focus, same schema minus `claim`.
    """

    @staticmethod
    def _prompt_validation(comp_names, cases, focus, strict: bool = False) -> str:
        rules = LAYERED_COREF_RULES if strict else LAYERED_ENTITY_RULES
        return f"""Validate components in a document. {focus}

COMPONENTS: {', '.join(comp_names)}

{rules}

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "approve": true}}]}}
JSON only:"""


class P1Only(SLinker25):
    """Full-name judge with the relevance pass alone.

    The paper has to state that the full-name judge runs two passes while the
    coreference judge runs one. Unmeasured, that asymmetry reads as tuning. This
    arm prices the second pass; `CorefTwoPass` prices the one the coreference
    judge does not have.
    """

    def _run_validation_pass(self, comp_names, cases, focus, phase_tag=None,
                             strict=False):
        if focus is P2_FOCUS:
            return {index: True for index in range(len(cases))}
        return super()._run_validation_pass(
            comp_names, cases, focus, phase_tag=phase_tag, strict=strict)


class CorefTwoPass(SLinker25):
    """Coreference judge with a second, uniqueness-style pass added."""

    def _run_validation_pass(self, comp_names, cases, focus, phase_tag=None,
                             strict=False):
        first = super()._run_validation_pass(
            comp_names, cases, focus, phase_tag=phase_tag, strict=strict)
        if not strict:
            return first
        second = super()._run_validation_pass(
            comp_names, cases, P2_FOCUS, phase_tag=phase_tag, strict=strict)
        return {index: bool(first.get(index)) and bool(second.get(index))
                for index in set(first) | set(second)}


class JudgeBatchOne(SLinker25):
    """One candidate per judging call, so no verdict can see its neighbours.

    The `sequence` pilot showed batch composition is not inert: removing the
    already-linked cases from the coreference judge's batches moved seven false
    positives among the cases that stayed. A reviewer can turn that into "your
    per-link decisions are not per-link". This arm prices the answer that ends
    the objection outright -- independent decisions -- against the batched form.
    """

    JUDGE_BATCH = 1


class NarrowWindow(SLinker25):
    """Half the evidence window and half the anchor list.

    Both constants are asserted in the module to change how much text a judge
    sees and never what counts as a link. That is a claim about the code, not
    about the model; this arm measures whether the output is stable under it.
    """

    CONTEXT_SENTENCES = 2
    ANCHOR_LIMIT = 2


class SinglePassExtraction(SLinker25):
    """One extraction sample instead of two unioned samples."""

    def _extract_named_mentions(self, sentences, components, name_to_id, sent_map):
        comp_names = [c.name for c in components]
        mappings = (
            [f"{term}={entry.component}"
             for term, entry in self.doc_knowledge.aliases.items()
             if entry.scope == "global"]
            if self.doc_knowledge else []
        )
        single = self._run_extraction_pass(
            sentences, comp_names, mappings, name_to_id, sent_map,
            pass_label="[C1] ", phase_tag="phase_25_full_name_extract1")
        print(f"    Single pass: {len(single)}")
        return single


class NoAmbiguityMap(SLinker25):
    """No model-understanding call, and no ambiguity line in the bundle."""

    def _analyze_model(self, components):
        return ModelKnowledge()

    def _format_evidence(self, bundle) -> str:
        lines = [
            f"  Evidence: source={bundle.source}, span=\"{bundle.matched_span}\", "
            f"mention={bundle.mention_type}",
            f"  Rationale: {bundle.extraction_rationale}",
        ]
        if bundle.preceding_text:
            lines.append(f"  [prev: \"{bundle.preceding_text}\"]")
        if bundle.anchor_sentences:
            lines.append("  Anchors (confirmed refs):")
            for anchor in bundle.anchor_sentences:
                lines.append(f"    {anchor}")
        return "\n".join(lines)


def unscoped(doc_knowledge):
    """The same alias table with every entry marked global.

    Nothing but the extraction prompt read `scope`, so this was exactly the
    "no scope distinction" design without touching a line of the linker. That
    design is now the linker's, and `scope` is gone with it -- against the
    current file this is the identity, which is what makes the arm re-runnable
    only against `s_linker25_pre_pilot_baseline.py`.
    """
    out = DocumentKnowledge()
    for term, entry in doc_knowledge.aliases.items():
        if hasattr(entry, "component"):
            out.aliases[term] = type(entry)(component=entry.component,
                                            scope="global")
        else:
            out.aliases[term] = entry
    return out


# ── plumbing ─────────────────────────────────────────────────────────────────

def new_linker(cls=SLinker25, doc_knowledge=None, model_knowledge=None):
    linker = cls(backend=LLMBackend.OPENAI, model=MODEL)
    linker.doc_knowledge = doc_knowledge
    linker.model_knowledge = model_knowledge
    return linker


def cached(path, build):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("rb") as handle:
            print(f"  [cache] {path.name}")
            return pickle.load(handle)
    value = build()
    with path.open("wb") as handle:
        pickle.dump(value, handle)
    return value


def parallel(units, fn):
    out = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fn, *u): u for u in units}
        for future in as_completed(futures):
            key = futures[future]
            try:
                out[key] = future.result()
            except Exception as exc:                        # noqa: BLE001
                print(f"    !! {key} FAILED: {exc}")
                out[key] = None
    return out


def collect(arm, units_fn, projects):
    """Run every (run, project) unit and fold each run into one global set."""
    units = [(run, name) for run in range(1, RUNS + 1) for name in projects]
    got = parallel(units, units_fn)
    return [set().union(*[got[(run, n)] or set() for n in projects])
            for run in range(1, RUNS + 1)]


def scorers(gold_by_project):
    def tp(pairs):
        return sum(1 for p, pair in pairs if tuple(pair) in gold_by_project[p])

    def fp(pairs):
        return len(pairs) - tp(pairs)

    return {"TP": tp, "FP": fp}


def report(name, stats, extra=None):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    with path.open("w") as handle:
        json.dump({"model": MODEL, "runs": RUNS, "source_run": str(SOURCE_RUN),
                   "stats": stats, **(extra or {})}, handle, indent=2, default=str)
    print(f"report -> {path}")


def normalise_aliases(doc_knowledge):
    """Checkpoints predating the scope removal hold AliasEntry, not a name.

    The current linker reads the alias table as term -> component name, the
    declared `DocumentKnowledge` type. Normalising on load keeps every pilot
    able to reuse an older promoted run.
    """
    for term, entry in list(doc_knowledge.aliases.items()):
        doc_knowledge.aliases[term] = getattr(entry, "component", entry)
    return doc_knowledge


def inputs_with_gold():
    data = {}
    for name in PROJECTS:
        project = load_project(name)
        project["gold"] = load_gold(name)
        knowledge = load_phase(SOURCE_RUN, name, "knowledge")
        knowledge["doc_knowledge"] = normalise_aliases(knowledge["doc_knowledge"])
        knowledge.setdefault("model_knowledge", None)
        project["knowledge"] = knowledge
        data[name] = project
    return data


# ── pilot: sequence ──────────────────────────────────────────────────────────

def pilot_sequence(inputs):
    """Subtract already-linked pairs before the coreference judge."""
    print("\n### sequence — subtract already-linked pairs before the coref judge")
    print("Upstream fixed from the promoted run: the full-name and partial-name")
    print("links define `prior`, and the coreference proposals that reach the")
    print("judge are that run's, so both arms judge the same resolutions.")

    prepared = {}
    for name, item in inputs.items():
        full = load_phase(SOURCE_RUN, name, "linker_full_name")
        partial = load_phase(SOURCE_RUN, name, "linker_partial_name")
        coref = load_phase(SOURCE_RUN, name, "linker_coreference")
        prior = {(l.sentence_number, l.component_id) for l in full["links"]}
        prior |= {(l.sentence_number, l.component_id) for l in partial["links"]}
        name_to_id = item["name_to_id"]
        proposals = [
            SadSamLink(c["sentence"], name_to_id[c["component"]], c["component"],
                       source="coreference")
            for c in coref["feedback"]["candidates"]
            if c["component"] in name_to_id
        ]
        fresh = [l for l in proposals
                 if (l.sentence_number, l.component_id) not in prior]
        prepared[name] = {"prior": prior, "all": proposals, "fresh": fresh}
        print(f"  {name:14s} prior {len(prior):3d} | proposals {len(proposals):3d} "
              f"-> fresh {len(fresh):3d}")

    def one(arm, run, name):
        item = inputs[name]
        state = prepared[name]
        links = state["all"] if arm == "A_all_proposals" else state["fresh"]
        final = set(state["prior"])
        if links:
            linker = new_linker(doc_knowledge=item["knowledge"]["doc_knowledge"],
                                model_knowledge=item["knowledge"]["model_knowledge"])
            approved, _ = linker._validate_coref_links(
                links, item["sent_map"], item["components"])
            final |= {(l.sentence_number, l.component_id) for l in approved}
        return {(name, pair) for pair in final}

    arms = {}
    for arm in ("A_all_proposals", "B_unlinked_only"):
        arms[arm] = collect(arm, lambda run, name, a=arm: one(a, run, name), inputs)
        print(f"  {arm}: {[len(s) for s in arms[arm]]}")

    gold = {n: inputs[n]["gold"] for n in inputs}
    stats = permutation_report(arms, scorers(gold),
                               title="sequence — final set, all vs unlinked-only")
    report("sequence_subtraction", stats, {
        "judge_cases": {n: {"all": len(v["all"]), "fresh": len(v["fresh"])}
                        for n, v in prepared.items()},
    })


# ── pilot: alias scope ───────────────────────────────────────────────────────

def pilot_alias(inputs):
    """Offer every alias to the extraction prompt, not only the global ones."""
    print("\n### alias — drop the scope distinction in the extraction prompt")
    for name, item in inputs.items():
        aliases = item["knowledge"]["doc_knowledge"].aliases
        local = [t for t, e in aliases.items() if e.scope == "local"]
        print(f"  {name:14s} aliases {len(aliases):2d} | newly offered (local): {local}")

    def one(arm, run, name):
        item = inputs[name]
        knowledge = item["knowledge"]["doc_knowledge"]
        if arm == "B_all_aliases":
            knowledge = unscoped(knowledge)
        linker = new_linker(doc_knowledge=knowledge,
                            model_knowledge=item["knowledge"]["model_knowledge"])
        raw = linker._extract_named_mentions(
            item["sentences"], item["components"], item["name_to_id"],
            item["sent_map"])
        candidates = linker._keep_stated_names(list(raw.values()))
        candidates = linker._add_spelling_variants(
            candidates, item["sentences"], item["components"])
        bundles = {
            (c.sentence_number, c.component_id):
                linker._build_evidence_bundle(c, item["sent_map"])
            for c in candidates
        }
        approved, _ = linker._validate_with_evidence(
            candidates, bundles, item["components"], item["sent_map"],
            p1_tag="pilot_alias_p1", p2_tag="pilot_alias_p2",
            stage_label="full_name")
        return {(name, (c.sentence_number, c.component_id)) for c in approved}

    arms = {}
    for arm in ("A_global_only", "B_all_aliases"):
        arms[arm] = collect(arm, lambda run, name, a=arm: one(a, run, name), inputs)
        print(f"  {arm}: {[len(s) for s in arms[arm]]}")

    gold = {n: inputs[n]["gold"] for n in inputs}
    stats = permutation_report(arms, scorers(gold),
                               title="alias — extraction prompt alias set")
    report("alias_scope", stats)


# ── pilot: coreference judge passes ──────────────────────────────────────────

def _coref_inputs(inputs):
    """Prior links and the post-subtraction coreference proposals per project."""
    prepared = {}
    for name, item in inputs.items():
        full = load_phase(SOURCE_RUN, name, "linker_full_name")
        partial = load_phase(SOURCE_RUN, name, "linker_partial_name")
        coref = load_phase(SOURCE_RUN, name, "linker_coreference")
        prior = {(l.sentence_number, l.component_id) for l in full["links"]}
        prior |= {(l.sentence_number, l.component_id) for l in partial["links"]}
        name_to_id = item["name_to_id"]
        fresh = [
            SadSamLink(c["sentence"], name_to_id[c["component"]], c["component"],
                       source="coreference")
            for c in coref["feedback"]["candidates"]
            if c["component"] in name_to_id
            and (c["sentence"], name_to_id[c["component"]]) not in prior
        ]
        prepared[name] = {"prior": prior, "fresh": fresh}
    return prepared


def pilot_corefpass(inputs):
    """Does the coreference judge need the second pass the full-name judge has?

    The paper must state that one judge runs two passes and the other one. This
    arm adds the missing pass so the asymmetry is a measured choice rather than
    an unexplained one. Proposals are the promoted run's, already reduced by the
    subtraction the `sequence` pilot adopted.
    """
    print("\n### corefpass — second judging pass for the coreference judge")
    prepared = _coref_inputs(inputs)
    for name, state in prepared.items():
        print(f"  {name:14s} prior {len(state['prior']):3d} | fresh proposals "
              f"{len(state['fresh']):3d}")

    def one(arm, run, name):
        item = inputs[name]
        state = prepared[name]
        final = set(state["prior"])
        if state["fresh"]:
            cls = SLinker25 if arm == "A_one_pass" else CorefTwoPass
            linker = new_linker(cls, item["knowledge"]["doc_knowledge"],
                                item["knowledge"]["model_knowledge"])
            approved, _ = linker._validate_coref_links(
                state["fresh"], item["sent_map"], item["components"])
            final |= {(l.sentence_number, l.component_id) for l in approved}
        return {(name, pair) for pair in final}

    arms = {}
    for arm in ("A_one_pass", "B_two_passes"):
        arms[arm] = collect(arm, lambda run, name, a=arm: one(a, run, name), inputs)
        print(f"  {arm}: {[len(s) for s in arms[arm]]}")

    gold = {n: inputs[n]["gold"] for n in inputs}
    stats = permutation_report(arms, scorers(gold),
                               title="corefpass — one vs two judging passes")
    report("coref_judge_passes", stats)


# ── pilot: extraction union ──────────────────────────────────────────────────

def pilot_union(inputs):
    """Two sampled extraction passes unioned, against one pass.

    Extraction is the varying stage here, so neither arm can reuse the cached
    candidate set; the judge that follows is identical in both.
    """
    print("\n### union — two extraction samples unioned vs one")

    def one(arm, run, name):
        item = inputs[name]
        cls = SLinker25 if arm == "A_union" else SinglePassExtraction
        linker = new_linker(cls, item["knowledge"]["doc_knowledge"],
                            item["knowledge"]["model_knowledge"])
        raw = linker._extract_named_mentions(
            item["sentences"], item["components"], item["name_to_id"],
            item["sent_map"])
        candidates = linker._keep_stated_names(list(raw.values()))
        candidates = linker._add_spelling_variants(
            candidates, item["sentences"], item["components"])
        bundles = {
            (c.sentence_number, c.component_id):
                linker._build_evidence_bundle(c, item["sent_map"])
            for c in candidates
        }
        approved, _ = linker._validate_with_evidence(
            candidates, bundles, item["components"], item["sent_map"],
            p1_tag="pilot_union_p1", p2_tag="pilot_union_p2",
            stage_label="full_name")
        return {(name, (c.sentence_number, c.component_id)) for c in approved}

    arms = {}
    for arm in ("A_union", "B_single_pass"):
        arms[arm] = collect(arm, lambda run, name, a=arm: one(a, run, name), inputs)
        print(f"  {arm}: {[len(s) for s in arms[arm]]}")

    gold = {n: inputs[n]["gold"] for n in inputs}
    stats = permutation_report(arms, scorers(gold),
                               title="union — second extraction sample")
    report("extraction_union", stats)


# ── pilot: judge family (claim check, ambiguity map) ─────────────────────────

def _judge_stage(cls, item, name, knowledge, ambiguity, tag):
    linker = new_linker(cls, knowledge, ambiguity)
    candidates = linker._keep_stated_names(item["extraction"])
    candidates = linker._add_spelling_variants(
        candidates, item["sentences"], item["components"])
    bundles = {
        (c.sentence_number, c.component_id):
            linker._build_evidence_bundle(c, item["sent_map"])
        for c in candidates
    }
    approved, _ = linker._validate_with_evidence(
        candidates, bundles, item["components"], item["sent_map"],
        p1_tag=f"{tag}_p1", p2_tag=f"{tag}_p2", stage_label="full_name")
    return {(name, (c.sentence_number, c.component_id)) for c in approved}


def _prepare_extraction(inputs):
    """One extraction realisation per project, shared by every judge arm."""
    for name, item in inputs.items():
        linker = new_linker(doc_knowledge=item["knowledge"]["doc_knowledge"])
        raw = cached(
            OUT / f"extraction_{name}.pkl",
            lambda linker=linker, item=item: linker._extract_named_mentions(
                item["sentences"], item["components"], item["name_to_id"],
                item["sent_map"]),
        )
        item["extraction"] = list(raw.values())
        print(f"  {name:14s} extraction {len(item['extraction'])} candidates")


def pilot_judges(inputs, which):
    """Two A/B comparisons over one shared control arm.

    The full-name judge is the only stage that varies, so extraction is computed
    once per project and every arm judges the same candidate set.
    """
    print("\n### judge family — one control, two candidate changes")
    _prepare_extraction(inputs)

    gold = {n: inputs[n]["gold"] for n in inputs}
    control = collect("A", lambda run, name: _judge_stage(
        SLinker25, inputs[name], name,
        inputs[name]["knowledge"]["doc_knowledge"],
        inputs[name]["knowledge"]["model_knowledge"], "pilot_judge_a"), inputs)
    print(f"  A_current: {[len(s) for s in control]}")

    if "claim" in which:
        arm = collect("B", lambda run, name: _judge_stage(
            ClaimChecked, inputs[name], name,
            inputs[name]["knowledge"]["doc_knowledge"],
            inputs[name]["knowledge"]["model_knowledge"], "pilot_judge_claim"),
            inputs)
        print(f"  B_claim_checked: {[len(s) for s in arm]}")
        stats = permutation_report({"A_current": control, "B_claim_checked": arm},
                                   scorers(gold),
                                   title="claim — contiguity instruction + check")
        report("claim_check", stats)

    if "noclaim" in which:
        arm = collect("B", lambda run, name: _judge_stage(
            NoClaimRequest, inputs[name], name,
            inputs[name]["knowledge"]["doc_knowledge"],
            inputs[name]["knowledge"]["model_knowledge"], "pilot_judge_noclaim"),
            inputs)
        print(f"  B_no_claim: {[len(s) for s in arm]}")
        stats = permutation_report({"A_current": control, "B_no_claim": arm},
                                   scorers(gold),
                                   title="noclaim — quote request removed")
        report("no_claim_request", stats)

    if "p2" in which:
        arm = collect("B", lambda run, name: _judge_stage(
            P1Only, inputs[name], name,
            inputs[name]["knowledge"]["doc_knowledge"],
            inputs[name]["knowledge"]["model_knowledge"], "pilot_judge_p1only"),
            inputs)
        print(f"  B_p1_only: {[len(s) for s in arm]}")
        stats = permutation_report({"A_current": control, "B_p1_only": arm},
                                   scorers(gold),
                                   title="p2 — full-name uniqueness pass removed")
        report("full_name_second_pass", stats)

    if "batch" in which:
        arm = collect("B", lambda run, name: _judge_stage(
            JudgeBatchOne, inputs[name], name,
            inputs[name]["knowledge"]["doc_knowledge"],
            inputs[name]["knowledge"]["model_knowledge"], "pilot_judge_batch1"),
            inputs)
        print(f"  B_batch_one: {[len(s) for s in arm]}")
        stats = permutation_report({"A_current": control, "B_batch_one": arm},
                                   scorers(gold),
                                   title="batch — one candidate per judging call")
        report("judge_batch", stats)

    if "window" in which:
        arm = collect("B", lambda run, name: _judge_stage(
            NarrowWindow, inputs[name], name,
            inputs[name]["knowledge"]["doc_knowledge"],
            inputs[name]["knowledge"]["model_knowledge"], "pilot_judge_window"),
            inputs)
        print(f"  B_narrow_window: {[len(s) for s in arm]}")
        stats = permutation_report({"A_current": control, "B_narrow_window": arm},
                                   scorers(gold),
                                   title="window — anchors and context halved")
        report("evidence_window", stats)

    if "ambiguity" in which:
        arm = collect("B", lambda run, name: _judge_stage(
            NoAmbiguityMap, inputs[name], name,
            inputs[name]["knowledge"]["doc_knowledge"],
            ModelKnowledge(), "pilot_judge_noambig"), inputs)
        print(f"  B_no_ambiguity: {[len(s) for s in arm]}")
        stats = permutation_report({"A_current": control, "B_no_ambiguity": arm},
                                   scorers(gold),
                                   title="ambiguity — map removed from the bundle")
        report("ambiguity_map", stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", nargs="+", required=True,
                        choices=["sequence", "alias", "union", "claim", "noclaim",
                                 "batch", "window", "ambiguity", "p2",
                                 "corefpass"])
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY unset (map OAI_KEY into it inline)")
    started = time.time()
    inputs = inputs_with_gold()
    if "sequence" in args.pilot:
        pilot_sequence(inputs)
    if "alias" in args.pilot:
        pilot_alias(inputs)
    if "union" in args.pilot:
        pilot_union(inputs)
    if "corefpass" in args.pilot:
        pilot_corefpass(inputs)
    judges = [p for p in args.pilot if p in ("claim", "noclaim", "batch", "window", "ambiguity", "p2")]
    if judges:
        pilot_judges(inputs, judges)
    print(f"\ntotal {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
