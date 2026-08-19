"""Ablate one prompt at one stage, against fixed recorded inputs. No E2E runs.

`prompt_audit.py` sizes a clause off traces; that says what a clause *could* touch,
not what it does. The E2E rounds say what a change costs end to end, at ~20 minutes a
run and with a null offset of 0.7 macro F1 to fight (`results/null_calibration/`).
Between those sits the measurement this file makes: replay **one stage** with the two
prompt wordings against the *same* recorded inputs, N samples a side, and test the
stage's own output.

That isolates what a prompt change does, because everything upstream is a fixed
checkpoint rather than a fresh LLM sample. It is a screen, not a verdict — this
branch has eight instances of a stage arm pointing opposite to the composed pipeline,
always on precision, because the three linkers subtract from one another. Read a
stage arm as "does this wording change this stage's behaviour, and in which
direction".

Pilots, one per prompt the s55 result nominates:

    aliasextract   the alias *proposer* rules (`DOC_KNOWLEDGE_EXTRACTION_RULES` +
                   `ALIAS_EXCLUSION_RULES`), s49's wording against s51's. Input: the
                   document and the component list, which is all this call ever sees.
                   Output: proposed terms, then the *unchanged* s49 judge, then the
                   alias table — so the arm measures the proposer with its own judge
                   downstream, which is the question s55 raised.
    aliasjudge     the alias judge rubric, s49's against s51's, on the recorded
                   proposal list. Proposer held fixed, so this is the judge alone.
    entityextract  `ENTITY_EXTRACTION_RULES`, s49's against s51's, with the alias
                   table pinned to the checkpoint's. Output: proposed (sentence,
                   component) candidates, scored against gold.
    fullnamejudge  the two judging rubrics (`P1_FOCUS`/`P2_FOCUS`/
                   `LAYERED_ENTITY_RULES`), on the recorded candidate set.
    corefprompt    the coreference resolution prompt: s55's wording against s56's,
                   which deletes the opening instruction paragraph that restates
                   `COREF_RULES` inside the same prompt.
    corefjudge     the coreference judging rubric, s49's against s51's, on the
                   recorded resolutions.

Usage (from approach/):
    AB_RUNS=3 ../.venv/bin/python pilot/prompt_stage_pilots.py --pilot aliasextract
    AB_RUNS=5 ../.venv/bin/python pilot/prompt_stage_pilots.py --pilot all
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from ab_stats import permutation_report                              # noqa: E402
import design_pilots                                                 # noqa: E402
from design_pilots import (                                          # noqa: E402
    MODEL, PROJECTS, RUNS, collect, load_gold, load_project, report,
)
from llm_sad_sam.core.data_types_v2 import DocumentKnowledge         # noqa: E402
from llm_sad_sam.llm_client import LLMBackend                        # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker49 as L49       # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker51 as L51       # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker49 import SLinker49    # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker55 import SLinker55    # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker56 import SLinker56    # noqa: E402

OUT = Path(os.environ.get("AB_OUT", "../results/prompt_stage_pilots"))
SOURCE_RUN = Path(os.environ.get(
    "AB_SOURCE_RUN", "../results/s5051_e2e_r1_20260813"))
SOURCE_VARIANT = os.environ.get("AB_SOURCE_VARIANT", "s_linker49")

# `report()` writes into design_pilots.OUT; point it at this round's directory.
design_pilots.OUT = OUT


# ── fixed inputs, read once from a recorded run ──────────────────────────────

def phase(project, name):
    import pickle
    path = (SOURCE_RUN / "phase_states" / SOURCE_VARIANT / "openai" / project
            / f"{name}.pkl")
    with path.open("rb") as handle:
        return pickle.load(handle)


_INPUTS: dict = {}


def inputs(project):
    if project not in _INPUTS:
        info = load_project(project)
        info["gold"] = load_gold(project)
        info["knowledge"] = phase(project, "knowledge")["doc_knowledge"]
        info["full_name"] = phase(project, "linker_full_name")
        info["coreference"] = phase(project, "linker_coreference")
        _INPUTS[project] = info
    return _INPUTS[project]


_PROPOSALS: dict = {}


def recorded_proposals(project):
    """Every term the alias proposer offered in the source run, before judging."""
    import json
    import re
    if project in _PROPOSALS:
        return _PROPOSALS[project]
    info = load_project(project)
    names = {c.name for c in info["components"]}
    out: dict[str, str] = {}
    for path in (SOURCE_RUN / "llm_logs").glob(
            f"{SOURCE_VARIANT}_openai_{project}_*_calls.json"):
        with path.open() as handle:
            calls = json.load(handle)
        for call in calls:
            if call.get("phase") != "phase_25_doc_extract":
                continue
            body = (call.get("response_text") or "").strip()
            fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
            if fence:
                body = fence.group(1).strip()
            start, end = body.find("{"), body.rfind("}")
            if start < 0 or end <= start:
                continue
            try:
                data = json.loads(body[start:end + 1])
            except json.JSONDecodeError:
                continue
            for key in ("abbreviations", "synonyms"):
                for item in data.get(key, []) or []:
                    term, comp = item.get("term"), item.get("component")
                    if term and comp in names:
                        out[str(term)] = str(comp)
    _PROPOSALS[project] = out
    return out


def linker(cls=SLinker49, knowledge=None):
    obj = cls(backend=LLMBackend.OPENAI, model=MODEL)
    obj.doc_knowledge = knowledge
    obj.model_knowledge = None
    return obj


def with_rules(base_cls, **overrides):
    """A one-off subclass whose prompt builders read the overridden constants.

    The builders reference module-level names, so swapping a constant means
    re-defining the builder with the alternative wording bound in. Done here rather than by
    editing a variant file, because these arms are questions, not designs.
    """
    rules = {name: overrides.get(name, getattr(base_cls._RULES_MODULE, name))
             for name in RULE_NAMES}

    class Arm(base_cls):
        _VARIANT_NAME = base_cls._VARIANT_NAME + "_arm"

        @staticmethod
        def _prompt_doc_knowledge_extract(comp_names, doc_lines) -> str:
            return f"""Find all alternative names used for these components in the document.

COMPONENTS: {', '.join(comp_names)}

{rules['DOC_KNOWLEDGE_EXTRACTION_RULES']}

{rules['ALIAS_EXCLUSION_RULES']}

DOCUMENT:
{chr(10).join(doc_lines)}

Return JSON:
{{
  "abbreviations": [{{"term": "short_form", "component": "FullComponent"}}],
  "synonyms":      [{{"term": "specific_alternative_name", "component": "FullComponent"}}]
}}
JSON only:"""

        @staticmethod
        def _prompt_doc_knowledge_judge(comp_names, mapping_list) -> str:
            return f"""JUDGE: Review these component name mappings for correctness.

COMPONENTS: {', '.join(comp_names)}

PROPOSED MAPPINGS:
{chr(10).join(mapping_list)}



{rules['DOC_KNOWLEDGE_JUDGE_RULES']}

Return JSON:
{{"approved": ["term1", "term2"]}}
JSON only:"""

        @staticmethod
        def _prompt_extraction(comp_names, mappings, batch) -> str:
            return f"""Extract ALL references to components from this document.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings)}' if mappings else ''}

{rules['ENTITY_EXTRACTION_RULES']}

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{{"references": [{{"sentence": N_INTEGER, "component": "Name", "matched_text": "text found in sentence"}}]}}
JSON only:"""

        @staticmethod
        def _prompt_validation(comp_names, cases, focus, strict: bool = False) -> str:
            block = (rules['LAYERED_COREF_RULES'] if strict
                     else rules['LAYERED_ENTITY_RULES'])
            return f"""Validate components in a document. {focus}

COMPONENTS: {', '.join(comp_names)}

{block}

For each case, first quote the EXACT words from the sentence that state the
architectural claim about the component (or write "none" if the sentence makes no
such claim), then decide approve true/false based on that claim.

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "claim": "<exact quote or none>", "approve": true}}]}}
JSON only:"""

    Arm.RULES = rules
    return Arm


RULE_NAMES = ("DOC_KNOWLEDGE_EXTRACTION_RULES", "ALIAS_EXCLUSION_RULES",
              "DOC_KNOWLEDGE_JUDGE_RULES", "ENTITY_EXTRACTION_RULES",
              "P1_FOCUS", "P2_FOCUS", "LAYERED_ENTITY_RULES",
              "COREF_VALIDATION_FOCUS", "LAYERED_COREF_RULES", "COREF_RULES")
SLinker49._RULES_MODULE = L49
SLinker55._RULES_MODULE = L49          # s55 differs only on the coref family
SLinker56._RULES_MODULE = L49


# ── scoring ──────────────────────────────────────────────────────────────────

def alias_yield(project, table):
    """A deterministic proxy for what an alias table is worth downstream.

    An alias earns links by making a sentence that carries the term, and not the
    component's own name, a full-name candidate. So for every term in the table,
    count the (sentence, component) pairs it alone could produce, split by gold.
    No LLM call, and it prices two tables against each other without running the
    stages that consume them.
    """
    info = inputs(project)
    gold = info["gold"]
    name_to_id = info["name_to_id"]
    hit_gold, hit_other = set(), set()
    for term, comp in table.items():
        cid = name_to_id.get(comp)
        if cid is None:
            continue
        for sentence in info["sentences"]:
            text = sentence.text.casefold()
            if term.casefold() not in text:
                continue
            if comp.casefold() in text:          # the name is stated; no alias needed
                continue
            key = (sentence.number, cid)
            (hit_gold if key in gold else hit_other).add((project, key))
    return hit_gold, hit_other


def pair_scorers():
    gold = {p: load_gold(p) for p in PROJECTS}

    def tp(pairs):
        return sum(1 for project, key in pairs if key in gold[project])

    def fp(pairs):
        return len(pairs) - tp(pairs)

    return {"TP": tp, "FP": fp}


# ── pilots ───────────────────────────────────────────────────────────────────

def pilot_aliasextract():
    """Proposer wording varied; the s49 judge runs unchanged behind both arms."""
    arms = {
        "s49 proposer": with_rules(SLinker49),
        "general proposer": with_rules(
            SLinker49,
            DOC_KNOWLEDGE_EXTRACTION_RULES=L51.DOC_KNOWLEDGE_EXTRACTION_RULES,
            ALIAS_EXCLUSION_RULES=L51.ALIAS_EXCLUSION_RULES),
    }
    return _knowledge_arms(arms, "aliasextract")


def pilot_aliasjudge():
    """Judge wording varied; both arms see the same recorded proposal list."""
    arms = {
        "s49 judge": with_rules(SLinker49),
        "general judge": with_rules(
            SLinker49, DOC_KNOWLEDGE_JUDGE_RULES=L51.DOC_KNOWLEDGE_JUDGE_RULES),
    }
    return _knowledge_arms(arms, "aliasjudge", fixed_proposals=True)


def _knowledge_arms(arms, name, fixed_proposals=False):
    sets, tables = {}, {}

    def unit(arm_cls, run, project):
        info = inputs(project)
        obj = linker(arm_cls)
        if fixed_proposals:
            # The judge must see the *proposed* list, not the approved table: the
            # checkpoint only keeps what already passed s49's judge, so judging that
            # again is the identity for any rubric. The proposals come from the run's
            # own doc_extract response.
            recorded = recorded_proposals(project)
            comp_names = [c.name for c in info["components"]]
            mapping_list = [f"'{k}' -> {v}" for k, v in recorded.items()]
            data = obj._ask(arm_cls._prompt_doc_knowledge_judge(comp_names, mapping_list),
                            timeout=120, label="alias judge", require="approved")
            approved = set((data or {}).get("approved", []))
            table = {t: c for t, c in recorded.items() if t in approved}
        else:
            knowledge = obj._learn_document_knowledge(
                info["sentences"], info["components"])
            table = dict(knowledge.aliases)
        gold_pairs, other_pairs = alias_yield(project, table)
        tables.setdefault((arm_cls.__name__, run, project), sorted(table))
        return gold_pairs | other_pairs

    for label, cls in arms.items():
        sets[label] = collect(label, lambda run, p, c=cls: unit(c, run, p), PROJECTS)
        sizes = [len(s) for s in sets[label]]
        print(f"  {label:20s} alias-reachable pairs per run: {sizes}")

    stats = permutation_report(sets, quality=pair_scorers(),
                               title=f"{name}: stage arm")
    report(name, stats, extra={"tables": {str(k): v for k, v in tables.items()}})
    return stats


def pilot_entityextract():
    """Extraction wording varied; the alias table is pinned to the checkpoint's."""
    arms = {
        "s49 extraction": with_rules(SLinker49),
        "general extraction": with_rules(
            SLinker49, ENTITY_EXTRACTION_RULES=L51.ENTITY_EXTRACTION_RULES),
    }
    sets = {}

    def unit(arm_cls, run, project):
        info = inputs(project)
        obj = linker(arm_cls, knowledge=info["knowledge"])
        candidates = obj._extract_named_mentions(
            info["sentences"], info["components"], info["name_to_id"],
            info["sent_map"])
        return {(project, (c.sentence_number, c.component_id))
                for c in candidates.values()}

    for label, cls in arms.items():
        sets[label] = collect(label, lambda run, p, c=cls: unit(c, run, p), PROJECTS)
        print(f"  {label:20s} candidates per run: {[len(s) for s in sets[label]]}")
    stats = permutation_report(sets, quality=pair_scorers(),
                               title="entityextract: stage arm")
    report("entityextract", stats)
    return stats


def pilot_fullnamejudge():
    """Judging rubric varied; both arms judge the same recorded candidate set."""
    arms = {
        "s49 rubric": with_rules(SLinker49),
        "general rubric": with_rules(
            SLinker49, P1_FOCUS=L51.P1_FOCUS,
            LAYERED_ENTITY_RULES=L51.LAYERED_ENTITY_RULES),
    }
    return _fullname_arms(arms, "fullnamejudge")


def _fullname_arms(arms, name):
    sets = {}

    def unit(arm_cls, run, project):
        info = inputs(project)
        obj = linker(arm_cls, knowledge=info["knowledge"])
        recorded = info["full_name"]["feedback"]["candidates"]
        cand_objs = _rebuild_candidates(obj, info, recorded)
        bundles = {(c.sentence_number, c.component_id):
                   obj._build_evidence_bundle(c, info["sent_map"])
                   for c in cand_objs}
        approved, _ = obj._validate_with_evidence(
            cand_objs, bundles, info["components"], info["sent_map"],
            "pilot_p1", "pilot_p2", "full_name_twopass")
        return {(project, (c.sentence_number, c.component_id)) for c in approved}

    for label, cls in arms.items():
        sets[label] = collect(label, lambda run, p, c=cls: unit(c, run, p), PROJECTS)
        print(f"  {label:30s} approved per run: {[len(s) for s in sets[label]]}")
    stats = permutation_report(sets, quality=pair_scorers(),
                               title=f"{name}: stage arm")
    report(name, stats)
    return stats


def _rebuild_candidates(obj, info, recorded):
    """Turn a checkpoint's candidate dicts back into the linker's candidate type.

    `mention_type` is re-derived rather than read, because the checkpoint's
    `candidates` view does not carry it and the classifier is deterministic.
    """
    from llm_sad_sam.core.data_types_v2 import CandidateLink
    out = []
    for item in recorded:
        cid = info["name_to_id"].get(item["component"])
        sentence = info["sent_map"].get(item["sentence"])
        if cid is None or sentence is None:
            continue
        matched = item.get("matched_text") or item["component"]
        out.append(CandidateLink(
            sentence_number=item["sentence"], sentence_text=sentence.text,
            component_name=item["component"], component_id=cid,
            matched_text=matched,
            source=item.get("source", "full_name_candidate"),
            mention_type=obj._classify_mention_typed(
                item["component"], sentence.text),
            alias_used=None))
    return out


def pilot_corefprompt():
    """s55's coreference prompt against s56's, which drops the duplicated paragraph."""
    sets = {}
    arms = {"s55 prompt": SLinker55, "s56 prompt (no preamble)": SLinker56}

    def unit(arm_cls, run, project):
        info = inputs(project)
        obj = linker(arm_cls, knowledge=info["knowledge"])
        links, _ = obj._resolve_references(
            info["sentences"], info["components"], info["name_to_id"],
            info["sent_map"])
        return {(project, (link.sentence_number, link.component_id))
                for link in links}

    for label, cls in arms.items():
        sets[label] = collect(label, lambda run, p, c=cls: unit(c, run, p), PROJECTS)
        print(f"  {label:24s} resolutions per run: {[len(s) for s in sets[label]]}")
    stats = permutation_report(sets, quality=pair_scorers(),
                               title="corefprompt: stage arm")
    report("corefprompt", stats)
    return stats


def pilot_corefjudge():
    """Coreference judging rubric varied on the recorded resolutions."""
    arms = {
        "s49 rubric": with_rules(SLinker49),
        "general rubric": with_rules(
            SLinker49, COREF_VALIDATION_FOCUS=L51.COREF_VALIDATION_FOCUS,
            LAYERED_COREF_RULES=L51.LAYERED_COREF_RULES),
    }
    sets = {}

    def unit(arm_cls, run, project):
        from llm_sad_sam.core.data_types_v2 import SadSamLink
        info = inputs(project)
        obj = linker(arm_cls, knowledge=info["knowledge"])
        recorded = info["coreference"]["feedback"]["candidates"]
        links = []
        for item in recorded:
            cid = info["name_to_id"].get(item["component"])
            if cid is None:
                continue
            links.append(SadSamLink(item["sentence"], cid, item["component"],
                                    source="coreference"))
        validated, _ = obj._validate_coref_links(
            links, info["sent_map"], info["components"])
        return {(project, (link.sentence_number, link.component_id))
                for link in validated}

    for label, cls in arms.items():
        sets[label] = collect(label, lambda run, p, c=cls: unit(c, run, p), PROJECTS)
        print(f"  {label:20s} approved per run: {[len(s) for s in sets[label]]}")
    stats = permutation_report(sets, quality=pair_scorers(),
                               title="corefjudge: stage arm")
    report("corefjudge", stats)
    return stats




def _coref_arm(head_text):
    """s55's coreference prompt with the opening paragraph replaced by `head_text`."""

    class Arm(SLinker55):
        _VARIANT_NAME = "s_linker55_corefarm"

        @staticmethod
        def _prompt_coref(comp_names, cases) -> str:
            prompt = f"""Resolve references (pronouns and noun phrases that refer back) to components.

COMPONENTS: {', '.join(comp_names)}
{head_text}"""
            for i, case in enumerate(cases):
                prompt += f"--- Case {i+1}: S{case['sent'].number} ---\n"
                prompt += "CONTEXT:\n" + "\n".join(case["context"]) + "\n"
                prompt += f"TARGET: S{case['sent'].number} (marked with >>>)\n\n"
            prompt += f"""{SLinker55._RULES_MODULE and L49 and ''}{__import__('llm_sad_sam.linkers.experimental.s_linker55', fromlist=['COREF_RULES']).COREF_RULES}

Return JSON:
{{"resolutions": [{{"case": 1, "sentence": N_INTEGER, "reference": "the server", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

JSON only:"""
            return prompt

    return Arm


FULL_HEAD = """
For each TARGET sentence below, identify any pronoun or noun phrase that
refers back to a component listed above. If a target sentence has no such
reference to a listed component, return no resolution for it. Be conservative \u2014 only include resolutions you are CERTAIN about.

"""
TASK_ONLY_HEAD = """
For each TARGET sentence below, identify any pronoun or noun phrase that
refers back to a component listed above. If a target sentence has no such
reference to a listed component, return no resolution for it.

"""
CONSERVATIVE_ONLY_HEAD = """
Be conservative \u2014 only include resolutions you are CERTAIN about.

"""
NOTHING_HEAD = """
"""


def _coref_resolution_arms(arms, name):
    sets = {}

    def unit(arm_cls, run, project):
        info = inputs(project)
        obj = linker(arm_cls, knowledge=info["knowledge"])
        links, _ = obj._resolve_references(
            info["sentences"], info["components"], info["name_to_id"],
            info["sent_map"])
        return {(project, (link.sentence_number, link.component_id))
                for link in links}

    for label, cls in arms.items():
        sets[label] = collect(label, lambda run, p, c=cls: unit(c, run, p), PROJECTS)
        print(f"  {label:34s} resolutions per run: {[len(s) for s in sets[label]]}")
    stats = permutation_report(sets, quality=pair_scorers(),
                               title=f"{name}: stage arm")
    report(name, stats)
    return stats


def pilot_corefpre_task():
    """Is it the task sentences or the strictness sentence that the paragraph buys?

    `corefprompt` showed that deleting the whole opening paragraph costs 16.2
    resolutions worth of true positives at this stage. The paragraph is two things:
    a statement of the per-case protocol (which case block is the target, and that a
    target with no reference yields nothing) and one strictness sentence. This arm
    keeps the protocol and drops the strictness sentence.
    """
    return _coref_resolution_arms(
        {"full paragraph (s55)": _coref_arm(FULL_HEAD),
         "protocol, no strictness": _coref_arm(TASK_ONLY_HEAD)},
        "corefpre_task")


def pilot_corefpre_strict():
    """The mirror: keep only the strictness sentence, drop the protocol."""
    return _coref_resolution_arms(
        {"full paragraph (s55)": _coref_arm(FULL_HEAD),
         "strictness only": _coref_arm(CONSERVATIVE_ONLY_HEAD)},
        "corefpre_strict")


def pilot_fullnamerubric():
    """The full-name judge's cost, split: the rubric alone."""
    arms = {
        "s49 rubric": with_rules(SLinker49),
        "general LAYERED_ENTITY_RULES": with_rules(
            SLinker49, LAYERED_ENTITY_RULES=L51.LAYERED_ENTITY_RULES),
    }
    return _fullname_arms(arms, "fullnamerubric")


def pilot_fullnamefocus():
    """The full-name judge's cost, split: the P1 focus line alone."""
    arms = {
        "s49 focus": with_rules(SLinker49),
        "general P1_FOCUS": with_rules(SLinker49, P1_FOCUS=L51.P1_FOCUS),
    }
    return _fullname_arms(arms, "fullnamefocus")



def pilot_composed59():
    """The three cleared clauses together, through the two stages they touch.

    Every arm in this file varies one prompt at one stage. This one composes the
    three that cleared — the coreference family (already confirmed end-to-end as
    s55), `P1_FOCUS` and `DOC_KNOWLEDGE_JUDGE_RULES` — and runs the knowledge stage
    and the whole full-name linker behind them, so the alias table each arm builds is
    the one its own judge approved and the candidates each arm judges are the ones
    its own table admitted. That is the composition risk this branch has been caught
    by eight times, checked without an end-to-end run.

    Coreference is excluded on purpose: its clauses are the ones already confirmed at
    six paired runs, and including its 40 calls per sample would triple the cost of
    the arm for a question already answered.
    """
    arms = {
        "s49": with_rules(SLinker49),
        "s59 (three cleared clauses)": with_rules(
            SLinker49, P1_FOCUS=L51.P1_FOCUS,
            DOC_KNOWLEDGE_JUDGE_RULES=L51.DOC_KNOWLEDGE_JUDGE_RULES),
    }
    sets = {}

    def unit(arm_cls, run, project):
        info = inputs(project)
        obj = linker(arm_cls)
        obj.doc_knowledge = obj._learn_document_knowledge(
            info["sentences"], info["components"])
        links = obj._run_full_name_linker(
            info["sentences"], info["components"], info["name_to_id"], set(),
            info["sent_map"])
        links = links[0] if isinstance(links, tuple) else links
        return {(project, (link.sentence_number, link.component_id))
                for link in links}

    for label, cls in arms.items():
        sets[label] = collect(label, lambda run, p, c=cls: unit(c, run, p), PROJECTS)
        print(f"  {label:30s} full-name links per run: {[len(s) for s in sets[label]]}")
    stats = permutation_report(sets, quality=pair_scorers(),
                               title="composed59: knowledge + full-name")
    report("composed59", stats)
    return stats



def pilot_mergedalias():
    """The variant the s26 line never built: merged proposal, dedicated judge.

    `s_linker26` folded alias proposal into the entity-extraction batches **and**
    deleted the alias judge, and lost 2.2 macro F1. Its diagnosis named two causes:

      (a) a batch cannot see a definition stated elsewhere, so the reading loses the
          short forms defined once and used far away (`ui`, `webui`, `e2e`, `gae`);
      (b) nothing judges what the reading collects, so the table gains descriptive
          phrases and generic words (`client`, `core`, `other layers`) and even the
          dotted forms the alias rubric forbids -- "the rule is followed by a
          dedicated prompt and violated when appended to an extraction prompt".

    Cause (b) is fixable without touching (a): keep the merged reading and put the
    *dedicated* alias judge back behind it. No variant in the s26-s34 line does this
    -- s26 and s28 have no judge at all, s29-s34 keep the separate proposer and move
    the judging instead. This arm builds it and prices it against the two-stage
    design at the only place the two differ, the alias table.

    The prediction the diagnosis licenses: the judge should remove most of what (b)
    added, and the table should still be missing the globally-defined short forms
    that only a document-wide pass finds. If it is, the two-stage design's remaining
    justification is exactly (a), and that is a granularity argument rather than a
    judging one.
    """
    from llm_sad_sam.linkers.experimental.s_linker26 import SLinker26

    sets = {}

    def two_stage(run, project):
        info = inputs(project)
        obj = linker(with_rules(SLinker49))
        table = dict(obj._learn_document_knowledge(
            info["sentences"], info["components"]).aliases)
        _TABLES[("two-stage", run, project)] = table
        g, o = alias_yield(project, table)
        return g | o

    def merged_then_judge(run, project):
        info = inputs(project)
        reader = linker(SLinker26)
        reader.no_knowledge = False
        reader._extract_named_mentions(
            info["sentences"], info["components"], info["name_to_id"],
            info["sent_map"])
        proposed = dict(reader.doc_knowledge.aliases)
        judge = linker(with_rules(SLinker49))
        comp_names = [c.name for c in info["components"]]
        if proposed:
            data = judge._ask(
                SLinker49._prompt_doc_knowledge_judge(
                    comp_names, [f"'{k}' -> {v}" for k, v in proposed.items()]),
                timeout=120, label="alias judge", require="approved")
            approved = set((data or {}).get("approved", []))
            table = {t: c for t, c in proposed.items() if t in approved}
        else:
            table = {}
        _TABLES[("merged+judge", run, project)] = table
        _TABLES[("merged proposed", run, project)] = proposed
        g, o = alias_yield(project, table)
        return g | o

    for label, fn in (("two-stage (s49)", two_stage),
                      ("merged reading + judge", merged_then_judge)):
        sets[label] = collect(label, fn, PROJECTS)
        print(f"  {label:24s} alias-reachable pairs per run: "
              f"{[len(s) for s in sets[label]]}")

    for project in PROJECTS:
        base = set().union(*[set(_TABLES.get(("two-stage", r, project), ()))
                             for r in range(1, RUNS + 1)])
        proposed = set().union(*[set(_TABLES.get(("merged proposed", r, project), ()))
                                 for r in range(1, RUNS + 1)])
        kept = set().union(*[set(_TABLES.get(("merged+judge", r, project), ()))
                             for r in range(1, RUNS + 1)])
        print(f"  {project:14s} two-stage {len(base):3d} terms | merged proposed "
              f"{len(proposed):3d} -> judge kept {len(kept):3d} | "
              f"only two-stage finds {sorted(base - kept)} | "
              f"only merged keeps {sorted(kept - base)}")

    stats = permutation_report(sets, quality=pair_scorers(),
                               title="mergedalias: alias table")
    report("mergedalias", stats,
           extra={"tables": {str(k): sorted(v) for k, v in _TABLES.items()}})
    return stats


_TABLES: dict = {}



def standalone_name_words(info):
    """Words of a multi-word component name that the document uses on their own.

    General naming convention, no benchmark vocabulary: if a document writes one
    word of a multi-word component name in a sentence that does not carry the whole
    name, that word is a candidate short form for it. The dedicated alias judge then
    decides — this proposer only offers.

    Motivated by a deterministic measurement, not by inspection. s60's merged reading
    misses exactly one term that matters, `GAE` for `GAE Datastore`, and adding that
    single term to its table takes teammates' partial-name candidates from 40 to 30
    with no gold lost — which is the whole of s60's measured +13.5 spurious
    partial-name links.
    """
    import re as _re
    out = {}
    for component in info["components"]:
        parts = [w for w in _re.split(r"[\s\-]+", component.name.strip())
                 if len(w) >= 3]
        if len(parts) < 2:
            continue
        for word in parts:
            if word.casefold() == component.name.casefold():
                continue
            for sentence in info["sentences"]:
                text = sentence.text
                if component.name.casefold() in text.casefold():
                    continue
                if _re.search(rf"(?<!\w){_re.escape(word)}(?!\w)", text, _re.I):
                    out[word] = component.name
                    break
    return out


def pilot_namewordalias():
    """Can a deterministic proposer restore the suppression the merged reading loses?

    s60's table is better for *admission* (stage arm: FP -16.6) and worse for
    *suppression*: the alias table doubles as the partial-name linker's exclusion
    list, so a tighter table frees partial-name candidates. Measured end to end,
    that is +13.5 spurious partial-name links per run, and deterministically it is
    almost all one missing term.

    This arm offers every standalone name word to the **unchanged** dedicated judge
    and measures what the surviving table does to the partial-name linker. The two
    failure modes are both real and both visible here: the judge approves too little
    and the suppression is not restored, or it approves too much and the partial-name
    linker is suppressed out of existence — an all-name-words table with no judge at
    all takes bigbluebutton from 31 candidates to 3, destroying 16 gold ones.
    """
    from llm_sad_sam.core.data_types_v2 import DocumentKnowledge
    from llm_sad_sam.linkers.experimental.s_linker60 import SLinker60

    print("  proposing standalone name words, judging them with s49's alias judge\n")
    rows = {}
    for project in PROJECTS:
        info = inputs(project)
        gold = load_gold(project)
        proposed = standalone_name_words(info)
        comp_names = [c.name for c in info["components"]]
        counts = []
        for sample in range(RUNS):
            judge = linker(with_rules(SLinker49))
            if proposed:
                data = judge._ask(
                    SLinker49._prompt_doc_knowledge_judge(
                        comp_names,
                        [f"'{t}' -> {c}" for t, c in proposed.items()]),
                    timeout=120, label="alias judge", require="approved")
                approved = set((data or {}).get("approved", []))
            else:
                approved = set()
            kept = {t: c for t, c in proposed.items() if t in approved}
            obj = SLinker60.__new__(SLinker60)
            obj.doc_knowledge = DocumentKnowledge()
            obj.doc_knowledge.aliases.update(kept)
            cands = obj._name_word_candidates(info["sentences"], info["components"])
            g = sum(1 for c in cands
                    if (c.sentence_number, c.component_id) in gold)
            counts.append((len(kept), len(cands), g))
        rows[project] = (len(proposed), counts)
        kept_n = sum(c[0] for c in counts) / len(counts)
        cand_n = sum(c[1] for c in counts) / len(counts)
        gold_n = sum(c[2] for c in counts) / len(counts)
        print(f"  {project:14s} proposed {len(proposed):3d} -> judge kept "
              f"{kept_n:5.1f} | partial-name candidates {cand_n:5.1f} "
              f"({gold_n:4.1f} gold)")
    print("\n  compare: the same count with s49's own table, and with s60's, is in"
          "\n  the deterministic sizing printed by this round's README.")
    report("namewordalias", {"rows": {k: v for k, v in rows.items()}})
    return rows


PILOTS = {
    "aliasextract": pilot_aliasextract,
    "aliasjudge": pilot_aliasjudge,
    "entityextract": pilot_entityextract,
    "fullnamejudge": pilot_fullnamejudge,
    "corefprompt": pilot_corefprompt,
    "corefjudge": pilot_corefjudge,
    "corefpre_task": pilot_corefpre_task,
    "corefpre_strict": pilot_corefpre_strict,
    "fullnamerubric": pilot_fullnamerubric,
    "fullnamefocus": pilot_fullnamefocus,
    "composed59": pilot_composed59,
    "mergedalias": pilot_mergedalias,
    "namewordalias": pilot_namewordalias,
}


def assert_builder_parity():
    """`with_rules(SLinker49)` must be s49, byte for byte, on real project data.

    The arms re-declare the prompt builders so a constant can be swapped in. That is
    only a valid way to ask the question if the un-swapped version is identical to
    the linker's own — otherwise every arm is measuring the re-declaration.
    """
    identity = with_rules(SLinker49)
    project = "mediastore"
    info = inputs(project)
    names = [c.name for c in info["components"]]
    batch = info["sentences"][:6]
    mappings = [f"'{t}' -> {c}" for t, c in list(info["knowledge"].aliases.items())]
    pairs = [
        (identity._prompt_doc_knowledge_extract(names, [s.text for s in batch]),
         SLinker49._prompt_doc_knowledge_extract(names, [s.text for s in batch])),
        (identity._prompt_doc_knowledge_judge(names, mappings),
         SLinker49._prompt_doc_knowledge_judge(names, mappings)),
        (identity._prompt_extraction(names, mappings, batch),
         SLinker49._prompt_extraction(names, mappings, batch)),
        (identity._prompt_validation(names, ["Case 1"], L49.P1_FOCUS),
         SLinker49._prompt_validation(names, ["Case 1"], L49.P1_FOCUS)),
        (identity._prompt_validation(names, ["Case 1"], L49.COREF_VALIDATION_FOCUS,
                                     strict=True),
         SLinker49._prompt_validation(names, ["Case 1"], L49.COREF_VALIDATION_FOCUS,
                                      strict=True)),
    ]
    bad = [i for i, (a, b) in enumerate(pairs) if a != b]
    assert not bad, f"re-declared builders differ from s49's at {bad}"
    print("  builder parity: 5 of 5 prompts byte-identical to s_linker49's")


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
