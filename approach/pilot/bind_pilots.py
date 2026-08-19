"""Bind a deterministic rule into a prompt and replay the stage it lives in.

`pilot/bind_audit.py` prices every remaining rule off recorded runs: what a
prompt-bound extractor would have to newly produce, and what a prompt-bound judge
would have to work out for itself. This file pays for the arms that audit nominates,
one stage at a time against fixed recorded inputs, in the branch's standing order
(stage pilot -> `composition_check.py` -> E2E only if the composition risk is
non-zero).

Every arm is a *relocation*, not a removal: the rule leaves the code and its content
enters a prompt. Each pilot therefore carries three arms where the audit says the rule
is worth something -- the linker as it stands, the rule deleted with no compensation
(what the rule is worth), and the rule deleted with its content stated in the prompt
(what the binding recovers) -- so a null result can be told apart from a rule that was
never worth anything.

Pilots:

    bindscans     the two tight scans (`SCANS[stated_name]`, `SCANS[spelling]`) into
                  the extraction prompt. Audit: the extractor already proposes 111 of
                  112 as-spelled pairs and all 6 spelling-variant pairs, so the
                  binding gap is 1.0 pair per run. The risk is not the gap, it is
                  perturbing the other ~205 proposals (cf. `s_linker58`, +20.2 FP).
    bindcontract  `_keep_stated_names` into the extraction prompt. Audit: the filter
                  drops 24.8 proposals per run, 8.0 gold, and later linkers recover
                  7.7 of those, so it is a router, not a gate.
    bindlabel     `_classify_mention_typed` into the judging prompt. Audit: 143.5 of
                  182.5 labels per run are computable from the sentence the judge is
                  already shown; the other 39.0 are `via known alias`, and the judging
                  prompt is the one prompt in the workflow that never sees the alias
                  table -- so this binding is extractor-side *and* judge-side.
    bindpartial   `SCANS[name_word]` into the extraction prompt. Audit: the binding
                  gap is 53.8 pairs and 15.8 gold per run -- pairs the extraction call
                  sees and declines. Priced, not expected to hold.

Usage (from approach/):
    AB_RUNS=5 ../.venv/bin/python pilot/bind_pilots.py --pilot bindscans
    AB_RUNS=5 ../.venv/bin/python pilot/bind_pilots.py --pilot all
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
    MODEL, RUNS, collect, load_gold, report,
)
from bind_audit import (                                             # noqa: E402
    PROJECTS, extractor_pairs, phase_state, project,
)
from llm_sad_sam.core.data_types_v2 import CandidateLink             # noqa: E402
from llm_sad_sam.llm_client import LLMBackend                        # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker65 as L65       # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker65 import (            # noqa: E402
    SCANS, SLinker65,
)

OUT = Path(os.environ.get("AB_OUT", "../results/bind_round"))
SOURCE_RUN = Path(os.environ.get(
    "AB_SOURCE_RUN", "../results/s64_e2e_r1_20260814"))
SOURCE_VARIANT = os.environ.get("AB_SOURCE_VARIANT", "s_linker64")

design_pilots.OUT = OUT


# ── the clauses the rules become ─────────────────────────────────────────────
#
# Generic English, no benchmark vocabulary (GATE-06). Each states what the deleted
# predicate does, in the register of the prompt it joins.

#: `SCANS[stated_name]` + `SCANS[spelling]`, as one instruction to the extractor.
SCAN_CLAUSE = (
    "Report a reference for every sentence that writes a component's name the way "
    "the COMPONENTS list spells it, however incidental the mention, and count a name "
    "written with different spacing, hyphenation or compound joining as that name."
)

#: `_keep_stated_names`, as the contract the extractor is currently not held to.
CONTRACT_RULES = (
    "Include a reference only when the sentence itself writes the component's name or "
    "one of the KNOWN ALIASES. Exclude a component that the sentence only implies as a "
    "participant in a described interaction without naming it, and exclude a name that "
    "appears only inside a code-level path -- even if the compound identifier is "
    "semantically related to the component -- or as ordinary English with no "
    "architectural intent. Favor inclusion among the sentences that do name it."
)

#: `SCANS[name_word]`, as an instruction to the extractor.
PARTIAL_CLAUSE = (
    "Report separately, under \"partial\", any sentence that uses a single word of a "
    "multi-word component name on its own to mean that component, when only one "
    "component's name contains that word."
)

#: `_classify_mention_typed`, as a question the judge answers for itself.
LABEL_CLAUSE = (
    "For each case, first state how the name is present in the sentence: as the "
    "component's own name, only as part of a longer qualified identifier, or through "
    "a short form the document introduced for it."
)


# ── arms ─────────────────────────────────────────────────────────────────────

def _extraction_builder(rules=L65.ENTITY_EXTRACTION_RULES, extra="", partial=False):
    """`_prompt_extraction`, with a clause added and/or the rules swapped.

    Re-declared rather than edited into a variant file: these arms are questions.
    `assert_builder_parity` checks the un-swapped version against s65's own bytes.
    """
    fields = ('{"references": [{"sentence": N_INTEGER, "component": "Name", '
              '"matched_text": "text found in sentence"}]}')
    if partial:
        fields = ('{"references": [{"sentence": N_INTEGER, "component": "Name", '
                  '"matched_text": "text found in sentence"}], '
                  '"partial": [{"sentence": N_INTEGER, "component": "Name", '
                  '"matched_text": "the single word found in sentence"}]}')

    @staticmethod
    def _prompt_extraction(comp_names, mappings, batch) -> str:
        body = rules + (f"\n\n{extra}" if extra else "")
        return f"""Extract ALL references to components from this document.

COMPONENTS: {', '.join(comp_names)}
{f'KNOWN ALIASES: {", ".join(mappings)}' if mappings else ''}

{body}

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{fields}
JSON only:"""

    return _prompt_extraction


def arm(name, **attrs):
    """A one-off `SLinker65` subclass carrying one relocation."""
    return type(f"Arm_{name}", (SLinker65,),
                {"_VARIANT_NAME": f"s_linker65_{name}", **attrs})


def no_scans(self, candidates, sentences, components, scan_name):
    """`_add_scan` with the two tight scans deleted."""
    return candidates


def keep_all(self, candidates):
    """`_keep_stated_names` deleted."""
    return list(candidates)


def linker(cls, knowledge=None):
    obj = cls(backend=LLMBackend.OPENAI, model=MODEL)
    obj.doc_knowledge = knowledge
    obj.model_knowledge = None
    return obj


# ── fixed inputs ─────────────────────────────────────────────────────────────

_INPUTS: dict = {}


def inputs(name):
    if name not in _INPUTS:
        info = dict(project(name))
        knowledge = phase_state(SOURCE_RUN, SOURCE_VARIANT, name, "knowledge")
        info["knowledge"] = knowledge["doc_knowledge"]
        info["full_name"] = phase_state(
            SOURCE_RUN, SOURCE_VARIANT, name, "linker_full_name")
        info["extractor"] = extractor_pairs(SOURCE_RUN, SOURCE_VARIANT, name)
        _INPUTS[name] = info
    return _INPUTS[name]


def pair_scorers():
    gold = {p: load_gold(p) for p in PROJECTS}

    def tp(pairs):
        return sum(1 for name, key in pairs if key in gold[name])

    def fp(pairs):
        return len(pairs) - tp(pairs)

    return {"TP": tp, "FP": fp}


def rebuilt_candidates(obj, info):
    """The recorded full-name candidate set, as the linker's own candidate objects.

    `mention_type` is re-derived because the checkpoint's candidate view does not
    carry it and the classifier is deterministic. `bind_audit.py --only B0` checks
    this set against the extraction log with a zero residue.
    """
    out = []
    for item in info["full_name"]["feedback"]["candidates"]:
        cid = info["name_to_id"].get(item["component"])
        sentence = info["sent_map"].get(item["sentence"])
        if cid is None or sentence is None:
            continue
        out.append(CandidateLink(
            sentence_number=item["sentence"], sentence_text=sentence.text,
            component_name=item["component"], component_id=cid,
            matched_text=item.get("matched_text") or item["component"],
            source=item.get("source", "full_name_candidate"),
            mention_type=obj._classify_mention_typed(
                item["component"], sentence.text),
            alias_used=None))
    return out


# ── the extraction-side pilots ───────────────────────────────────────────────

def _extraction_arms(arms, name, note=""):
    """Score each arm on the candidate set the full-name judge would receive."""
    sets = {}

    def unit(cls, run, project_name):
        info = inputs(project_name)
        obj = linker(cls, knowledge=info["knowledge"])
        candidates = obj._extract_named_mentions(
            info["sentences"], info["components"], info["name_to_id"],
            info["sent_map"])
        kept = obj._keep_stated_names(list(candidates.values()))
        kept = obj._add_scan(kept, info["sentences"], info["components"], "spelling")
        kept = obj._add_scan(kept, info["sentences"], info["components"],
                             "stated_name")
        return {(project_name, (c.sentence_number, c.component_id)) for c in kept}

    for label, cls in arms.items():
        sets[label] = collect(label, lambda run, p, c=cls: unit(c, run, p), PROJECTS)
        print(f"  {label:34s} candidates per run: {[len(s) for s in sets[label]]}")
    return against_first(sets, name, note)


def against_first(sets, name, note=""):
    """Test every later arm against the first, the way `score_runs.py` does.

    `permutation_report` compares two arms; a relocation pilot carries three, so the
    control (the rule deleted) and the binding (the rule stated in the prompt) are
    each read against the linker as it stands.
    """
    labels = list(sets)
    stats = {}
    for other in labels[1:]:
        stats[other] = permutation_report(
            {labels[0]: sets[labels[0]], other: sets[other]},
            quality=pair_scorers(), title=f"{name}: {other} vs {labels[0]}{note}")
    report(name, stats, extra={"sizes": {k: [len(s) for s in v]
                                         for k, v in sets.items()}})
    return stats


def pilot_bindscans():
    """The two tight scans, relocated into the extraction prompt."""
    arms = {
        "s65 (scans in code)": arm("base"),
        "scans deleted": arm("noscan", _add_scan=no_scans),
        "scans bound to the prompt": arm(
            "boundscan", _add_scan=no_scans,
            _prompt_extraction=_extraction_builder(extra=SCAN_CLAUSE)),
    }
    return _extraction_arms(arms, "bindscans")


def pilot_bindcontract():
    """The admission filter, relocated into the extraction prompt."""
    arms = {
        "s65 (filter in code)": arm("base2"),
        "filter deleted": arm("nofilter", _keep_stated_names=keep_all),
        "contract bound to the prompt": arm(
            "boundcontract", _keep_stated_names=keep_all,
            _prompt_extraction=_extraction_builder(rules=CONTRACT_RULES)),
    }
    return _extraction_arms(arms, "bindcontract")


# ── the judge-side pilot ─────────────────────────────────────────────────────

def _evidence_without_label(self, bundle) -> str:
    lines = [f"  Evidence: source={bundle.source}, span=\"{bundle.matched_span}\""]
    if bundle.preceding_text:
        lines.append(f"  [prev: \"{bundle.preceding_text}\"]")
    if bundle.anchor_sentences:
        lines.append("  Anchors (confirmed refs):")
        for a in bundle.anchor_sentences:
            lines.append(f"    {a}")
    return "\n".join(lines)


def _validation_builder(ask_label=False, aliases=None):
    """`_prompt_validation`, optionally asking the judge for the mention itself.

    ``aliases`` is the second half of this binding: the judging prompt is the only
    prompt in the workflow that never sees the alias table, and 39.0 of the 182.5
    labels per run are `via known alias`, so a judge asked to work the label out
    needs what the label was computed from.
    """
    @staticmethod
    def _prompt_validation(comp_names, cases, focus, strict: bool = False) -> str:
        block = L65.LAYERED_COREF_RULES if strict else L65.LAYERED_ENTITY_RULES
        alias_line = (f"\nKNOWN ALIASES: {', '.join(aliases)}" if aliases else "")
        label_line = f"\n{LABEL_CLAUSE}\n" if ask_label else ""
        fields = ('{"validations": [{"case": 1, '
                  + ('"mention": "<how the name is present>", ' if ask_label else '')
                  + '"claim": "<exact quote or none>", "approve": true}]}')
        return f"""Validate components in a document. {focus}

COMPONENTS: {', '.join(comp_names)}{alias_line}

{block}
{label_line}
For each case, first quote the EXACT words from the sentence that state the
architectural claim about the component (or write "none" if the sentence makes no
such claim), then decide approve true/false based on that claim.

CASES:
{chr(10).join(cases)}

Return JSON:
{fields}
JSON only:"""

    return _prompt_validation


def pilot_bindlabel():
    """The mention label, relocated into the judging prompt. Same candidates."""
    sets = {}
    arms = {
        "s65 (label computed)": lambda info: arm("base3"),
        "label dropped": lambda info: arm(
            "nolabel", _format_evidence=_evidence_without_label),
        "label asked of the judge": lambda info: arm(
            "askedlabel", _format_evidence=_evidence_without_label,
            _prompt_validation=_validation_builder(ask_label=True)),
        "label asked + alias table": lambda info: arm(
            "askedlabel_alias", _format_evidence=_evidence_without_label,
            _prompt_validation=_validation_builder(
                ask_label=True,
                aliases=[f"{t}={c}" for t, c
                         in (info["knowledge"].aliases or {}).items()])),
    }

    def unit(make, run, project_name):
        info = inputs(project_name)
        obj = linker(make(info), knowledge=info["knowledge"])
        candidates = rebuilt_candidates(obj, info)
        bundles = {(c.sentence_number, c.component_id):
                   obj._build_evidence_bundle(c, info["sent_map"])
                   for c in candidates}
        approved, _ = obj._validate_with_evidence(
            candidates, bundles, info["components"], info["sent_map"],
            "pilot_p1", "pilot_p2", "full_name_twopass")
        return {(project_name, (c.sentence_number, c.component_id))
                for c in approved}

    for label, make in arms.items():
        sets[label] = collect(label, lambda run, p, m=make: unit(m, run, p), PROJECTS)
        print(f"  {label:34s} approved per run: {[len(s) for s in sets[label]]}")
    return against_first(sets, "bindlabel")


# ── the partial-name pilot ───────────────────────────────────────────────────

def pilot_bindpartial():
    """`SCANS[name_word]` against an extractor asked for the same pairs.

    Both arms are judged by the unchanged denotation judge, so what varies is only
    where the partial-name linker's candidates come from.
    """
    sets = {}

    def scan_unit(run, project_name):
        info = inputs(project_name)
        obj = linker(arm("base4"), knowledge=info["knowledge"])
        candidates = obj._scan(info["sentences"], info["components"],
                               SCANS["name_word"])
        approved, _ = obj._judge_partial_names(candidates, info["sentences"])
        return {(project_name, (c.sentence_number, c.component_id))
                for c in approved}

    def bound_unit(run, project_name):
        info = inputs(project_name)
        cls = arm("boundpartial",
                  _prompt_extraction=_extraction_builder(
                      extra=PARTIAL_CLAUSE, partial=True))
        obj = linker(cls, knowledge=info["knowledge"])
        comp_names = [c.name for c in info["components"]]
        mappings = [f"{t}={c}" for t, c in (info["knowledge"].aliases or {}).items()]
        candidates = []
        seen = set()
        for _, batch in obj._iter_batches(info["sentences"], obj.EXTRACTION_BATCH):
            data = obj._ask(
                cls._prompt_extraction(comp_names, mappings, batch),
                timeout=240, label="batch", require="references")
            for ref in (data or {}).get("partial", []):
                cname = ref.get("component")
                raw = str(ref.get("sentence", ""))
                snum = int(raw.lstrip("Ss")) if raw.lstrip("Ss").isdigit() else None
                cid = info["name_to_id"].get(cname)
                sentence = info["sent_map"].get(snum)
                if cid is None or sentence is None or (snum, cid) in seen:
                    continue
                matched = ref.get("matched_text", "")
                if matched and matched.lower() not in sentence.text.lower():
                    continue
                seen.add((snum, cid))
                candidates.append(CandidateLink(
                    sentence_number=snum, sentence_text=sentence.text,
                    component_name=cname, component_id=cid,
                    matched_text=matched or cname,
                    source="partial_name_candidate"))
        approved, _ = obj._judge_partial_names(candidates, info["sentences"])
        return {(project_name, (c.sentence_number, c.component_id))
                for c in approved}

    for label, fn in (("s65 (scan in code)", scan_unit),
                      ("scan bound to the prompt", bound_unit)):
        sets[label] = collect(label, fn, PROJECTS)
        print(f"  {label:34s} approved per run: {[len(s) for s in sets[label]]}")
    return against_first(sets, "bindpartial")


def pilot_bindboth():
    """Every extractor-side rule at once: the contract and the two tight scans.

    The audit's B6 shows why the two cannot be priced separately. Of the 2.0 pairs
    per run that reach the candidate set only through a scan, 1.0 is a pair the
    extractor never proposed (`stated_name`) and 1.0 is a pair it *did* propose and
    the admission filter would have dropped (`spelling`, whose surfaces do not write
    the name at ANY_CASE). So the spelling row is a widening of the filter, not a
    proposer, and deleting the filter makes it redundant.
    """
    arms = {
        "s65 (filter + scans in code)": arm("base5"),
        "all extractor rules bound": arm(
            "boundall", _keep_stated_names=keep_all, _add_scan=no_scans,
            _prompt_extraction=_extraction_builder(
                rules=CONTRACT_RULES, extra=SCAN_CLAUSE)),
    }
    return _extraction_arms(arms, "bindboth")


# ── the cutting arms: what is left once the extractor side is bound ──────────
#
# After `s_linker67` the deterministic layer is one relation, one `SCANS` row, the
# mention label and four structural predicates. These two arms ask whether the last
# two *decisions* in that list earn their code, one predicate at a time. Both are
# cuts, not relocations: nothing moves into a prompt, a distinction simply goes.


def label_without_code_token(self, comp_name: str, text: str):
    """`_classify_mention_typed` without its qualified-path branch.

    The branch is the only consumer of `_all_occurrences_in_qualified_path`, the
    predicate that lowercases the name and searches the raw sentence (rule_audit A4:
    28 labels as written, 25 handled consistently). It fires on 7.7 cases per run,
    which the judge approves at 69.6% against 95-99% for every other value, so it is
    the one value besides the alias one that separates -- and the one whose evidence
    the judge can see for itself in the sentence it is shown.

    Every wording is s65's: the affected cases fall back to the two stated-name
    values, which `s_linker44` measured as a pair that must NOT be merged. Nothing
    else about the field changes.
    """
    from llm_sad_sam.linkers.experimental.s_linker65 import MentionType
    matched = self._find_exact_form(text, comp_name)
    if matched:
        return (MentionType.PROPER_STANDALONE if matched == comp_name
                else MentionType.LOWERCASE_PROSE)
    for alias in self._names_by_component().get(comp_name, ()):
        if self._find_exact_form(text, alias):
            return MentionType.VIA_ALIAS
    return MentionType.INDIRECT


def pilot_cutcodetoken():
    """Drop the label's qualified-path value. Same candidates, same judge."""
    sets = {}
    arms = {
        "s65 (five-value label)": arm("base6"),
        "no qualified-path value": arm(
            "nocodetoken", _classify_mention_typed=label_without_code_token),
    }

    def unit(cls, run, project_name):
        info = inputs(project_name)
        obj = linker(cls, knowledge=info["knowledge"])
        candidates = rebuilt_candidates(obj, info)
        bundles = {(c.sentence_number, c.component_id):
                   obj._build_evidence_bundle(c, info["sent_map"])
                   for c in candidates}
        approved, _ = obj._validate_with_evidence(
            candidates, bundles, info["components"], info["sent_map"],
            "pilot_p1", "pilot_p2", "full_name_twopass")
        return {(project_name, (c.sentence_number, c.component_id))
                for c in approved}

    for label, cls in arms.items():
        sets[label] = collect(label, lambda run, p, c=cls: unit(c, run, p), PROJECTS)
        print(f"  {label:34s} approved per run: {[len(s) for s in sets[label]]}")
    return against_first(sets, "cutcodetoken")


def pilot_cutqualified():
    """Drop the span-boundary test from the one scan `s_linker67` keeps.

    Under s67 `_inside_qualified_identifier` has exactly one consumer left, the
    partial-name scan. If the target-blind denotation judge behind it is as good at
    rejecting a glued span as the predicate is, the predicate goes -- and with it the
    `"" in "-_"` defect this branch documents as a validity threat (41 candidate
    spans suppressed over the five documents, repaired at FP +1.2 in `s_linker63`).
    """
    from dataclasses import replace
    sets = {}
    scans = {
        "s65 (boundary test in code)": SCANS["name_word"],
        "no boundary test": replace(SCANS["name_word"], skip_qualified=False),
    }

    def unit(scan, run, project_name):
        info = inputs(project_name)
        obj = linker(arm("base7"), knowledge=info["knowledge"])
        candidates = obj._scan(info["sentences"], info["components"], scan)
        approved, _ = obj._judge_partial_names(candidates, info["sentences"])
        return {(project_name, (c.sentence_number, c.component_id))
                for c in approved}

    for label, scan in scans.items():
        sets[label] = collect(label, lambda run, p, s=scan: unit(s, run, p), PROJECTS)
        print(f"  {label:34s} approved per run: {[len(s) for s in sets[label]]}")
    return against_first(sets, "cutqualified")


PILOTS = {
    "bindboth": pilot_bindboth,
    "cutcodetoken": pilot_cutcodetoken,
    "cutqualified": pilot_cutqualified,
    "bindscans": pilot_bindscans,
    "bindcontract": pilot_bindcontract,
    "bindlabel": pilot_bindlabel,
    "bindpartial": pilot_bindpartial,
}


def assert_builder_parity():
    """The re-declared builders must be s65's, byte for byte, on real project data.

    Without this every arm measures the re-declaration instead of the relocation.
    """
    info = inputs("mediastore")
    names = [c.name for c in info["components"]]
    batch = info["sentences"][:6]
    mappings = [f"{t}={c}" for t, c in (info["knowledge"].aliases or {}).items()]
    extraction = _extraction_builder().__func__
    validation = _validation_builder().__func__
    pairs = [
        (extraction(names, mappings, batch),
         SLinker65._prompt_extraction(names, mappings, batch)),
        (validation(names, ["Case 1"], L65.P1_FOCUS),
         SLinker65._prompt_validation(names, ["Case 1"], L65.P1_FOCUS)),
        (validation(names, ["Case 1"], L65.COREF_VALIDATION_FOCUS, strict=True),
         SLinker65._prompt_validation(names, ["Case 1"], L65.COREF_VALIDATION_FOCUS,
                                      strict=True)),
    ]
    bad = [i for i, (a, b) in enumerate(pairs) if a != b]
    assert not bad, f"re-declared builders differ from s65's at {bad}"
    print("  builder parity: 3 of 3 re-declared prompts byte-identical to s65's")


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
