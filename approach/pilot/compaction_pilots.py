"""The compaction round: what a prompt repeats, priced one repetition at a time.

Replays ONE stage of `s_linker87` against the checkpoints of a recorded run, so an
arm costs that stage's calls and nothing else, and every arm in an invocation sees
the same recorded aliases. Same harness as `pilot/typed_prompt_pilots.py`, one head
later, and aimed at what `pilot/judge_prompt_bytes.py` says the bytes actually are:
**not rules**. Rule constants are 5.3% of a full-name judging call; 27.9% of it is
anchor sentences the same call already showed, and 25.4% of a resolver call is table
rows for sentences the same call prints inline as a TARGET.

Groups, one clause or one repetition each, every arm designed off `pilot/clause_audit.py`:

  resolve3  the resolver AND the strict judge behind it -- what a resolver proposes
            is only a link if that gate keeps it.

              ctl           `s_linker87` as it stands
              nocasectx     minus the per-case "CONTEXT: sentences Sx-Sy above."
                            line. The audit reads it as non-binding: 16.3 antecedents
                            per five-project run on terra and 42.7 on luna already sit
                            outside the range their case declares, because the table
                            is the union of ten windows and the model reads the table.
                            7.3% of the resolver prompt.
              notargetrows  minus the table rows for sentences the same call prints
                            below as a TARGET. Their text is in the call either way,
                            once instead of twice; 25.4% of the resolver prompt, the
                            largest single item in it. Counter-evidence it must
                            survive: 159.0 of 186.0 antecedents per run on terra cite
                            a sentence that IS a target of their own call, so this is
                            the population the arm rewrites the framing of, and s82
                            measured a target losing salience at 15 spurious
                            resolutions per run.

  fullname5 the full-name judge, over ONE extraction pass shared by every arm.

              ctl           `s_linker87` as it stands
              anchorblock   the anchor sentences hoisted out of the cases into one
                            ANCHORS block per call, each sentence once, indexed by
                            the component whose name it writes. Removes 43.4 kB per
                            five-project run of literal repetition and moves the
                            evidence away from the case.
              anchorref     the same de-duplication with the evidence kept local: the
                            first case for a component carries its anchors, later ones
                            say which case to read them in. Same bytes removed, the
                            locality kept -- the two arms separate "the repetition
                            costs nothing" from "the position costs nothing".
              nosource      minus `source=` from the evidence line, which the s38
                            audit measured at one value in 99% of renderings and left
                            alone. The line's `span=` is deliberately NOT touched:
                            removing it was stage-neutral and pipeline-negative on the
                            s25 base (`../results/s25_ablate_all/`).

  denot2    the partial-name linker: the deterministic scan, then the denotation
            judge that is the only LLM step behind it.

              ctl           `s_linker87` as it stands
              nodenotqual   minus `QUALIFIED_CLAUSE` from the denotation prompt. The
                            scan has no qualified-identifier skip, so the clause does
                            guard a real population -- and the audit sizes that
                            population at 2.0 candidates per five-project run on both
                            models, 0 of them gold, 5 of 6 already answered
                            `associated`.

  fullname6 the same stage again, with the arm the first group's result is worth
            shipping -- and the correction it needed.

              ctl           `s_linker87` as it stands
              anchorunion   `s_linker88`'s own code, run as an arm: each component's
                            anchors are the UNION of what every case for it in the
                            batch would show, written into the first such case and
                            referenced by the rest. `anchorref` and `anchorblock`
                            above showed the FIRST case's list to every later case,
                            which is not lossless -- the per-case lists differ (each
                            drops the case's own sentence and stops at ANCHOR_LIMIT),
                            and only 19 of 121 pairs are equal over the five projects.
                            The union form loses nothing for any of 169 cases
                            (`pilot/test_s88_anchors.py`) and still removes 32.8% of
                            the judging prompt.

  coref5    the strict judge, over the resolutions recorded in the same run.

              ctl           `s_linker87` as it stands
              noartifact    minus `LAYERED_COREF_RULES`'s longest clause, the one
                            naming what a component acts on or produces. The recorded
                            objections cite that ground 1.0 times per run on terra and
                            1.7 on luna, 0 gold in both -- but the unclassified residue
                            is full of the same shape in other words, which is what the
                            arm decides.

Usage (from approach/), one model per invocation:

    OPENAI_API_KEY="$OAI_KEY" LLM_BACKEND=openai OPENAI_MODEL_NAME=gpt-5.6-terra \
    OPENAI_SERVICE_TIER=flex OPENAI_REASONING_EFFORT=none \
    LLM_LOG_DIR=../results/compaction_round/llm_logs_terra_resolve3 \
    AB_OUT=../results/compaction_round \
      ../.venv/bin/python pilot/compaction_pilots.py \
        --group resolve3 --model terra --runs 3
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
os.environ.setdefault("PHASE_CACHE_DIR", "/tmp/compaction_pilots_cache")

from llm_sad_sam.core.data_types_v2 import SadSamLink                  # noqa: E402
from llm_sad_sam.core.document_loader_v2 import (                      # noqa: E402
    build_sent_map, load_sentences,
)
from llm_sad_sam.llm_client import LLMBackend                          # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository             # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker87 as L87         # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker87 import SLinker87      # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker88 import SLinker88      # noqa: E402

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

#: The recorded runs each model's stages are replayed against: the head's own
#: end-to-end batch, whose `s_linker87` checkpoints supply the aliases every arm
#: starts from and the other two stages the arm's stage is composed with.
RECORDED = {
    "terra": ("dedup_e2e_terra_r*_20260821", "s_linker87"),
    "luna": ("dedup_e2e_luna_r*_20260821", "s_linker87"),
}

ORIG_COREF_PROMPT = SLinker87._prompt_coref
ORIG_VALIDATE_EVIDENCE = SLinker87._validate_with_evidence
ORIG_FORMAT_EVIDENCE = SLinker87._format_evidence
ORIG_VALIDATION = SLinker87._prompt_validation
ORIG_LAYERED_COREF = L87.LAYERED_COREF_RULES

# ── the resolver prompt, re-declared once and patched twice ──────────────────

#: `drop_context` removes the per-case range line; `inline_targets` removes the table
#: rows for sentences the same call prints as a TARGET. With both False this renders
#: byte-identically to `s_linker87._prompt_coref`, which `check_parity` asserts.
def coref_prompt(drop_context=False, inline_targets=False):
    def build(comp_names, sentence_table, targets) -> str:
        if inline_targets:
            target_numbers = {t["target"] for t in targets}
            sentence_table = [row for row in sentence_table
                              if row["sentence"] not in target_numbers]
        blocks = [
            f"--- Case {t['case']} ---\n"
            f"TARGET S{t['target']}: {t['text']}"
            + ("" if drop_context else
               f"\nCONTEXT: sentences S{min(t['context'])}-S{max(t['context'])} above.")
            for t in targets
        ]
        return f"""Resolve references (pronouns and noun phrases that refer back) to components.

COMPONENTS: {', '.join(comp_names)}

SENTENCES (the document text the cases are drawn from)
{json.dumps(sentence_table)}

For each TARGET sentence below, identify any pronoun or noun phrase in THAT sentence
that refers back to a component listed above. Read the TARGET's context in SENTENCES.
If a target sentence has no such reference to a listed component, return no resolution
for it. Be conservative — only include resolutions you are CERTAIN about.

{chr(10).join(blocks)}

{L87.COREF_RULES}

Return JSON:
{{"resolutions": [{{"case": 1, "sentence": N_INTEGER, "reference": "the server", "component": "Name", "antecedent_sentence": M_INTEGER, "antecedent_text": "exact quote with component name"}}]}}

JSON only:"""

    return build


# ── the judging prompt, with a place to put a shared block ───────────────────

#: `s_linker87._prompt_validation` with one addition: a per-call block rendered
#: between the rubric and the claim-first instruction. With the block empty this
#: renders byte-identically to the original, asserted by `check_parity`.
def validation_prompt_with_block(get_block):
    def build(comp_names, cases, focus, strict: bool = False) -> str:
        rules = L87.LAYERED_COREF_RULES if strict else L87.LAYERED_ENTITY_RULES
        decide = (
            " then decide approve true/false based on that claim."
            if not strict else
            " then state the strongest ground there is for rejecting this case under the\n"
            "rules above (or \"none\" if there is none), then decide: approve unless that "
            "ground is one\nthe rules above make decisive. An objection you could raise "
            "against most sentences is not\na ground for rejecting this one."
        )
        field = "" if not strict else ', "objection": "<strongest ground to reject, or none>"'
        tail = ("" if strict else
                f"\n{L87.QUALIFIED_CLAUSE}\n{L87.STRICTER_CLAUSE}\n")
        block = "" if strict else get_block()
        return f"""Validate components in a document.{f" {focus}" if focus else ""}

COMPONENTS: {', '.join(comp_names)}

{rules}
{tail}{block}
For each case, first quote the EXACT words from the sentence that state the
architectural claim about the component (or write "none" if the sentence makes no
such claim),{decide}

CASES:
{chr(10).join(cases)}

Return JSON:
{{"validations": [{{"case": 1, "claim": "<exact quote or none>"{field}, "approve": true}}]}}
JSON only:"""

    return build


#: The block the anchor arms fill in, set per judging call and read by the builder.
_ANCHOR_BLOCK = {"text": ""}


def _format_evidence_no_source(self, bundle) -> str:
    """`_format_evidence` minus the `source=` field, which is one value in 99% of
    renderings (the s38 audit). Everything else is byte for byte the original."""
    mention = f", mention={bundle.mention_type}" if bundle.mention_type else ""
    lines = [f'  Evidence: span="{bundle.matched_span}"{mention}']
    if bundle.preceding_text:
        lines.append(f'  [prev: "{bundle.preceding_text}"]')
    if bundle.anchor_sentences:
        lines.append("  Anchors (confirmed refs):")
        for a in bundle.anchor_sentences:
            lines.append(f"    {a}")
    return "\n".join(lines)


def _format_evidence_no_anchors(self, bundle) -> str:
    """The evidence line without its anchor list -- the anchor arms print those once."""
    mention = f", mention={bundle.mention_type}" if bundle.mention_type else ""
    lines = [f'  Evidence: source={bundle.source}, span="{bundle.matched_span}"'
             f'{mention}']
    if bundle.preceding_text:
        lines.append(f'  [prev: "{bundle.preceding_text}"]')
    return "\n".join(lines)


def validate_with_deduped_anchors(shared_block: bool):
    """`_validate_with_evidence` with each anchor sentence rendered once per call.

    Two placements, one argument apart. ``shared_block=True`` hoists every anchor
    into one ANCHORS section above the cases, indexed by component; ``False`` keeps
    the anchors in the first case that needs them and points later cases at it. The
    judging protocol, the batching, the rubric and the decisions recorded are the
    original's; what changes is how many times a sentence is written down.
    """

    def _validate(self, candidates, bundles, components, sent_map,
                  phase_tag, stage_label):
        if not candidates:
            return [], {}
        comp_names = L87.get_comp_names(components)
        decisions: dict = {}
        approved = []
        for _, batch in self._iter_batches(candidates, self.JUDGE_BATCH):
            cases = []
            seen: dict = {}          # component -> case number that showed its anchors
            block_lines: list = []
            for i, c in enumerate(batch):
                p = self._prev_prefix(c.sentence_number, sent_map)
                bundle = bundles.get((c.sentence_number, c.component_id))
                evidence = _format_evidence_no_anchors(self, bundle) if bundle else ""
                anchors = list(bundle.anchor_sentences) if bundle else []
                if anchors:
                    first = seen.get(c.component_name)
                    if first is None:
                        seen[c.component_name] = i + 1
                        if shared_block:
                            block_lines.append(f"  {c.component_name}:")
                            block_lines += [f"    {a}" for a in anchors]
                        else:
                            evidence += "\n  Anchors (confirmed refs):\n" + "\n".join(
                                f"    {a}" for a in anchors)
                    elif not shared_block:
                        evidence += (f"\n  Anchors (confirmed refs): as shown in "
                                     f"Case {first}.")
                cases.append((
                    f'Case {i+1}: "{c.matched_text}" -> {c.component_name}\n'
                    f'  {p}"{c.sentence_text}"\n'
                    f'{evidence}',
                    c,
                ))
            _ANCHOR_BLOCK["text"] = (
                "ANCHORS (sentences elsewhere in the document that name a component,\n"
                "read them for the case whose component they are listed under):\n"
                + "\n".join(block_lines) + "\n"
                if shared_block and block_lines else "")
            case_strings = [ct for ct, _ in cases]
            verdicts = self._run_validation_pass(
                comp_names, case_strings, "", phase_tag)
            _ANCHOR_BLOCK["text"] = ""
            for i, (_case_text, c) in enumerate(cases):
                ok, claim, _ = verdicts.get(i, (False, "", ""))
                decisions[(c.sentence_number, c.component_id)] = {
                    "approved": ok,
                    "claim": claim,
                    "path": f"{stage_label}_judged" if ok
                            else f"{stage_label}_rejected",
                    "stage": f"{stage_label}_judge",
                }
                if ok:
                    approved.append(c)
        return approved, decisions

    return _validate


def classify_denotations(with_qualified_clause: bool):
    """`_classify_denotations` with its one rule clause made an argument.

    Re-declared rather than patched through the module global, so the arm removes
    the clause AND the blank line it sat on; `check_parity` captures both prompts
    on real project data and asserts the `True` form is the head's bytes.
    """

    def _classify(self, candidates, sentences):
        sent_map = {s.number: s for s in sentences}
        decisions = {}
        for _, batch in self._iter_batches(candidates, self.JUDGE_BATCH):
            evidence_ids = {
                sentence.number
                for candidate in batch
                for sentence in self._window(candidate.sentence_number, sentences)
            }
            sentence_table = [
                {"sentence": n, "text": sent_map[n].text}
                for n in sorted(evidence_ids)
            ]
            cases = [
                {"case": n, "source": c.sentence_number, "expression": c.matched_text}
                for n, c in enumerate(batch, 1)
            ]
            clause = f"\n{L87.QUALIFIED_CLAUSE}\n" if with_qualified_clause else ""
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
                valid = denotation in {"participant", "associated"} and bool(claim)
                decisions[(candidate.sentence_number, candidate.component_id)] = {
                    "approved": False,
                    "requested_keep": False,
                    "evidence_valid": valid,
                    "claim": claim,
                    "denotation": denotation,
                    "alternative": "not reviewed",
                    "path": "denotation",
                    "stage": "partial_name",
                }
        participants = [
            c for c in candidates
            if decisions.get((c.sentence_number, c.component_id), {}).get(
                "denotation") == "participant"
            and decisions[(c.sentence_number, c.component_id)]["evidence_valid"]
        ]
        return participants, decisions

    return _classify


ORIG_CLASSIFY_DENOTATIONS = SLinker87._classify_denotations

#: `LAYERED_COREF_RULES` minus the clause that enumerates what a component acts on
#: or produces. Everything else in the rubric is byte for byte the head's.
ARTIFACT_CLAUSE = (
    "An expression denoting what a component acts on or produces -- the data, "
    "the artifact, the request, the result -- refers to that thing and not to the "
    "component, however clearly the component is the one acting on it. "
)
NOARTIFACT_COREF = ORIG_LAYERED_COREF.replace(ARTIFACT_CLAUSE, "")

ARMS = {
    "resolve3": {
        "ctl": {},
        "nocasectx": {"coref_prompt": coref_prompt(drop_context=True)},
        "notargetrows": {"coref_prompt": coref_prompt(inline_targets=True)},
    },
    "fullname5": {
        "ctl": {},
        "anchorblock": {"validate_evidence": validate_with_deduped_anchors(True),
                        "validation": validation_prompt_with_block(
                            lambda: _ANCHOR_BLOCK["text"])},
        "anchorref": {"validate_evidence": validate_with_deduped_anchors(False)},
        "nosource": {"format_evidence": _format_evidence_no_source},
    },
    "fullname6": {
        "ctl": {},
        "anchorunion": {"validate_evidence": SLinker88._validate_with_evidence,
                        "format_evidence": SLinker88._format_evidence,
                        "anchor_union": SLinker88._anchor_union},
    },
    "coref5": {
        "ctl": {},
        "noartifact": {"layered_coref": NOARTIFACT_COREF},
    },
    "denot2": {
        "ctl": {},
        "nodenotqual": {"classify": classify_denotations(False)},
    },
}


class Arm:
    """The patches an arm applies, applied for the duration of its stage."""

    def __init__(self, name, spec):
        self.name, self.spec = name, spec

    def __enter__(self):
        L87.LAYERED_COREF_RULES = self.spec.get("layered_coref", ORIG_LAYERED_COREF)
        SLinker87._prompt_coref = staticmethod(
            self.spec.get("coref_prompt", ORIG_COREF_PROMPT))
        SLinker87._validate_with_evidence = self.spec.get(
            "validate_evidence", ORIG_VALIDATE_EVIDENCE)
        SLinker87._format_evidence = self.spec.get(
            "format_evidence", ORIG_FORMAT_EVIDENCE)
        SLinker87._prompt_validation = staticmethod(
            self.spec.get("validation", ORIG_VALIDATION))
        SLinker87._classify_denotations = self.spec.get(
            "classify", ORIG_CLASSIFY_DENOTATIONS)
        # `_anchor_union` exists only on the arm that needs it; s_linker87 has no
        # such method, so it is attached for the arm and removed on exit.
        if "anchor_union" in self.spec:
            SLinker87._anchor_union = staticmethod(self.spec["anchor_union"])
        return self

    def __exit__(self, *exc):
        L87.LAYERED_COREF_RULES = ORIG_LAYERED_COREF
        SLinker87._prompt_coref = staticmethod(ORIG_COREF_PROMPT)
        SLinker87._validate_with_evidence = ORIG_VALIDATE_EVIDENCE
        SLinker87._format_evidence = ORIG_FORMAT_EVIDENCE
        SLinker87._prompt_validation = staticmethod(ORIG_VALIDATION)
        SLinker87._classify_denotations = ORIG_CLASSIFY_DENOTATIONS
        if hasattr(SLinker87, "_anchor_union"):
            del SLinker87._anchor_union


def check_parity():
    """Every re-declared builder must render the head's bytes before any arm runs.

    An arm measures the clause it removes only if the surrounding prompt is the one
    the head sends; a re-declaration that drifted would price the drift instead.
    """
    comp_names = ["Alpha", "Beta"]
    table = [{"sentence": 1, "text": "One."}, {"sentence": 2, "text": "Two."}]
    targets = [{"case": 1, "target": 2, "text": "Two.", "context": [1, 2, 3]}]
    mine = coref_prompt()(comp_names, table, targets)
    theirs = ORIG_COREF_PROMPT(comp_names, table, targets)
    assert mine == theirs, "re-declared _prompt_coref drifted from s_linker87's"

    cases = ['Case 1: "Alpha" -> Alpha\n  "A sentence."']
    for strict in (False, True):
        mine = validation_prompt_with_block(lambda: "")(
            comp_names, cases, "", strict=strict)
        theirs = ORIG_VALIDATION(comp_names, cases, "", strict=strict)
        assert mine == theirs, f"re-declared _prompt_validation drifted (strict={strict})"

    class _B:
        source, matched_span, mention_type = "full_name", "Alpha", ""
        preceding_text, anchor_sentences = "", ["S3: Alpha runs."]
    body = ORIG_FORMAT_EVIDENCE(None, _B())
    assert _format_evidence_no_source(None, _B()) == body.replace(
        "source=full_name, ", ""), "nosource is not the head's line minus source="
    assert ARTIFACT_CLAUSE in ORIG_LAYERED_COREF, "the artifact clause moved"

    # The denotation prompt is built inline in a method, so parity is checked by
    # capturing what each version sends on a real project's own scan.
    comps = parse_pcm_repository(os.path.join(
        BASE, "benchmark", "teammates/model_2021/pcm/teammates.repository"))
    sents = load_sentences(os.path.join(
        BASE, "benchmark", "teammates/text_2021/teammates.txt"))
    captured = []

    def capture(self, prompt, **kw):
        captured.append(prompt)
        return {}

    lk = SLinker87.__new__(SLinker87)
    lk.doc_knowledge = None
    lk._ask = capture.__get__(lk)
    cands = lk._scan(sents, comps)
    assert cands, "the parity check needs a project whose scan proposes something"
    ORIG_CLASSIFY_DENOTATIONS(lk, cands, sents)
    classify_denotations(True)(lk, cands, sents)
    classify_denotations(False)(lk, cands, sents)
    n = len(captured) // 3
    theirs, mine, cut = (captured[:n], captured[n:2 * n], captured[2 * n:])
    assert mine == theirs, "re-declared _classify_denotations drifted from s_linker87's"
    assert all(len(a) - len(b) == len(L87.QUALIFIED_CLAUSE) + 2
               for a, b in zip(theirs, cut)), \
        "nodenotqual is not the head's prompt minus exactly the clause and its line"
    print(f"prompt parity: re-declared builders render s_linker87's bytes "
          f"({n} denotation prompts compared)")


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
    lk = SLinker87(backend=LLMBackend.OPENAI)
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


#: Which recorded stages an arm's own stage is composed with.
OTHER_STAGES = {
    "resolve3": ("full_name", "partial_name"),
    "fullname5": ("partial_name", "coreference"),
    "fullname6": ("partial_name", "coreference"),
    "coref5": ("full_name", "partial_name"),
    "denot2": ("full_name", "coreference"),
}


def run_group(group, model, runs, out_dir):
    run_dirs, variant = recorded_runs(model)
    if not run_dirs:
        sys.exit(f"no recorded runs for {model}: {RECORDED[model][0]}")
    arms = ARMS[group]
    kept = {arm: collections.defaultdict(dict) for arm in arms}
    stage_tot = {arm: collections.Counter() for arm in arms}
    prompt_chars = {arm: [] for arm in arms}
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
                lk = linker_for(knowledge, sink)
                with Arm("ctl", {}):
                    shared_candidates = lk._extract_named_mentions(
                        sents, comps, name_to_id, sent_map)
                calls["extract"] += len(sink)

            for arm, spec in arms.items():
                sink = []
                lk = linker_for(knowledge, sink)
                with Arm(arm, spec):
                    if group.startswith("fullname"):
                        pairs = judge_fullname(lk, list(shared_candidates.values()),
                                               comps, sent_map)
                    elif group == "denot2":
                        links, _ = lk._run_partial_name_linker(
                            sents, comps, sent_map)
                        pairs = {(l.sentence_number, l.component_id) for l in links}
                    elif group == "resolve3":
                        links, _ = lk._run_coreference_linker(
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
        total = sum(prompt_chars[arm]) / runs if prompt_chars[arm] else 0
        print(f"  {arm:<13} gold {stage_tot[arm]['g'] / runs:6.1f}  "
              f"spurious {stage_tot[arm]['n'] / runs:6.1f}  "
              f"calls {calls[arm] / runs:5.1f}  mean prompt {chars:7.0f} chars  "
              f"stage bytes/run {total:9.0f}")
    compose(group, model, runs, arms, kept, variant, run_dirs)


def compose(group, model, runs, arms, kept, variant, run_dirs):
    """The exact pipeline score: the arm's kept pairs, unioned with the same
    recorded run's other two stages, which no prompt here touches."""
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
    ap.add_argument("--projects", nargs="*", default=None)
    ap.add_argument("--out", default=os.environ.get(
        "AB_OUT", "../results/compaction_round"))
    args = ap.parse_args()
    check_parity()
    if args.projects:
        for name in list(PROJECTS):
            if name not in args.projects:
                del PROJECTS[name]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_group(args.group, args.model, args.runs, out_dir)


if __name__ == "__main__":
    main()
