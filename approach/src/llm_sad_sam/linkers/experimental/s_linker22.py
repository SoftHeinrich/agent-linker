"""s_linker22 — s21 workflow with typed extraction and F2-oriented validation.

EXPERIMENTAL. This is not a post-processor over frozen s21 outputs. It subclasses
`SLinker21` and keeps the same live six-phase workflow, but changes two seams:

* Phase 2 keeps s21 Framing-C extraction as the live floor, then adds a typed
  extraction pass that emits (`AFFIRMATIVE`, `CONTRAST`, `IMPLICIT`, `ANAPHORA`,
  `CODEPATH`) for candidates the floor missed.
* Phase 4 validates floor candidates with s21's unchanged P1/P2 validator.
  Typed-only `AFFIRMATIVE` candidates use the same validator after a generic
  exact/terminal/no-code evidence filter. Typed-only `CONTRAST` candidates use a
  contrast-specific claim-before-verdict validator.

The design is the inline version of the pilot's best F2 policy
`exact_or_terminal_no_code`: same philosophy and measured target behavior, but
inside the linker workflow instead of loading frozen outputs and augmenting them.
"""
from __future__ import annotations

import re

from llm_sad_sam.core.data_types_v2 import CandidateLink
from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names, parse_snum
from llm_sad_sam.linkers.experimental.s_linker21 import (
    P1_FOCUS,
    P2_FOCUS,
    SLinker21,
)


MODES = ("AFFIRMATIVE", "CONTRAST", "IMPLICIT", "ANAPHORA", "CODEPATH")

CODE_HINT_RE = re.compile(
    r"(^|[^A-Za-z0-9])("
    r"[a-z][a-z0-9_]*(?:\.[a-zA-Z_][A-Za-z0-9_]*)+|"
    r"[A-Za-z_][A-Za-z0-9_]*(?:Servlet|Factory|Action|Controller|Util|Test|Tests|Socket|Layer)|"
    r"[A-Za-z0-9]+\s+tests?"
    r")($|[^A-Za-z0-9])",
    re.IGNORECASE,
)


CONTRAST_RULES = (
    "The component appears inside a negation, contrast, or exclusion. Approve when "
    "the sentence still asserts a fact ABOUT THIS component's role in the system: "
    "it is compared against, excluded from, or offered as an alternative to something. "
    "Reject only when the sentence denies that this component is part of the system at "
    "all, or the token is a different entity / product-brand name."
)


class SLinker22(SLinker21):
    """s21 pipeline with typed extraction + F2 validation policy."""

    _VARIANT_NAME = "s_linker22"

    @staticmethod
    def _prompt_typed_extraction(comp_names, mappings, batch) -> str:
        alias_line = f'KNOWN ALIASES: {", ".join(mappings)}' if mappings else ""
        return f"""Extract documentation references to architecture components.

COMPONENT CATALOG:
{chr(10).join(f"- {name}" for name in comp_names)}
{alias_line}

Read each numbered sentence. Choose components only from the catalog.

For every reference, output:
- sentence: the sentence number
- component: exact catalog name
- quote: exact words in the sentence carrying the reference
- mode: one of {", ".join(MODES)}

Mode meanings:
AFFIRMATIVE = the component is plainly named or aliased as an architecture participant.
CONTRAST = the component is named or aliased inside contrast/negation/exclusion.
IMPLICIT = generic role/example phrase without the proper name.
ANAPHORA = pronoun or role phrase pointing back to a component.
CODEPATH = reference occurs only inside a code/package/member path.

Do not output a component unless specific quoted words in the sentence support it.

DOCUMENT:
{chr(10).join([f"S{s.number}: {s.text}" for s in batch])}

Return JSON:
{{"references": [{{"sentence": 1, "component": "Name", "quote": "exact words", "mode": "AFFIRMATIVE"}}]}}
JSON only:"""

    def _run_framing_c(self, sentences, components, name_to_id, sent_map) -> dict:
        self._s22_candidate_modes = {}
        base = SLinker21._run_framing_c(self, sentences, components, name_to_id, sent_map)
        comp_names = get_comp_names(components)
        mappings = (
            [f"{term}={entry.component}" for term, entry in self.doc_knowledge.aliases.items()
             if entry.scope == "global"]
            if self.doc_knowledge else []
        )
        typed_results = self._run_parallel({
            "pass1": lambda: self._run_typed_extraction_pass(
                sentences, comp_names, mappings, name_to_id, sent_map,
                pass_label="[T1] ", phase_tag="phase_2_typed_pass1"),
            "pass2": lambda: self._run_typed_extraction_pass(
                sentences, comp_names, mappings, name_to_id, sent_map,
                pass_label="[T2] ", phase_tag="phase_2_typed_pass2"),
        })
        typed_pass1, typed_pass2 = typed_results["pass1"], typed_results["pass2"]
        typed_union = {**typed_pass2, **typed_pass1}
        for key in base:
            self._s22_candidate_modes.pop(key, None)
        merged = {**typed_union, **base}
        print(
            f"    Typed add-on: P1={len(typed_pass1)} P2={len(typed_pass2)} "
            f"union={len(typed_union)} typed_only={len(merged) - len(base)}"
        )
        self._s22_typed_pass1 = typed_pass1
        self._s22_typed_pass2 = typed_pass2
        self._s22_typed_union = typed_union
        return merged

    def _run_typed_extraction_pass(self, sentences, comp_names, mappings,
                                   name_to_id, sent_map, pass_label="", phase_tag=None):
        if phase_tag:
            self.llm.set_phase(phase_tag)
        if not hasattr(self, "_s22_candidate_modes"):
            self._s22_candidate_modes = {}
        batch_size = 50
        candidates: dict = {}
        for batch_num, batch in self._iter_batches(sentences, batch_size):
            if len(sentences) > batch_size:
                print(f"    {pass_label}batch {batch_num}: "
                      f"S{batch[0].number}-S{batch[-1].number} ({len(batch)} sents)")
            prompt = self._prompt_typed_extraction(comp_names, mappings, batch)
            data = self._ask(prompt, timeout=240,
                             label=f"{pass_label}batch", require="references")
            if not data:
                continue
            for ref in data.get("references", []):
                cname = ref.get("component")
                snum = parse_snum(ref.get("sentence"))
                if snum is None or not cname or cname not in name_to_id:
                    continue
                sent = sent_map.get(snum)
                if not sent:
                    continue
                matched = ref.get("quote") or ref.get("matched_text") or ""
                if matched and matched.lower() not in sent.text.lower():
                    continue
                mode = str(ref.get("mode", "AFFIRMATIVE")).upper().strip()
                if mode not in MODES:
                    mode = "AFFIRMATIVE"
                key = (snum, name_to_id[cname])
                # Preserve first accepted mode for this pass; the two-pass union keeps
                # pass1 over pass2, as in s21.
                if key not in candidates:
                    candidates[key] = CandidateLink(
                        snum, sent.text, cname, name_to_id[cname],
                        matched, source="entity",
                    )
                    self._s22_candidate_modes.setdefault(key, mode)
        return candidates

    def _validate_with_evidence(self, candidates, bundles, components, sent_map,
                                p1_tag, p2_tag, stage_label):
        if not candidates:
            return [], {}
        comp_names = get_comp_names(components)
        decisions: dict = {}
        approved = []
        for _, batch in self._iter_batches(candidates, 25):
            standard_cases = []
            contrast_cases = []
            for c in batch:
                key = (c.sentence_number, c.component_id)
                mode = getattr(self, "_s22_candidate_modes", {}).get(key)
                if mode is None:
                    standard_cases.append((self._s22_base_case_text(c, bundles, sent_map), c, "BASE"))
                    continue
                if mode not in ("AFFIRMATIVE", "CONTRAST"):
                    decisions[key] = {
                        "approved": False, "p1": False, "p2": False,
                        "mode": mode, "path": f"{stage_label}_mode_reject",
                        "stage": f"{stage_label}_typed",
                    }
                    continue
                if mode == "AFFIRMATIVE" and not self._s22_affirmative_candidate_allowed(c):
                    decisions[key] = {
                        "approved": False, "p1": False, "p2": False,
                        "mode": mode, "path": f"{stage_label}_evidence_filter_reject",
                        "stage": f"{stage_label}_typed",
                    }
                    continue
                case_text = self._s22_case_text(c, bundles, sent_map, mode)
                if mode == "CONTRAST":
                    contrast_cases.append((case_text, c))
                else:
                    standard_cases.append((case_text, c, "AFFIRMATIVE"))

            if standard_cases:
                case_strings = [
                    ct.replace("Case {case}:", f"Case {i + 1}:")
                    for i, (ct, _c, _mode) in enumerate(standard_cases)
                ]
                r1 = self._run_validation_pass(comp_names, case_strings, P1_FOCUS, p1_tag)
                r2 = self._run_validation_pass(comp_names, case_strings, P2_FOCUS, p2_tag)
                for i, (_case_text, c, mode) in enumerate(standard_cases):
                    p1 = r1.get(i, False)
                    p2 = r2.get(i, False)
                    ok = p1 and p2
                    key = (c.sentence_number, c.component_id)
                    decisions[key] = {
                        "approved": ok, "p1": p1, "p2": p2, "mode": mode,
                        "path": f"{stage_label}_twopass" if ok else f"{stage_label}_twopass_reject",
                        "stage": f"{stage_label}_typed",
                    }
                    if ok:
                        approved.append(c)

            if contrast_cases:
                case_strings = [ct for ct, _ in contrast_cases]
                r = self._run_contrast_validation_pass(comp_names, case_strings, p1_tag)
                for i, (_case_text, c) in enumerate(contrast_cases):
                    ok = r.get(i, False)
                    key = (c.sentence_number, c.component_id)
                    decisions[key] = {
                        "approved": ok, "p1": ok, "p2": True, "mode": "CONTRAST",
                        "path": f"{stage_label}_contrast" if ok else f"{stage_label}_contrast_reject",
                        "stage": f"{stage_label}_typed",
                    }
                    if ok:
                        approved.append(c)
        return approved, decisions

    def _s22_base_case_text(self, candidate, bundles, sent_map) -> str:
        p = self._prev_prefix(candidate.sentence_number, sent_map)
        bundle = bundles.get((candidate.sentence_number, candidate.component_id))
        evidence_block = self._format_evidence(bundle) if bundle else ""
        return (
            f'Case {{case}}: "{candidate.matched_text}" -> {candidate.component_name}\n'
            f'  {p}"{candidate.sentence_text}"\n'
            f'{evidence_block}'
        )

    def _s22_case_text(self, candidate, bundles, sent_map, mode: str) -> str:
        p = self._prev_prefix(candidate.sentence_number, sent_map)
        bundle = bundles.get((candidate.sentence_number, candidate.component_id))
        evidence_block = self._format_evidence(bundle) if bundle else ""
        return (
            f'Case {{case}}: "{candidate.matched_text}" -> {candidate.component_name}\n'
            f'  mode={mode}\n'
            f'  {p}"{candidate.sentence_text}"\n'
            f'{evidence_block}'
        )

    def _run_contrast_validation_pass(self, comp_names, cases, phase_tag=None):
        if phase_tag:
            self.llm.set_phase(f"{phase_tag}_contrast")
        prompt = self._prompt_contrast_validation(comp_names, cases)
        data = self._ask(prompt, timeout=120, label="Contrast validation pass",
                         require="validations")
        results: dict[int, bool] = {}
        if data:
            for v in data.get("validations", []):
                idx = v.get("case", 0) - 1
                if 0 <= idx < len(cases):
                    val = v.get("approve", False)
                    results[idx] = val is True or (isinstance(val, str)
                                                   and val.lower() == "true")
        return results

    @staticmethod
    def _prompt_contrast_validation(comp_names, cases) -> str:
        numbered = []
        for i, case in enumerate(cases, 1):
            numbered.append(case.replace("Case {case}:", f"Case {i}:"))
        return f"""Validate trace links where the component is named in CONTRAST or NEGATION.

COMPONENTS: {', '.join(comp_names)}

{CONTRAST_RULES}

For each case, FIRST quote the exact contrast/negation words, THEN decide approve
true/false.

CASES:
{chr(10).join(numbered)}

Return JSON:
{{"validations": [{{"case": 1, "claim": "<quote>", "approve": true}}]}}
JSON only:"""

    def _s22_affirmative_candidate_allowed(self, candidate) -> bool:
        quote = (candidate.matched_text or "").strip()
        sentence = candidate.sentence_text or ""
        component = candidate.component_name or ""
        if not quote or quote.lower() not in sentence.lower():
            return False
        if self._s22_code_like(quote) or self._s22_code_like(sentence[:160]):
            return False
        return (
            self._s22_contains_name(quote, component)
            or self._s22_contains_name(sentence, component)
            or self._s22_terminal_quote(quote, component)
        )

    @staticmethod
    def _s22_code_like(text: str) -> bool:
        return bool(CODE_HINT_RE.search(text or ""))

    @staticmethod
    def _s22_contains_name(text: str, component: str) -> bool:
        if not text or not component:
            return False
        return re.search(
            rf"(?<![A-Za-z0-9]){re.escape(component)}(?![A-Za-z0-9])",
            text,
            re.IGNORECASE,
        ) is not None

    @staticmethod
    def _s22_terminal_quote(quote: str, component: str) -> bool:
        q = re.sub(r"^(the|a|an)\s+", "", (quote or "").strip().lower())
        tokens = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+", component or "")
        if not tokens:
            return False
        terminal = tokens[-1].lower()
        return q == terminal or q.endswith(" " + terminal)
