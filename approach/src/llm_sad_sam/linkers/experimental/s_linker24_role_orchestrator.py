"""S24 replacement controller with a project-specific catalog-handle tool."""
from __future__ import annotations

import json
import re
import unicodedata

from llm_sad_sam.core.data_types_v2 import CandidateLink, SadSamLink
from llm_sad_sam.core.document_loader_v2 import load_sentences
from llm_sad_sam.linkers.experimental._s_linker24_orchestrator_base import (
    _SLinker24OrchestratorBase,
)


class SLinker24RoleOrchestrator(_SLinker24OrchestratorBase):
    """Replacement controller with non-overlapping reference-mode tools."""

    _VARIANT_NAME = "s_linker24_role_orchestrator"
    PHASE_TOOLS = (
        "entity_pipeline",
        "coreference_pipeline",
        "relation_role_resolution",
    )

    def _tool_catalog(self):
        return """- entity_pipeline: exact catalog-name, approved-alias, and
  catalog-equivalent orthographic identity evidence, followed by the existing
  two-pass validator.
- coreference_pipeline: pronoun, demonstrative, and anaphoric reference
  resolution, followed by its existing validator.
- relation_role_resolution: resolve unique terminal participant nouns from
  compound catalog names through grounded local discourse evidence.
- finalize: return the union of links produced by completed tools."""

    def _choose_tool(self, profile, remaining, history, current, sent_map):
        prompt = f"""You orchestrate trace-linking tools for one software project.
There is no base linker and no protected floor. Choose one available,
evidence-backed tool. You cannot propose, validate, add, or remove links.

TOOLS
{self._tool_catalog()}

PROJECT PROFILE
{json.dumps(profile)}

COMPLETED TOOL FEEDBACK
{json.dumps(history)}

CURRENT LINKS
{json.dumps(self._controller_link_view(current))}

AVAILABLE ACTIONS
{json.dumps(remaining)}

Select an unused capability only when exact document words show its evidence
mode remains unresolved. Rejected evidence must not be retried through another
tool. Every listed capability already has project-profile evidence; order them
by the unresolved evidence, then finish after all listed capabilities run.
Counts are observations, not thresholds.

Return JSON only:
{{"action":"one available action",
"evidence_quotes":["exact source words supporting the call"],
"unresolved_obligation":"evidence mode or none",
"reason":"brief workflow reason"}}
"""
        data = self._ask(
            prompt,
            phase=f"phase_24_role_orchestrator_{len(history) + 1}",
            require_present="action",
            label="S24 role controller",
        )
        action = str(data.get("action", "")).strip()
        if action not in remaining:
            raise RuntimeError(f"invalid replacement action: {action!r}")
        quotes = [
            str(quote).strip().strip("\"'")
            for quote in data.get("evidence_quotes", [])
            if str(quote).strip().strip("\"'")
        ]
        document = "\n".join(
            item["text"] for item in profile["document"]
        ).casefold()
        grounded_quotes = [
            quote for quote in quotes if quote.casefold() in document
        ]
        if not grounded_quotes:
            raise RuntimeError(
                f"ungrounded workflow evidence for {action!r}: {quotes!r}"
            )
        decision = {
            "evidence_quotes": grounded_quotes,
            "discarded_evidence_quotes": [
                quote for quote in quotes if quote not in grounded_quotes
            ],
            "unresolved_obligation": str(
                data.get("unresolved_obligation", "")
            ).strip(),
            "reason": str(data.get("reason", "")).strip(),
        }
        return action, decision

    def _project_profile(self, sentences, components):
        profile = super()._project_profile(sentences, components)
        handles = self._catalog_role_handles(components)
        profile["role_handle_evidence"] = [
            {
                **handle,
                "occurrences": [
                    {
                        "sentence": sentence.number,
                        "quote": matched,
                    }
                    for sentence in sentences
                    if (
                        matched := self._find_handle(
                            sentence.text, handle["expression"]
                        )
                    )
                ],
            }
            for handle in handles
            if any(
                self._find_handle(sentence.text, handle["expression"])
                for sentence in sentences
            )
        ]
        profile["entity_orthographic_evidence"] = self._link_view(
            [
                SadSamLink(
                    candidate.sentence_number,
                    candidate.component_id,
                    candidate.component_name,
                    source="entity_orthographic_candidate",
                )
                for candidate in self._lexical_entity_candidates(
                    sentences, components
                )
            ],
            {sentence.number: sentence for sentence in sentences},
        )
        return profile

    def _available_tools(self, profile):
        tools = super()._available_tools(profile)
        if not profile.get("role_handle_evidence"):
            tools.remove("relation_role_resolution")
        return tools

    def _select_entity_candidates(self, candidates, sent_map):
        forms_by_component = self._identity_forms_by_component()
        return [
            candidate
            for candidate in candidates
            if any(
                self._find_exact_form(candidate.sentence_text, form)
                for form in (
                    candidate.component_name,
                    *forms_by_component.get(candidate.component_name, []),
                )
            )
        ]

    def _identity_forms_by_component(self):
        aliases = getattr(
            getattr(self, "doc_knowledge", None), "aliases", {}
        )
        forms_by_component = {}
        for term, entry in aliases.items():
            component = getattr(entry, "component", entry)
            forms_by_component.setdefault(component, []).append(term)
        return forms_by_component

    def _augment_entity_candidates(
        self, candidates, sentences, components
    ):
        merged = {
            (candidate.sentence_number, candidate.component_id): candidate
            for candidate in candidates
        }
        for candidate in self._lexical_entity_candidates(
            sentences, components
        ):
            merged.setdefault(
                (candidate.sentence_number, candidate.component_id),
                candidate,
            )
        return list(merged.values())

    @staticmethod
    def _entity_link_source(candidate):
        if candidate.source == "entity_orthographic":
            return "s24_entity_orthographic"
        return "entity"

    def _run_selected_tool(
        self,
        action,
        sentences,
        components,
        name_to_id,
        current,
        sent_map,
    ):
        if action == "relation_role_resolution":
            return self._run_relation_role_tool(
                sentences, components, current, sent_map
            )
        return super()._run_selected_tool(
            action,
            sentences,
            components,
            name_to_id,
            current,
            sent_map,
        )

    def _run_relation_role_tool(
        self, sentences, components, current_links, sent_map
    ):
        handles = self._catalog_role_handles(components)
        candidates = self._apply_role_handles(
            handles, sentences, components, current_links
        )
        full_sentences = (
            load_sentences(self._current_text_path)
            if self._current_text_path
            else sentences
        )
        approved, decisions = self._review_role_candidates(
            candidates, full_sentences
        )
        links = [
            SadSamLink(
                candidate.sentence_number,
                candidate.component_id,
                candidate.component_name,
                source="s24_relation_role",
            )
            for candidate in approved
        ]
        return links, {
            "role_handles": handles,
            "proposed": self._link_view(
                [
                    SadSamLink(
                        candidate.sentence_number,
                        candidate.component_id,
                        candidate.component_name,
                        source="relation_role_candidate",
                    )
                    for candidate in candidates
                ],
                sent_map,
            ),
            "accepted": self._link_view(links, sent_map),
            "handle_decisions": self._decision_view(decisions),
        }

    def _review_role_candidates(self, candidates, full_sentences):
        if not candidates:
            return [], {}
        sent_map = {
            sentence.number: sentence for sentence in full_sentences
        }
        forms_by_component = self._identity_forms_by_component()
        targets = sorted({
            candidate.component_name for candidate in candidates
        })
        profiles = []
        for target in targets:
            forms = [target, *forms_by_component.get(target, [])]
            anchors = []
            for sentence in full_sentences:
                matches = [
                    self._find_exact_form(sentence.text, form)
                    for form in forms
                ]
                matches = [match for match in matches if match]
                if matches:
                    anchors.append({
                        "sentence": sentence.number,
                        "text": sentence.text,
                        "identity_forms": matches,
                    })
            profiles.append({"target": target, "anchors": anchors})
        anchors_by_target = {
            item["target"]: {
                anchor["sentence"]: anchor
                for anchor in item["anchors"]
            }
            for item in profiles
        }
        cases = [
            {
                "case": number,
                "sentence": candidate.sentence_number,
                "participant": candidate.matched_text,
                "target": candidate.component_name,
                "text": candidate.sentence_text,
            }
            for number, candidate in enumerate(candidates, 1)
        ]
        document = [
            {"sentence": sentence.number, "text": sentence.text}
            for sentence in full_sentences
        ]
        prompt = f"""Resolve generic or inflected architectural participants
using the document's discourse structure. For each case, keep the mapping only
when the highlighted participant denotes the listed target component in its
section or discourse chain.

An approval must identify an explicit target identity anchor, the participant's
architectural role in the source sentence, and the strongest competing
referent. A common role noun may validly denote a deployed component instance,
its service endpoint, its owned state, or its user-facing instances; do not
reject it merely for being generic or plural when the active discourse scope
and architectural claim select the target. A competing referent must be an
explicit locally plausible entity, not an invented unnamed deployment. When
an apparent alternate label is resolved to the target by a local discourse
chain, cite the exact bridge sentence that makes that resolution possible.

Reject physical host capacity, users or browsers, protocol/technology use,
code paths, and cases where multiple explicit referents remain plausible.
The highlighted noun must itself be a semantic participant in a finite
architectural claim. Reject fragments and uses where it merely modifies the
name of a workflow, process, stage, diagram, artifact, or technology. Full
names, aliases, pronouns, and orthographic variants are owned by other tools.

DOCUMENT
{json.dumps(document)}

TARGET IDENTITY ANCHORS
{json.dumps(profiles)}

CASES
{json.dumps(cases)}

Return JSON only:
{{"judgments":[{{"case":1,"keep":true,
"section_anchor":"exact document quote",
"identity_anchor_sentence":1,
"identity_anchor":"exact quote from that anchor sentence",
"scope_bridge_sentence":1,
"scope_bridge":"exact quote establishing the local discourse chain",
"claim":"exact quote from the source sentence",
"participant_role":"brief role",
"competing_referent":"strongest alternative or none"}}]}}
"""
        data = self._ask(
            prompt,
            phase="phase_24_discourse_scope_review",
            require_present="judgments",
            label="S24 discourse-scope review",
            timeout=240,
        )
        document_text = "\n".join(
            sentence.text for sentence in full_sentences
        ).casefold()
        by_case = {}
        for item in data.get("judgments", []):
            if not str(item.get("case", "")).isdigit():
                continue
            number = int(item["case"])
            if not 1 <= number <= len(candidates):
                continue
            candidate = candidates[number - 1]
            anchor_value = str(
                item.get("identity_anchor_sentence", "")
            )
            bridge_value = str(
                item.get("scope_bridge_sentence", "")
            )
            anchor_sentence = (
                int(anchor_value) if anchor_value.isdigit() else 0
            )
            bridge_sentence = (
                int(bridge_value) if bridge_value.isdigit() else 0
            )
            anchor = str(item.get("identity_anchor", "")).strip()
            bridge = str(item.get("scope_bridge", "")).strip()
            section = str(item.get("section_anchor", "")).strip()
            claim = str(item.get("claim", "")).strip()
            role = str(item.get("participant_role", "")).strip()
            competing = str(
                item.get("competing_referent", "")
            ).strip()
            allowed_anchor = anchors_by_target.get(
                candidate.component_name, {}
            ).get(anchor_sentence)
            allowed_bridge = sent_map.get(bridge_sentence)
            evidence_valid = (
                bool(section)
                and section.casefold() in document_text
                and allowed_anchor is not None
                and bool(anchor)
                and anchor.casefold()
                in allowed_anchor["text"].casefold()
                and allowed_bridge is not None
                and bool(bridge)
                and bridge.casefold()
                in allowed_bridge.text.casefold()
                and bool(claim)
                and claim.casefold()
                in candidate.sentence_text.casefold()
                and bool(role)
                and bool(competing)
            )
            by_case[number] = {
                "approved": (
                    item.get("keep") is True and evidence_valid
                ),
                "requested_keep": item.get("keep") is True,
                "evidence_valid": evidence_valid,
                "section_anchor": section,
                "identity_anchor_sentence": anchor_sentence,
                "identity_anchor": anchor,
                "scope_bridge_sentence": bridge_sentence,
                "scope_bridge": bridge,
                "claim": claim,
                "participant_role": role,
                "competing_referent": competing,
            }
        approved = [
            candidate
            for number, candidate in enumerate(candidates, 1)
            if by_case.get(number, {}).get("approved") is True
        ]
        decisions = {
            (candidate.sentence_number, candidate.component_id): {
                **by_case.get(number, {
                    "approved": False,
                    "requested_keep": False,
                    "evidence_valid": False,
                    "competing_referent": "missing judgment",
                }),
                "path": "discourse_scope_review",
                "stage": "relation_role_resolution",
            }
            for number, candidate in enumerate(candidates, 1)
        }
        return approved, decisions

    def _review_identity_candidates(
        self,
        candidates,
        full_sentences,
        *,
        case_key,
        instruction,
        phase,
        label,
        path,
        stage,
    ):
        if not candidates:
            return [], {}
        forms_by_component = self._identity_forms_by_component()
        targets = sorted({
            candidate.component_name for candidate in candidates
        })
        profiles = []
        for target in targets:
            forms = [target, *forms_by_component.get(target, [])]
            profiles.append({
                "target": target,
                "identity_anchors": [
                    {
                        "sentence": sentence.number,
                        "text": sentence.text,
                    }
                    for sentence in full_sentences
                    if any(
                        self._find_exact_form(sentence.text, form)
                        for form in forms
                    )
                ],
            })
        cases = [
            {
                "case": number,
                "sentence": candidate.sentence_number,
                case_key: candidate.matched_text,
                "target": candidate.component_name,
                "text": candidate.sentence_text,
            }
            for number, candidate in enumerate(candidates, 1)
        ]
        prompt = f"""{instruction}
For each case, keep the mapping only when the highlighted expression refers to the
listed target component. Identity anchors show explicit project usage.

TARGET PROFILES
{json.dumps(profiles)}

CASES
{json.dumps(cases)}

JSON only:
{{"judgments":[{{"case":1,"keep":true,"referent":"brief referent"}}]}}
"""
        data = self._ask(
            prompt,
            phase=phase,
            require_present="judgments",
            label=label,
            timeout=240,
        )
        by_case = {
            int(item["case"]): {
                "approved": item.get("keep") is True,
                "referent": str(item.get("referent", "")).strip(),
            }
            for item in data.get("judgments", [])
            if str(item.get("case", "")).isdigit()
        }
        approved = [
            candidate
            for number, candidate in enumerate(candidates, 1)
            if by_case.get(number, {}).get("approved") is True
        ]
        decisions = {
            (candidate.sentence_number, candidate.component_id): {
                **by_case.get(number, {
                    "approved": False,
                    "referent": "missing judgment",
                }),
                "path": path,
                "stage": stage,
            }
            for number, candidate in enumerate(candidates, 1)
        }
        return approved, decisions

    @staticmethod
    def _lexical_signature(expression):
        normalized = unicodedata.normalize("NFKC", expression)
        normalized = normalized.replace("-", " ").replace("_", " ")
        return tuple(
            token.casefold()
            for token in re.findall(
                r"[A-Z]+(?=[A-Z][a-z]|\b)|"
                r"[A-Z]?[a-z]+|[A-Z]+|\d+",
                normalized,
            )
        )

    @staticmethod
    def _qualified_identifier_boundary(text, start, end):
        before = text[start - 1] if start else ""
        after = text[end] if end < len(text) else ""
        dotted_before = (
            before == "." and start > 1 and text[start - 2].isalnum()
        )
        dotted_after = (
            after == "."
            and end + 1 < len(text)
            and text[end + 1].isalnum()
        )
        joined_before = before in "-_" or (
            before and before.isalnum()
        )
        joined_after = after in "-_" or (
            after and after.isalnum()
        )
        return (
            dotted_before
            or dotted_after
            or joined_before
            or joined_after
        )

    @classmethod
    def _lexical_entity_candidates(cls, sentences, components):
        """Find exact, catalog-unique orthographic entity variants."""
        word_pattern = re.compile(r"[A-Za-z0-9]+")
        owners = {}
        for component in components:
            signature = cls._lexical_signature(component.name)
            if signature:
                owners.setdefault(signature, []).append(component)
        max_tokens = max((len(item) for item in owners), default=0)
        candidates = {}
        for sentence in sentences:
            words = list(word_pattern.finditer(sentence.text))
            for start_index, first in enumerate(words):
                for end_index in range(
                    start_index,
                    min(len(words), start_index + max_tokens),
                ):
                    last = words[end_index]
                    if end_index > start_index:
                        separator = sentence.text[
                            words[end_index - 1].end():last.start()
                        ]
                        if not re.fullmatch(r"[\s_-]+", separator):
                            break
                    start, end = first.start(), last.end()
                    if cls._qualified_identifier_boundary(
                        sentence.text, start, end
                    ):
                        continue
                    surface = sentence.text[start:end]
                    targets = owners.get(
                        cls._lexical_signature(surface), ()
                    )
                    if len(targets) != 1:
                        continue
                    component = targets[0]
                    if surface.casefold() == component.name.casefold():
                        continue
                    key = (sentence.number, component.id)
                    candidates[key] = CandidateLink(
                        sentence.number,
                        sentence.text,
                        component.name,
                        component.id,
                        surface,
                        source="entity_orthographic",
                    )
        return list(candidates.values())

    @staticmethod
    def _catalog_role_handles(components):
        tokens_by_component = {
            component.name: re.findall(
                r"[A-Za-z]+[A-Za-z0-9]*|\d+", component.name
            )
            for component in components
        }
        owners = {}
        for component_name, tokens in tokens_by_component.items():
            if len(tokens) > 1:
                owners.setdefault(
                    tokens[-1].casefold(), set()
                ).add(component_name)
        handles = []
        for component_name, tokens in tokens_by_component.items():
            if len(tokens) < 2:
                continue
            terminal = tokens[-1]
            if len(owners[terminal.casefold()]) != 1:
                continue
            if re.search(
                r"(?:tion|sion|ment|ance|ence|ing)$",
                terminal,
                re.IGNORECASE,
            ):
                continue
            expressions = [terminal]
            if (
                terminal.isalpha()
                and not terminal.casefold().endswith("s")
            ):
                expressions.append(f"{terminal}s")
            handles.extend(
                {
                    "expression": expression,
                    "component": component_name,
                }
                for expression in expressions
            )
        return handles

    def _apply_role_handles(
        self, handles, sentences, components, current_links
    ):
        name_to_id = {component.name: component.id for component in components}
        forms_by_component = self._identity_forms_by_component()
        current = {
            (link.sentence_number, link.component_id)
            for link in current_links
        }
        candidates = {}
        for handle in handles:
            component = handle["component"]
            component_id = name_to_id[component]
            expression = handle["expression"]
            for sentence in sentences:
                key = (sentence.number, component_id)
                matched = self._find_handle(sentence.text, expression)
                identity_forms = [
                    component,
                    *forms_by_component.get(component, []),
                ]
                if (
                    key in current
                    or not matched
                    or any(
                        self._find_exact_form(sentence.text, form)
                        for form in identity_forms
                    )
                ):
                    continue
                candidates[key] = CandidateLink(
                    sentence.number,
                    sentence.text,
                    component,
                    component_id,
                    matched,
                    source="relation_role",
                )
        return list(candidates.values())

    @staticmethod
    def _find_handle(text, expression):
        for match in re.finditer(
            rf"(?<![\w-]){re.escape(expression)}(?![\w-])",
            text,
            re.IGNORECASE,
        ):
            if not SLinker24RoleOrchestrator._qualified_identifier_boundary(
                text, match.start(), match.end()
            ):
                return match.group(0)
        return ""

    @staticmethod
    def _find_exact_form(text, expression):
        match = re.search(
            rf"(?<!\w){re.escape(expression)}(?!\w)",
            text,
            re.IGNORECASE,
        )
        return match.group(0) if match else ""
