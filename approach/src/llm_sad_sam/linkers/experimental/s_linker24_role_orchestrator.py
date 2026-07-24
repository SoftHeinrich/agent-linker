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
    TOOL_CONTRACTS = {
        "entity_pipeline": "named components",
        "coreference_pipeline": "references to introduced components",
        "relation_role_resolution": "contextual participant nouns",
    }
    _REFERENCE = re.compile(
        r"\b(?:it|its|they|their|them|this|these|those|"
        r"such|former|latter)\b",
        re.IGNORECASE,
    )

    def _choose_tool(self, profile, remaining, history, current, sent_map):
        del current, sent_map
        signals = self._compact_signals(profile, remaining)
        prompt = f"""Choose the next trace-linking tool.
You schedule tools; tools alone decide links.

TOOLS
{json.dumps(self.TOOL_CONTRACTS)}

EVIDENCE
{json.dumps(signals)}

DONE
{json.dumps(self._compact_outcomes(history))}

ACTIONS
{json.dumps(remaining)}

Use a tool only for its evidence kind. Do not retry rejected evidence through
another tool. Only evidence-bearing tools are listed; completion is automatic
when none remain.

JSON only:
{{"action":"action","evidence":[1],"reason":"brief reason"}}
"""
        data = self._ask(
            prompt,
            phase=f"phase_24_simple_controller_{len(history) + 1}",
            require_present="action",
            label="S24 simple controller",
        )
        action = str(data.get("action", "")).strip()
        if action not in remaining:
            raise RuntimeError(f"invalid simple action: {action!r}")
        cited = [
            int(value)
            for value in data.get("evidence", [])
            if str(value).isdigit()
        ]
        available = {
            item["sentence"]
            for item in profile["tool_evidence"].get(action, [])
        }
        grounded = [value for value in cited if value in available]
        if not grounded:
            raise RuntimeError(
                f"ungrounded simple action: {action!r}, {cited!r}"
            )
        return action, {
            "evidence": grounded,
            "reason": str(data.get("reason", "")).strip(),
        }

    @staticmethod
    def _compact_outcomes(history):
        return [
            {
                "tool": step["action"],
                "accepted": len(
                    step.get("feedback", {}).get("accepted", [])
                ),
                "rejected": len(
                    step.get("feedback", {}).get("rejected", [])
                ),
            }
            for step in history
        ]

    @staticmethod
    def _compact_signals(profile, remaining):
        evidence = profile.get("tool_evidence", {})
        return {
            tool: {
                "count": len(evidence.get(tool, [])),
                "examples": evidence.get(tool, [])[:6],
            }
            for tool in remaining
        }

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
        aliases = self._identity_forms_by_component()
        identity = []
        for sentence in sentences:
            for component in components:
                forms = [
                    component.name,
                    *aliases.get(component.name, []),
                ]
                match = next(
                    (
                        self._find_exact_form(sentence.text, form)
                        for form in forms
                        if self._find_exact_form(sentence.text, form)
                    ),
                    "",
                )
                if match:
                    identity.append({
                        "sentence": sentence.number,
                        "quote": match,
                        "target": component.name,
                    })
        identity.extend(
            {
                "sentence": item["sentence"],
                "quote": item["text"],
                "target": item["component"],
            }
            for item in profile["entity_orthographic_evidence"]
        )
        references = [
            {
                "sentence": sentence.number,
                "quote": match.group(0),
            }
            for sentence in sentences
            if (match := self._REFERENCE.search(sentence.text))
        ]
        participants = [
            {
                "sentence": occurrence["sentence"],
                "quote": occurrence["quote"],
                "target": item["component"],
            }
            for item in profile["role_handle_evidence"]
            for occurrence in item["occurrences"]
        ]
        profile["tool_evidence"] = {
            "entity_pipeline": identity,
            "coreference_pipeline": references,
            "relation_role_resolution": participants,
        }
        return profile

    def _available_tools(self, profile):
        evidence = profile.get("tool_evidence", {})
        return [
            tool
            for tool in self.PHASE_TOOLS
            if evidence.get(tool)
        ]

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
        forms_by_component = self._identity_forms_by_component()
        anchors_by_target = {}
        for target in {
            candidate.component_name for candidate in candidates
        }:
            forms = [target, *forms_by_component.get(target, [])]
            anchors_by_target[target] = [
                {
                        "sentence": sentence.number,
                        "text": sentence.text,
                }
                for sentence in full_sentences
                if any(
                    self._find_exact_form(sentence.text, form)
                    for form in forms
                )
            ]
        sent_map = {
            sentence.number: sentence for sentence in full_sentences
        }
        cases = []
        allowed_anchors = {}
        evidence_sentences = set()
        for number, candidate in enumerate(candidates, 1):
            anchors = sorted(
                anchors_by_target.get(candidate.component_name, []),
                key=lambda item: (
                    abs(item["sentence"] - candidate.sentence_number),
                    item["sentence"],
                ),
            )[:3]
            anchor_ids = [
                anchor["sentence"] for anchor in anchors
            ]
            context = [
                sentence.number
                for sentence in full_sentences
                if abs(
                    sentence.number - candidate.sentence_number
                ) <= 4
            ]
            allowed_anchors[number] = set(anchor_ids)
            evidence_sentences.update(context)
            evidence_sentences.update(anchor_ids)
            cases.append({
                "case": number,
                "source": candidate.sentence_number,
                "participant": candidate.matched_text,
                "target": candidate.component_name,
                "context": context,
                "anchors": anchor_ids,
            })
        sentence_table = [
            {
                "sentence": number,
                "text": sent_map[number].text,
            }
            for number in sorted(evidence_sentences)
        ]
        prompt = f"""Does each participant denote its target component?
A generic singular or plural may denote the target when context and an anchor
select it. Reject a stronger local referent, a hardware host/machine,
type/model/technology, or another named component. Hardware capacity is host
evidence; the word "server" alone is not. Keep only architectural claims.

SENTENCES
{json.dumps(sentence_table)}

CASES
{json.dumps(cases)}

Use only a listed case anchor. Claim must be one contiguous exact substring of
the source sentence; do not abbreviate it or use ellipses.

JSON only:
{{"judgments":[{{"case":1,"keep":true,"anchor_sentence":1,
"claim":"exact source quote","alternative":"strongest alternative or none"}}]}}
"""
        data = self._ask(
            prompt,
            phase="phase_24_simple_participant_review",
            require_present="judgments",
            label="S24 simple participant review",
            timeout=240,
        )
        by_case = {}
        for item in data.get("judgments", []):
            case_value = str(item.get("case", ""))
            anchor_value = str(item.get("anchor_sentence", ""))
            if not case_value.isdigit():
                continue
            number = int(case_value)
            if not 1 <= number <= len(candidates):
                continue
            candidate = candidates[number - 1]
            anchor = int(anchor_value) if anchor_value.isdigit() else 0
            claim = str(item.get("claim", "")).strip()
            claim = claim.strip("\"'“”‘’")
            alternative = str(item.get("alternative", "")).strip()
            evidence_valid = (
                anchor in allowed_anchors[number]
                and bool(claim)
                and claim.casefold()
                in candidate.sentence_text.casefold()
                and bool(alternative)
            )
            by_case[number] = {
                "approved": (
                    item.get("keep") is True and evidence_valid
                ),
                "requested_keep": item.get("keep") is True,
                "evidence_valid": evidence_valid,
                "anchor_sentence": anchor,
                "claim": claim,
                "alternative": alternative,
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
                    "alternative": "missing judgment",
                }),
                "path": "simple_participant_review",
                "stage": "relation_role_resolution",
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
