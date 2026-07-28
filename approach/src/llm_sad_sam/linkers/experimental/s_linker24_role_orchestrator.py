"""S24 controller whose small tools own candidate discovery and decisions."""
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
    """Multi-turn controller over non-overlapping, self-discovering tools."""

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
    def _choose_tool(self, profile, remaining, history, current, sent_map):
        del profile, current, sent_map
        prompt = f"""Choose the next trace-linking tool.
You schedule tools; tools alone decide links.

TOOLS
{json.dumps(self.TOOL_CONTRACTS)}

DONE
{json.dumps(self._compact_outcomes(history))}

ACTIONS
{json.dumps(remaining)}

Do not retry rejected candidates through another tool. Each tool discovers and
decides only its own evidence kind. Completion is automatic when none remain.

JSON only:
{{"action":"action","reason":"brief reason"}}
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
        return action, {
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

    def _project_profile(self, sentences, components):
        return {
            "sentence_count": len(sentences),
            "component_count": len(components),
        }

    def _available_tools(self, profile):
        del profile
        return list(self.PHASE_TOOLS)

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
        candidates = self._catalog_overlap_candidates(
            sentences, components, current_links
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
            "participant_decisions": self._decision_view(decisions),
        }

    def _review_role_candidates(self, candidates, full_sentences):
        participant_candidates, decisions = self._classify_denotations(
            candidates, full_sentences
        )
        approved, reviewed = self._review_role_batch(
            participant_candidates, full_sentences
        )
        for key, decision in reviewed.items():
            decisions[key] = {
                **decisions.get(key, {}),
                **decision,
            }
        return approved, decisions

    def _classify_denotations(self, candidates, full_sentences):
        sent_map = {
            sentence.number: sentence for sentence in full_sentences
        }
        decisions = {}
        for start in range(0, len(candidates), 25):
            batch = candidates[start:start + 25]
            evidence_ids = {
                sentence.number
                for candidate in batch
                for sentence in full_sentences
                if abs(sentence.number - candidate.sentence_number) <= 2
            }
            sentence_table = [
                {
                    "sentence": number,
                    "text": sent_map[number].text,
                }
                for number in sorted(evidence_ids)
            ]
            cases = [
                {
                    "case": number,
                    "source": candidate.sentence_number,
                    "expression": candidate.matched_text,
                }
                for number, candidate in enumerate(batch, 1)
            ]
            prompt = f"""Classify what each expression itself denotes in its
local context: participant for a software participant, or associated for
something merely associated with software.

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
                prompt,
                phase="phase_24_general_denotation",
                require_present="judgments",
                label="S24 general denotation",
                timeout=240,
            )
            for item in data.get("judgments", []):
                case_value = str(item.get("case", ""))
                if not case_value.isdigit():
                    continue
                number = int(case_value)
                if not 1 <= number <= len(batch):
                    continue
                candidate = batch[number - 1]
                claim = str(item.get("claim", "")).strip()
                claim = claim.strip("\"'“”‘’")
                denotation = str(item.get("denotation", "")).strip()
                valid = (
                    denotation in {"participant", "associated"}
                    and bool(claim)
                    and claim.casefold()
                    in candidate.sentence_text.casefold()
                )
                key = (
                    candidate.sentence_number,
                    candidate.component_id,
                )
                decisions[key] = {
                    "approved": False,
                    "requested_keep": False,
                    "evidence_valid": valid,
                    "claim": claim,
                    "denotation": denotation,
                    "alternative": "not reviewed",
                    "path": "general_denotation",
                    "stage": "relation_role_resolution",
                }
        participant_candidates = [
            candidate
            for candidate in candidates
            if decisions.get(
                (candidate.sentence_number, candidate.component_id),
                {},
            ).get("denotation") == "participant"
            and decisions[
                (candidate.sentence_number, candidate.component_id)
            ]["evidence_valid"]
        ]
        return participant_candidates, decisions

    def _review_role_batch(self, candidates, full_sentences):
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
        prompt = f"""For each case, do the expression and target denote the
same participant? A longer or shorter label may denote the same participant.
Reject when a distinct referent is better supported. Keep only architectural
claims.

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
            phase="phase_24_general_identity",
            require_present="judgments",
            label="S24 general identity",
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
                "path": "general_identity",
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

    def _catalog_overlap_candidates(
        self, sentences, components, current_links
    ):
        tokens_by_component = {
            component.id: [
                token.casefold()
                for token in re.findall(
                    r"[A-Za-z]+[A-Za-z0-9]*|\d+",
                    component.name,
                )
            ]
            for component in components
        }
        forms_by_component = self._identity_forms_by_component()
        current = {
            (link.sentence_number, link.component_id)
            for link in current_links
        }
        candidates = {}
        for sentence in sentences:
            for match in re.finditer(
                r"[A-Za-z]+[A-Za-z0-9]*|\d+", sentence.text
            ):
                if self._qualified_identifier_boundary(
                    sentence.text, match.start(), match.end()
                ):
                    continue
                surface = match.group(0).casefold()
                owners = [
                    component
                    for component in components
                    if any(
                        surface.startswith(token)
                        for token in tokens_by_component[component.id]
                    )
                ]
                if len(owners) != 1:
                    continue
                component = owners[0]
                key = (sentence.number, component.id)
                identity_forms = [
                    component.name,
                    *forms_by_component.get(component.name, []),
                ]
                if (
                    key in current
                    or any(
                        self._find_exact_form(sentence.text, form)
                        for form in identity_forms
                    )
                ):
                    continue
                candidates[key] = CandidateLink(
                    sentence.number,
                    sentence.text,
                    component.name,
                    component.id,
                    match.group(0),
                    source="catalog_overlap",
                )
        return list(candidates.values())

    @staticmethod
    def _find_exact_form(text, expression):
        match = re.search(
            rf"(?<!\w){re.escape(expression)}(?!\w)",
            text,
            re.IGNORECASE,
        )
        return match.group(0) if match else ""
