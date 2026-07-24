"""S24 replacement controller with a project-specific catalog-handle tool."""
from __future__ import annotations

import json
import re

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
        "catalog_identifier_resolution",
    )

    def _tool_catalog(self):
        return """- entity_pipeline: exact catalog-name and approved-alias identity evidence,
  followed by the existing two-pass validator.
- coreference_pipeline: pronoun, demonstrative, and anaphoric reference
  resolution, followed by its existing validator.
- relation_role_resolution: derive project-specific shortened handles from
  compound catalog names and apply exact standalone occurrences.
- catalog_identifier_resolution: resolve standalone alternate spellings whose
  complete token sequence equals one runtime catalog component.
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
        profile["catalog_identifier_evidence"] = self._link_view(
            [
                SadSamLink(
                    candidate.sentence_number,
                    candidate.component_id,
                    candidate.component_name,
                    source="catalog_identifier_candidate",
                )
                for candidate in self._catalog_identifier_candidates(
                    sentences, components, []
                )
            ],
            {sentence.number: sentence for sentence in sentences},
        )
        return profile

    def _available_tools(self, profile):
        tools = super()._available_tools(profile)
        if not profile.get("role_handle_evidence"):
            tools.remove("relation_role_resolution")
        if not profile.get("catalog_identifier_evidence"):
            tools.remove("catalog_identifier_resolution")
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
        if action == "catalog_identifier_resolution":
            return self._run_catalog_identifier_tool(
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
        return self._review_identity_candidates(
            candidates,
            full_sentences,
            case_key="handle",
            instruction=(
                "Resolve shortened component handles in project context."
            ),
            phase="phase_24_role_context_review",
            label="S24 role-context review",
            path="role_context_review",
            stage="relation_role_resolution",
        )

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

    def _run_catalog_identifier_tool(
        self, sentences, components, current_links, sent_map
    ):
        candidates = self._catalog_identifier_candidates(
            sentences, components, current_links
        )
        approved, decisions = self._review_identity_candidates(
            candidates,
            sentences,
            case_key="identifier",
            instruction=(
                "Resolve catalog-equivalent standalone component identifiers."
            ),
            phase="phase_24_catalog_identifier_review",
            label="S24 catalog-identifier review",
            path="catalog_identifier_resolution",
            stage="catalog_identifier_resolution",
        )
        links = [
            SadSamLink(
                candidate.sentence_number,
                candidate.component_id,
                candidate.component_name,
                source="s24_catalog_identifier",
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
                        source="catalog_identifier_candidate",
                    )
                    for candidate in candidates
                ],
                sent_map,
            ),
            "accepted": self._link_view(links, sent_map),
            "identifier_decisions": self._decision_view(decisions),
        }

    def _catalog_identifier_candidates(
        self, sentences, components, current_links
    ):
        current = {
            (link.sentence_number, link.component_id)
            for link in current_links
        }
        forms_by_component = self._identity_forms_by_component()
        tokens_by_component = {
            component.name: self._identifier_tokens(component.name)
            for component in components
        }
        pattern = re.compile(
            r"(?<![\w-])(?<!\w\.)(?:"
            r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+"
            r"|[A-Za-z]+[A-Z][A-Za-z0-9]*"
            r")(?![\w-])(?!\.\w)"
        )
        candidates = {}
        for sentence in sentences:
            for match in pattern.finditer(sentence.text):
                expression = match.group(0)
                expression_tokens = self._identifier_tokens(expression)
                for component in components:
                    if (
                        expression_tokens
                        != tokens_by_component[component.name]
                    ):
                        continue
                    identity_forms = [
                        component.name,
                        *forms_by_component.get(component.name, []),
                    ]
                    key = (sentence.number, component.id)
                    if (
                        key in current
                        or any(
                            expression.casefold() == form.casefold()
                            for form in identity_forms
                        )
                    ):
                        continue
                    candidates[key] = CandidateLink(
                        sentence.number,
                        sentence.text,
                        component.name,
                        component.id,
                        expression,
                        source="catalog_identifier",
                    )
        return list(candidates.values())

    @staticmethod
    def _identifier_tokens(expression):
        return [
            token.casefold()
            for token in re.findall(
                r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|\d+",
                expression.replace("-", " "),
            )
        ]

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
            for token in set(item.casefold() for item in tokens):
                owners.setdefault(token, set()).add(component_name)
        return [
            {"expression": token, "component": component_name}
            for component_name, tokens in tokens_by_component.items()
            if len(tokens) > 1
            for token in (
                tokens if "-" in component_name else tokens[-1:]
            )
            if len(owners[token.casefold()]) == 1
        ]

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
        match = re.search(
            rf"(?<![\w.-]){re.escape(expression)}(?![\w.-])",
            text,
            re.IGNORECASE,
        )
        return match.group(0) if match else ""

    @staticmethod
    def _find_exact_form(text, expression):
        match = re.search(
            rf"(?<!\w){re.escape(expression)}(?!\w)",
            text,
            re.IGNORECASE,
        )
        return match.group(0) if match else ""
