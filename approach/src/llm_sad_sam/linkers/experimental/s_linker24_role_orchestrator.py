"""S24 replacement controller with a project-specific catalog-handle tool."""
from __future__ import annotations

import json
import re

from llm_sad_sam.core.data_types_v2 import CandidateLink, SadSamLink
from llm_sad_sam.linkers.experimental.s_linker24_orchestrator import (
    SLinker24Orchestrator,
)


class SLinker24RoleOrchestrator(SLinker24Orchestrator):
    """Replacement controller with non-overlapping reference-mode tools."""

    _VARIANT_NAME = "s_linker24_role_orchestrator"
    PHASE_TOOLS = (
        "entity_pipeline",
        "coreference_pipeline",
        "relation_role_resolution",
    )

    def _tool_catalog(self):
        return """- entity_pipeline: exact catalog-name and approved-alias identity evidence,
  followed by the existing two-pass validator.
- coreference_pipeline: pronoun, demonstrative, and anaphoric reference
  resolution, followed by its existing validator.
- relation_role_resolution: derive project-specific shortened handles from
  compound catalog names and apply exact standalone occurrences.
- finalize: return the union of links produced by completed tools."""

    def _choose_tool(self, profile, remaining, history, current, sent_map):
        prompt = f"""You orchestrate trace-linking tools for one software project.
There is no base linker and no protected floor. Choose one available tool or
finalize. You cannot propose, validate, add, or remove links.

TOOLS
{self._tool_catalog()}

PROJECT PROFILE
{json.dumps(profile)}

COMPLETED TOOL FEEDBACK
{json.dumps(history)}

CURRENT LINKS
{json.dumps(self._controller_link_view(current))}

AVAILABLE ACTIONS
{json.dumps(remaining + ["finalize"])}

Select an unused capability only when exact document words show its evidence
mode remains unresolved. Rejected evidence must not be retried through another
tool. Counts are observations, not thresholds.

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
        if action not in remaining + ["finalize"]:
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
        if action != "finalize" and not grounded_quotes:
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
        return profile

    def _available_tools(self, profile):
        tools = super()._available_tools(profile)
        if not profile.get("role_handle_evidence"):
            tools.remove("relation_role_resolution")
        return tools

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
        approved = candidates
        decisions = {
            (candidate.sentence_number, candidate.component_id): {
                "approved": True,
                "path": "catalog_unique_handle",
                "stage": "relation_role_resolution",
            }
            for candidate in candidates
        }
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
                if key in current or not matched:
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
            rf"(?<![\w-]){re.escape(expression)}(?![\w-])",
            text,
            re.IGNORECASE,
        )
        return match.group(0) if match else ""
