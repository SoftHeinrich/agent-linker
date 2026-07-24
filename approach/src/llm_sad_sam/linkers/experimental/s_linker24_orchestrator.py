"""S24 replacement orchestrator over reusable trace-linking phase tools.

This linker does not call ``SLinker21.link`` and has no protected S21 floor.
It reuses S21's phase implementations as tools, while a controller selects their
order from runtime document/component profiles and prior tool feedback.

Autonomy is bounded structurally: each state-transforming tool can run once and
is then removed from the available registry. There is no numeric step budget or
score threshold in runtime decisions.
"""
from __future__ import annotations

import json
import time

from llm_sad_sam.core.data_types_v2 import (
    CandidateLink,
    DocumentKnowledge,
    ModelKnowledge,
    SadSamLink,
)
from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.s_linker21 import SLinker21


PHASE_TOOLS = ("entity_pipeline", "coreference_pipeline", "coverage_audit")


class SLinker24Orchestrator(SLinker21):
    """Project-profile controller that assembles a replacement phase workflow."""

    _VARIANT_NAME = "s_linker24_orchestrator"

    def link(self, text_path, model_path, **_kwargs):
        self._phase_log = []
        self._llm_calls.clear()
        self._phase_metrics = {}
        self._current_text_path = text_path
        started = time.time()

        components = parse_pcm_repository(model_path)
        sentences = load_sentences(text_path)
        name_to_id = {component.name: component.id for component in components}
        sent_map = build_sent_map(sentences)
        print(
            f"Loaded {len(components)} components, {len(sentences)} sentences"
        )

        print("\n[Profile tools] Model and document knowledge")
        if self.no_knowledge:
            self.model_knowledge = ModelKnowledge()
            self.doc_knowledge = DocumentKnowledge()
        else:
            knowledge = self._run_parallel(
                {
                    "model": lambda: self._analyze_model(components),
                    "doc": lambda: self._learn_document_knowledge(
                        sentences, components
                    ),
                }
            )
            self.model_knowledge = knowledge["model"]
            self.doc_knowledge = knowledge["doc"]
        self._save_phase(
            text_path,
            "profile",
            {
                "model_knowledge": self.model_knowledge,
                "doc_knowledge": self.doc_knowledge,
            },
        )

        profile = self._project_profile(sentences, components)
        remaining = list(PHASE_TOOLS)
        current: list[SadSamLink] = []
        history = []
        while remaining:
            action, decision = self._choose_tool(
                profile, remaining, history, current, sent_map
            )
            if action == "finalize":
                history.append({"action": "finalize", **decision})
                break

            print(f"\n[Controller tool] {action}")
            if action == "entity_pipeline":
                produced, feedback = self._run_entity_tool(
                    sentences, components, name_to_id, sent_map
                )
            elif action == "coreference_pipeline":
                produced, feedback = self._run_coreference_tool(
                    sentences, components, name_to_id, sent_map
                )
            else:
                produced, feedback = self._run_coverage_audit_tool(
                    sentences, components, current, sent_map
                )

            current = self._union(current, produced)
            history.append(
                {
                    "action": action,
                    **decision,
                    "feedback": self._controller_feedback(feedback),
                }
            )
            remaining.remove(action)
            self._save_phase(
                text_path,
                f"tool_{action}",
                {
                    "links": produced,
                    "feedback": feedback,
                    "workflow": history,
                },
            )
        else:
            history.append(
                {
                    "action": "finalize",
                    "reason": "all phase capabilities consumed",
                }
            )

        self.orchestrator_workflow = history
        self._log(
            "s24_replacement_summary",
            {"profile": profile},
            {
                "workflow": history,
                "final": len(current),
                "elapsed_s": round(time.time() - started, 2),
                "llm_calls": len(self._llm_calls),
            },
            current,
        )
        self._save_phase(
            text_path,
            "final",
            {
                "final": current,
                "workflow": history,
                "elapsed_s": round(time.time() - started, 2),
            },
        )
        self._save_log(text_path)
        print(
            f"\nFinal: {len(current)} links "
            f"({time.time() - started:.1f}s, {len(self._llm_calls)} LLM calls)"
        )
        return current

    def _project_profile(self, sentences, components):
        aliases = getattr(self.doc_knowledge, "aliases", {})
        return {
            "document": [
                {"sentence": sentence.number, "text": sentence.text}
                for sentence in sentences
            ],
            "component_catalog": [
                component.name for component in components
            ],
            "ambiguous_components": sorted(
                getattr(self.model_knowledge, "ambiguous_names", set())
            ),
            "approved_aliases": [
                {
                    "alias": term,
                    "target": getattr(entry, "component", entry),
                    "scope": getattr(entry, "scope", "global"),
                }
                for term, entry in aliases.items()
            ],
        }

    @staticmethod
    def _link_view(links, sent_map):
        return [
            {
                "sentence": link.sentence_number,
                "text": sent_map[link.sentence_number].text,
                "component": link.component_name,
                "source": link.source,
            }
            for link in links
            if link.sentence_number in sent_map
        ]

    @staticmethod
    def _controller_feedback(feedback):
        """Normalize detailed tool evidence into non-redundant controller state."""
        proposed = feedback.get("candidates", feedback.get("proposed", []))
        accepted = feedback.get("accepted", [])
        accepted_keys = {
            (item["sentence"], item["component"]) for item in accepted
        }

        def reference(item):
            return {
                "sentence": item["sentence"],
                "component": item["component"],
            }

        return {
            "accepted": [reference(item) for item in accepted],
            "rejected": [
                reference(item)
                for item in proposed
                if (item["sentence"], item["component"])
                not in accepted_keys
            ],
        }

    @staticmethod
    def _controller_link_view(links):
        return [
            {
                "sentence": link.sentence_number,
                "component": link.component_name,
                "source": link.source,
            }
            for link in links
        ]

    @staticmethod
    def _union(existing, additions):
        result = list(existing)
        keys = {
            (link.sentence_number, link.component_id) for link in existing
        }
        for link in additions:
            key = (link.sentence_number, link.component_id)
            if key not in keys:
                result.append(link)
                keys.add(key)
        return result

    def _choose_tool(self, profile, remaining, history, current, sent_map):
        prompt = f"""You orchestrate trace-linking phase tools for one software project.
There is no base linker and no protected floor. Choose one available tool or
finalize the current result. You cannot propose, validate, add, or remove links.

TOOLS
- entity_pipeline: knowledge-aware entity extraction followed by the existing
  two-pass evidence validator.
- coreference_pipeline: reference-resolution extraction followed by its
  reference validator.
- coverage_audit: inspect the document, catalog, aliases, and CURRENT LINKS for
  semantically grounded omissions, then use the existing two-pass validator.
- finalize: return the union of links produced by completed tools.

WORKFLOW METHOD
- Use the document's naming/reference style, component ambiguity and aliases,
  current coverage, and concrete tool feedback.
- Entity, coreference, and audit tools cover different evidence modes; choose
  and order only those warranted by this project.
- Treat candidate/link counts as observations, never thresholds.
- Do not infer gold, use project identity, or create link decisions yourself.
- Finalize when supported reference modes are covered and no unused tool has
  complementary evidence.

PROJECT PROFILE
{json.dumps(profile, indent=2)}

COMPLETED TOOL FEEDBACK
{json.dumps(history, indent=2)}

CURRENT LINKS
{json.dumps(self._controller_link_view(current), indent=2)}

AVAILABLE ACTIONS
{json.dumps(remaining + ["finalize"])}

Return JSON only:
{{
  "action": "one available action",
  "document_reference_style": "brief runtime assessment",
  "component_profile_effect": "brief runtime assessment",
  "coverage_gap": "complementary evidence still missing or none",
  "reason": "brief workflow reason"
}}
"""
        data = self._ask(
            prompt,
            phase=f"phase_24_orchestrator_{len(history) + 1}",
            require_present="action",
            label="S24 replacement controller",
        )
        action = str(data.get("action", "")).strip()
        if action not in remaining + ["finalize"]:
            raise RuntimeError(f"invalid replacement action: {action!r}")
        decision = {
            key: str(data.get(key, "")).strip()
            for key in (
                "document_reference_style",
                "component_profile_effect",
                "coverage_gap",
                "reason",
            )
        }
        return action, decision

    def _run_entity_tool(
        self, sentences, components, name_to_id, sent_map
    ):
        candidates_by_key = self._run_framing_c(
            sentences, components, name_to_id, sent_map
        )
        candidates = list(candidates_by_key.values())
        bundles = {
            (candidate.sentence_number, candidate.component_id):
                self._build_evidence_bundle(candidate, sent_map)
            for candidate in candidates
        }
        approved, decisions = self._validate_with_evidence(
            candidates,
            bundles,
            components,
            sent_map,
            p1_tag="phase_24_entity_p1",
            p2_tag="phase_24_entity_p2",
            stage_label="entity",
        )
        links = [
            SadSamLink(
                candidate.sentence_number,
                candidate.component_id,
                candidate.component_name,
                source="entity",
            )
            for candidate in approved
        ]
        return links, {
            "candidates": self._link_view(
                [
                    SadSamLink(
                        candidate.sentence_number,
                        candidate.component_id,
                        candidate.component_name,
                        source="entity_candidate",
                    )
                    for candidate in candidates
                ],
                sent_map,
            ),
            "accepted": self._link_view(links, sent_map),
            "validator_decisions": self._decision_view(decisions),
        }

    def _run_coreference_tool(
        self, sentences, components, name_to_id, sent_map
    ):
        raw, metadata = self._run_coreference(
            sentences, components, name_to_id, sent_map
        )
        approved, decisions = self._validate_coref_links(
            raw, sent_map, components
        )
        return approved, {
            "candidates": self._link_view(raw, sent_map),
            "accepted": self._link_view(approved, sent_map),
            "metadata": [
                {
                    "sentence": sentence,
                    "component_id": component,
                    **value,
                }
                for (sentence, component), value in metadata.items()
            ],
            "validator_decisions": self._decision_view(decisions),
        }

    @staticmethod
    def _decision_view(decisions):
        return [
            {
                "sentence": sentence,
                "component_id": component,
                **decision,
            }
            for (sentence, component), decision in decisions.items()
        ]

    def _run_coverage_audit_tool(
        self, sentences, components, current_links, sent_map
    ):
        candidates = self._coverage_candidates(
            sentences, components, current_links, sent_map
        )
        bundles = {
            (candidate.sentence_number, candidate.component_id):
                self._build_evidence_bundle(
                    candidate,
                    sent_map,
                    rationale=(
                        "semantic coverage audit: identity and architectural "
                        "participation"
                    ),
                )
            for candidate in candidates
        }
        approved, decisions = self._validate_with_evidence(
            candidates,
            bundles,
            components,
            sent_map,
            p1_tag="phase_24_audit_p1",
            p2_tag="phase_24_audit_p2",
            stage_label="coverage_audit",
        )
        links = [
            SadSamLink(
                candidate.sentence_number,
                candidate.component_id,
                candidate.component_name,
                source="s24_coverage_audit",
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
                        source="coverage_audit_candidate",
                    )
                    for candidate in candidates
                ],
                sent_map,
            ),
            "accepted": self._link_view(links, sent_map),
            "validator_decisions": self._decision_view(decisions),
        }

    def _coverage_candidates(
        self, sentences, components, current_links, sent_map
    ):
        current_by_sentence = {}
        for link in current_links:
            current_by_sentence.setdefault(link.sentence_number, []).append(
                link.component_name
            )
        items = "\n\n".join(
            f"ITEM S{sentence.number}\n"
            f"  SENTENCE: {sentence.text}\n"
            f"  CURRENT LINKS: "
            f"{json.dumps(current_by_sentence.get(sentence.number, []))}"
            for sentence in sentences
        )
        profile = self._project_profile(sentences, components)
        prompt = f"""Audit trace-link coverage for one software design document.
Find supported component references missing from CURRENT LINKS. This is a
candidate-generation tool; a separate validator makes final decisions.

COMPONENT CATALOG
{json.dumps(profile["component_catalog"])}

APPROVED DOCUMENT ALIASES
{json.dumps(profile["approved_aliases"])}

For each proposed omission, establish:
- REFERENT IDENTITY: the exact quoted phrase denotes or unambiguously invokes
  the catalog component itself, not merely an associated user, browser,
  algorithm, implementation, package, connection, data item, action, or result;
- ARCHITECTURAL PARTICIPATION: the sentence states an architectural fact in
  which that component participates as subject, object, endpoint, service,
  boundary, or contextual referent;
- COMPETING REFERENT: the strongest alternative interpretation, or "none".

Enumerate every distinct participating catalog component; do not stop after the
most salient one. Negated and contrastive relations still specify architectural
boundaries. A descriptive heading or caption is evidence when it identifies a
component-owned architectural flow. Do not repeat CURRENT LINKS. Reject
code/package-only mentions and unsupported group references. Use only exact
catalog names and exact source quotes.

DOCUMENT
{items}

Return JSON only:
{{"omissions":[{{"sentence":1,"component":"exact catalog name",
"quote":"exact source words","referent":"thing denoted",
"architectural_role":"how component participates",
"competing_referent":"alternative or none"}}]}}
"""
        data = self._ask(
            prompt,
            phase="phase_24_coverage_audit",
            require_present="omissions",
            label="S24 coverage audit",
            timeout=240,
        )
        names = {component.name: component.id for component in components}
        current_keys = {
            (link.sentence_number, link.component_id)
            for link in current_links
        }
        candidates = {}
        for item in data.get("omissions", []):
            try:
                sentence_number = int(item["sentence"])
                component_name = str(item["component"]).strip()
                quote = str(item["quote"]).strip()
            except Exception:
                continue
            sentence = sent_map.get(sentence_number)
            if (
                component_name not in names
                or not sentence
                or not quote
                or quote.casefold() not in sentence.text.casefold()
            ):
                continue
            key = (sentence_number, names[component_name])
            if key in current_keys:
                continue
            candidates.setdefault(
                key,
                CandidateLink(
                    sentence_number,
                    sentence.text,
                    component_name,
                    names[component_name],
                    quote,
                    source="coverage_audit",
                ),
            )
        return list(candidates.values())
