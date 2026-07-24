"""S24 dynamic controller — sequential, profile-aware phase orchestration.

Unlike ``SLinker24Agentic``'s one-shot inventory router, this controller sees a
runtime profile of the document, component catalog, S21 floor, and grounded
candidate sketches.  It selects one bounded phase at a time, observes the
phase's accepted yield, and may then stop or select the remaining phase.

The controller never proposes or accepts a link.  The unchanged S21 Phase-4
validator and S24 anchored-reference validator retain that authority.
"""
from __future__ import annotations

import json
from collections import Counter

from llm_sad_sam.core.data_types_v2 import SadSamLink
from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.s_linker21 import SLinker21
from llm_sad_sam.linkers.experimental.s_linker24_agentic import (
    SLinker24Agentic,
    TOOL_NAMES,
)


class SLinker24Dynamic(SLinker24Agentic):
    """S21 floor plus a sequential controller over bounded recovery phases."""

    _VARIANT_NAME = "s_linker24_dynamic"

    def link(self, text_path, model_path, **kwargs):
        floor = SLinker21.link(self, text_path, model_path, **kwargs)
        self.dynamic_steps = []
        self.agentic_tool_calls = []
        self.agentic_plan_reason = ""
        self._s24_stats = {"eligible": 0, "resolver_approved": 0, "kept": 0}
        try:
            final = self._augment_floor(text_path, model_path, floor)
        except Exception as exc:
            print(f"  [s24-dynamic] controller failed; keeping S21 floor: {exc}")
            final = floor
        self._log(
            "phase_24_dynamic_summary",
            {"floor": len(floor), "profile": getattr(self, "dynamic_profile", {})},
            {
                "final": len(final),
                "additions": len(final) - len(floor),
                "steps": self.dynamic_steps,
            },
            final,
        )
        self._save_log(text_path)
        return final

    def _augment_floor(self, text_path, model_path, floor):
        components = parse_pcm_repository(model_path)
        sentences = load_sentences(text_path)
        aliases = self._alias_candidates(sentences, components, floor)
        anchors = self._anchor_candidate_sketches(sentences, components, floor)
        candidates = {
            "alias_phase4": aliases,
            "anchored_reference": anchors,
        }
        self.agentic_inventory = {
            name: len(items) for name, items in candidates.items()
        }
        self.dynamic_profile = self._build_runtime_profile(
            components, sentences, floor, aliases, anchors
        )
        remaining = [
            name for name in TOOL_NAMES if self.agentic_inventory[name] > 0
        ]
        self.dynamic_steps = []
        self.agentic_tool_calls = []
        additions = []

        # There is nothing for a controller to decide when no phase is eligible.
        while remaining:
            action, reason, assessment = self._next_action(
                self.dynamic_profile, remaining, self.dynamic_steps
            )
            if action == "stop":
                self.dynamic_steps.append(
                    {
                        "action": "stop",
                        "assessment": assessment,
                        "reason": reason,
                    }
                )
                break
            if action not in remaining:
                raise RuntimeError(
                    f"controller selected unavailable tool: {action!r}"
                )
            before = {
                (link.sentence_number, link.component_id)
                for link in floor + additions
            }
            if action == "alias_phase4":
                raw = self._run_alias_phase4(aliases, components, sentences)
                validation = self._alias_validation_feedback
            else:
                raw = self._remove_competing_aliases(
                    self._recover(text_path, model_path, floor + additions),
                    sentences,
                )
                validation = {
                    "resolver_approved": self._s24_stats[
                        "resolver_approved"
                    ],
                    "validator_approved": self._s24_stats["kept"],
                }
            accepted = [
                link
                for link in raw
                if (link.sentence_number, link.component_id) not in before
            ]
            additions.extend(accepted)
            self.agentic_tool_calls.append(action)
            self.dynamic_steps.append(
                {
                    "action": action,
                    "assessment": assessment,
                    "reason": reason,
                    "feedback": {
                        "input": {
                            "eligible": self.agentic_inventory[action],
                            "distinct_targets": len(
                                {
                                    (
                                        item.component_name
                                        if action == "alias_phase4"
                                        else item["target"]
                                    )
                                    for item in candidates[action]
                                }
                            ),
                        },
                        "validation": validation,
                        "output": {
                            "new_links": len(accepted),
                            "accepted_targets": sorted(
                                {link.component_name for link in accepted}
                            ),
                        },
                    },
                }
            )
            remaining.remove(action)

        self.agentic_plan_reason = " | ".join(
            step["reason"] for step in self.dynamic_steps
        )
        unique = {}
        floor_keys = {
            (link.sentence_number, link.component_id) for link in floor
        }
        for link in additions:
            key = (link.sentence_number, link.component_id)
            if key not in floor_keys:
                unique.setdefault(key, link)
        print(
            f"  [s24-dynamic] workflow={self.agentic_tool_calls or ['none']} "
            f"steps={len(self.dynamic_steps)} additions={len(unique)}"
        )
        return floor + list(unique.values())

    def _build_runtime_profile(
        self, components, sentences, floor, aliases, anchors
    ) -> dict:
        names = [component.name for component in components]
        ambiguous = sorted(
            getattr(
                getattr(self, "model_knowledge", None),
                "ambiguous_names",
                set(),
            )
        )
        linked_components = Counter(link.component_name for link in floor)
        linked_sentences = {link.sentence_number for link in floor}
        alias_entries = getattr(
            getattr(self, "doc_knowledge", None), "aliases", {}
        )
        scopes = Counter(
            getattr(entry, "scope", "global")
            for entry in alias_entries.values()
        )
        floor_sources = Counter(link.source or "unspecified" for link in floor)
        return {
            "document": {
                "sentences": len(sentences),
                "sentences_with_floor_links": len(linked_sentences),
                "floor_sentence_coverage": round(
                    len(linked_sentences) / max(1, len(sentences)), 3
                ),
            },
            "components": {
                "catalog_size": len(names),
                "catalog": names,
                "ambiguous_names": ambiguous,
                "ambiguous_ratio": round(
                    len(ambiguous) / max(1, len(names)), 3
                ),
                "without_floor_links": [
                    name for name in names if not linked_components[name]
                ],
                "floor_link_counts": dict(linked_components),
            },
            "knowledge": {
                "approved_aliases": len(alias_entries),
                "alias_scopes": dict(scopes),
                "alias_profile": [
                    {
                        "alias": term,
                        "target": getattr(entry, "component", entry),
                        "scope": getattr(entry, "scope", "global"),
                    }
                    for term, entry in alias_entries.items()
                ],
            },
            "floor": {
                "links": len(floor),
                "source_mix": dict(floor_sources),
            },
            "recovery_evidence": {
                "alias_phase4": {
                    "eligible": len(aliases),
                    "samples": [
                        {
                            "sentence": item.sentence_number,
                            "text": item.sentence_text,
                            "matched_alias": item.matched_text,
                            "target": item.component_name,
                            "target_ambiguous": item.component_name in ambiguous,
                        }
                        for item in aliases
                    ],
                },
                "anchored_reference": {
                    "eligible": len(anchors),
                    "samples": anchors,
                },
            },
        }

    def _next_action(self, profile, remaining, history):
        prompt = f"""You are the workflow controller after an S21 trace-linking floor.

Choose exactly ONE next action from the available bounded phases or stop. You
cannot propose, approve, reject, or edit links. Return a compact decision record,
not hidden chain-of-thought.

PHASES
- alias_phase4: exact occurrences of aliases already approved by the document
  knowledge phase, followed by S21's unchanged two-pass entity validator.
- anchored_reference: local structural-sibling or unique technical-prefix
  references, followed by the dedicated anchored-reference resolver and validator.

DECISION PRINCIPLES
- First classify the documentation regime from the actual evidence as
  architecture_prose, mixed, technical_inventory, or caption_heavy.
- Assess catalog risk from ambiguous names, naming overlap, and concentration of
  unlinked or weakly linked components.
- Rank the remaining phase evidence by semantic specificity, locality, expected
  novel recall, validator risk, and cost. Counts alone are not evidence quality.
- Prefer evidence whose referring phrase and target are specific in this project.
- After observing a phase funnel, use its yield and validator agreement. Call
  another only when its evidence is complementary enough to justify its
  validation cost and false-positive exposure.
- A zero-yield phase exhausts only that evidence channel; it is not evidence
  against an independent remaining phase. Reassess the remaining examples on
  their own semantics.
- The validators operate per candidate. A noisy pool can still justify a call
  when at least one grounded example is specific and plausibly component-
  denoting; do not require most of the pool to look clean.
- Stop when no remaining phase has sufficiently specific project-grounded evidence.
- Never infer gold links and never use project identity.

RUNTIME PROFILE
{json.dumps(profile, indent=2)}

OBSERVED WORKFLOW
{json.dumps(history, indent=2)}

AVAILABLE ACTIONS
{json.dumps(remaining + ["stop"])}

Return JSON only:
{{
  "assessment": {{
    "document_regime": "architecture_prose|mixed|technical_inventory|caption_heavy",
    "catalog_risk": "low|medium|high",
    "best_evidence": "one available phase or none",
    "expected_gain": "low|medium|high",
    "false_positive_risk": "low|medium|high"
  }},
  "action": "one available action",
  "reason": "brief reason citing profile fields and candidate or feedback evidence"
}}
"""
        data = self._ask(
            prompt,
            phase=f"phase_24_dynamic_plan_{len(history) + 1}",
            require_present="action",
            label="S24 dynamic plan",
        )
        action = str(data.get("action", "")).strip()
        if action not in remaining + ["stop"]:
            raise RuntimeError(f"invalid controller action: {action!r}")
        assessment = data.get("assessment", {})
        required = {
            "document_regime",
            "catalog_risk",
            "best_evidence",
            "expected_gain",
            "false_positive_risk",
        }
        if not isinstance(assessment, dict) or not required.issubset(assessment):
            raise RuntimeError("controller omitted structured assessment")
        return action, str(data.get("reason", "")).strip(), assessment

    def _run_alias_phase4(self, candidates, components, sentences):
        """Run unchanged S21 validation and retain its agreement funnel."""
        if not candidates:
            self._alias_validation_feedback = {
                "pass1_approved": 0,
                "pass2_approved": 0,
                "consensus_approved": 0,
                "pass_disagreements": 0,
            }
            return []
        sent_map = build_sent_map(sentences)
        bundles = {
            (candidate.sentence_number, candidate.component_id):
                self._build_evidence_bundle(
                    candidate,
                    sent_map,
                    rationale="exact Phase-1-approved alias",
                )
            for candidate in candidates
        }
        approved, decisions = self._validate_with_evidence(
            candidates,
            bundles,
            components,
            sent_map,
            p1_tag="phase_24_dynamic_alias_p1",
            p2_tag="phase_24_dynamic_alias_p2",
            stage_label="dynamic_alias",
        )
        self._alias_validation_feedback = {
            "pass1_approved": sum(
                item["p1"] for item in decisions.values()
            ),
            "pass2_approved": sum(
                item["p2"] for item in decisions.values()
            ),
            "consensus_approved": len(approved),
            "pass_disagreements": sum(
                item["p1"] != item["p2"] for item in decisions.values()
            ),
        }
        return [
            SadSamLink(
                candidate.sentence_number,
                candidate.component_id,
                candidate.component_name,
                source="s24_dynamic_alias",
            )
            for candidate in approved
        ]

    def _anchor_candidate_sketches(self, sentences, components, floor):
        """Expose S24's deterministic candidate evidence to the controller."""
        _, cases = self._anchored_cases(sentences, components, floor)
        return [
            {
                "sentence": sentence.number,
                "text": sentence.text,
                "target": name,
                "anchor_sentence": anchor.number,
                "anchor_text": anchor.text,
                "basis": basis,
            }
            for sentence, name, anchor, basis in cases
        ]
