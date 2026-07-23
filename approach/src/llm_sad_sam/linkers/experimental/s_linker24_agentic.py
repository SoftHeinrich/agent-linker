"""S24 agentic tools — bounded project-adaptive recovery over the S21 floor.

The controller has authority to select tools, not links.  Each selected tool
constructs grounded candidates and delegates acceptance to a bounded validator:

* ``alias_phase4`` scans exact occurrences of Phase-1-approved aliases and reuses
  S21's unchanged Phase-4 two-pass validator.
* ``anchored_reference`` reuses S24's narrow local-anchor resolver and dedicated
  anchored-reference validator.

The S21 result is preserved setwise on every path.  This does not mathematically
guarantee an F1 floor—any added false positive can lower F1—so marginal precision
is an explicit evaluation gate rather than an architectural claim.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from llm_sad_sam.core.data_types_v2 import CandidateLink, SadSamLink
from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.linkers.experimental.s_linker21 import SLinker21
from llm_sad_sam.linkers.experimental.s_linker24 import SLinker24


TOOL_NAMES = ("alias_phase4", "anchored_reference")


@dataclass(frozen=True)
class ToolInventory:
    alias_phase4: int
    anchored_reference: int


class SLinker24Agentic(SLinker24):
    """S21 floor plus an agent-selected, validator-bounded S24 tool pass."""

    _VARIANT_NAME = "s_linker24_agentic"

    def link(self, text_path, model_path, **kwargs):
        # Bypass SLinker24.link(): its anchored recovery is an optional tool here,
        # not an always-on post-pass.  The canonical S21 implementation is untouched.
        floor = SLinker21.link(self, text_path, model_path, **kwargs)
        self.agentic_tool_calls: list[str] = []
        self.agentic_plan_reason = ""
        self._s24_stats = {"eligible": 0, "resolver_approved": 0, "kept": 0}
        try:
            final = self._augment_floor(text_path, model_path, floor)
        except Exception as exc:
            print(f"  [s24-agentic] tool pass failed; keeping S21 floor: {exc}")
            final = floor
        self._log(
            "phase_24_agentic_summary",
            {"floor": len(floor), "selected_tools": self.agentic_tool_calls},
            {
                "final": len(final),
                "additions": len(final) - len(floor),
                "plan_reason": self.agentic_plan_reason,
            },
            final,
        )
        self._save_log(text_path)
        return final

    def _augment_floor(self, text_path, model_path, floor):
        """Run agent-selected tools over an already-computed floor.

        Kept separate from ``link`` for deterministic fixed-floor replay and unit
        testing without resampling S21.
        """
        components = parse_pcm_repository(model_path)
        sentences = load_sentences(text_path)
        aliases = self._alias_candidates(sentences, components, floor)
        inventory = ToolInventory(
            alias_phase4=len(aliases),
            anchored_reference=self._anchored_opportunity_count(
                text_path, model_path, floor
            ),
        )
        self.agentic_inventory = asdict(inventory)
        selected, reason = self._plan_tools(
            inventory, [component.name for component in components], floor
        )
        self.agentic_tool_calls = selected
        self.agentic_plan_reason = reason

        additions: list[SadSamLink] = []
        if "alias_phase4" in selected:
            additions.extend(
                self._run_alias_phase4(aliases, components, sentences)
            )
        if "anchored_reference" in selected:
            anchored = self._recover(text_path, model_path, floor)
            additions.extend(self._remove_competing_aliases(anchored, sentences))

        existing = {(link.sentence_number, link.component_id) for link in floor}
        unique: dict[tuple[int, str], SadSamLink] = {}
        for link in additions:
            key = (link.sentence_number, link.component_id)
            if key not in existing:
                unique.setdefault(key, link)
        print(
            f"  [s24-agentic] tools={selected or ['none']} "
            f"additions={len(unique)} inventory={asdict(inventory)}"
        )
        return floor + list(unique.values())

    def _alias_candidates(self, sentences, components, floor) -> list[CandidateLink]:
        """Ground exact residual mentions in Phase-1-approved global aliases."""
        knowledge = getattr(self, "doc_knowledge", None)
        if not knowledge:
            return []
        name_to_id = {component.name: component.id for component in components}
        existing = {(link.sentence_number, link.component_id) for link in floor}
        ambiguous = set(
            getattr(getattr(self, "model_knowledge", None), "ambiguous_names", set())
        )
        found: dict[tuple[int, str], CandidateLink] = {}
        for term, entry in knowledge.aliases.items():
            component = getattr(entry, "component", entry)
            scope = getattr(entry, "scope", "global")
            if (
                scope != "global"
                or component not in name_to_id
                or term.casefold() == component.casefold()
            ):
                continue
            # A plain lexical synonym is not an exact identifier when Phase 1 has
            # already classified the target name as ordinary-word ambiguous.
            weak_lexical_alias = (
                component in ambiguous
                and not any(char.isdigit() for char in term)
                and not any(char.isupper() for char in term[1:])
                and len(re.findall(r"[A-Za-z0-9]+", term)) <= 2
            )
            if weak_lexical_alias:
                continue
            pattern = re.compile(
                rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.I
            )
            for sentence in sentences:
                match = pattern.search(sentence.text)
                key = (sentence.number, name_to_id[component])
                if not match or key in existing:
                    continue
                found.setdefault(
                    key,
                    CandidateLink(
                        sentence.number,
                        sentence.text,
                        component,
                        name_to_id[component],
                        match.group(0),
                        source="agent_alias",
                    ),
                )
        return list(found.values())

    def _remove_competing_aliases(self, additions, sentences) -> list[SadSamLink]:
        """A longer approved alias for another target defeats a short target."""
        sent_map = build_sent_map(sentences)
        aliases = getattr(getattr(self, "doc_knowledge", None), "aliases", {})
        kept = []
        for link in additions:
            sentence = sent_map.get(link.sentence_number)
            if not sentence:
                continue
            target = link.component_name.casefold()
            competing = False
            for term, entry in aliases.items():
                other = getattr(entry, "component", entry)
                target_in_alias = re.search(
                    rf"(?<![A-Za-z0-9]){re.escape(target)}(?![A-Za-z0-9])",
                    term.casefold(),
                )
                if other == link.component_name or not target_in_alias:
                    continue
                if re.search(
                    rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])",
                    sentence.text,
                    re.I,
                ):
                    competing = True
                    break
            if not competing:
                kept.append(link)
        return kept

    def _plan_tools(self, inventory, component_names, floor) -> tuple[list[str], str]:
        prompt = f"""You control optional recovery tools after a trace-linking floor has completed.

You may call zero, one, or both tools. You cannot create links yourself.

TOOLS
- alias_phase4: deterministically scan exact occurrences of aliases already approved
  by the floor's knowledge phase, then send only floor-missed candidates through
  the floor's unchanged two-pass entity validator.
- anchored_reference: run the narrow anchored-reference resolver and validator for
  local structural sibling or unique technical-prefix shorthand.

RUNTIME PROJECT SNAPSHOT
- component catalog: {json.dumps(component_names)}
- accepted floor links: {len(floor)}
- eligible alias_phase4 candidates: {inventory.alias_phase4}
- eligible anchored_reference candidates: {inventory.anchored_reference}

Choose tools only when their eligible count is nonzero. Prefer a bounded, evidence-
grounded tool over broad re-extraction. Do not infer missing gold links and do not
use project identity. Return JSON only:
{{"calls":["alias_phase4","anchored_reference"],"reason":"brief runtime-only reason"}}
"""
        data = self._ask(
            prompt,
            phase="phase_24_agent_plan",
            require_present="calls",
            label="S24 agent plan",
        )
        calls = data.get("calls", [])
        if not isinstance(calls, list):
            raise RuntimeError("controller returned non-list calls")
        selected = []
        for name in calls:
            if name not in TOOL_NAMES:
                raise RuntimeError(f"controller selected unknown tool: {name!r}")
            if name not in selected:
                selected.append(name)
        if len(selected) > 2:
            raise RuntimeError("controller exceeded two-tool budget")
        counts = asdict(inventory)
        if any(counts[name] == 0 for name in selected):
            raise RuntimeError("controller selected a tool with no eligible candidates")
        return selected, str(data.get("reason", "")).strip()

    def _run_alias_phase4(self, candidates, components, sentences) -> list[SadSamLink]:
        if not candidates:
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
        approved, _ = self._validate_with_evidence(
            candidates,
            bundles,
            components,
            sent_map,
            p1_tag="phase_24_alias_p1",
            p2_tag="phase_24_alias_p2",
            stage_label="agent_alias",
        )
        return [
            SadSamLink(
                candidate.sentence_number,
                candidate.component_id,
                candidate.component_name,
                source="s24_agent_alias",
            )
            for candidate in approved
        ]

    @staticmethod
    def _anchored_opportunity_count(text_path, model_path, floor) -> int:
        """Count S24's deterministic structural opportunities without an LLM."""
        components = parse_pcm_repository(model_path)
        names = {component.name: component.id for component in components}
        sentences = load_sentences(text_path)
        families: dict[str, list[tuple[str, str]]] = {}
        for name in names:
            match = re.match(r"(.+?)\s+(Client|Server)$", name, re.I)
            if match:
                families.setdefault(match.group(1).lower(), []).append(
                    (name, match.group(2).lower())
                )
        prefixes: dict[str, list[str]] = {}
        for name in names:
            prefix = re.split(r"[-\s]", name)[0]
            if any(char.isupper() for char in prefix) or any(
                char.isdigit() for char in prefix
            ):
                prefixes.setdefault(prefix.lower(), []).append(name)

        def anchor_before(index, component_name):
            needle = re.compile(rf"\b{re.escape(component_name)}\b", re.I)
            return next(
                (
                    sentence
                    for sentence in reversed(sentences[max(0, index - 5) : index])
                    if needle.search(sentence.text)
                ),
                None,
            )

        floor_keys = {(link.sentence_number, link.component_id) for link in floor}
        cases: set[tuple[int, str]] = set()
        for index, sentence in enumerate(sentences):
            low = sentence.text.lower()
            context = " ".join(
                item.text.lower()
                for item in sentences[max(0, index - 5) : index]
            )
            for base, members in families.items():
                if base not in context:
                    continue
                for name, role in members:
                    if (
                        anchor_before(index, name)
                        and re.search(rf"\b{re.escape(role)}\b", low)
                        and name.lower() not in low
                    ):
                        cases.add((sentence.number, names[name]))
            for prefix, component_names in prefixes.items():
                if (
                    len(component_names) != 1
                    or len(prefix) < 4
                    or not re.search(rf"\b{re.escape(prefix)}\b", low)
                ):
                    continue
                name = component_names[0]
                if anchor_before(index, name) and name.lower() != low:
                    cases.add((sentence.number, names[name]))
        return len(cases - floor_keys)
