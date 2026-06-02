"""ILinker4 — Voyager-native standalone seed extractor.

Standalone file (per user preference). Does NOT inherit from ilinker3.py.
Same two-pass extraction + merge logic as ILinker3, but:
  - Pass A and Pass B prompts are structural scaffolding only (no inline behavioral rules)
  - SEED_EXTRACTION_RULES and SEED_ACTOR_RULES are first-class bank slots injected at call time
  - Empty-string injection produces output equivalent to ILinker3 baseline behavior

  Pass A: Extraction-framed (find all mentions)
  Pass B: Actor/subject-framed (what is each sentence about?)
  Merge: exact from either → accept; synonym/partial → intersection only
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_sad_sam.core.data_types_v2 import SadSamLink
from llm_sad_sam.core.document_loader_v2 import Sentence
from llm_sad_sam.pcm_parser_v2 import ArchitectureComponent
from llm_sad_sam.llm_client import LLMClient

BATCH_SIZE = 50
BATCH_OVERLAP = 5

_LEARNED_HEADER = "\n\nLEARNED PATTERNS (apply when relevant; do not contradict the rules above):"


@dataclass
class ExtractedLink:
    sentence_number: int
    component_name: str
    component_id: str
    matched_text: str
    match_type: str  # exact, synonym, partial


class ILinker4:
    """Voyager-native seed extractor — 2 LLM passes, v2 stack, first-class SEED slots.

    Constructor parameters:
      seed_extraction_rules: str  — bank patterns for SEED_EXTRACTION_RULES slot (empty = axiom-only)
      seed_actor_rules:      str  — bank patterns for SEED_ACTOR_RULES slot (empty = axiom-only)

    With both slots empty, produces output equivalent to ILinker3 baseline behavior.
    The training harness injects learned patterns into these slots to augment seed extraction.
    """

    def __init__(
        self,
        llm: LLMClient,
        seed_extraction_rules: str = "",
        seed_actor_rules: str = "",
    ):
        self.llm = llm
        self._seed_extraction_rules = seed_extraction_rules
        self._seed_actor_rules = seed_actor_rules

    def extract(
        self,
        sentences: list[Sentence],
        components: list[ArchitectureComponent],
    ) -> list[SadSamLink]:
        """Extract explicit trace links from pre-loaded sentences and components."""
        name_to_id = {c.name: c.id for c in components}
        comp_block = self._build_comp_block(components)
        batches = self._make_batches(sentences)

        print(f"  ILinker4: {len(sentences)} sentences, {len(components)} components")
        print(f"    Batches: {len(batches)}")

        pass_a = self._run_pass_batched(batches, comp_block, name_to_id, self._prompt_extract)
        print(f"    Pass A (extract): {len(pass_a)} links")

        pass_b = self._run_pass_batched(batches, comp_block, name_to_id, self._prompt_actor)
        print(f"    Pass B (actor):   {len(pass_b)} links")

        merged = self._merge(pass_a, pass_b)
        print(f"    Merged: {len(merged)} links")

        return [
            SadSamLink(
                sentence_number=l.sentence_number,
                component_id=l.component_id,
                component_name=l.component_name,
                source="seed",
            )
            for l in merged
        ]

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_comp_block(components: list[ArchitectureComponent]) -> str:
        return "\n".join(f"  {i+1}. {c.name}" for i, c in enumerate(components))

    @staticmethod
    def _make_batches(sentences: list[Sentence]) -> list[list[Sentence]]:
        if len(sentences) <= BATCH_SIZE:
            return [sentences]
        batches, start = [], 0
        while start < len(sentences):
            end = min(start + BATCH_SIZE, len(sentences))
            batches.append(sentences[start:end])
            if end >= len(sentences):
                break
            start = end - BATCH_OVERLAP
        return batches

    def _run_pass_batched(self, batches, comp_block, name_to_id, prompt_fn) -> list[ExtractedLink]:
        seen: dict[tuple[int, str], ExtractedLink] = {}
        for i, batch in enumerate(batches):
            doc_block = "\n".join(f"S{s.number}: {s.text}" for s in batch)
            prompt = prompt_fn(doc_block, comp_block)
            links = self._query_and_parse(prompt, name_to_id)
            for link in links:
                key = (link.sentence_number, link.component_id)
                if key not in seen:
                    seen[key] = link
            if len(batches) > 1:
                print(f"      batch {i+1}/{len(batches)}: +{len(links)} (total {len(seen)})")
        return list(seen.values())

    # ── prompts (structural scaffolding + injected bank slots) ───────────

    def _prompt_extract(self, doc_block: str, comp_block: str) -> str:
        slot_block = (
            f"{_LEARNED_HEADER}\n{self._seed_extraction_rules}\n"
            if self._seed_extraction_rules
            else ""
        )
        return f"""ARCHITECTURE COMPONENTS:
{comp_block}

DOCUMENT:
{doc_block}

TASK: For each sentence, find architecture components EXPLICITLY mentioned or referenced.

Valid: exact name, synonym, abbreviation, or unambiguous partial name in the sentence text.
Invalid: names inside dotted paths, generic English words, or no clear textual evidence.
{slot_block}
Return JSON:
{{"links": [{{"s": N_INTEGER, "c": "ComponentName", "text": "matched text", "type": "exact|synonym|partial"}}]}}
Precision is critical."""

    def _prompt_actor(self, doc_block: str, comp_block: str) -> str:
        slot_block = (
            f"{_LEARNED_HEADER}\n{self._seed_actor_rules}\n"
            if self._seed_actor_rules
            else ""
        )
        return f"""ARCHITECTURE COMPONENTS:
{comp_block}

DOCUMENT:
{doc_block}

TASK: For each sentence, find components that are ARCHITECTURALLY RELEVANT — the sentence
describes their role, behavior, interactions, or responsibilities.

Report ALL participating components (not just grammatical subject). "X connects to Y" → both X and Y.

CAUTION with single-word names (e.g., "Scheduler", "Dispatcher"): only report when the sentence
discusses that component's architectural role, not generic English usage.

Rules: Must be explicitly named/abbreviated in text. Skip pronouns. Skip dotted paths. Skip generic word usage.
{slot_block}
Return JSON:
{{"links": [{{"s": N_INTEGER, "c": "ComponentName", "text": "evidence", "type": "exact|synonym|partial"}}]}}"""

    # ── LLM + parse ─────────────────────────────────────────────────────

    def _query_and_parse(self, prompt: str, name_to_id: dict) -> list[ExtractedLink]:
        response = self.llm.query(prompt, timeout=300)
        if not response.success:
            print(f"      LLM error: {response.error}")
            return []

        data = self.llm.extract_json(response)
        if not data or "links" not in data:
            print("      Failed to parse JSON")
            return []

        links = []
        for item in data["links"]:
            snum = item.get("s")
            cname = item.get("c", "")
            if not snum or not cname:
                continue
            if isinstance(snum, str):
                snum = snum.lstrip("S")
            try:
                snum = int(snum)
            except (ValueError, TypeError):
                continue

            cid = name_to_id.get(cname)
            if not cid:
                for name, nid in name_to_id.items():
                    if name.lower() == cname.lower():
                        cid, cname = nid, name
                        break
            if not cid:
                continue

            links.append(ExtractedLink(
                sentence_number=snum,
                component_name=cname,
                component_id=cid,
                matched_text=item.get("text", ""),
                match_type=item.get("type", "unknown"),
            ))
        return links

    # ── merge ────────────────────────────────────────────────────────────

    def _merge(self, pass_a: list[ExtractedLink], pass_b: list[ExtractedLink]) -> list[ExtractedLink]:
        """Exact from either pass → accept. Non-exact → intersection only."""
        result: dict[tuple[int, str], ExtractedLink] = {}

        for link in pass_a + pass_b:
            key = (link.sentence_number, link.component_id)
            if link.match_type == "exact":
                result[key] = link

        a_keys = {(l.sentence_number, l.component_id) for l in pass_a if l.match_type != "exact"}
        b_keys = {(l.sentence_number, l.component_id) for l in pass_b if l.match_type != "exact"}

        lookup = {}
        for link in pass_b + pass_a:
            key = (link.sentence_number, link.component_id)
            if link.match_type != "exact":
                lookup[key] = link

        for key in a_keys & b_keys:
            if key not in result:
                result[key] = lookup[key]

        return list(result.values())
