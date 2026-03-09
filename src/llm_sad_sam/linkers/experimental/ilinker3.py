"""ILinker3 — ILinker2 with simplified prompts (26% fewer prompt tokens).

Same architecture as ILinker2: two LLM passes + merge.
Prompts condensed from test_prompt_simplify_wide.py testing — verified equivalent
or better on all 5 benchmark datasets.

  Pass A: Extraction-framed (simplified — 13 lines vs 25)
  Pass B: Actor/subject-framed (simplified — 18 lines vs 35)
  Merge: exact from either → accept; synonym/partial → intersection only

Output: list[SadSamLink] with source="ilinker2" — same shape as ILinker2 links.
"""

import json
import logging
import re
from dataclasses import dataclass

from llm_sad_sam.core.data_types import SadSamLink
from llm_sad_sam.core.document_loader import DocumentLoader, Sentence
from llm_sad_sam.llm_client import LLMClient, LLMBackend
from llm_sad_sam.pcm_parser import parse_pcm_repository

logger = logging.getLogger(__name__)

BATCH_SIZE = 50
BATCH_OVERLAP = 5


@dataclass
class ExtractedLink:
    sentence_number: int
    component_name: str
    component_id: str
    matched_text: str
    match_type: str  # exact, synonym, partial


class ILinker3:
    """High-precision explicit extractor — simplified prompts, same merge logic."""

    def __init__(self, backend: LLMBackend = LLMBackend.CLAUDE):
        self.llm = LLMClient(backend=backend)

    def link(self, text_path: str, model_path: str, transarc_csv: str = None) -> list[SadSamLink]:
        """Extract explicit trace links. transarc_csv is accepted but ignored."""
        sentences = DocumentLoader.load_sentences(text_path)
        components = parse_pcm_repository(model_path)
        name_to_id = {c.name: c.id for c in components}

        print(f"  ILinker3: {len(sentences)} sentences, {len(components)} components")

        comp_block = self._build_comp_block(components)
        batches = self._make_batches(sentences)
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
                confidence=0.92,
                source="ilinker2",
            )
            for l in merged
        ]

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_comp_block(components) -> str:
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

    # ── prompts (simplified) ─────────────────────────────────────────────

    def _prompt_extract(self, doc_block: str, comp_block: str) -> str:
        return f"""ARCHITECTURE COMPONENTS:
{comp_block}

DOCUMENT:
{doc_block}

TASK: For each sentence, find architecture components EXPLICITLY mentioned or referenced.

Valid: exact name, synonym, abbreviation, or unambiguous partial name in the sentence text.
Invalid: names inside dotted paths, generic English words, or no clear textual evidence.

Return JSON:
{{"links": [{{"s": N_INTEGER, "c": "ComponentName", "text": "matched text", "type": "exact|synonym|partial"}}]}}
Precision is critical."""

    def _prompt_actor(self, doc_block: str, comp_block: str) -> str:
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
        agreed = a_keys & b_keys

        lookup = {}
        for link in pass_b + pass_a:
            key = (link.sentence_number, link.component_id)
            if link.match_type != "exact":
                lookup[key] = link

        for key in agreed:
            if key not in result:
                result[key] = lookup[key]

        return list(result.values())
