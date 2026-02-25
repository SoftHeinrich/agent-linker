"""ILinker2 — High-precision explicit link extractor to replace TransArc baseline.

Designed as a drop-in replacement for TransArc CSV input to V26-family linkers.
Two LLM passes focused solely on explicit mentions (no contextual/coref — V26 handles that).

  Pass A: Extraction-framed (find all mentions)
  Pass B: Actor/subject-framed (what is each sentence about?)
  Merge: exact from either → accept; synonym/partial → intersection only

Output: list[SadSamLink] with source="ilinker2" — same shape as TransArc links.
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


class ILinker2:
    """High-precision explicit extractor — 2 LLM calls per batch, no contextual."""

    def __init__(self, backend: LLMBackend = LLMBackend.CLAUDE):
        self.llm = LLMClient(backend=backend)

    def link(self, text_path: str, model_path: str, transarc_csv: str = None) -> list[SadSamLink]:
        """Extract explicit trace links. transarc_csv is accepted but ignored."""
        sentences = DocumentLoader.load_sentences(text_path)
        components = parse_pcm_repository(model_path)
        name_to_id = {c.name: c.id for c in components}

        print(f"  ILinker2: {len(sentences)} sentences, {len(components)} components")

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

    # ── prompts ──────────────────────────────────────────────────────────

    def _prompt_extract(self, doc_block: str, comp_block: str) -> str:
        return f"""You are a software architecture traceability expert.

ARCHITECTURE COMPONENTS:
{comp_block}

DOCUMENT:
{doc_block}

TASK: For each sentence, identify which architecture components are EXPLICITLY mentioned or referenced.

A valid reference is:
- Exact name: the component name appears verbatim in the sentence
- Synonym: a well-known alternative name for the component (e.g., "code generator" → "CodeGenerator")
- Abbreviation: a shortened form (e.g., "AST" → "AbstractSyntaxTree")
- Partial name: a distinctive sub-phrase of the component name that unambiguously identifies it (e.g., "the scheduler" → "TaskScheduler")

NOT a valid reference:
- A component name that only appears inside a dotted path (e.g., "renderer.utils.config" does NOT reference "Renderer")
- A generic English word used in its ordinary sense (e.g., "optimized code" does NOT reference "Optimizer")
- A sentence that merely describes related functionality without naming or clearly referring to the component

Return ONLY valid JSON:
{{"links": [{{"s": N_INTEGER, "c": "ComponentName", "text": "matched text", "type": "exact|synonym|partial"}}]}}

Precision is critical — only include links with clear textual evidence."""

    def _prompt_actor(self, doc_block: str, comp_block: str) -> str:
        return f"""You are a software architecture traceability expert performing an independent review.

ARCHITECTURE COMPONENTS:
{comp_block}

DOCUMENT:
{doc_block}

TASK: For each sentence, determine which architecture component is the SUBJECT or primary ACTOR.

Ask yourself:
- Which component is this sentence ABOUT?
- Which component PERFORMS or RECEIVES the described action?
- Is the component named, abbreviated, or referred to by a recognizable alias?

Rules:
- Only report components that are explicitly named, abbreviated, or identified by a clear synonym/partial name IN THE SENTENCE TEXT.
- Do NOT report contextual or pronoun-based references (e.g., "It does X" — skip these).
- Do NOT match component names inside dotted package paths (e.g., "renderer.utils.config" does NOT reference "Renderer").
- Do NOT match generic English words used in their ordinary sense (e.g., "optimized code" does NOT reference "Optimizer").

Return ONLY valid JSON:
{{"links": [{{"s": N_INTEGER, "c": "ComponentName", "text": "evidence", "type": "exact|synonym|partial"}}]}}

Include links where the component is clearly the subject or actor. Omit pronoun-only references."""

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
            # Handle "S101" format from some backends (strip "S" prefix)
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

        # Exact from either → accept
        for link in pass_a + pass_b:
            key = (link.sentence_number, link.component_id)
            if link.match_type == "exact":
                result[key] = link

        # Non-exact → intersection
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
