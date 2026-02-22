"""CNR + I2 Linker — Component Name Recovery + ILinker2 two-pass extraction.

Discovers component names from the SAD text alone, then runs ILinker2-style
two-pass extraction (extract + actor framing, intersection merge) on the
discovered components.

Architecture:
  Phase CNR: Component Name Discovery (from cnr_linker.py)
  Pass A: Extraction-framed (find all mentions)
  Pass B: Actor/subject-framed (what is each sentence about?)
  Merge: exact from either → accept; synonym/partial → intersection only
  Eval Bridge: Map discovered names → PCM IDs (for evaluation only)
"""

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

from ...core.data_types import SadSamLink
from ...core.document_loader import DocumentLoader, Sentence
from ...llm_client import LLMClient, LLMBackend
from ...pcm_parser import ArchitectureComponent, parse_pcm_repository
from .cnr_linker import CNRLinker

BATCH_SIZE = 50
BATCH_OVERLAP = 5


@dataclass
class ExtractedLink:
    sentence_number: int
    component_name: str
    component_id: str
    matched_text: str
    match_type: str  # exact, synonym, partial


class CNRI2Linker:
    """Component Name Recovery + ILinker2-style two-pass extraction."""

    def __init__(self, backend: LLMBackend = LLMBackend.CLAUDE):
        os.environ.setdefault("CLAUDE_MODEL", "sonnet")
        self.llm = LLMClient(backend=backend)
        self._cnr = CNRLinker(backend=backend)
        print(f"CNRI2Linker: Component Name Recovery + I2 two-pass extraction")
        print(f"  Backend: {self.llm.backend.value}, Model: {os.environ.get('CLAUDE_MODEL', 'default')}")

    def link(self, text_path: str, model_path: str = None,
             transarc_csv: str = None) -> list[SadSamLink]:
        """Discover components from text, run I2 extraction, optionally map to PCM IDs."""
        t0 = time.time()

        sentences = DocumentLoader.load_sentences(text_path)
        print(f"Loaded {len(sentences)} sentences")

        # ── Phase CNR: Discover component names ──
        components = self._cnr.discover_components_from_sentences(sentences)
        name_to_id = {c.name: c.id for c in components}

        print(f"\n[I2 Extraction] Two-pass extraction on {len(components)} discovered components")

        comp_block = self._build_comp_block(components)
        # Include discovered aliases in extraction prompts
        alias_info = []
        for canonical, aliases in self._cnr._discovered_aliases.items():
            for a in aliases:
                alias_info.append(f"{a} = {canonical}")
        alias_block = f"\nKNOWN ALIASES: {', '.join(alias_info)}" if alias_info else ""

        batches = self._make_batches(sentences)
        print(f"  Batches: {len(batches)}")

        pass_a = self._run_pass_batched(batches, comp_block, alias_block, name_to_id,
                                         self._prompt_extract)
        print(f"  Pass A (extract): {len(pass_a)} links")

        pass_b = self._run_pass_batched(batches, comp_block, alias_block, name_to_id,
                                         self._prompt_actor)
        print(f"  Pass B (actor):   {len(pass_b)} links")

        merged = self._merge(pass_a, pass_b)
        print(f"  Merged: {len(merged)} links")

        links = [
            SadSamLink(
                sentence_number=l.sentence_number,
                component_id=l.component_id,
                component_name=l.component_name,
                confidence=0.92,
                source="cnr_i2",
            )
            for l in merged
        ]

        # ── Eval bridge (if model_path provided) ──
        if model_path:
            pcm_components = parse_pcm_repository(model_path)
            bridge = self._cnr._build_eval_bridge(components, pcm_components)
            links = self._remap_links(links, bridge)
            print(f"  After eval bridge remap: {len(links)} links")

        elapsed = time.time() - t0
        print(f"\nFinal: {len(links)} links ({elapsed:.0f}s)")
        return links

    # ── Helpers ──────────────────────────────────────────────────────────

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

    def _run_pass_batched(self, batches, comp_block, alias_block, name_to_id,
                          prompt_fn) -> list[ExtractedLink]:
        seen: dict[tuple[int, str], ExtractedLink] = {}
        for i, batch in enumerate(batches):
            doc_block = "\n".join(f"S{s.number}: {s.text}" for s in batch)
            prompt = prompt_fn(doc_block, comp_block, alias_block)
            links = self._query_and_parse(prompt, name_to_id)
            for link in links:
                key = (link.sentence_number, link.component_id)
                if key not in seen:
                    seen[key] = link
            if len(batches) > 1:
                print(f"      batch {i+1}/{len(batches)}: +{len(links)} (total {len(seen)})")
        return list(seen.values())

    # ── Prompts ─────────────────────────────────────────────────────────

    def _prompt_extract(self, doc_block: str, comp_block: str, alias_block: str) -> str:
        return f"""You are a software architecture traceability expert.

ARCHITECTURE COMPONENTS:
{comp_block}
{alias_block}

DOCUMENT:
{doc_block}

TASK: For each sentence, identify which architecture components are EXPLICITLY mentioned or referenced.

A valid reference is:
- Exact name: the component name appears verbatim in the sentence
- Synonym: a well-known alternative name for the component (e.g., "code generator" for "CodeGenerator")
- Abbreviation: a shortened form (e.g., "AST" for "AbstractSyntaxTree")
- Partial name: a distinctive sub-phrase of the component name that unambiguously identifies it

NOT a valid reference:
- A component name that only appears inside a dotted path (e.g., "renderer.utils.config" does NOT reference "Renderer")
- A generic English word used in its ordinary sense (e.g., "optimized code" does NOT reference "Optimizer")
- A sentence that merely describes related functionality without naming or clearly referring to the component

Return ONLY valid JSON:
{{"links": [{{"s": <sentence_number>, "c": "<ComponentName>", "text": "<matched text>", "type": "exact|synonym|partial"}}]}}

Precision is critical — only include links with clear textual evidence.
JSON only:"""

    def _prompt_actor(self, doc_block: str, comp_block: str, alias_block: str) -> str:
        return f"""You are a software architecture traceability expert performing an independent review.

ARCHITECTURE COMPONENTS:
{comp_block}
{alias_block}

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
- Do NOT match component names inside dotted package paths.
- Do NOT match generic English words used in their ordinary sense.

Return ONLY valid JSON:
{{"links": [{{"s": <sentence_number>, "c": "<ComponentName>", "text": "<evidence>", "type": "exact|synonym|partial"}}]}}

Be conservative — omit uncertain links.
JSON only:"""

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
            if snum is None or not cname:
                continue

            cid = name_to_id.get(cname)
            if not cid:
                for name, nid in name_to_id.items():
                    if name.lower() == cname.lower():
                        cid, cname = nid, name
                        break
            if not cid:
                # Check discovered aliases
                for canonical, aliases in self._cnr._discovered_aliases.items():
                    if cname in aliases or cname.lower() in [a.lower() for a in aliases]:
                        cid = name_to_id.get(canonical)
                        cname = canonical
                        break
            if not cid:
                continue

            links.append(ExtractedLink(
                sentence_number=int(snum),
                component_name=cname,
                component_id=cid,
                matched_text=item.get("text", ""),
                match_type=item.get("type", "unknown"),
            ))
        return links

    # ── Merge ───────────────────────────────────────────────────────────

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

    # ── Eval bridge remap ───────────────────────────────────────────────

    @staticmethod
    def _remap_links(links: list[SadSamLink], bridge: dict[str, str]) -> list[SadSamLink]:
        """Remap links from cnr IDs to PCM IDs. Drop unmatched."""
        remapped = []
        for lk in links:
            pcm_id = bridge.get(lk.component_id)
            if pcm_id:
                remapped.append(SadSamLink(
                    sentence_number=lk.sentence_number,
                    component_id=pcm_id,
                    component_name=lk.component_name,
                    confidence=lk.confidence,
                    source=lk.source,
                ))
        return remapped
