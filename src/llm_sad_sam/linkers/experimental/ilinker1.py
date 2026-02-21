"""ILinker1 — Pure prompt-based traceability link recovery.

3-call precision cascade: no TransArc dependency, no Java.
  Pass A: Full-document explicit extraction
  Pass B: Independent re-extraction (actor/subject framing)
  Pass C: Contextual references (pronouns, continuations, implicit)
  Merge: exact from either → auto-accept; non-exact → intersection; Pass C → accept

Documents > BATCH_SIZE sentences are split into overlapping batches.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from llm_sad_sam.core.data_types import SadSamLink
from llm_sad_sam.core.document_loader import DocumentLoader, Sentence
from llm_sad_sam.llm_client import LLMClient, LLMBackend
from llm_sad_sam.pcm_parser import parse_pcm_repository

logger = logging.getLogger(__name__)

BATCH_SIZE = 50       # sentences per batch
BATCH_OVERLAP = 5     # overlap for context continuity


@dataclass
class ExtractedLink:
    """A link extracted from a single LLM pass."""
    sentence_number: int
    component_name: str
    component_id: str
    matched_text: str
    match_type: str  # exact, synonym, partial, contextual


class ILinker1:
    """Pure LLM linker using a 3-call precision cascade."""

    def __init__(self, backend: LLMBackend = LLMBackend.CLAUDE):
        self.llm = LLMClient(backend=backend)

    def link(self, text_path: str, model_path: str, transarc_csv: str = None) -> list[SadSamLink]:
        """Recover trace links between documentation and architecture model.

        Args:
            text_path: Path to documentation text file (one sentence per line)
            model_path: Path to PCM .repository file
            transarc_csv: Ignored (pure LLM approach, no TransArc dependency)

        Returns:
            List of SadSamLink objects
        """
        sentences = DocumentLoader.load_sentences(text_path)
        components = parse_pcm_repository(model_path)
        name_to_id = {c.name: c.id for c in components}

        print(f"  ILinker1: {len(sentences)} sentences, {len(components)} components")

        comp_block = "\n".join(f"  {i+1}. {c.name}" for i, c in enumerate(components))
        batches = self._make_batches(sentences)
        print(f"    Batches: {len(batches)} (size={BATCH_SIZE}, overlap={BATCH_OVERLAP})")

        # Run 3 passes across all batches
        pass_a = self._run_pass_batched(batches, comp_block, name_to_id, self._prompt_explicit)
        print(f"    Pass A: {len(pass_a)} links")

        pass_b = self._run_pass_batched(batches, comp_block, name_to_id, self._prompt_reextract)
        print(f"    Pass B: {len(pass_b)} links")

        pass_c = self._run_pass_batched(batches, comp_block, name_to_id, self._prompt_contextual)
        print(f"    Pass C: {len(pass_c)} links")

        merged = self._merge_passes(pass_a, pass_b, pass_c)
        print(f"    Merged: {len(merged)} links")

        result = []
        for link in merged:
            result.append(SadSamLink(
                sentence_number=link.sentence_number,
                component_id=link.component_id,
                component_name=link.component_name,
                confidence=0.90,
                source=f"ilinker1_{link.match_type}",
            ))
        return result

    # ------------------------------------------------------------------ #
    #  Batching
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_batches(sentences: list[Sentence]) -> list[list[Sentence]]:
        """Split sentences into overlapping batches."""
        if len(sentences) <= BATCH_SIZE:
            return [sentences]
        batches = []
        start = 0
        while start < len(sentences):
            end = min(start + BATCH_SIZE, len(sentences))
            batches.append(sentences[start:end])
            if end >= len(sentences):
                break
            start = end - BATCH_OVERLAP
        return batches

    def _run_pass_batched(self, batches: list[list[Sentence]], comp_block: str,
                          name_to_id: dict, prompt_fn) -> list[ExtractedLink]:
        """Run a single pass across all batches and deduplicate."""
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
                print(f"      batch {i+1}/{len(batches)}: +{len(links)} links (total {len(seen)})")
        return list(seen.values())

    # ------------------------------------------------------------------ #
    #  Prompt builders (return prompt string, don't call LLM)
    # ------------------------------------------------------------------ #

    def _prompt_explicit(self, doc_block: str, comp_block: str) -> str:
        """Pass A: Full-document explicit extraction."""
        return f"""You are a software architecture traceability expert.

ARCHITECTURE COMPONENTS:
{comp_block}

DOCUMENT:
{doc_block}

TASK: Identify which architecture components are explicitly referenced in each sentence.

A reference includes:
- Exact name match (the component name appears in the sentence)
- Clear synonym (e.g., "code generator" for a component named "CodeGenerator")
- Abbreviation (e.g., "AST" for "AbstractSyntaxTree")
- Partial name match (e.g., "the scheduler" for "TaskScheduler")

NOT a reference:
- Component name inside a dotted package path (e.g., "renderer.utils.config" does NOT reference component "Renderer")
- Merely mentioning a concept related to the component without referring to it as an architectural element
- Generic English words that happen to match a component name when used in their ordinary sense (e.g., "optimized code" is NOT a reference to component "Optimizer")

Return ONLY valid JSON with no additional text:
{{"links": [{{"s": <sentence_number>, "c": "<ComponentName>", "text": "<matched text in sentence>", "type": "exact|synonym|partial"}}]}}

Precision is critical — only include links you are confident about. Every link must have clear textual evidence in the sentence."""

    def _prompt_reextract(self, doc_block: str, comp_block: str) -> str:
        """Pass B: Independent re-extraction with actor/subject framing."""
        return f"""You are a software architecture traceability expert performing an independent analysis.

ARCHITECTURE COMPONENTS:
{comp_block}

DOCUMENT:
{doc_block}

TASK: For each sentence, determine which architecture component (if any) is the primary ACTOR, SUBJECT, or OBJECT being described or operated on.

Focus on:
- What component is this sentence ABOUT?
- What component is PERFORMING the action described?
- What component is RECEIVING or being AFFECTED by the action?

Include:
- Direct name mentions (exact, synonym, abbreviation, partial name)
- Cases where the component is clearly the topic even if referred to indirectly

Exclude:
- Component names appearing only inside dotted package paths (e.g., "renderer.utils.config")
- Generic English words used in their ordinary sense
- Vague or uncertain associations

Return ONLY valid JSON with no additional text:
{{"links": [{{"s": <sentence_number>, "c": "<ComponentName>", "text": "<evidence text>", "type": "exact|synonym|partial"}}]}}

Be precise — only include links where the component is clearly referenced or discussed."""

    def _prompt_contextual(self, doc_block: str, comp_block: str) -> str:
        """Pass C: Contextual references (pronouns, continuations, implicit)."""
        return f"""You are a software architecture traceability expert specializing in discourse analysis.

ARCHITECTURE COMPONENTS:
{comp_block}

DOCUMENT:
{doc_block}

TASK: Find INDIRECT references to architecture components — cases where a sentence discusses a component without naming it explicitly.

Look for:
1. Pronoun references: "It handles...", "They communicate...", "This component..."
   → Resolve the pronoun to the most likely component based on preceding sentences.
2. Continuation references: A sentence that continues describing a component mentioned in a previous sentence without repeating its name.
3. Implicit references: A sentence describes functionality that clearly belongs to a specific component based on architectural context.

For each link, cite which earlier sentence establishes the antecedent.

Exclude:
- Any sentence that already contains the component name (those are explicit, not contextual)
- Uncertain or speculative associations
- Component names inside dotted package paths

Return ONLY valid JSON with no additional text:
{{"links": [{{"s": <sentence_number>, "c": "<ComponentName>", "antecedent_s": <sentence_number_of_antecedent>, "type": "contextual"}}]}}

Only include links where the contextual reference is clear and unambiguous."""

    # ------------------------------------------------------------------ #
    #  LLM query + JSON parsing
    # ------------------------------------------------------------------ #

    def _query_and_parse(self, prompt: str, name_to_id: dict) -> list[ExtractedLink]:
        """Send prompt to LLM and parse the JSON response into ExtractedLink list."""
        response = self.llm.query(prompt, timeout=300)
        if not response.success:
            logger.warning(f"LLM query failed: {response.error}")
            print(f"      LLM query failed: {response.error}")
            return []

        data = self.llm.extract_json(response)
        if not data or "links" not in data:
            logger.warning("Failed to parse LLM response as JSON")
            print("      Failed to parse LLM response as JSON")
            return []

        links = []
        for item in data["links"]:
            snum = item.get("s")
            cname = item.get("c", "")
            match_type = item.get("type", "unknown")
            matched_text = item.get("text", "")

            if not snum or not cname:
                continue

            # Resolve component name to ID
            cid = name_to_id.get(cname)
            if not cid:
                # Try case-insensitive match
                for name, nid in name_to_id.items():
                    if name.lower() == cname.lower():
                        cid = nid
                        cname = name
                        break
            if not cid:
                logger.debug(f"Unknown component '{cname}' in S{snum}, skipping")
                continue

            links.append(ExtractedLink(
                sentence_number=int(snum),
                component_name=cname,
                component_id=cid,
                matched_text=matched_text,
                match_type=match_type,
            ))

        return links

    # ------------------------------------------------------------------ #
    #  Merge logic
    # ------------------------------------------------------------------ #

    def _merge_passes(self, pass_a: list[ExtractedLink], pass_b: list[ExtractedLink],
                      pass_c: list[ExtractedLink]) -> list[ExtractedLink]:
        """Merge results from 3 passes using precision-focused rules.

        - Exact name matches from either Pass A or B → auto-accept
        - Non-exact (synonym/partial) → require both passes to agree (intersection)
        - Pass C contextual links → accept if not already covered
        """
        result: dict[tuple[int, str], ExtractedLink] = {}

        # Exact matches from either pass → auto-accept
        for link in pass_a + pass_b:
            key = (link.sentence_number, link.component_id)
            if link.match_type == "exact":
                result[key] = link

        # Non-exact: require intersection (both passes agree)
        a_nonexact = {(l.sentence_number, l.component_id) for l in pass_a if l.match_type != "exact"}
        b_nonexact = {(l.sentence_number, l.component_id) for l in pass_b if l.match_type != "exact"}
        agreed = a_nonexact & b_nonexact

        # Build lookup for non-exact links (prefer pass_a version)
        nonexact_lookup = {}
        for link in pass_b + pass_a:  # pass_a overwrites pass_b
            key = (link.sentence_number, link.component_id)
            if link.match_type != "exact":
                nonexact_lookup[key] = link

        for key in agreed:
            if key not in result:
                result[key] = nonexact_lookup[key]

        # Pass C contextual links → accept if not already covered
        for link in pass_c:
            key = (link.sentence_number, link.component_id)
            if key not in result:
                result[key] = link

        return list(result.values())
