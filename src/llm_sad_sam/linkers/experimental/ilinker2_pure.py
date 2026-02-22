"""ILinker2-Pure — Zero heuristics, 100% LLM + set operations.

Extends ILinker2 with additional LLM phases to close the recall gap,
but without ANY regex filters, deterministic overrides, or hand-coded patterns.

Pipeline:
  Phase 1: LLM synonym/abbreviation discovery (1 call)
  Phase 2: ILinker2 two-pass extraction with synonyms (2-10 calls)
  Phase 3: LLM coreference resolution (1-2 calls)
  Phase 4: LLM validation of non-exact links (1 call)
  Merge: set operations only

Total: ~5-15 LLM calls. Zero regex. Zero heuristics.
"""

import os

from llm_sad_sam.core.data_types import SadSamLink
from llm_sad_sam.core.document_loader import DocumentLoader, Sentence
from llm_sad_sam.llm_client import LLMClient, LLMBackend
from llm_sad_sam.pcm_parser import parse_pcm_repository, ArchitectureComponent

BATCH_SIZE = 50
BATCH_OVERLAP = 5


class ILinker2Pure:
    """Pure LLM linker — no heuristics, no regex, no deterministic overrides."""

    def __init__(self, backend: LLMBackend = LLMBackend.CLAUDE):
        os.environ.setdefault("CLAUDE_MODEL", "sonnet")
        self.llm = LLMClient(backend=backend)
        print(f"ILinker2Pure: Zero-heuristic LLM linker")
        print(f"  Backend: {self.llm.backend.value}, Model: {os.environ.get('CLAUDE_MODEL', 'default')}")

    def link(self, text_path: str, model_path: str, transarc_csv: str = None) -> list[SadSamLink]:
        sentences = DocumentLoader.load_sentences(text_path)
        components = parse_pcm_repository(model_path)
        name_to_id = {c.name: c.id for c in components}
        comp_names = [c.name for c in components]

        print(f"Loaded {len(components)} components, {len(sentences)} sentences")

        # ── Phase 1: Synonym/abbreviation discovery ──
        print("\n[Phase 1] Synonym Discovery")
        synonyms = self._discover_synonyms(sentences, comp_names)
        print(f"  Discovered: {len(synonyms)} mappings")
        for alt, canonical in synonyms.items():
            print(f"    {alt} -> {canonical}")

        # ── Phase 2: Two-pass extraction (with synonyms) ──
        print("\n[Phase 2] Two-Pass Extraction")
        comp_block = "\n".join(f"  {i+1}. {c.name}" for i, c in enumerate(components))

        # Build synonym block for prompts
        syn_lines = [f"  {alt} = {canon}" for alt, canon in synonyms.items()]
        syn_block = "\nKNOWN ALIASES:\n" + "\n".join(syn_lines) if syn_lines else ""

        batches = self._make_batches(sentences)
        print(f"  Batches: {len(batches)}")

        pass_a = self._run_pass(batches, comp_block, syn_block, name_to_id, synonyms,
                                self._prompt_extract)
        print(f"  Pass A (extract): {len(pass_a)} links")

        pass_b = self._run_pass(batches, comp_block, syn_block, name_to_id, synonyms,
                                self._prompt_actor)
        print(f"  Pass B (actor):   {len(pass_b)} links")

        # Merge: exact from either, non-exact intersection
        merged = self._merge(pass_a, pass_b)
        print(f"  Merged: {len(merged)} links")

        # ── Phase 3: Coreference resolution ──
        print("\n[Phase 3] Coreference Resolution")
        coref_links = self._resolve_coreferences(sentences, components, name_to_id,
                                                  synonyms, merged)
        print(f"  Coref links: {len(coref_links)}")

        # ── Phase 4: Validation of uncertain links ──
        # Separate high-confidence (exact match, both passes agree) from uncertain
        a_keys = {(s, c) for s, c, _, _ in pass_a}
        b_keys = {(s, c) for s, c, _, _ in pass_b}
        both_agree = a_keys & b_keys
        exact_keys = {(s, c) for s, c, _, t in pass_a + pass_b if t == "exact"}

        confident = exact_keys | both_agree
        uncertain_coref = [(s, c, n) for s, c, n in coref_links if (s, c) not in confident]

        if uncertain_coref:
            print(f"\n[Phase 4] Validation ({len(uncertain_coref)} uncertain coref links)")
            validated_coref = self._validate_links(uncertain_coref, sentences, components,
                                                    name_to_id, synonyms)
            print(f"  Validated: {len(validated_coref)} of {len(uncertain_coref)}")
        else:
            print(f"\n[Phase 4] Validation — nothing to validate")
            validated_coref = []

        # ── Final merge ──
        final_map: dict[tuple[int, str], SadSamLink] = {}

        # Extraction links (already merged)
        for snum, cid, cname, mtype in merged:
            key = (snum, cid)
            if key not in final_map:
                final_map[key] = SadSamLink(snum, cid, cname, 0.92, "ilinker2")

        # Confident coref links
        confident_coref = [(s, c, n) for s, c, n in coref_links if (s, c) in confident]
        for snum, cid, cname in confident_coref:
            key = (snum, cid)
            if key not in final_map:
                final_map[key] = SadSamLink(snum, cid, cname, 0.85, "coreference")

        # Validated coref links
        for snum, cid, cname in validated_coref:
            key = (snum, cid)
            if key not in final_map:
                final_map[key] = SadSamLink(snum, cid, cname, 0.80, "coreference")

        final = list(final_map.values())
        print(f"\nFinal: {len(final)} links")
        return final

    # ── Phase 1: Synonym discovery ───────────────────────────────────────

    def _discover_synonyms(self, sentences: list[Sentence],
                           comp_names: list[str]) -> dict[str, str]:
        """Ask LLM to find abbreviations, synonyms, partial names for components."""
        doc_lines = [f"S{s.number}: {s.text}" for s in sentences[:100]]

        prompt = f"""You are a software architecture traceability expert.

ARCHITECTURE COMPONENTS: {', '.join(comp_names)}

DOCUMENT:
{chr(10).join(doc_lines)}

Find all alternative names used for these components in the document:

1. ABBREVIATIONS: Short forms explicitly defined (e.g., "Full Name (FN)" defines FN)
2. SYNONYMS: Alternative names that specifically refer to exactly one component
3. PARTIAL NAMES: Shorter forms of multi-word component names used unambiguously

Rules:
- Each alternative must clearly refer to exactly ONE component
- Reject generic descriptions ("the system", "the service") that could mean anything
- Only include mappings supported by the document text

Return ONLY valid JSON:
{{"mappings": [{{"alt": "alternative_name", "component": "ExactComponentName"}}]}}

Be precise — wrong mappings cause false links."""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=150))
        result = {}
        if data and "mappings" in data:
            comp_set = set(comp_names)
            comp_lower = {n.lower(): n for n in comp_names}
            for item in data["mappings"]:
                alt = item.get("alt", "").strip()
                comp = item.get("component", "").strip()
                if not alt or not comp:
                    continue
                # Resolve component name
                if comp in comp_set:
                    result[alt] = comp
                elif comp.lower() in comp_lower:
                    result[alt] = comp_lower[comp.lower()]
        return result

    # ── Phase 2: Two-pass extraction ─────────────────────────────────────

    def _run_pass(self, batches, comp_block, syn_block, name_to_id, synonyms,
                  prompt_fn) -> list[tuple[int, str, str, str]]:
        """Run one extraction pass. Returns list of (snum, cid, cname, type)."""
        seen: dict[tuple[int, str], tuple] = {}
        comp_lower = {n.lower(): n for n in name_to_id}
        syn_lower = {a.lower(): c for a, c in synonyms.items()}

        for i, batch in enumerate(batches):
            doc_block = "\n".join(f"S{s.number}: {s.text}" for s in batch)
            prompt = prompt_fn(doc_block, comp_block, syn_block)
            response = self.llm.query(prompt, timeout=300)
            if not response.success:
                continue
            data = self.llm.extract_json(response)
            if not data or "links" not in data:
                continue
            for item in data["links"]:
                snum = item.get("s")
                cname = item.get("c", "")
                if snum is None or not cname:
                    continue
                # Resolve: exact → case-insensitive → synonym
                cid = name_to_id.get(cname)
                if not cid and cname.lower() in comp_lower:
                    cname = comp_lower[cname.lower()]
                    cid = name_to_id.get(cname)
                if not cid and cname.lower() in syn_lower:
                    cname = syn_lower[cname.lower()]
                    cid = name_to_id.get(cname)
                if not cid and cname in synonyms:
                    cname = synonyms[cname]
                    cid = name_to_id.get(cname)
                if not cid:
                    continue
                key = (int(snum), cid)
                if key not in seen:
                    seen[key] = (int(snum), cid, cname,
                                 item.get("type", "unknown"))
            if len(batches) > 1:
                print(f"    batch {i+1}/{len(batches)}: total {len(seen)}")
        return list(seen.values())

    def _prompt_extract(self, doc_block, comp_block, syn_block):
        return f"""You are a software architecture traceability expert.

ARCHITECTURE COMPONENTS:
{comp_block}
{syn_block}

DOCUMENT:
{doc_block}

TASK: For each sentence, identify which architecture components are EXPLICITLY mentioned or referenced.

A valid reference is:
- Exact name: the component name appears verbatim
- Known alias: a synonym or abbreviation listed above
- Partial name: a distinctive sub-phrase that unambiguously identifies the component

NOT a valid reference:
- A component name inside a dotted package path (e.g., "logic.api" does NOT reference "Logic" as a component)
- A generic English word used in its ordinary sense
- A sentence describing related functionality without naming the component

Return ONLY valid JSON:
{{"links": [{{"s": <sentence_number>, "c": "<ComponentName>", "text": "<matched text>", "type": "exact|synonym|partial"}}]}}

Precision is critical — only include links with clear textual evidence.
JSON only:"""

    def _prompt_actor(self, doc_block, comp_block, syn_block):
        return f"""You are a software architecture traceability expert performing an independent review.

ARCHITECTURE COMPONENTS:
{comp_block}
{syn_block}

DOCUMENT:
{doc_block}

TASK: For each sentence, determine which architecture component is the SUBJECT or primary ACTOR.

Ask yourself:
- Which component is this sentence ABOUT?
- Which component PERFORMS or RECEIVES the described action?
- Is the component named, abbreviated, or referred to by a recognizable alias?

Rules:
- Only report components explicitly named, abbreviated, or identified by a clear synonym/partial.
- Do NOT report contextual or pronoun-based references.
- Do NOT match component names inside dotted package paths.
- Do NOT match generic English words used in their ordinary sense.

Return ONLY valid JSON:
{{"links": [{{"s": <sentence_number>, "c": "<ComponentName>", "text": "<evidence>", "type": "exact|synonym|partial"}}]}}

Be conservative — omit uncertain links.
JSON only:"""

    # ── Phase 3: Coreference ─────────────────────────────────────────────

    def _resolve_coreferences(self, sentences, components, name_to_id,
                              synonyms, extracted) -> list[tuple[int, str, str]]:
        """Ask LLM to find pronoun/coreference links for sentences not yet linked."""
        # Build set of already-linked sentences
        linked_sents = {s for s, _, _, _ in extracted}
        # Find unlinked sentences that might have coref
        unlinked = [s for s in sentences if s.number not in linked_sents]
        if not unlinked:
            return []

        comp_names = [c.name for c in components]
        comp_lower = {n.lower(): n for n in name_to_id}
        syn_lower = {a.lower(): c for a, c in synonyms.items()}

        # Build context: show extracted links so LLM knows what's nearby
        link_context = {}
        for snum, cid, cname, _ in extracted:
            link_context.setdefault(snum, []).append(cname)

        results = []
        # Batch unlinked sentences
        batch_size = 30
        for start in range(0, len(unlinked), batch_size):
            batch = unlinked[start:start + batch_size]

            # Build context window: for each unlinked sentence, show surrounding linked ones
            context_lines = []
            for s in batch:
                # Show 3 sentences before and after with their links
                window = [ss for ss in sentences
                          if abs(ss.number - s.number) <= 3]
                for ws in window:
                    linked = link_context.get(ws.number, [])
                    link_str = f" [LINKED: {', '.join(linked)}]" if linked else ""
                    marker = " <<<" if ws.number == s.number else ""
                    context_lines.append(
                        f"S{ws.number}: {ws.text}{link_str}{marker}")
                context_lines.append("---")

            prompt = f"""You are resolving coreference in a software architecture document.

COMPONENTS: {', '.join(comp_names)}

Below are document excerpts. Sentences marked with <<< are UNLINKED — they may refer
to a component via pronoun ("it", "they", "this"), demonstrative, or implicit reference.
Sentences with [LINKED: X] are already linked to component X.

CONTEXT:
{chr(10).join(context_lines)}

For each unlinked sentence (<<<), determine if it refers to a specific component
through coreference (pronoun, demonstrative, or continuation of a topic).

Rules:
- Only assign a component if the reference is clear from context
- "It" or "This" typically refers to the most recently mentioned component
- Skip sentences that are generic or don't refer to any specific component
- Skip sentences where the referent is ambiguous between multiple components

Return ONLY valid JSON:
{{"coref_links": [{{"s": <sentence_number>, "c": "<ComponentName>", "reason": "<brief explanation>"}}]}}

Be conservative — skip ambiguous cases.
JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=300))
            if data and "coref_links" in data:
                for item in data["coref_links"]:
                    snum = item.get("s")
                    cname = item.get("c", "")
                    if snum is None or not cname:
                        continue
                    cid = name_to_id.get(cname)
                    if not cid and cname.lower() in comp_lower:
                        cname = comp_lower[cname.lower()]
                        cid = name_to_id.get(cname)
                    if not cid and cname.lower() in syn_lower:
                        cname = syn_lower[cname.lower()]
                        cid = name_to_id.get(cname)
                    if not cid:
                        continue
                    results.append((int(snum), cid, cname))
                    reason = item.get("reason", "")
                    print(f"    Coref: S{snum} -> {cname} ({reason[:50]})")

        return results

    # ── Phase 4: Validation ──────────────────────────────────────────────

    def _validate_links(self, uncertain: list[tuple[int, str, str]],
                        sentences, components, name_to_id,
                        synonyms) -> list[tuple[int, str, str]]:
        """Ask LLM to validate uncertain links."""
        sent_map = {s.number: s for s in sentences}

        link_lines = []
        for snum, cid, cname in uncertain:
            sent = sent_map.get(snum)
            text = sent.text[:120] if sent else "?"
            link_lines.append(f"  S{snum} -> {cname}: \"{text}\"")

        prompt = f"""Review these proposed trace links between sentences and architecture components.

COMPONENTS: {', '.join(c.name for c in components)}

PROPOSED LINKS:
{chr(10).join(link_lines)}

For each link, decide APPROVE or REJECT:
- APPROVE if the sentence clearly refers to or is about that component (even via coreference)
- REJECT if the connection is too weak, generic, or the sentence is about something else

Return ONLY valid JSON:
{{"approved": [<sentence_numbers>], "rejected": [<sentence_numbers>]}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
        if not data:
            return []

        approved_snums = set(data.get("approved", []))
        return [(s, c, n) for s, c, n in uncertain if s in approved_snums]

    # ── Merge ────────────────────────────────────────────────────────────

    @staticmethod
    def _merge(pass_a, pass_b):
        """Exact from either pass → accept. Non-exact → intersection only."""
        result: dict[tuple[int, str], tuple] = {}

        for link in pass_a + pass_b:
            key = (link[0], link[1])
            if link[3] == "exact":
                result[key] = link

        a_keys = {(l[0], l[1]) for l in pass_a if l[3] != "exact"}
        b_keys = {(l[0], l[1]) for l in pass_b if l[3] != "exact"}
        agreed = a_keys & b_keys

        lookup = {}
        for link in pass_b + pass_a:
            key = (link[0], link[1])
            if link[3] != "exact":
                lookup[key] = link

        for key in agreed:
            if key not in result:
                result[key] = lookup[key]

        return list(result.values())

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _make_batches(sentences):
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
