"""CNR Linker — Component Name Recovery from text + simple extraction.

Discovers architecture component names from the SAD text alone (no PCM model needed),
then extracts trace links using a simple LLM-based extraction pass.

CNR Discovery Pipeline:
  Phase DA (optional): Document Analysis — extract entities, synonyms, abbreviations
  CNR-0: Section heading extraction (from DA if enabled, else LLM call)
  CNR-1: Broad extraction (recall-focused, DA entities as hints if enabled)
  CNR-2: Actor/responsibility extraction (independent pass, DA entities as hints)
  CNR-3: Judge filter (reject concepts, keep components; DA stats if enabled)
  CNR-4: Deduplication & alias grouping (DA synonym groups as starting points)
  → Synthetic ArchitectureComponent objects

Eval Bridge: Maps discovered names → PCM IDs for evaluation only.
"""

import os
import re
import time
from dataclasses import dataclass, field

from ...core.data_types import SadSamLink
from ...core.document_loader import DocumentLoader, Sentence
from ...llm_client import LLMClient, LLMBackend
from ...pcm_parser import ArchitectureComponent, parse_pcm_repository

BATCH_SIZE = 80  # sentences per batch for discovery


@dataclass
class DocumentAnalysis:
    """Results of document-level analysis (no component list needed)."""
    named_entities: list[str] = field(default_factory=list)
    synonym_groups: list[list[str]] = field(default_factory=list)
    abbreviations: dict[str, str] = field(default_factory=dict)  # short → long
    section_headings: list[str] = field(default_factory=list)
    # Deterministic stats (computed after LLM extraction)
    term_frequency: dict[str, int] = field(default_factory=dict)  # entity → sentence count
    term_spread: dict[str, float] = field(default_factory=dict)  # entity → spread (0-1)
    co_occurrence: dict[str, list[str]] = field(default_factory=dict)  # entity → list of co-occurring entities


class CNRLinker:
    """Component Name Recovery + simple extraction linker."""

    def __init__(self, backend: LLMBackend = LLMBackend.CLAUDE, enable_da: bool = False):
        os.environ.setdefault("CLAUDE_MODEL", "sonnet")
        self.llm = LLMClient(backend=backend)
        self._eval_bridge: dict[str, str] = {}
        self._discovered_aliases: dict[str, list[str]] = {}
        self._enable_da = enable_da
        self._doc_analysis: DocumentAnalysis | None = None
        da_str = " + Document Analysis" if enable_da else ""
        print(f"CNRLinker: Component Name Recovery + simple extraction{da_str}")
        print(f"  Backend: {self.llm.backend.value}, Model: {os.environ.get('CLAUDE_MODEL', 'default')}")

    # ═════════════════════════════════════════════════════════════════════
    # Public API
    # ═════════════════════════════════════════════════════════════════════

    def link(self, text_path: str, model_path: str = None,
             transarc_csv: str = None) -> list[SadSamLink]:
        """Discover components from text, extract links, optionally map to PCM IDs.

        Args:
            text_path: Path to SAD text file.
            model_path: Optional PCM model path (used ONLY for eval bridge, never for discovery).
            transarc_csv: Accepted but ignored.

        Returns:
            list[SadSamLink] with PCM IDs if model_path given, else synthetic cnr_N IDs.
        """
        t0 = time.time()

        sentences = DocumentLoader.load_sentences(text_path)
        print(f"Loaded {len(sentences)} sentences")

        # ── Phase CNR: Discover component names ──
        components = self.discover_components_from_sentences(sentences)
        name_to_id = {c.name: c.id for c in components}

        # ── Simple extraction ──
        print(f"\n[Extraction] Simple LLM extraction")
        links = self._simple_extract(sentences, components, name_to_id)
        print(f"  Extracted: {len(links)} links")

        # ── Eval bridge (if model_path provided) ──
        if model_path:
            pcm_components = parse_pcm_repository(model_path)
            self._eval_bridge = self._build_eval_bridge(components, pcm_components)
            links = self._remap_links(links)
            print(f"  After eval bridge remap: {len(links)} links")

        elapsed = time.time() - t0
        print(f"\nFinal: {len(links)} links ({elapsed:.0f}s)")
        return links

    def discover_components(self, text_path: str) -> list[ArchitectureComponent]:
        """Public API: discover component names from text file."""
        sentences = DocumentLoader.load_sentences(text_path)
        return self.discover_components_from_sentences(sentences)

    def discover_components_from_sentences(self, sentences: list[Sentence]) -> list[ArchitectureComponent]:
        """Discover component names from pre-loaded sentences."""
        # Phase DA: Document Analysis (optional, replaces CNR-0 heading LLM call)
        da = None
        if self._enable_da:
            print(f"\n[Phase DA] Document Analysis")
            da = self._document_analysis(sentences)
            self._doc_analysis = da
            print(f"  Entities: {len(da.named_entities)}")
            print(f"  Synonym groups: {len(da.synonym_groups)}")
            print(f"  Abbreviations: {len(da.abbreviations)}")
            print(f"  Headings: {len(da.section_headings)}")

        # CNR-0: Section heading extraction
        print(f"\n[CNR-0] Section heading extraction")
        if da and da.section_headings:
            # Use DA headings directly — no separate LLM call needed
            heading_names = list(da.section_headings)
            # Still run abbreviation scan (deterministic, no LLM)
            heading_names = self._add_abbreviation_scan(heading_names, sentences)
            print(f"  Heading-derived names (from DA): {len(heading_names)}")
        else:
            heading_names = self._extract_section_headings(sentences)
            print(f"  Heading-derived names: {len(heading_names)}")
        for h in heading_names:
            print(f"    - {h}")

        # DA entity hints for CNR-1/2
        da_entity_hints = da.named_entities if da else None

        print(f"\n[CNR-1] Broad extraction")
        pass1 = self._extract_pass_broad(sentences, heading_names, da_entity_hints)
        print(f"  Found: {len(pass1)} candidates")

        print(f"\n[CNR-2] Actor extraction")
        pass2 = self._extract_pass_actor(sentences, heading_names, da_entity_hints)
        print(f"  Found: {len(pass2)} candidates")

        # Combine all candidates — heading names get auto-included as if from both passes
        all_candidates = sorted(set(pass1) | set(pass2) | set(heading_names))
        print(f"\n  Combined unique candidates: {len(all_candidates)}")

        # Compute document statistics for each candidate
        print(f"\n[CNR-S] Candidate statistics")
        candidate_stats = self._compute_candidate_stats(all_candidates, sentences)

        print(f"\n[CNR-3] Judge filter")
        da_synonym_groups = da.synonym_groups if da else None
        filtered = self._judge_filter(all_candidates, pass1, pass2, sentences,
                                       heading_names, candidate_stats, da_synonym_groups)
        print(f"  Accepted: {len(filtered)}")

        print(f"\n[CNR-4] Deduplication")
        canonical_names = self._deduplicate_names(filtered, sentences, da_synonym_groups)
        print(f"  Canonical names: {len(canonical_names)}")
        for name in sorted(canonical_names):
            aliases = self._discovered_aliases.get(name, [])
            alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
            print(f"    - {name}{alias_str}")

        components = self._build_synthetic_components(canonical_names)
        return components

    def get_eval_bridge(self) -> dict[str, str]:
        """cnr_id → pcm_id mapping (available after link() if model_path given)."""
        return self._eval_bridge

    # ═════════════════════════════════════════════════════════════════════
    # Phase DA: Document Analysis
    # ═════════════════════════════════════════════════════════════════════

    def _document_analysis(self, sentences: list[Sentence]) -> DocumentAnalysis:
        """Run document-level analysis: extract entities, synonyms, abbreviations, headings.

        One LLM call (batched if doc > 80 sentences), plus deterministic stats.
        No component list needed — this is a pre-discovery pass.
        """
        da = DocumentAnalysis()

        # Build document text for LLM
        batches = self._make_batches(sentences)
        all_entities = set()
        all_synonym_groups = []
        all_abbreviations = {}
        all_headings = []

        for i, batch in enumerate(batches):
            doc_block = "\n".join(f"S{s.number}: {s.text}" for s in batch)
            prompt = f"""Read this software architecture document. Identify:

1. All NAMED SOFTWARE ENTITIES: proper nouns, CamelCase identifiers, acronyms,
   package names, hyphenated identifiers, lowercase terms used as architectural
   layer/module names. Include anything that looks like it could be a named
   software component, service, library, or subsystem.

2. SYNONYM GROUPS: terms that refer to the same thing in this document.
   Group alternative names, abbreviations, and variant spellings together.

3. ABBREVIATION DEFINITIONS: "Full Name (ABBR)" patterns found in the text.
   Extract both the full name and the abbreviation.

4. SECTION HEADINGS: short sentences (1-6 words) that appear to name a
   software component, subsystem, or architectural element. Strip generic
   suffixes like "overview", "architecture", "diagram".

DOCUMENT:
{doc_block}

Return ONLY valid JSON:
{{"entities": ["Entity1", "Entity2", ...],
  "synonym_groups": [["term1a", "term1b"], ["term2a", "term2b"], ...],
  "abbreviations": {{"ABBR": "Full Name", ...}},
  "headings": ["ComponentName1", "ComponentName2", ...]}}
JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=240))
            if data:
                all_entities.update(data.get("entities", []))
                for group in data.get("synonym_groups", []):
                    if isinstance(group, list) and len(group) >= 2:
                        all_synonym_groups.append(group)
                for abbr, full in data.get("abbreviations", {}).items():
                    if isinstance(abbr, str) and isinstance(full, str):
                        all_abbreviations[abbr] = full
                for h in data.get("headings", []):
                    if isinstance(h, str) and len(h) >= 2:
                        all_headings.append(h)
                if len(batches) > 1:
                    print(f"    Batch {i+1}/{len(batches)}: {len(all_entities)} entities")

        da.named_entities = sorted(all_entities)
        da.synonym_groups = all_synonym_groups
        da.abbreviations = all_abbreviations
        da.section_headings = list(dict.fromkeys(all_headings))  # dedupe preserving order

        # Compute deterministic stats from text + DA entities
        self._compute_da_stats(da, sentences)

        return da

    @staticmethod
    def _compute_da_stats(da: DocumentAnalysis, sentences: list[Sentence]):
        """Compute deterministic term frequency, spread, and co-occurrence for DA entities."""
        total_sents = len(sentences)
        if not da.named_entities or total_sents == 0:
            return

        # Build mention maps
        entity_mentions: dict[str, list[int]] = {}
        for entity in da.named_entities:
            try:
                pattern = re.compile(rf'\b{re.escape(entity)}\b', re.IGNORECASE)
            except re.error:
                continue
            mentions = []
            for sent in sentences:
                if pattern.search(sent.text):
                    mentions.append(sent.number)
            if mentions:
                entity_mentions[entity] = mentions

        # Term frequency
        da.term_frequency = {e: len(m) for e, m in entity_mentions.items()}

        # Term spread
        for entity, mentions in entity_mentions.items():
            if len(mentions) >= 2 and total_sents > 1:
                positions = [s / total_sents for s in mentions]
                da.term_spread[entity] = round(max(positions) - min(positions), 2)
            else:
                da.term_spread[entity] = 0.0

        # Co-occurrence
        mention_sets = {e: set(m) for e, m in entity_mentions.items()}
        for entity, eset in mention_sets.items():
            co = []
            for other, oset in mention_sets.items():
                if other != entity and eset & oset:
                    co.append(other)
            da.co_occurrence[entity] = co

    def _add_abbreviation_scan(self, heading_names: list[str],
                                sentences: list[Sentence]) -> list[str]:
        """Deterministic abbreviation scan (extracted from _extract_section_headings)."""
        abbr_pattern = re.compile(
            r'(\b[A-Z][a-zA-Z]+(?:\s+[A-Z]?[a-zA-Z]+){1,4})\s+\(([A-Z][A-Za-z0-9]{1,8})\)')
        for sent in sentences:
            for m in abbr_pattern.finditer(sent.text):
                full_name = m.group(1)
                abbr = m.group(2)
                abbr_used = any(
                    re.search(rf'\b{re.escape(abbr)}\b', other.text)
                    for other in sentences if other.number != sent.number
                )
                if abbr_used and abbr not in heading_names:
                    heading_names.append(abbr)
                    print(f"    Abbreviation defined: {full_name} ({abbr})")
        return heading_names

    # ═════════════════════════════════════════════════════════════════════
    # CNR-0: Section Heading Extraction
    # ═════════════════════════════════════════════════════════════════════

    def _extract_section_headings(self, sentences: list[Sentence]) -> list[str]:
        """Extract component name signals from short heading-like sentences.

        Uses a simple structural filter (short sentences = candidate headings),
        then asks the LLM to identify which ones name software components.
        """
        # Step 1: Structural filter — collect short sentences (likely headings)
        candidates = []
        for sent in sentences:
            text = sent.text.strip()
            words = text.rstrip(".").split()
            # Short sentence: 1-6 words, < 60 chars
            if 1 <= len(words) <= 6 and len(text) < 60:
                candidates.append(sent)

        if not candidates:
            return []

        # Step 2: Ask LLM which short sentences are component/subsystem headings
        heading_block = "\n".join(
            f"  S{s.number}: {s.text}" for s in candidates
        )
        prompt = f"""These SHORT SENTENCES were extracted from a software architecture document.
Some are SECTION HEADINGS that name a software component or subsystem.
Others are just short phrases, labels, or non-component headings.

SHORT SENTENCES:
{heading_block}

Which of these sentences are headings that NAME a software component, subsystem, or
a specific named process/function of the architecture?

For each one that names a component or architectural function, extract the name
(strip generic suffixes like "flow", "overview", "architecture", "diagram").

EXAMPLE:
  "S10: Scheduler module." → component: "Scheduler"
  "S20: General overview." → not a component (too generic)
  "S30: Cache layer." → component: "Cache layer"
  "S35: Payment processing flow." → component: "Payment processing" (specific named function)
  "S40: See below." → not a component

Return ONLY valid JSON:
{{"headings": [{{"s": <sentence_number>, "component": "<extracted name>"}}]}}

Include headings that name specific architectural components OR specific named processes/functions
of the system. Exclude purely generic labels.
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
        heading_names = []
        if data and "headings" in data:
            for item in data["headings"]:
                name = item.get("component", "").strip()
                if name and len(name) >= 2:
                    heading_names.append(name)

        # Step 3: Scan for parenthetical abbreviation definitions
        # Pattern: "Full Name (ABBR)" where ABBR is uppercase and reused elsewhere
        abbr_pattern = re.compile(r'(\b[A-Z][a-zA-Z]+(?:\s+[A-Z]?[a-zA-Z]+){1,4})\s+\(([A-Z][A-Za-z0-9]{1,8})\)')
        for sent in sentences:
            for m in abbr_pattern.finditer(sent.text):
                full_name = m.group(1)
                abbr = m.group(2)
                abbr_used = any(
                    re.search(rf'\b{re.escape(abbr)}\b', other.text)
                    for other in sentences if other.number != sent.number
                )
                if abbr_used and abbr not in heading_names:
                    heading_names.append(abbr)
                    print(f"    Abbreviation defined: {full_name} ({abbr})")

        return heading_names

    # ═════════════════════════════════════════════════════════════════════
    # CNR-1: Broad Extraction
    # ═════════════════════════════════════════════════════════════════════

    def _extract_pass_broad(self, sentences: list[Sentence], heading_names: list[str] = None,
                            da_entity_hints: list[str] = None) -> list[str]:
        """Recall-focused extraction of candidate component names."""
        all_names = set()
        if heading_names:
            all_names.update(heading_names)
        batches = self._make_batches(sentences)

        # Build DA entity hint block if available
        da_hint_block = ""
        if da_entity_hints:
            da_hint_block = f"""
PRELIMINARY ENTITIES (from document scan — not all are components):
  {', '.join(da_entity_hints[:40])}

Confirm which of these are architecture components and find any others.
"""

        for i, batch in enumerate(batches):
            doc_block = "\n".join(f"S{s.number}: {s.text}" for s in batch)
            prompt = f"""You are a software architecture expert. Read this document and extract ALL names
of software architecture components mentioned in it.

A software architecture component is a named software module, service, subsystem, package,
layer, or framework that has defined responsibilities within the system. Components:
- Are often capitalized as proper nouns (e.g., "the Scheduler handles job queuing")
- Have defined responsibilities ("manages", "handles", "provides", "processes", "is responsible for")
- Participate in architectural relationships ("communicates with", "sends to", "depends on")
- May be introduced by SHORT SECTION HEADINGS (1-3 word sentences that name a subsystem)
- May be lowercase package/layer names described as having distinct roles in the architecture

EXAMPLE (from a compiler system):
  Document mentions: "The Lexer tokenizes input. The Parser builds an AST from tokens.
  The CodeGenerator emits bytecode."
  Components: ["Lexer", "Parser", "CodeGenerator"]

EXAMPLE (from an e-commerce system):
  Document mentions: "The OrderService validates orders before passing them to the
  PaymentGateway. The InventoryManager tracks stock levels."
  Components: ["OrderService", "PaymentGateway", "InventoryManager"]

EXAMPLE (from a layered architecture):
  Document mentions: "The core handles data structures. The engine is responsible for
  processing, and the ui provides the user interface. The config stores settings."
  Components: ["core", "engine", "ui", "config"]
{da_hint_block}
DOCUMENT:
{doc_block}

Extract ALL software component names from the document. Include names that appear as:
- Standalone capitalized terms used as proper nouns
- CamelCase identifiers
- Hyphenated identifiers (e.g., "api-gw")
- Acronyms/abbreviations used as component names (e.g., "API Gateway")
- Short heading sentences that introduce a subsystem (e.g., "S42: Scheduler module." → "Scheduler")
- Names defined by parenthetical abbreviations (e.g., "Distributed Task Queue (DTQ)" → both "DTQ" and "Distributed Task Queue")
- Lowercase package or layer names that have defined architectural roles (e.g., "the core depends only on...")

Do NOT include:
- Generic concepts used as common nouns WITHOUT defined responsibilities
- External tools or technologies used by the system ("uses X for conversion") — these are NOT components of the system itself
- File names, paths, or URLs
- Data objects or message types (unless they are named components)

Return ONLY valid JSON:
{{"components": ["Name1", "Name2", ...]}}
JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=240))
            if data and "components" in data:
                names = data["components"]
                all_names.update(names)
                if len(batches) > 1:
                    print(f"    Batch {i+1}/{len(batches)}: +{len(names)} (total unique: {len(all_names)})")

        return sorted(all_names)

    # ═════════════════════════════════════════════════════════════════════
    # CNR-2: Actor Extraction
    # ═════════════════════════════════════════════════════════════════════

    def _extract_pass_actor(self, sentences: list[Sentence], heading_names: list[str] = None,
                            da_entity_hints: list[str] = None) -> list[str]:
        """Independent pass: what named software components perform actions?"""
        all_names = set()
        if heading_names:
            all_names.update(heading_names)
        batches = self._make_batches(sentences)

        # Build DA entity hint block if available
        da_hint_block = ""
        if da_entity_hints:
            da_hint_block = f"""
PRELIMINARY ENTITIES (from document scan — not all are components):
  {', '.join(da_entity_hints[:40])}

Confirm which of these act as subjects/actors and find any others.
"""

        for i, batch in enumerate(batches):
            doc_block = "\n".join(f"S{s.number}: {s.text}" for s in batch)
            prompt = f"""You are analyzing a software architecture document. Your task is to identify
all NAMED SOFTWARE COMPONENTS that PERFORM ACTIONS or HAVE RESPONSIBILITIES in this text.

For each sentence, ask: "What named software entity is the actor or subject here?"
Also check for short section headings (1-3 word sentences) that introduce subsystems.

EXAMPLE (from an OS textbook):
  "The Scheduler assigns CPU time to processes. The MemoryManager handles page allocation.
  When a process requests memory, the MemoryManager allocates a free page."
  Active components: ["Scheduler", "MemoryManager"]

EXAMPLE (from a web application):
  "The AuthService validates user credentials. After authentication, the SessionManager
  creates a new session. The Router directs requests to the appropriate handler."
  Active components: ["AuthService", "SessionManager", "Router"]
{da_hint_block}
DOCUMENT:
{doc_block}

List ALL named software components that act as subjects/actors in the document.
Only include components that:
1. Are referred to by name (capitalized, CamelCase, acronym, or lowercase package/layer name)
2. Perform at least one described action OR are described as having responsibilities
3. Are part of the software system being documented

Also include names that:
- Appear as section headings (short sentences naming a subsystem)
- Are defined via parenthetical abbreviation (e.g., "X Event Y Layer (XYL)" → "XYL")
- Are lowercase package or layer names with defined architectural roles

Do NOT include:
- Users, roles, or human actors
- Generic terms ("the system", "the application", "the component") when used generically
- External tools or technologies not part of the system itself

Return ONLY valid JSON:
{{"components": ["Name1", "Name2", ...]}}
JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=240))
            if data and "components" in data:
                names = data["components"]
                all_names.update(names)
                if len(batches) > 1:
                    print(f"    Batch {i+1}/{len(batches)}: +{len(names)} (total unique: {len(all_names)})")

        return sorted(all_names)

    # ═════════════════════════════════════════════════════════════════════
    # CNR-S: Candidate Statistics
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_candidate_stats(candidates: list[str],
                                  sentences: list[Sentence]) -> dict[str, dict]:
        """Compute document-level statistics for each candidate name.

        Returns dict mapping name → {
            "mention_count": int,        # total sentences mentioning it
            "spread": float,             # spread across document (0-1)
            "has_caps": bool,            # appears capitalized (not just sentence-initial)
            "near_heading": bool,        # appears near short sentences (headings)
            "co_occurs_with": int,       # count of other candidates appearing in same sentences
            "is_subject": bool,          # appears as sentence subject (first noun)
        }
        """
        total_sents = len(sentences)
        candidate_set = set(candidates)
        stats = {}

        for name in candidates:
            pattern = re.compile(rf'\b{re.escape(name)}\b', re.IGNORECASE)

            # Basic mention stats
            mention_sents = []
            has_caps = False
            is_subject = False
            for sent in sentences:
                m = pattern.search(sent.text)
                if m:
                    mention_sents.append(sent.number)
                    # Capitalized not just at sentence start
                    if m.start() > 0 and sent.text[m.start()].isupper():
                        has_caps = True
                    # Sentence-initial with original capitalization
                    if sent.text.startswith(name):
                        has_caps = True
                        is_subject = True

            mention_count = len(mention_sents)

            # Spread: how evenly distributed across the document
            if mention_count >= 2 and total_sents > 1:
                positions = [s / total_sents for s in mention_sents]
                spread = max(positions) - min(positions)
            else:
                spread = 0.0

            # Near heading: any mention within 3 sentences of a short sentence (≤4 words)
            short_sent_nums = {s.number for s in sentences if len(s.text.split()) <= 4}
            near_heading = any(
                any(abs(ms - hs) <= 3 for hs in short_sent_nums)
                for ms in mention_sents
            )

            # Co-occurrence: how many other candidates appear in the same sentences
            co_occurs = set()
            for other in candidates:
                if other == name:
                    continue
                other_pat = re.compile(rf'\b{re.escape(other)}\b', re.IGNORECASE)
                for sent in sentences:
                    if sent.number in mention_sents and other_pat.search(sent.text):
                        co_occurs.add(other)
                        break

            stats[name] = {
                "mention_count": mention_count,
                "spread": round(spread, 2),
                "has_caps": has_caps,
                "near_heading": near_heading,
                "co_occurs_with": len(co_occurs),
                "is_subject": is_subject,
            }

        return stats

    # ═════════════════════════════════════════════════════════════════════
    # CNR-3: Judge Filter
    # ═════════════════════════════════════════════════════════════════════

    def _judge_filter(self, all_candidates: list[str], pass1: list[str],
                      pass2: list[str], sentences: list[Sentence],
                      heading_names: list[str] = None,
                      candidate_stats: dict[str, dict] = None,
                      da_synonym_groups: list[list[str]] = None) -> list[str]:
        """Review candidates using voting + statistics + LLM judge."""
        set1 = set(pass1)
        set2 = set(pass2)
        heading_set = set(heading_names) if heading_names else set()
        intersection = set1 & set2
        single_pass_only = (set1 | set2) - intersection - heading_set
        stats = candidate_stats or {}

        # Auto-accept intersection candidates AND heading-derived names
        accepted = set(intersection) | heading_set
        print(f"    Intersection (auto-accept): {len(intersection)}")
        if heading_set:
            print(f"    Heading-derived (auto-accept): {len(heading_set)}")

        if not single_pass_only:
            return sorted(accepted)

        # Build synonym group lookup for DA-informed judging
        da_synonym_map: dict[str, list[str]] = {}  # name → group members
        if da_synonym_groups:
            for group in da_synonym_groups:
                for member in group:
                    da_synonym_map[member] = [m for m in group if m != member]

        # Statistics-based pre-filter for single-pass candidates
        # Minimum bar: at least 2 mentions (capitalization is a signal, not a requirement)
        judgeable = []
        pre_filtered = []
        for name in sorted(single_pass_only):
            s = stats.get(name, {})
            mentions = s.get("mention_count", 0)

            if mentions >= 2:
                judgeable.append(name)
            else:
                pre_filtered.append(name)

        if pre_filtered:
            print(f"    Pre-filtered out ({len(pre_filtered)}): "
                  f"{', '.join(pre_filtered[:5])}{'...' if len(pre_filtered) > 5 else ''}")

        if not judgeable:
            return sorted(accepted)

        # Build stats summary for LLM judge
        candidate_lines = []
        for name in judgeable:
            s = stats.get(name, {})
            # Check if this candidate is in a synonym group with a confirmed component
            in_syn_group = "no"
            if da_synonym_map and name in da_synonym_map:
                group_members = da_synonym_map[name]
                confirmed_peers = [m for m in group_members if m in accepted]
                if confirmed_peers:
                    in_syn_group = f"yes (synonym of {confirmed_peers[0]})"
            stat_str = (f"mentions={s.get('mention_count', '?')}, "
                        f"spread={s.get('spread', '?')}, "
                        f"co-occurs={s.get('co_occurs_with', '?')}, "
                        f"subject={s.get('is_subject', '?')}, "
                        f"in_synonym_group={in_syn_group}")
            candidate_lines.append(f"  - {name}  [{stat_str}]")
        candidate_block = "\n".join(candidate_lines)
        accepted_list = ", ".join(sorted(accepted)[:20])

        prompt = f"""You are judging whether candidate names are SOFTWARE ARCHITECTURE COMPONENTS
in a software documentation text.

ALREADY CONFIRMED COMPONENTS (found by two independent passes):
{accepted_list if accepted_list else "(none yet)"}

CANDIDATES TO JUDGE (found by only one pass), with document statistics:
{candidate_block}

Statistics key:
- mentions: number of sentences mentioning this name
- spread: how distributed across the document (0=clustered, 1=full spread)
- co-occurs: number of other candidate names it appears alongside
- subject: whether it appears as sentence subject
- in_synonym_group: whether document analysis found this name in a synonym group with a confirmed component

For each candidate, decide ACCEPT or REJECT.

ACCEPT if the name:
- Refers to a specific software module, service, or subsystem
- Is used as a proper name (not a generic concept)
- Has strong statistics (high mentions, spread, co-occurrence with other components)
- Is in a synonym group with an already confirmed component (strong accept signal)

REJECT if the name:
- Is a generic concept used as a common noun
- Is a data object, not a component
- Is an EXTERNAL tool or technology the system uses, not a component OF the system
- Is a process, activity, protocol, or format name

Return ONLY valid JSON:
{{"accepted": ["Name1", "Name2", ...], "rejected": ["Name3", ...]}}
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=180))
        if data:
            judge_accepted = set(data.get("accepted", []))
            judge_rejected = set(data.get("rejected", []))
            accepted.update(judge_accepted & set(judgeable))
            print(f"    Judge accepted: {len(judge_accepted & set(judgeable))}, "
                  f"rejected: {len(judge_rejected & set(judgeable))}")
        else:
            accepted.update(judgeable)
            print(f"    Judge failed, accepting all {len(judgeable)} deterministic survivors")

        return sorted(accepted)

    # ═════════════════════════════════════════════════════════════════════
    # CNR-4: Deduplication & Alias Grouping
    # ═════════════════════════════════════════════════════════════════════

    def _deduplicate_names(self, names: list[str], sentences: list[Sentence],
                           da_synonym_groups: list[list[str]] = None) -> list[str]:
        """Group synonyms/aliases and pick canonical names."""
        if len(names) <= 1:
            self._discovered_aliases = {}
            return names

        # Deterministic dedup first: CamelCase normalization
        # "TaskScheduler" and "Task Scheduler" → same
        normalized_groups: dict[str, list[str]] = {}
        for name in names:
            # Normalize: CamelCase split, lowercase, sort
            norm = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
            norm = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', norm)
            norm_key = norm.lower().replace("-", " ").replace("_", " ")
            normalized_groups.setdefault(norm_key, []).append(name)

        # If all names have unique normalizations, no LLM needed for grouping
        has_duplicates = any(len(v) > 1 for v in normalized_groups.values())

        if not has_duplicates and len(names) <= 3:
            self._discovered_aliases = {}
            return names

        # Build DA synonym hint block if available
        da_syn_block = ""
        if da_synonym_groups:
            # Filter to groups where at least one member is in our names
            name_set = set(names)
            relevant_groups = [g for g in da_synonym_groups
                               if len(set(g) & name_set) >= 2]
            if relevant_groups:
                group_strs = [" = ".join(g) for g in relevant_groups]
                da_syn_block = f"""
PRELIMINARY SYNONYM GROUPS (from document scan):
{chr(10).join(f'  - {gs}' for gs in group_strs)}

Use these as starting points. Verify and refine.
"""

        # LLM dedup for remaining potential aliases
        name_list = "\n".join(f"  {i+1}. {name}" for i, name in enumerate(names))
        prompt = f"""These names were extracted from a software architecture document as potential
component names. Some may be aliases or alternative forms for the same component.

NAMES:
{name_list}
{da_syn_block}
Group names that refer to the SAME component. For each group, choose the most descriptive
form as canonical name.

Rules for MERGING (same component):
- "api-gw" and "API Gateway" → same component, canonical: "API Gateway"
- "FM" and "FileManager" → same, canonical: "FileManager"
- CamelCase and space-separated forms of the same words are the same
- An abbreviation and its EXACT expansion are the same

Rules for KEEPING SEPARATE (different components):
- Names that have different responsibilities in the system are SEPARATE even if they share words
- If two names are explicitly contrasted or decoupled in an architecture (e.g., one stores files,
  another stores metadata), they are DIFFERENT components
- A name that is a SUBSET of another (e.g., "Logger" vs "FileLogger") should only merge
  if they truly refer to the same thing — when in doubt, keep separate
- Prefer keeping names separate over incorrectly merging distinct components

Return ONLY valid JSON:
{{"groups": [{{"canonical": "BestName", "aliases": ["alt1", "alt2"]}}]}}

If a name has no aliases, list it with an empty aliases array.
JSON only:"""

        data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
        if data and "groups" in data:
            canonical_names = []
            self._discovered_aliases = {}
            seen = set()
            for group in data["groups"]:
                canonical = group.get("canonical", "")
                aliases = group.get("aliases", [])
                if canonical and canonical not in seen:
                    # Verify canonical is one of our discovered names (or an alias is)
                    all_group = {canonical} | set(aliases)
                    if all_group & set(names):
                        canonical_names.append(canonical)
                        if aliases:
                            self._discovered_aliases[canonical] = aliases
                        seen.add(canonical)
                        seen.update(aliases)

            # Add any names not in any group
            for name in names:
                if name not in seen:
                    canonical_names.append(name)

            return canonical_names
        else:
            # Fallback: deterministic dedup only
            canonical_names = []
            self._discovered_aliases = {}
            for norm_key, group in normalized_groups.items():
                # Pick the longest / CamelCase form as canonical
                canonical = max(group, key=lambda n: (len(n), n))
                canonical_names.append(canonical)
                aliases = [n for n in group if n != canonical]
                if aliases:
                    self._discovered_aliases[canonical] = aliases
            return canonical_names

    # ═════════════════════════════════════════════════════════════════════
    # Synthetic Model Construction
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_synthetic_components(names: list[str]) -> list[ArchitectureComponent]:
        """Build synthetic ArchitectureComponent objects from discovered names."""
        return [
            ArchitectureComponent(id=f"cnr_{i}", name=n, entity_name=n)
            for i, n in enumerate(sorted(names))
        ]

    # ═════════════════════════════════════════════════════════════════════
    # Simple Extraction
    # ═════════════════════════════════════════════════════════════════════

    def _simple_extract(self, sentences: list[Sentence],
                        components: list[ArchitectureComponent],
                        name_to_id: dict[str, str]) -> list[SadSamLink]:
        """Simple LLM extraction — one pass over batched sentences."""
        comp_block = "\n".join(f"  {i+1}. {c.name}" for i, c in enumerate(components))

        # Build alias lookup from discovered aliases
        alias_info = []
        for canonical, aliases in self._discovered_aliases.items():
            for a in aliases:
                alias_info.append(f"{a} = {canonical}")
        alias_block = f"\nKNOWN ALIASES:\n" + "\n".join(f"  {a}" for a in alias_info) if alias_info else ""

        all_links: dict[tuple[int, str], SadSamLink] = {}
        batches = self._make_batches(sentences)

        for i, batch in enumerate(batches):
            doc_block = "\n".join(f"S{s.number}: {s.text}" for s in batch)

            prompt = f"""You are a software architecture traceability expert.

ARCHITECTURE COMPONENTS:
{comp_block}
{alias_block}

DOCUMENT:
{doc_block}

TASK: For each sentence, identify which architecture components are EXPLICITLY mentioned or referenced.

A valid reference is:
- Exact name: the component name appears verbatim in the sentence
- Synonym/alias: a known alternative name for the component
- Partial name: a distinctive sub-phrase that unambiguously identifies the component

NOT a valid reference:
- A component name inside a dotted path (e.g., "renderer.utils.config")
- A generic English word used in its ordinary sense
- A sentence that merely describes related functionality without naming the component

Return ONLY valid JSON:
{{"links": [{{"s": <sentence_number>, "c": "<ComponentName>", "text": "<matched text>"}}]}}

Precision is critical — only include links with clear textual evidence.
JSON only:"""

            data = self.llm.extract_json(self.llm.query(prompt, timeout=300))
            if not data or "links" not in data:
                if len(batches) > 1:
                    print(f"    Batch {i+1}/{len(batches)}: parse error")
                continue

            batch_count = 0
            for item in data["links"]:
                snum = item.get("s")
                cname = item.get("c", "")
                if snum is None or not cname:
                    continue

                # Resolve component name (exact or case-insensitive)
                cid = name_to_id.get(cname)
                if not cid:
                    for name, nid in name_to_id.items():
                        if name.lower() == cname.lower():
                            cid, cname = nid, name
                            break
                if not cid:
                    # Check aliases
                    for canonical, aliases in self._discovered_aliases.items():
                        if cname in aliases or cname.lower() in [a.lower() for a in aliases]:
                            cid = name_to_id.get(canonical)
                            cname = canonical
                            break
                if not cid:
                    continue

                key = (int(snum), cid)
                if key not in all_links:
                    all_links[key] = SadSamLink(
                        sentence_number=int(snum),
                        component_id=cid,
                        component_name=cname,
                        confidence=0.85,
                        source="cnr",
                    )
                    batch_count += 1

            if len(batches) > 1:
                print(f"    Batch {i+1}/{len(batches)}: +{batch_count} (total: {len(all_links)})")

        return list(all_links.values())

    # ═════════════════════════════════════════════════════════════════════
    # Eval Bridge
    # ═════════════════════════════════════════════════════════════════════

    def _build_eval_bridge(self, discovered: list[ArchitectureComponent],
                           pcm_components: list[ArchitectureComponent]) -> dict[str, str]:
        """Map cnr_id → pcm_id for evaluation. Four-tier matching."""
        bridge: dict[str, str] = {}
        pcm_by_name = {c.name: c.id for c in pcm_components}
        unmatched_cnr = []

        for comp in discovered:
            matched_pcm_id = None

            # Tier 1: Exact match (case-insensitive)
            for pcm_name, pcm_id in pcm_by_name.items():
                if comp.name.lower() == pcm_name.lower():
                    matched_pcm_id = pcm_id
                    print(f"    Bridge [exact]: {comp.name} → {pcm_name} ({pcm_id})")
                    break

            # Tier 2: Normalized match (strip hyphens/spaces, CamelCase-split, compare word sets)
            if not matched_pcm_id:
                cnr_words = self._normalize_name_words(comp.name)
                for pcm_name, pcm_id in pcm_by_name.items():
                    pcm_words = self._normalize_name_words(pcm_name)
                    if cnr_words and pcm_words and cnr_words == pcm_words:
                        matched_pcm_id = pcm_id
                        print(f"    Bridge [normalized]: {comp.name} → {pcm_name} ({pcm_id})")
                        break

            # Tier 2b: Check aliases
            if not matched_pcm_id:
                aliases = self._discovered_aliases.get(comp.name, [])
                for alias in aliases:
                    for pcm_name, pcm_id in pcm_by_name.items():
                        if alias.lower() == pcm_name.lower():
                            matched_pcm_id = pcm_id
                            print(f"    Bridge [alias]: {comp.name} (alias: {alias}) → {pcm_name} ({pcm_id})")
                            break
                        alias_words = self._normalize_name_words(alias)
                        pcm_words = self._normalize_name_words(pcm_name)
                        if alias_words and pcm_words and alias_words == pcm_words:
                            matched_pcm_id = pcm_id
                            print(f"    Bridge [alias-norm]: {comp.name} (alias: {alias}) → {pcm_name} ({pcm_id})")
                            break
                    if matched_pcm_id:
                        break

            # Tier 3: Fuzzy match (rapidfuzz)
            if not matched_pcm_id:
                try:
                    from rapidfuzz import fuzz
                    already_matched = set(bridge.values())
                    best_score, best_pcm_name, best_pcm_id = 0, None, None
                    for pcm_name, pcm_id in pcm_by_name.items():
                        if pcm_id in already_matched:
                            continue
                        score = fuzz.ratio(comp.name.lower(), pcm_name.lower())
                        if score > best_score:
                            best_score, best_pcm_name, best_pcm_id = score, pcm_name, pcm_id
                    if best_score >= 85:
                        matched_pcm_id = best_pcm_id
                        print(f"    Bridge [fuzzy {best_score:.0f}%]: {comp.name} → {best_pcm_name} ({best_pcm_id})")
                except ImportError:
                    pass

            if matched_pcm_id:
                bridge[comp.id] = matched_pcm_id
            else:
                unmatched_cnr.append(comp)

        # Tier 4: LLM-assisted matching for remaining unmatched
        if unmatched_cnr:
            already_matched_pcm = set(bridge.values())
            remaining_pcm = [n for n, pid in pcm_by_name.items() if pid not in already_matched_pcm]

            if remaining_pcm:
                cnr_list = "\n".join(f"  {c.name}" for c in unmatched_cnr)
                pcm_list = "\n".join(f"  {n}" for n in remaining_pcm)

                prompt = f"""Match discovered component names to known architecture model component names.

DISCOVERED NAMES (from document analysis):
{cnr_list}

KNOWN MODEL COMPONENTS (unmatched):
{pcm_list}

For each discovered name, find the matching model component. A match means they refer to
the same software component, even if named differently (e.g., "Web Frontend" = "frontend-app").

Return ONLY valid JSON:
{{"matches": [{{"discovered": "DiscoveredName", "model": "ModelComponentName"}}]}}

Only include confident matches. If a discovered name has no match, omit it.
JSON only:"""

                data = self.llm.extract_json(self.llm.query(prompt, timeout=120))
                if data and "matches" in data:
                    for match in data["matches"]:
                        disc = match.get("discovered", "")
                        model = match.get("model", "")
                        pcm_id = pcm_by_name.get(model)
                        if pcm_id:
                            # Find the cnr component
                            for c in unmatched_cnr:
                                if c.name == disc:
                                    bridge[c.id] = pcm_id
                                    print(f"    Bridge [LLM]: {disc} → {model} ({pcm_id})")
                                    break

        # Report unmatched
        matched_pcm = set(bridge.values())
        unmatched_pcm = [n for n, pid in pcm_by_name.items() if pid not in matched_pcm]
        if unmatched_pcm:
            print(f"    WARNING: {len(unmatched_pcm)} PCM components unmatched: {unmatched_pcm}")
        unmatched_disc = [c.name for c in discovered if c.id not in bridge]
        if unmatched_disc:
            print(f"    INFO: {len(unmatched_disc)} discovered names not in PCM: {unmatched_disc}")

        return bridge

    def _remap_links(self, links: list[SadSamLink]) -> list[SadSamLink]:
        """Remap links from cnr IDs to PCM IDs using eval bridge. Drop unmatched, dedup."""
        remapped: dict[tuple[int, str], SadSamLink] = {}
        for lk in links:
            pcm_id = self._eval_bridge.get(lk.component_id)
            if pcm_id:
                key = (lk.sentence_number, pcm_id)
                if key not in remapped:
                    remapped[key] = SadSamLink(
                        sentence_number=lk.sentence_number,
                        component_id=pcm_id,
                        component_name=lk.component_name,
                        confidence=lk.confidence,
                        source=lk.source,
                    )
        return list(remapped.values())

    # ═════════════════════════════════════════════════════════════════════
    # Helpers
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def _normalize_name_words(name: str) -> set[str]:
        """Normalize a name to a set of lowercase words for comparison."""
        # CamelCase split
        split = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        split = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', split)
        # Replace hyphens/underscores with spaces
        split = split.replace("-", " ").replace("_", " ")
        words = {w.lower() for w in split.split() if len(w) >= 2}
        return words

    @staticmethod
    def _make_batches(sentences: list[Sentence]) -> list[list[Sentence]]:
        """Split sentences into overlapping batches."""
        if len(sentences) <= BATCH_SIZE:
            return [sentences]
        batches, start = [], 0
        overlap = 5
        while start < len(sentences):
            end = min(start + BATCH_SIZE, len(sentences))
            batches.append(sentences[start:end])
            if end >= len(sentences):
                break
            start = end - overlap
        return batches
