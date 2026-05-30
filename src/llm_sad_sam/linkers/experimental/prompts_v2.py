"""Prompt constants for the S-Linker pipeline.

Clean version: only constants actually used by the current pipeline.
All examples use safe SE textbook domains (compiler, OS, e-commerce, graphics).
Zero benchmark-derived terms. See BENCHMARK_TABOO.md for rules.

Constants are ordered by pipeline tier (1 -> 2 -> 3).
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — Model Analysis: Component Ambiguity Classification
# ═══════════════════════════════════════════════════════════════════════════════

AMBIGUITY_FEW_SHOT = """
EXAMPLE 1:
NAMES: Lexer, Parser, CodeGenerator, Optimizer, Core, Util, AST, SymbolTable, Base
→ architectural: ["Lexer", "Parser", "CodeGenerator", "Optimizer", "AST", "SymbolTable"]
→ ambiguous: ["Core", "Util", "Base"]
Reasoning: Lexer/Parser/Optimizer name specific compilation roles. Core/Util/Base are
organizational labels that tell you nothing about what the component does.

EXAMPLE 2:
NAMES: Scheduler, Dispatcher, MemoryManager, Monitor, Pool, Helper, ProcessTable
→ architectural: ["Scheduler", "Dispatcher", "MemoryManager", "ProcessTable"]
→ ambiguous: ["Monitor", "Pool", "Helper"]
Reasoning: Scheduler/Dispatcher name specific OS roles. Monitor and Pool are ordinary
English words regularly used generically ("monitor performance", "thread pool").
Helper is an organizational label.

EXAMPLE 3:
NAMES: RenderEngine, SceneGraph, Pipeline, Broker, Proxy, Multiplexer, Router
→ architectural: ["RenderEngine", "SceneGraph", "Multiplexer", "Router"]
→ ambiguous: ["Pipeline", "Broker", "Proxy"]
Reasoning: RenderEngine/SceneGraph are CamelCase compounds — always architectural.
Multiplexer/Router name specific networking roles. Pipeline/Broker/Proxy are ordinary words
used generically in documentation ("processing pipeline", "message broker", "behind a proxy").

EXAMPLE 4:
NAMES: PaymentGateway, InvoiceHandler, Connector, Controller, Wrapper, Worker, Agent
→ architectural: ["PaymentGateway", "InvoiceHandler", "Worker"]
→ ambiguous: ["Connector", "Controller", "Wrapper", "Agent"]
Reasoning: PaymentGateway/InvoiceHandler are CamelCase compounds naming specific roles.
Worker names a specific concurrency mechanism. But Connector/Controller/Wrapper/Agent
seem functional yet are GENERIC categories writers use without referring to any specific
component: "a network connector", "the main controller", "a data wrapper", "a
background agent". They describe WHAT KIND of thing it is, not WHICH specific mechanism
— so they are ambiguous.""".strip()


AMBIGUITY_RULES = """RULES:
1. ARCHITECTURAL: Names that refer to a specific role or responsibility. If the name tells you
   WHAT the component does (scheduling, parsing, rendering, storing data, managing users), it is
   architectural — even if the word also exists in a dictionary.
   Multi-word names, CamelCase compounds, and abbreviations (API, TCP, RPC) → always architectural.

2. AMBIGUOUS: Single words that writers regularly use generically in software documentation.
   This includes TWO categories:
   Category A — Organizational labels: core, util, base, helper (tell you nothing about function)
   Category B — Generic functional categories: connector, controller, wrapper, agent
   (describe WHAT KIND of thing, not WHICH specific mechanism)
   The test: "Could a technical writer naturally write this word in a sentence about ANY system
   without referring to a specific component?" If yes → ambiguous.
   Key: Scheduler/Router describe HOW (specific mechanism) → ARCHITECTURAL.
         Connector/Controller/Wrapper describe WHAT KIND (generic category) → AMBIGUOUS."""


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — Document Knowledge: Alias Discovery & Judging
# ═══════════════════════════════════════════════════════════════════════════════

DOC_KNOWLEDGE_EXTRACTION_RULES = """WHAT TO FIND:
1. ABBREVIATIONS: Short forms explicitly introduced in the document.
   Rule: The abbreviation must be defined in the text, e.g., "Full Name (FN)" introduces FN.
   Like "Abstract Syntax Tree (AST)" defines AST — look for the same parenthetical pattern.

2. SYNONYMS: Alternative names that SPECIFICALLY refer to one component.
   Rule: The alternative name must unambiguously identify exactly ONE component.
   This includes:
   - Proper names, role titles, or technical aliases used interchangeably with the component
   - Trailing words of multi-word component names when used alone to mean the full name
     (e.g., "Dispatcher" used alone to mean "TaskDispatcher")
   APPROVE: Only if the alternative name is unambiguous — it clearly means one component
   REJECT: Generic descriptions that could apply to anything (like "the system" or "the process"),
   or ordinary words with plain English meanings beyond the component"""


DOC_KNOWLEDGE_JUDGE_EXAMPLES = """EXAMPLES — study these to calibrate your judgment:

Example 1 — APPROVE (abbreviation from component name):
  'AST' -> AbstractSyntaxTree (abbrev)
  Verdict: APPROVE. "AST" is the initials of "AbstractSyntaxTree". Abbreviations
  formed from the component name's words are always valid.

Example 2 — APPROVE (trailing word of multi-word name):
  'Dispatcher' -> TaskDispatcher (synonym)
  Verdict: APPROVE. "Dispatcher" is the last word of "TaskDispatcher".
  If no other component ends in "Dispatcher", this synonym is unambiguous.

Example 3 — APPROVE (CamelCase identifier):
  'RenderEngine' -> GameRenderEngine (synonym)
  Verdict: APPROVE. CamelCase is a constructed identifier — always a proper name.

Example 4 — APPROVE (trailing word of multi-word name):
  'Table' -> SymbolTable (synonym)
  Verdict: APPROVE. "Table" is the trailing word of "SymbolTable" and
  likely refers to this specific component when no other component uses "Table".

Example 5 — APPROVE (multi-word descriptive phrase):
  'query execution layer' -> IndexManager (synonym)
  Verdict: APPROVE. A multi-word descriptive phrase that consistently refers
  to a specific component is a valid synonym, even when the phrase words
  differ from the component name.

Example 6 — REJECT (ordinary English verb/noun):
  'handle' -> InvoiceHandler (synonym)
  Verdict: REJECT. "handle" is an ordinary English verb used generically
  in many contexts ("handle requests", "the handler").

Example 7 — REJECT (refers to whole system):
  'system' -> PaymentSystem (synonym)
  Verdict: REJECT. "system" is too generic — it could refer to the overall system."""


DOC_KNOWLEDGE_JUDGE_RULES = """DECISION RULES (apply in order):

1. AUTO-APPROVE these — they are always valid mappings:
   - Abbreviations formed from the component name's initials or words
   - Trailing words of multi-word component names (if no other component shares that word)
   - CamelCase identifiers
   - Multi-word phrases that contain the component name

2. APPROVE if the term plausibly refers to exactly one component and is NOT
   a generic word like "system", "process", "utility", "component", "module".

3. REJECT only if the term is clearly generic and could refer to anything,
   or clearly refers to a different component or the system as a whole.

IMPORTANT: When in doubt, APPROVE. False approvals are filtered by later
pipeline stages; false rejections cause permanent recall loss."""


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy — Word Usage Classification (kept for pre-12c linkers)
# ═══════════════════════════════════════════════════════════════════════════════

WORD_USAGE_PROMPT = """WORD USAGE CLASSIFICATION

In this document, the word "{partial}" could be a short name for an architecture
component called "{comp_name}".

{calibration}Below are ALL sentences where "{partial}" appears WITHOUT the full name "{comp_name}".
Analyze how the word "{partial}" is used across these sentences:

{sent_block}

QUESTION: Is "{partial}" used as a standalone entity reference in ANY of these sentences?

Classify as NAME if the word appears as a standalone noun phrase referring to a specific
system entity in at least SOME sentences — even if other sentences use it generically.
Examples of entity reference: "the {partial_lower} connects to...", "sends data to the
{partial_lower}", "the {partial_lower} handles...", "on the {partial_lower}"

Classify as ORDINARY only if EVERY occurrence uses the word as part of a compound phrase,
modifier, or generic descriptor — never as a standalone entity.
Examples of purely ordinary: "{partial_lower} process", "automated {partial_lower}",
"{partial_lower} strategy", "{partial_lower}-based"

The threshold is: if even ONE sentence uses "{partial}" as a standalone entity reference,
classify as NAME. Only classify as ORDINARY when you see ZERO standalone entity uses.

Return JSON: {{"classification": "name" or "ordinary", "reason": "brief explanation"}}
JSON only:"""


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Entity Extraction & Validation
# ═══════════════════════════════════════════════════════════════════════════════

ENTITY_EXTRACTION_RULES = """RULES — include a reference when:
1. The component name (or known alias) appears directly in the sentence
2. A space-separated form matches a compound name (e.g., "Memory Manager" → MemoryManager)
3. The sentence describes what a specific component does by name or role
4. A known synonym is used
5. The component participates in an interaction described in the sentence (as sender, receiver, or target) — e.g., "X sends data to Y" references BOTH X and Y
6. The component is mentioned in a passive or prepositional phrase — e.g., "data is stored in X", "handled by X", "via X", "through X"

RULES — exclude when:
1. The name appears only inside a dotted path (e.g., com.example.name)
2. The name is used as an ordinary English word, not as a component reference

Favor inclusion over exclusion — later verification will filter borderline cases."""


VALIDATION_RULES = """DECISION RULES:
APPROVE when:
- The component is named as an architectural participant: it performs an operation, provides or receives a service, is being configured, or is explicitly introduced as part of the system
- A section heading names the component as its subject
- The sentence describes the component's responsibilities, behavior, or interactions with other parts of the system

REJECT when:
- The name is used as an ordinary technical or English word, not as a reference to this specific component
  (e.g., "proxy" in "proxy pattern" refers to a design pattern concept, not a Proxy component)
- The name modifies a noun phrase without being a standalone architectural reference
  (e.g., "observer pattern", "pipeline stage" — the word describes a type, not the component)
- The sentence describes an algorithm, subprocess, or implementation technique that shares the component's name but is not the component itself"""


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Coreference Resolution
# ═══════════════════════════════════════════════════════════════════════════════

COREF_RULES = """For each case, determine if any pronoun in the TARGET sentence refers to a component.

RULES (all must hold):
1. The component name (or known alias) MUST appear verbatim in the antecedent sentence
2. The antecedent MUST be within the provided context window; prefer the nearest matching reference
3. The pronoun MUST grammatically refer back to that component as its subject
4. If the pronoun could refer to multiple things, DO NOT resolve it
5. Do NOT resolve pronouns about subprocesses or implementation details

Like in technical writing: "The Scheduler assigns tasks to threads. It uses a priority queue internally."
— "It" clearly refers to "the Scheduler" because it was the subject of the previous sentence."""


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — Standalone-Mention Detection (EXT-01)
# ═══════════════════════════════════════════════════════════════════════════════

STANDALONE_MENTION_RULES_PRE_FILTERED = """STANDALONE-MENTION DETECTION — for each sentence, answer YES if the sentence contains a standalone reference to the named component (the name appears as a subject, object, or named participant — not only as an ordinary English word).

RULES:
1. YES when the component name appears as a standalone token — as the subject of an architectural action, in a list of components, or named as a participant.
2. NO when the name is used only as an ordinary English word with its dictionary meaning, with no architectural intent.
3. YES when the name is configured, queried, or named as the target of an interaction (e.g., "data is stored in X", "via X", "through X").
4. When uncertain between a surface mention and a generic English use, favor YES — downstream validators filter generic uses.

Return JSON: {"results": [{"component": "Name", "sentence": N_INTEGER, "standalone": true}]}
JSON only:"""


STANDALONE_MENTION_RULES_LLM_ONLY = """STANDALONE-MENTION DETECTION — for each sentence, answer YES if the sentence makes a standalone reference to the named component; NO if the name appears only as part of a longer code identifier or as an ordinary English word.

RULES:
1. YES when the component name appears as a standalone token, including as a subject, object, or in a list of components.
   Example: "The Parser consumes tokens emitted by the lexer." -> YES for Parser.
2. NO when the name appears only inside a qualified or dotted identifier.
   Example: "The class compiler.parser.ASTBuilder extends the base class." -> NO for Parser; Parser is a path segment, not a standalone reference.
3. NO when the name participates only in a hyphenated compound that denotes a different entity.
   Example: "Parser-style grammar" -> NO for Parser.
4. YES when the name is the subject of an architectural action — performs work, provides a service, is configured, receives input.
   Example: "Disk I/O is handled by the FileSystem." -> YES for FileSystem.
5. When uncertain between a surface mention and a generic English use, favor YES — downstream validators filter generic uses.

Return JSON: {"results": [{"component": "Name", "sentence": N_INTEGER, "standalone": true}]}
JSON only:"""


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — Standalone-Mention Detection (EXT-01) — Alias-Aware (Plan 06-05)
# ═══════════════════════════════════════════════════════════════════════════════
#
# These four constants extend the Plan 06-01 STANDALONE_MENTION_RULES_PRE_FILTERED
# and STANDALONE_MENTION_RULES_LLM_ONLY constants with knowledge blocks injected
# at call time by the linker (Plan 06-06). The blocks are substituted via
# `prompt.replace("{KNOWN_ALIASES_BLOCK}", ...)` — NOT `.format(...)` — because
# the JSON template at the end uses literal braces. Knowledge is whatever the
# upstream doc_knowledge / seed stages discovered: possibly EMPTY on projects
# without aliases or with no prior links yet, per D-11 (CONTEXT.md).


STANDALONE_MENTION_RULES_PRE_FILTERED_ALIAS_AWARE = """STANDALONE-MENTION DETECTION — for each sentence, answer YES if the sentence contains a standalone reference to the named component (the name appears as a subject, object, or named participant — not only as an ordinary English word).

You also have a list of KNOWN ALIASES discovered earlier in this document. An alias surface form referring to the named component counts as a standalone mention of that component.

KNOWN ALIASES (term -> Component):
{KNOWN_ALIASES_BLOCK}

RULES:
1. YES when the component name appears as a standalone token — as the subject of an architectural action, in a list of components, or named as a participant.
2. YES when a KNOWN ALIAS for the component appears as a standalone token in the sentence (e.g., the abbreviation, alternate form, or short name listed above).
3. NO when the name is used only as an ordinary English word with its dictionary meaning, with no architectural intent.
4. YES when the name (or one of its aliases) is configured, queried, or named as the target of an interaction (e.g., "data is stored in X", "via X", "through X").
5. When uncertain between a surface mention and a generic English use, favor YES — downstream validators filter generic uses.

Return JSON: {"results": [{"component": "Name", "sentence": N_INTEGER, "standalone": true}]}
JSON only:"""


STANDALONE_MENTION_RULES_LLM_ONLY_ALIAS_AWARE = """STANDALONE-MENTION DETECTION — for each sentence, answer YES if the sentence makes a standalone reference to the named component; NO if the name appears only as part of a longer code identifier or as an ordinary English word.

You also have a list of KNOWN ALIASES discovered earlier in this document. An alias surface form referring to the named component counts as a standalone mention of that component.

KNOWN ALIASES (term -> Component):
{KNOWN_ALIASES_BLOCK}

RULES:
1. YES when the component name appears as a standalone token, including as a subject, object, or in a list of components.
   Example: "The Parser consumes tokens emitted by the lexer." -> YES for Parser.
2. YES when a KNOWN ALIAS for the component appears as a standalone token.
   Example: alias list contains `SymTbl -> SymbolTable`; sentence "SymTbl is consulted before scope resolution." -> YES for SymbolTable.
3. NO when the name appears only inside a qualified or dotted identifier.
   Example: "The class compiler.parser.ASTBuilder extends the base class." -> NO for Parser; Parser is a path segment, not a standalone reference.
4. NO when the name participates only in a hyphenated compound that denotes a different entity.
   Example: "Parser-style grammar" -> NO for Parser.
5. YES when the name (or an alias) is the subject of an architectural action — performs work, provides a service, is configured, receives input.
   Example: "Disk I/O is handled by the FileSystem." -> YES for FileSystem.
6. When uncertain between a surface mention and a generic English use, favor YES — downstream validators filter generic uses.

Return JSON: {"results": [{"component": "Name", "sentence": N_INTEGER, "standalone": true}]}
JSON only:"""


STANDALONE_MENTION_RULES_PRE_FILTERED_FULL_KNOWLEDGE = """STANDALONE-MENTION DETECTION — for each sentence, answer YES if the sentence contains a standalone reference to the named component (the name appears as a subject, object, or named participant — not only as an ordinary English word).

You have access to two pieces of context discovered earlier in this document:
1. KNOWN ALIASES — alternative surface forms for the named component.
2. RUNNING LINK MAP — sentences already attributed to a component by earlier passes. A new sentence that anaphorically continues an already-linked component's discussion (using a pronoun like "it" / "the component" / "the service") counts as a standalone reference to that component.

KNOWN ALIASES (term -> Component):
{KNOWN_ALIASES_BLOCK}

RUNNING LINK MAP (already-attributed pairs):
{RUNNING_LINK_MAP_BLOCK}

RULES:
1. YES when the component name appears as a standalone token — as the subject of an architectural action, in a list of components, or named as a participant.
2. YES when a KNOWN ALIAS for the component appears as a standalone token.
3. YES when the sentence uses a pronoun or definite reference ("it", "the component", "the service") AND the RUNNING LINK MAP shows the named component was just attributed to an adjacent earlier sentence (within +-3 sentences).
4. NO when the name is used only as an ordinary English word with its dictionary meaning, with no architectural intent.
5. YES when the name (or an alias) is configured, queried, or named as the target of an interaction (e.g., "data is stored in X", "via X", "through X").
6. When uncertain between a surface mention and a generic English use, favor YES — downstream validators filter generic uses.

Return JSON: {"results": [{"component": "Name", "sentence": N_INTEGER, "standalone": true}]}
JSON only:"""


STANDALONE_MENTION_RULES_LLM_ONLY_FULL_KNOWLEDGE = """STANDALONE-MENTION DETECTION — for each sentence, answer YES if the sentence makes a standalone reference to the named component; NO if the name appears only as part of a longer code identifier or as an ordinary English word.

You have access to two pieces of context discovered earlier in this document:
1. KNOWN ALIASES — alternative surface forms for the named component.
2. RUNNING LINK MAP — sentences already attributed to a component by earlier passes. A new sentence that anaphorically continues an already-linked component's discussion (using a pronoun like "it" / "the component" / "the service") counts as a standalone reference to that component.

KNOWN ALIASES (term -> Component):
{KNOWN_ALIASES_BLOCK}

RUNNING LINK MAP (already-attributed pairs):
{RUNNING_LINK_MAP_BLOCK}

RULES:
1. YES when the component name appears as a standalone token, including as a subject, object, or in a list of components.
   Example: "The Parser consumes tokens emitted by the lexer." -> YES for Parser.
2. YES when a KNOWN ALIAS for the component appears as a standalone token.
   Example: alias list contains `SymTbl -> SymbolTable`; sentence "SymTbl is consulted before scope resolution." -> YES for SymbolTable.
3. YES when the sentence uses a pronoun or definite reference ("it", "the component", "the service") AND the RUNNING LINK MAP shows the named component was attributed to an adjacent earlier sentence (within +-3 sentences).
   Example: linkmap shows `S12: Scheduler`; sentence "S13: It then assigns the task to an idle worker." -> YES for Scheduler.
4. NO when the name appears only inside a qualified or dotted identifier.
   Example: "The class compiler.parser.ASTBuilder extends the base class." -> NO for Parser; Parser is a path segment.
5. NO when the name participates only in a hyphenated compound that denotes a different entity.
   Example: "Parser-style grammar" -> NO for Parser.
6. YES when the name (or an alias) is the subject of an architectural action — performs work, provides a service, is configured, receives input.
   Example: "Disk I/O is handled by the FileSystem." -> YES for FileSystem.
7. When uncertain between a surface mention and a generic English use, favor YES — downstream validators filter generic uses.

Return JSON: {"results": [{"component": "Name", "sentence": N_INTEGER, "standalone": true}]}
JSON only:"""


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Seed Reference Disambiguation
# ═══════════════════════════════════════════════════════════════════════════════

SEED_DISAMBIGUATION_RULES = """REFERENCE DISAMBIGUATION — determine what the name means in each sentence.

COMPONENT (approve): The sentence discusses this architectural component —
it performs actions, provides services, is described, configured, listed,
or referenced by name in any grammatical role.

OTHER (reject): The name clearly carries a different meaning:
- Code-level notation: the name appears inside a package path, qualified
  identifier, or a sentence that enumerates code-level identifiers
- Technique or methodology: the sentence describes an algorithm, pattern,
  or approach that shares the component's name — not what the component
  does as an architectural participant
- Embedded sub-entity: the name appears only as part of a longer proper
  name that denotes a different, more specific entity
- Different entity: the sentence refers to a similarly-named but distinct
  thing (the name partially overlaps but the full reference is different)
- Generic English: the word is used with its ordinary dictionary meaning

When uncertain, choose COMPONENT — these candidates passed independent extraction."""
