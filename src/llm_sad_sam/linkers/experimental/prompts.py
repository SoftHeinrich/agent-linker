"""Shared prompt constants for S-Linker pipeline variants.

All prompt examples use safe SE textbook domains (compiler, OS, e-commerce, graphics).
Zero benchmark-derived terms. See BENCHMARK_TABOO.md for rules.
"""

CONVENTION_GUIDE = """### STEP 1 — Hierarchical name reference (not about the component itself)?

The most common reason for NO_LINK: the sentence mentions the component name only
as part of a HIERARCHICAL/QUALIFIED NAME (dotted path, namespace, module path) that
refers to a nested sub-unit, not the component's own architectural role.

Software documentation commonly uses hierarchical naming (e.g., "X.utils", "X/handlers",
"X::impl") to refer to parts inside a component. The component name appears only as
a prefix, not as the subject.

Recognize these patterns — all are NO_LINK for component X:
- "X.utils provides helper functions" — dotted sub-unit reference
- "X.handlers, X.mappers, X.adapters follow a pipeline" — listing sub-packages of X
- "Classes in the X.impl package are not exported" — even with
  architectural language, if the subject is X's sub-unit → NO_LINK
- Bare name mixed with qualified paths: "X, Y.adapters, Y.transformers follow
  a pipeline design" — treat ALL as hierarchical references → NO_LINK

KEY DISTINCTION: Sentences that describe what X DOES or HOW X INTERACTS with other
components are LINK, even if they mention implementation details (e.g., "X uses Y
technology for Z" → LINK for X, because it describes X's behavior).

EXCEPTION: If the sentence also explicitly names the target component AS A PROPER
NOUN with the word "component" (e.g., "for the X component") → LINK.

Cross-reference rule: A sub-unit sentence mentioning a DIFFERENT component
in an architectural role is LINK for that other component.

### STEP 2 — Component name confused with a different entity?

**2a. Technology / methodology confusion:**

CRITICAL RULE: If a component IS NAMED AFTER a technology (e.g., the architecture
has a component called "Kafka Broker" or "Nginx Proxy" or "Zookeeper"), then ANY sentence
describing that technology's capabilities, role, or behavior IS about the component → LINK.
This rule applies because architecture components are often named after the technology they wrap.

NO_LINK ONLY when:
- The sentence describes a technology that is NOT one of our components
- The name appears in a compound entity unrelated to the component
  ("X Protocol specification" → NO_LINK for "X" if X is not about that protocol)
- Uses the name as part of a METHODOLOGY ("X testing in CI" → NO_LINK for "X")

LINK when:
- The technology IS one of our architecture components (always LINK)
- Components INTERACT with or connect THROUGH the technology

**2b. Generic word collision:**
NO_LINK — narrow, non-architectural sense:
- Process/activity modifier: "throttle X", "batch X", "polling X"
- Hardware/deployment: "a physical rack-mounted appliance", "multi-socket machine"
- Possessive/personal: "her bookmarks", "their account"

LINK — system-level architectural sense:
- System name + word: "the [System] gateway"
- Architectural role: "the orchestrator routes jobs to the gateway"

### STEP 3 — Default: LINK.
If neither Step 1 nor Step 2 applies → LINK.

### IMPORTANT GUARDRAILS:
- Multi-word component names (e.g., "Kafka Broker", "Nginx Proxy") are NEVER generic words → LINK
- CamelCase identifiers are NEVER generic words → LINK
- Sentences describing how components interact, connect, or communicate → LINK for ALL components involved (not just the grammatical subject). "X connects to Y" is LINK for both X and Y.
- Sentences about what a component does, provides, or handles → LINK
- A component does NOT need to be the grammatical subject to be relevant. If a sentence says "X sends data to Y", both X and Y get LINK.
- Only use NO_LINK when you are CONFIDENT the name is NOT used as a component reference

### Priority:
Be AGGRESSIVE with NO_LINK on sub-package descriptions (Step 1).
For Step 2, only NO_LINK when confident. Default to LINK."""


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
NAMES: RenderEngine, SceneGraph, Pipeline, Layer, Proxy, Socket, Router
→ architectural: ["RenderEngine", "SceneGraph", "Socket", "Router"]
→ ambiguous: ["Pipeline", "Layer", "Proxy"]
Reasoning: RenderEngine/SceneGraph are CamelCase compounds — always architectural.
Socket/Router name specific networking roles. Pipeline/Layer/Proxy are ordinary words
used generically in documentation ("processing pipeline", "network layer", "behind a proxy").

EXAMPLE 4:
NAMES: PaymentGateway, OrderProcessor, Connector, Controller, Adapter, Worker, Agent
→ architectural: ["PaymentGateway", "OrderProcessor", "Worker"]
→ ambiguous: ["Connector", "Controller", "Adapter", "Agent"]
Reasoning: PaymentGateway/OrderProcessor are CamelCase compounds naming specific roles.
Worker names a specific concurrency mechanism. But Connector/Controller/Adapter/Agent
seem functional yet are GENERIC categories writers use without referring to any specific
component: "a database connector", "the main controller", "a protocol adapter", "a
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
   Category B — Generic functional categories: connector, controller, adapter, agent
   (describe WHAT KIND of thing, not WHICH specific mechanism)
   The test: "Could a technical writer naturally write this word in a sentence about ANY system
   without referring to a specific component?" If yes → ambiguous.
   Key: Scheduler/Router describe HOW (specific mechanism) → ARCHITECTURAL.
         Connector/Controller/Adapter describe WHAT KIND (generic category) → AMBIGUOUS."""


DOC_KNOWLEDGE_JUDGE_EXAMPLES = """EXAMPLES — study these to calibrate your judgment:

Example 1 — APPROVE (abbreviation from component name):
  'AST' -> AbstractSyntaxTree (abbrev)
  Verdict: APPROVE. "AST" is the initials of "AbstractSyntaxTree". Abbreviations
  formed from the component name's words are always valid.

Example 2 — APPROVE (trailing word of multi-word name):
  'Dispatcher' -> EventDispatcher (partial)
  Verdict: APPROVE. "Dispatcher" is the last word of "EventDispatcher".
  If no other component ends in "Dispatcher", this partial is unambiguous.

Example 3 — APPROVE (CamelCase identifier):
  'RenderEngine' -> GameRenderEngine (synonym)
  Verdict: APPROVE. CamelCase is a constructed identifier — always a proper name.

Example 4 — APPROVE (trailing word of multi-word name):
  'Table' -> SymbolTable (partial)
  Verdict: APPROVE. "Table" is the trailing word of "SymbolTable" and
  likely refers to this specific component when no other component uses "Table".

Example 5 — REJECT (ordinary English verb/noun):
  'process' -> OrderProcessor (partial)
  Verdict: REJECT. "process" is an ordinary English verb/noun used generically
  in many contexts ("process requests", "the process").

Example 6 — REJECT (refers to whole system):
  'system' -> PaymentSystem (partial)
  Verdict: REJECT. "system" is too generic — it could refer to the overall system."""


DOC_KNOWLEDGE_JUDGE_RULES = """DECISION RULES (apply in order):

1. AUTO-APPROVE these — they are always valid mappings:
   - Abbreviations formed from the component name's initials or words
   - Trailing words of multi-word component names (if no other component shares that word)
   - CamelCase identifiers
   - Multi-word phrases that contain the component name

2. APPROVE if the term plausibly refers to exactly one component and is NOT
   a generic word like "system", "process", "service", "component", "module".

3. REJECT only if the term is clearly generic and could refer to anything,
   or clearly refers to a different component or the system as a whole.

IMPORTANT: When in doubt, APPROVE. False approvals are filtered by later
pipeline stages; false rejections cause permanent recall loss."""


ENTITY_EXTRACTION_RULES = """RULES — include a reference when:
1. The component name (or known alias) appears directly in the sentence
2. A space-separated form matches a compound name (e.g., "Memory Manager" → MemoryManager)
3. The sentence describes what a specific component does by name or role
4. A known synonym or partial reference is used
5. The component participates in an interaction described in the sentence (as sender, receiver, or target) — e.g., "X sends data to Y" references BOTH X and Y
6. The component is mentioned in a passive or prepositional phrase — e.g., "data is stored in X", "handled by X", "via X", "through X"

RULES — exclude when:
1. The name appears only inside a dotted path (e.g., com.example.name)
2. The name is used as an ordinary English word, not as a component reference

Favor inclusion over exclusion — later validation will filter borderline cases."""


VALIDATION_RULES = """DECISION RULES:
APPROVE when:
- The component is the grammatical actor or subject (the sentence is ABOUT the component)
- A section heading names the component (introduces that component's topic)
- The sentence describes what the component does, provides, or interacts with

REJECT when:
- The name is used as an ordinary English word, not as a proper name
  (Like "proxy" in "proxy pattern" is the design pattern concept, not the Proxy component — reject the component link)
- The name is a modifier inside a larger phrase, not a standalone reference
  (Like "observer" in "observer pattern" modifies pattern — reject if Observer is a component)
- The sentence is about a subprocess, algorithm, or implementation detail — not the component itself"""


COREF_RULES = """For each case, determine if any pronoun in the TARGET sentence refers to a component.

RULES (all must hold):
1. The component name (or known alias) MUST appear verbatim in the antecedent sentence
2. The antecedent MUST be within the previous 3 sentences
3. The pronoun MUST grammatically refer back to that component as its subject
4. If the pronoun could refer to multiple things, DO NOT resolve it
5. Do NOT resolve pronouns about subprocesses or implementation details

Like in technical writing: "The Scheduler assigns tasks to threads. It uses a priority queue internally."
— "It" clearly refers to "the Scheduler" because it was the subject of the previous sentence."""


DOC_KNOWLEDGE_EXTRACTION_RULES = """WHAT TO FIND:
1. ABBREVIATIONS: Short forms explicitly introduced in the document.
   Rule: The abbreviation must be defined in the text, e.g., "Full Name (FN)" introduces FN.
   Like "Abstract Syntax Tree (AST)" defines AST — look for the same parenthetical pattern.

2. SYNONYMS: Alternative names that SPECIFICALLY refer to one component.
   Rule: The alternative name must unambiguously identify exactly ONE component.
   APPROVE: A proper name, role title, or technical alias used interchangeably with the component
   REJECT: A generic description that could apply to anything (like "the system" or "the process")

3. PARTIAL REFERENCES: A shorter form of a multi-word component name used alone.
   Rule: A trailing word from a multi-word name that, in this document, consistently means the full name.
   APPROVE: Only if the short form is unambiguous — no other component shares this word
   REJECT: Ordinary words that have plain English meanings beyond the component"""
