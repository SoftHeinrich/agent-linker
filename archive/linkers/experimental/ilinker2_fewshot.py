"""ILinker2 with few-shot examples — calibrate extraction boundary.

Adds decision examples to Pass A and Pass B prompts to help LLMs (especially GPT)
distinguish component references from generic word usage.

All examples use safe SE textbook domains (compiler, OS, e-commerce).
"""

from llm_sad_sam.linkers.experimental.ilinker2 import ILinker2


class ILinker2FewShot(ILinker2):
    """ILinker2 with few-shot examples in extraction prompts."""

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

EXAMPLES (components: Scheduler, Monitor, MemoryManager, Pool, Logger):

S1: "The Scheduler assigns incoming jobs to available worker threads."
→ {{"s": 1, "c": "Scheduler", "text": "Scheduler", "type": "exact"}}
Reason: "Scheduler" is used as a standalone proper noun naming the component.

S2: "We scheduled weekly maintenance windows for the database."
→ (no link)
Reason: "scheduled" is an ordinary English verb, not a reference to the Scheduler component.

S3: "The memory manager reclaims unused pages after garbage collection."
→ {{"s": 3, "c": "MemoryManager", "text": "memory manager", "type": "synonym"}}
Reason: "memory manager" is a space-separated form of the CamelCase component name.

S4: "Classes in the Logger.impl package handle file rotation."
→ (no link)
Reason: "Logger" appears only as a prefix in a dotted package path, not as a standalone reference.

S5: "The pool of database connections is limited to 50."
→ (no link)
Reason: "pool" is a generic English word describing a collection, not a reference to the Pool component.

S6: "The Pool distributes connections across three database replicas."
→ {{"s": 6, "c": "Pool", "text": "Pool", "type": "exact"}}
Reason: "Pool" is capitalized and used as the sentence subject performing an architectural action.

S7: "The Monitor.alerts, Monitor.metrics, and Monitor.dashboards packages handle observability."
→ (no link)
Reason: "Monitor" appears only as a namespace prefix in dotted paths listing sub-packages. Even though Monitor is capitalized, it is NOT being discussed as a component here.

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

EXAMPLES (components: Parser, Optimizer, CodeGen, AST, Runtime):

S1: "The Parser transforms token streams into an abstract syntax tree."
→ Parser (exact): "Parser" is the subject performing the action.
→ AST (partial): "abstract syntax tree" is a space-separated form of AST.

S2: "We optimized the build pipeline for faster CI runs."
→ (no link): "optimized" is a generic verb, not a reference to the Optimizer component.

S3: "The code generation phase emits platform-specific bytecode."
→ CodeGen (synonym): "code generation" refers to the CodeGen component.

S4: "Runtime.utils.logging provides structured log output."
→ (no link): "Runtime" appears only inside a dotted package path.

S5: "The runtime environment handles memory allocation and thread scheduling."
→ Runtime (exact): "runtime" is used as a standalone noun naming the component.

S6: "The AST.visitors, AST.transforms, and AST.serializers modules implement tree processing."
→ (no link): "AST" appears only as a namespace prefix in dotted paths. The sentence describes sub-modules, not the AST component itself.

Rules:
- Only report components that are explicitly named, abbreviated, or identified by a clear synonym/partial name IN THE SENTENCE TEXT.
- Do NOT report contextual or pronoun-based references (e.g., "It does X" — skip these).
- Do NOT match component names inside dotted package paths (e.g., "renderer.utils.config" does NOT reference "Renderer").
- Do NOT match generic English words used in their ordinary sense (e.g., "optimized code" does NOT reference "Optimizer").

Return ONLY valid JSON:
{{"links": [{{"s": N_INTEGER, "c": "ComponentName", "text": "evidence", "type": "exact|synonym|partial"}}]}}

Be conservative — omit uncertain links."""
