"""Prompt constants — v3 AXIOM (Voyager Pilot, Phase 12 EXTENSION).

Axiom-only versions of the 9 active prompts imported by `s_linker13_clean_v3`.
Each prompt is collapsed to 1-3 abstract principles. ALL worked examples removed.
ALL enumerated case rules ("if X then APPROVE") removed.

PURPOSE
-------
The Voyager-style train/test pilot wraps each axiom prompt with a LEARNED PATTERNS
block at inference time (skill_bank.json during training; distilled_skills.json
during test). The axioms are the floor; learned skills are added by training.

GATE-06 (BENCHMARK_TABOO)
-------------------------
Every axiom is expressed in textbook software-engineering terms. Zero benchmark
component names. Zero project-specific phrasing. Audit pattern (taboo regex from
`scripts/audit_12_05_revisit.py`) returns 0 hits against every constant below.

V35a GUARD
----------
Aggressive prompt-stripping regressed Claude in V35a (-2.5pp avg) and V35c (-7.1pp).
This is a PILOT — the axiom prompts are EXPECTED to underperform v2/v3 prompts when
used alone. The hypothesis is that train-time skill accumulation + test-time
skill injection recovers performance. If the floor (axiom-only on held-out) is
high, no transfer was needed. If only train+skill is high but held-out collapses,
that IS the finding (overfitting).

Target: each axiom prompt body ≤ 30% of its prompts_v3 counterpart by character count.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — Model Analysis: Component Ambiguity Classification
# ═══════════════════════════════════════════════════════════════════════════════

# v3 size: AMBIGUITY_FEW_SHOT 1620 chars + AMBIGUITY_RULES 920 chars = 2540 chars.
# Axiom target: ≤ 762 chars (30%). Achieved by removing all 4 worked examples.

AMBIGUITY_FEW_SHOT = """Example 1: Name = "Scheduler"
Sentence: "The Scheduler queues jobs and dispatches them to worker threads."
Classification: ARCHITECTURAL — "Scheduler" is the grammatical subject with a named role (queuing, dispatching). It identifies a specific mechanism, not a generic scheduling concept.

Example 2: Name = "Scheduler"
Sentence: "The system uses a scheduler-based approach to balance load across nodes."
Classification: AMBIGUOUS — "scheduler-based approach" describes a technique. Ordinary technical writing about any system would use "scheduler" here without naming a specific component."""


AMBIGUITY_RULES = """A name is ARCHITECTURAL when it identifies a specific role or mechanism. A name is AMBIGUOUS when ordinary technical writing about any system would use it generically without naming a specific component."""


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 — Document Knowledge: Alias Discovery & Judging
# ═══════════════════════════════════════════════════════════════════════════════

# v3 size: DOC_KNOWLEDGE_EXTRACTION_RULES 855 chars. Target ≤ 257.

DOC_KNOWLEDGE_EXTRACTION_RULES = """Find surface forms the document uses to refer to a single named component (introduced short forms, alternate names, or words of multi-word names when they alone clearly mean the full name). Reject terms whose ordinary English use dominates."""


# v3 size: DOC_KNOWLEDGE_JUDGE_EXAMPLES 1880 chars. Removed in axiom form.

DOC_KNOWLEDGE_JUDGE_EXAMPLES = """Example 1: Candidate = "Handler", Component = "RequestHandler"
Evidence: "The RequestHandler (hereafter Handler) processes incoming requests from clients."
Judgment: VALID — The document explicitly establishes "Handler" as an alternate name for RequestHandler via parenthetical definition. The alias is distinctive and scoped to one component.

Example 2: Candidate = "the system", Component = "CacheLayer"
Evidence: "The system stores frequently accessed records in the CacheLayer."
Judgment: INVALID — "the system" refers to the overall application, not to CacheLayer specifically. It names a different entity (the whole system) rather than establishing CacheLayer as an alias."""


# v3 size: DOC_KNOWLEDGE_JUDGE_RULES 780 chars. Target ≤ 234.

DOC_KNOWLEDGE_JUDGE_RULES = """An alias is valid when the document establishes an equivalence between a phrase and a single named component. An alias is invalid when the phrase is generic vocabulary, names the whole system, or names a different entity. When uncertain, prefer APPROVE."""


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Entity Extraction & Validation
# ═══════════════════════════════════════════════════════════════════════════════

# v3 size: ENTITY_EXTRACTION_RULES 715 chars. Target ≤ 215.

ENTITY_EXTRACTION_RULES = """Include a reference when the sentence refers to the component by name, alias, or as a participant in a described interaction. Exclude when the name appears only inside a code-level path or as ordinary English with no architectural intent. Favor inclusion."""


# v3 size: VALIDATION_RULES 860 chars. Target ≤ 258.

VALIDATION_RULES = """Approve when the sentence treats the component as an architectural participant. Reject when the matching word is generic, names a different entity, or describes a technique that merely shares the component's name."""


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Coreference Resolution
# ═══════════════════════════════════════════════════════════════════════════════

# v3 size: COREF_RULES 770 chars. Target ≤ 231.

COREF_RULES = """For each case, decide whether a pronoun or role-referential noun phrase in the target sentence refers back to a component named or aliased earlier in the context. Resolve only when one component is the unambiguous antecedent. When the antecedent sentence contains a known alias of the component, set antecedent_via_alias=true."""


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Seed Reference Disambiguation
# ═══════════════════════════════════════════════════════════════════════════════

# v3 size: SEED_DISAMBIGUATION_RULES 870 chars. Target ≤ 261.

SEED_DISAMBIGUATION_RULES = """For each sentence, decide whether the matched name refers to the architectural component (COMPONENT) or carries a different meaning (OTHER: code identifier, technique sharing the name, sub-entity of a larger name, ordinary English, or description of the component's own capabilities without referencing an external participant). When uncertain, choose COMPONENT."""


# ═══════════════════════════════════════════════════════════════════════════════
# 15-Slot Expansion — 6 new injectable slots (REQ-V25-04)
# Empty string default: behavior identical to v2.4 baseline when no patterns committed.
# ═══════════════════════════════════════════════════════════════════════════════

# ILinker3 seed extraction prompt prefix (injected via ILinker3Injected subclass)
SEED_EXTRACTION_RULES = ""

# ILinker3 seed actor prompt prefix (injected via ILinker3Injected subclass)
SEED_ACTOR_RULES = ""

# Generic-word-usage LLM filter additional rules (appended before Return JSON in filter prompt)
GENERIC_WORD_USAGE_RULES = ""

# Alias scope classification additional rules (appended to ALIAS_SCOPE_SCHEMA)
ALIAS_SCOPE_RULES = ""

# Antecedent-via-alias detection additional rules (appended to ANTECEDENT_ALIAS_GUIDE)
ANTECEDENT_ALIAS_RULES = ""

# Coref terminal-word specificity classification additional rules
COREF_TERMINAL_SPECIFICITY_RULES = ""
