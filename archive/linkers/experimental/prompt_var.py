"""Pareto prompt variants for cross-model robustness (Claude + GPT).

Each constant is a drop-in replacement for its prompts_v2.py counterpart.
Goal: improve GPT without regressing Claude. Keep prompts elegant.

Import selectively in variant linkers:
    from .prompt_var import WORD_USAGE_PROMPT_V as WORD_USAGE_PROMPT
"""

# ═══════════════════════════════════════════════════════════════════════════════
# P1 — Word Usage: "recurring pattern" threshold (replaces "even ONE")
#
# Original: ultra-low threshold causes GPT to classify almost everything as NAME.
# Variant: require recurring pattern — moderate tightening, not majority.
# Claude impact: ~neutral (10b already 95.56% with original).
# GPT impact: prevents literal interpreters from over-approving on single cases.
# ═══════════════════════════════════════════════════════════════════════════════

# Uses str.format() with named placeholders: partial, partial_lower, comp_name, calibration, sent_block
WORD_USAGE_PROMPT_V = """WORD USAGE CLASSIFICATION

In this document, the word "{partial}" could be a short name for an architecture
component called "{comp_name}".

{calibration}Below are ALL sentences where "{partial}" appears WITHOUT the full name "{comp_name}".
Analyze how the word "{partial}" is used across these sentences:

{sent_block}

QUESTION: Is "{partial}" consistently used as a standalone entity reference in these sentences?

Classify as NAME if the word regularly appears as a standalone noun phrase referring to
a specific system entity — it should be a RECURRING pattern, not a single ambiguous case.
Examples of entity reference: "the {partial_lower} connects to...", "sends data to the
{partial_lower}", "the {partial_lower} handles...", "on the {partial_lower}"

Classify as ORDINARY if the word is primarily used as a modifier, in compound phrases,
or as a generic descriptor. A single borderline case among many generic uses is not enough.
Examples of purely ordinary: "{partial_lower} process", "automated {partial_lower}",
"{partial_lower} strategy", "{partial_lower}-based"

The test: look at ALL sentences together. If standalone entity uses are a clear,
recurring pattern, classify as NAME. If the word is mostly generic with only
occasional standalone uses, classify as ORDINARY.

Return JSON: {{"classification": "name" or "ordinary", "reason": "brief explanation"}}
JSON only:"""


# ═══════════════════════════════════════════════════════════════════════════════
# P2 — Judge Rules: scoped doubt clause (replaces blanket "when in doubt")
#
# Original: GPT takes "when in doubt, APPROVE" literally, over-approves synonyms.
# Variant: scope the doubt rule to proper names/technical terms, not dictionary words.
# Claude impact: ~zero (Claude is already conservative here).
# GPT impact: closes the common-noun synonym exploit.
# ═══════════════════════════════════════════════════════════════════════════════

DOC_KNOWLEDGE_JUDGE_RULES_V = """DECISION RULES (apply in order):

1. AUTO-APPROVE these — they are always valid mappings:
   - Abbreviations formed from the component name's initials or words
   - Trailing words of multi-word component names (if no other component shares that word)
   - CamelCase identifiers
   - Multi-word phrases that contain the component name

2. APPROVE if the term plausibly refers to exactly one component and is NOT
   a generic word like "system", "process", "service", "component", "module".

3. REJECT only if the term is clearly generic and could refer to anything,
   or clearly refers to a different component or the system as a whole.

IMPORTANT: When genuinely uncertain about a proper name or technical term,
lean toward APPROVE — false rejections cause permanent recall loss.
But do NOT approve common single-word English nouns (server, client, service,
store, manager, handler) as synonyms: these appear in every software document
and approval floods later stages with noise. The doubt rule applies to
constructed identifiers and multi-word terms, not dictionary words."""


# ═══════════════════════════════════════════════════════════════════════════════
# P3 — Alias Rule: evidential framing (replaces "IS a reference")
#
# Original: GPT treats "IS a reference" as default APPROVE for alias candidates.
# Variant: "CAN refer" reframes alias as hypothesis, not declaration.
# Claude impact: ~zero (Claude already applies nuanced judgment).
# GPT impact: requires GPT to actually evaluate each case, not rubber-stamp.
# ═══════════════════════════════════════════════════════════════════════════════

ALIAS_RULE_V = ("\n- When a KNOWN ALIAS is indicated, this confirms the word CAN refer to "
                "the component. Still evaluate whether THIS sentence discusses the "
                "component's architectural role or behavior — an alias provides context, "
                "not automatic approval")


# ═══════════════════════════════════════════════════════════════════════════════
# P4 — Extraction Rules: prefix disambiguation (adds exclusion rule 3)
#
# Original: only 2 exclusion rules; GPT matches prefixes of multi-word names.
# Variant: explicit prefix-of-multiword exclusion.
# Claude impact: ~zero (Claude handles this implicitly).
# GPT impact: prevents "GAE" in "GAE server" matching "GAE Datastore".
# ═══════════════════════════════════════════════════════════════════════════════

ENTITY_EXTRACTION_RULES_V = """RULES — include a reference when:
1. The component name (or known alias) appears directly in the sentence
2. A space-separated form matches a compound name (e.g., "Memory Manager" → MemoryManager)
3. The sentence describes what a specific component does by name or role
4. A known synonym or partial reference is used
5. The component participates in an interaction described in the sentence (as sender, receiver, or target) — e.g., "X sends data to Y" references BOTH X and Y
6. The component is mentioned in a passive or prepositional phrase — e.g., "data is stored in X", "handled by X", "via X", "through X"

RULES — exclude when:
1. The name appears only inside a dotted path (e.g., com.example.name)
2. The name is used as an ordinary English word, not as a component reference
3. Only a PREFIX of a multi-word component name appears, in a phrase about
   something else (e.g., for component "Cloud Database", the word "Cloud" alone
   in "Cloud server" or "Cloud environment" is NOT a reference to that component)

Favor inclusion over exclusion — later validation will filter borderline cases."""


# ═══════════════════════════════════════════════════════════════════════════════
# P5 — Generic Detection: anchor contrast instruction
#
# Original: abstract distinction ("names entity" vs "describes activity").
# Variant: grounds decision in the anchor sentences already in the prompt.
# Claude impact: ~zero (Claude uses anchors implicitly).
# GPT impact: directs GPT to compare against concrete correct examples.
# ═══════════════════════════════════════════════════════════════════════════════

GENERIC_DETECTION_DISTINCTION_V = (
    "Key distinction: Compare each case to the FULL-NAME REFERENCES above. In those anchor\n"
    "sentences, the component is the SUBJECT or a named participant. Does the lowercase\n"
    "usage function the same way? If the word is part of a compound phrase, a modifier,\n"
    "or describes a general concept rather than naming the specific entity, it is GENERIC."
)


# ═══════════════════════════════════════════════════════════════════════════════
# P3a — Alias Rule: two-tier (strict for generic aliases, lenient for clear)
#
# Original P3: uniform "CAN refer" — too strict for BBB (killed 2 TPs).
# This variant: CamelCase/abbreviation aliases get "IS a reference" (strong);
# single-word/generic aliases get "CAN refer" (evidential).
# Claude impact: ~zero (same as P3 for ambiguous, preserves original for clear).
# GPT impact: prevents rubber-stamp on ambiguous aliases, preserves recall on clear.
# ═══════════════════════════════════════════════════════════════════════════════

ALIAS_RULE_V2 = (
    "\n- When a KNOWN ALIAS is indicated:"
    "\n  • For CamelCase identifiers, abbreviations, or multi-word aliases: the word IS a "
    "reference to that component unless the sentence clearly uses it in an unrelated sense"
    "\n  • For single common-word aliases (e.g., a trailing word of a multi-word name): "
    "the alias confirms the word CAN refer to the component — still evaluate whether "
    "THIS sentence discusses the component's architectural role, not just mentions the word"
)


# ═══════════════════════════════════════════════════════════════════════════════
# P3b — Alias Rule: role-verification framing
#
# Original P3: "CAN refer" is a prior probability reframe.
# This variant: keeps "IS a reference" but adds an explicit role-verification step.
# Targets Pattern 1 (GAE→GAE Datastore in platform context sentences).
# Claude impact: ~zero (Claude already does implicit role check).
# GPT impact: forces GPT to verify the sentence is about the component's FUNCTION.
# ═══════════════════════════════════════════════════════════════════════════════

ALIAS_RULE_V3 = (
    "\n- When a KNOWN ALIAS is indicated, the word IS a reference to that component."
    "\n  HOWEVER: verify the sentence discusses the component's specific architectural "
    "role or function — not a broader platform, tool, or infrastructure that happens "
    "to share the same prefix. For example, if \"Cloud\" is an alias for \"Cloud Database\", "
    "\"Cloud server handles routing\" is NOT about the database component."
)


# ═══════════════════════════════════════════════════════════════════════════════
# P3c — Alias Rule: combined CAN-refer + role-verification
#
# Merges P3 evidential framing with P3b's role-verification instruction.
# Strongest anti-rubber-stamp variant.
# ═══════════════════════════════════════════════════════════════════════════════

ALIAS_RULE_V4 = (
    "\n- When a KNOWN ALIAS is indicated, this confirms the word CAN refer to "
    "the component — but an alias is not automatic approval."
    "\n  Verify: does THIS sentence discuss the component's specific architectural "
    "role or function? If the sentence is about a broader platform, infrastructure, "
    "or different subsystem that shares the alias word, REJECT the match."
)


# ═══════════════════════════════════════════════════════════════════════════════
# P4a — Extraction Rules: prefix rule + abbreviation exception
#
# Original P4: prefix exclusion regressed MS (-3.0pp, 3 DB FPs).
# Root cause: LLM variance on re-run, not the rule itself.
# This variant: same prefix rule but clarifies abbreviations are NOT prefixes.
# Claude impact: ~zero.
# GPT impact: same prefix protection as P4, safer for short-name components.
# ═══════════════════════════════════════════════════════════════════════════════

ENTITY_EXTRACTION_RULES_V2 = """RULES — include a reference when:
1. The component name (or known alias) appears directly in the sentence
2. A space-separated form matches a compound name (e.g., "Memory Manager" → MemoryManager)
3. The sentence describes what a specific component does by name or role
4. A known synonym or partial reference is used
5. The component participates in an interaction described in the sentence (as sender, receiver, or target) — e.g., "X sends data to Y" references BOTH X and Y
6. The component is mentioned in a passive or prepositional phrase — e.g., "data is stored in X", "handled by X", "via X", "through X"

RULES — exclude when:
1. The name appears only inside a dotted path (e.g., com.example.name)
2. The name is used as an ordinary English word, not as a component reference
3. Only a PREFIX of a multi-word component name appears, and the sentence is about
   something else (e.g., for component "Cloud Database", the word "Cloud" alone
   in "Cloud server" is NOT a reference to the database component).
   NOTE: This does NOT apply to abbreviations — if "DB" is a known alias for
   "Database", then "DB" in a sentence IS a valid reference.

Favor inclusion over exclusion — later validation will filter borderline cases."""


# ═══════════════════════════════════════════════════════════════════════════════
# V_ROLE — Validation Rules: explicit architectural-role verification
#
# Original: validation checks actor role and direct reference.
# This variant adds: for alias matches, verify the component's specific function
# (not just word presence). Targets GAE→GAE Datastore cascade.
# Claude impact: ~zero (Claude does implicit role verification).
# GPT impact: prevents approving alias matches in wrong-context sentences.
# ═══════════════════════════════════════════════════════════════════════════════

VALIDATION_RULES_V = """DECISION RULES:
APPROVE when:
- The component is the grammatical actor or subject (the sentence is ABOUT the component)
- A section heading names the component (introduces that component's topic)
- The sentence describes what the component does, provides, or interacts with

REJECT when:
- The name is used as an ordinary English word, not as a proper name
  (Like "proxy" in "proxy pattern" is the design pattern concept, not the Proxy component — reject the component link)
- The name is a modifier inside a larger phrase, not a standalone reference
  (Like "observer" in "observer pattern" modifies pattern — reject if Observer is a component)
- The sentence is about a subprocess, algorithm, or implementation detail — not the component itself
- An alias matches but the sentence discusses a DIFFERENT aspect of the system that
  merely shares the same word (e.g., a platform name used as component alias, but
  the sentence is about the platform's hosting/routing, not the component's function)"""


# ═══════════════════════════════════════════════════════════════════════════════
# V_FOCUS — Alternative validation focus prompts
#
# Original focus strings are somewhat abstract. These variants are more concrete.
# ═══════════════════════════════════════════════════════════════════════════════

VALIDATION_FOCUS_1_V = ("Focus on SPECIFICITY: is this sentence about the specific "
                        "component named, or about a broader concept that shares the word?")

VALIDATION_FOCUS_2_V = ("Focus on FUNCTION: does the sentence describe this component's "
                        "architectural role (what it does, provides, or connects to), "
                        "or just mention the word in passing?")
