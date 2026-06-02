---
created: 2026-06-01T14:00:13.469Z
title: Implement refined v3-style axiom diffs from feasibility study
area: tooling
resolves_phase: 33
files:
  - src/llm_sad_sam/linkers/experimental/prompts_v3_axiom.py
  - src/llm_sad_sam/linkers/experimental/s_linker14_voyager.py:719
  - src/llm_sad_sam/linkers/experimental/s_linker14_voyager.py:946
  - src/llm_sad_sam/linkers/experimental/s_linker14_voyager.py:1004
---

## Problem

Feasibility study on todo `2026-06-01-design-better-axioms-section-context-responsibility.md`
found that the proposed axiom additions (SCN, gerund rejection, alias coref) were too
specific — they named surface forms ("the W", "W-side", "verb-ing without explicit subject",
"≤10 sentences") rather than expressing semantic principles in v3 style.

v3 axioms operate at intent/meaning level. The proposals dropped to syntactic pattern level.
Additionally, Gap 1 (SCN) was mis-placed: DOC_KNOWLEDGE_EXTRACTION_RULES is document-global
alias discovery; SCN is context-local anaphoric resolution → belongs in COREF_RULES.

## Solution

Three concrete diffs, all at v3 abstraction level (no surface forms, no magic numbers):

### 1. COREF_RULES — covers Gap 1 (SCN) + Gap 3 (alias coref) together

```diff
- For each case, decide whether a pronoun in the target sentence grammatically
- refers back to a component named verbatim earlier in the context. Resolve only
- when one component is the unambiguous antecedent.

+ For each case, decide whether a pronoun or role-referential noun phrase in the
+ target sentence refers back to a component named or aliased earlier in the
+ context. Resolve only when one component is the unambiguous antecedent.
```

### 2. SEED_DISAMBIGUATION_RULES — covers Gap 2 (gerund/responsibility-list FPs)

```diff
- OTHER: code identifier, technique sharing the name, sub-entity of a larger name,
- or ordinary English

+ OTHER: code identifier, technique sharing the name, sub-entity of a larger name,
+ ordinary English, or description of the component's own capabilities without
+ referencing an external participant
```

### 3. Code change required alongside COREF_RULES axiom (s_linker14_voyager.py)

COREF_RULES axiom extension has NO EFFECT without widening the sentence filter.
Line 949 currently gates on `PRONOUN_PATTERN` — SCN sentences ("The server handles X")
contain no pronoun and never reach the LLM batch.

Required changes in `_coref_cases_in_context`:

a. Build runtime definite-NP pattern from component terminal words:
```python
comp_terminals = {c.name.split()[-1].lower() for c in components if len(c.name.split()) > 1}
role_ref_pat = re.compile(
    r'\bthe (' + '|'.join(re.escape(w) for w in comp_terminals) + r')\b',
    re.IGNORECASE
) if comp_terminals else None
```

b. Expand filter (line 949):
```python
anaphoric_sents = [
    s for s in sentences
    if self.PRONOUN_PATTERN.search(s.text)
    or (role_ref_pat and role_ref_pat.search(s.text))
]
```

c. Update prompt header: "pronoun references" → "anaphoric references (pronouns and
role-referential noun phrases)"

d. Update JSON template: `"pronoun": "it"` → `"reference": "the server"`

e. Variable rename: `pronoun_sents` / `pronoun_count` → `anaphoric_sents` / `anaphoric_count`
   (lines 719 and 949; code never parses `pronoun` field so no parsing changes needed)

Line 1004 gate (`has_standalone_mention`) needs no change — SCN antecedent sentences
contain the full component name, so gate passes correctly.

## Prerequisites before deploying Gap 2

Scan MS and TS gold links for legitimate gerund trace links before adding SEED_DISAMBIGUATION
change. If "Providing X to Y" (cross-component gerund) appears as gold TP in training data,
the rule fires incorrectly. The safety valve ("referencing an external participant") should
handle this, but verify empirically first.

## Risk

- Gap 3 (alias coref in COREF_RULES): low. One phrase change.
- Gap 2 (gerund in SEED_DISAMBIGUATION): medium. Verify on training data first.
- Gap 1 (SCN via COREF_RULES): medium. Requires code change + adds sentences to coref batches
  (batch size may increase; test on BBB/TM). Gate at line 1004 is unchanged and correct.
