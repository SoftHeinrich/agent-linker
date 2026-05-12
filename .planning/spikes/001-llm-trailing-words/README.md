---
spike: 001
name: llm-trailing-words
validates: "Given components+document, when single LLM call replaces structural gate + verify step, then alias output parity achievable with only a light evidence guardrail (not a gate)"
verdict: VALIDATED
related: [002-rules-audit]
tags: [llm-only, enrichment, trailing-words]
---

# Spike 001: LLM-only Trailing-Word Enrichment

## What This Validates

**Given** a component list + document sentences,
**when** a single LLM call does both discovery AND verification of trailing-word aliases (no CamelCase regex, no uniqueness gate, no sentence-presence gate),
**then** the output matches the current structural+verify pipeline on clean input, and hallucinations are rejected by a light evidence-sentence guardrail.

## How to Run

```bash
python .planning/spikes/001-llm-trailing-words/spike.py
```

## What to Expect

Four tests pass:
1. **Happy path** — LLM-only returns 2 aliases, no regex used in the pipeline.
2. **Hallucination rejected** — alias word absent from cited sentence → dropped.
3. **Full-name-in-evidence rejected** — sentence contains both alias + full name → not standalone use → dropped.
4. **Parity** — LLM-only and current structural+verify produce the same alias map on a clean fixture.

## Results

**VERDICT: VALIDATED ✓**

- All 4 self-verifying tests pass.
- Drop-in replacement `fully_llm_driven(knowledge, sentences, components, llm_call)` has the same signature shape as `_enrich_trailing_words` (sans the intermediate regex helpers).
- **Removed from s_linker12c if adopted:**
  - `_split_component_name` (CamelCase regex splitter)
  - Structural candidate-gate loop inside `_enrich_trailing_words` (~15 lines of code)
- **Retained as guardrail (not as gate):** evidence-sentence membership check — pure post-condition, no regex beyond lowercase substring.
- **Prompt cost:** one LLM call instead of one. (Current: structural gate is free, then 1 LLM verify call. Proposed: 1 LLM discovery+verify call. Net equal or smaller.)
- **Risk surfaced:** LLM hallucination. Mitigation = evidence-sentence guardrail, demonstrated in test #2.

## Signal for Full Pipeline

The enrichment can go fully LLM-driven with essentially zero behavior loss provided the LLM is asked to **cite evidence**. This pattern (LLM returns facts *with* evidence pointers; code only sanity-checks the pointer) generalizes to the other rule-based helpers audited in Spike 002.
