---
spike: 003
name: llm-mention-classifier
validates: "Given LLM-emitted mention_type enum per candidate, when the consumer formats it for prompts, then output is byte-identical to current _classify_mention strings with zero regex calls"
verdict: VALIDATED
related: [001-llm-trailing-words, 002-rules-audit]
tags: [llm-only, mention-classification, prompt-schema]
---

# Spike 003: LLM-only Mention Classification

## What This Validates

**Given** the entity-extraction LLM pass already reads `(comp_name, sentence, known_aliases)`,
**when** the prompt is extended to emit `mention_type` as an enum per candidate,
**then** the consumer-side formatter produces byte-identical strings to the current regex-based `_classify_mention`, with **zero `re` module references**.

## How to Run

```bash
python .planning/spikes/003-llm-mention-classifier/spike.py
```

## What to Expect

Four tests pass:
1. All 6 enum branches (5 types + aliased variant) produce correct strings.
2. Unknown enum falls back to `"indirect/unclear match"`.
3. Output strings match `_classify_mention` output byte-for-byte.
4. Consumer functions reference zero regex (bytecode-level check).

## Results

**VERDICT: VALIDATED ✓**

- No new LLM call needed — piggyback on existing `_extract_entities_enriched` prompt.
- **Removed from s_linker12c if adopted:** `_classify_mention` (4 regex branches, ~30 lines).
- **Net LLM cost delta:** zero.
- **Pattern:** same cite-evidence schema as Spike 001 — LLM emits structured data, code sanity-checks the enum, no regex.

## Signal for Full Pipeline

Combined with Spikes 001 + 002:
- Enrichment (Spike 001) → LLM-only with evidence guardrail.
- Mention classification (Spike 003) → LLM-only via prompt-schema extension (free).
- Audit (Spike 002) → 9 of 12 helpers follow the same pattern.

Only `_has_standalone_mention` remains as a surviving structural primitive. Everything else melts into LLM prompts + enum consumers.
