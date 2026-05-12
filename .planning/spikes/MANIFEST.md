# Spike Manifest

## Idea

Transform `s_linker12c` from a hybrid (structural-rule gate + LLM verify) pipeline into a **fully LLM-driven** pipeline. Audit every rule/heuristic function, replace where feasible via a cite-evidence LLM pattern, and identify which primitives (if any) must remain as code.

## Spikes

| # | Name | Validates | Verdict | Tags |
|---|------|-----------|---------|------|
| 001 | llm-trailing-words | Single LLM call replaces structural gate + LLM verify for trailing-word alias enrichment with evidence guardrail | ✓ VALIDATED | llm-only, enrichment, trailing-words |
| 002 | rules-audit | Every rule/heuristic function classified REPLACEABLE / RISKY / ESSENTIAL with ranked removal plan | ✓ VALIDATED | audit, rules, static-analysis |
| 003 | llm-mention-classifier | LLM enum emission replaces regex-based `_classify_mention` with byte-identical output strings and zero regex | ✓ VALIDATED | llm-only, mention-classification, prompt-schema |
