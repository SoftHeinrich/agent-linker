---
spike: 002
name: rules-audit
validates: "Given s_linker12c source, when every rule/heuristic function is classified for LLM-replaceability, then a ranked removal-order report is produced"
verdict: VALIDATED
related: [001-llm-trailing-words, 003-llm-mention-classifier]
tags: [audit, rules, static-analysis]
---

# Spike 002: Rules & Heuristics Audit

## What This Validates

**Given** `s_linker12c.py`,
**when** each rule/heuristic function is inspected and classified as REPLACEABLE / RISKY / ESSENTIAL,
**then** the audit shows how many are reachable by the LLM-driven goal and produces a ranked removal plan.

## How to Run

```bash
python .planning/spikes/002-rules-audit/audit.py
```

## What to Expect

- Two self-tests pass (all audited functions exist, report is generated).
- `AUDIT.md` appears in this directory with classification table + removal order + surviving-regex hotspots.

## Results

**VERDICT: VALIDATED ✓**

Counts (from `AUDIT.md`):

| Category | Count | Meaning |
|----------|-------|---------|
| REPLACEABLE | 6 | Can become LLM-driven via cite-evidence pattern (Spike 001) |
| ESSENTIAL | 3 | Parsers/accessors — not heuristics; keep |
| ESSENTIAL (removable via Spike 001) | 1 | `_split_component_name` |
| RISKY | 1 | `_has_standalone_mention` — latency-critical word-boundary primitive |
| MIXED | 1 | `_build_evidence_bundle` — orchestrator, becomes trivial after replacements |

**The only surviving rule after a full LLM-ification pass is `_has_standalone_mention`** — a regex word-boundary check used O(sentences × components) during anchor collection. It is performance-critical, not policy-critical. Everything else (mention classification, alias strength, ambiguity detection, CamelCase splitting) can be folded into existing LLM prompts via a cite-evidence schema.

See `AUDIT.md` for the full table and removal order.

## Signal for Full Pipeline

Target end-state: **one boundary regex + pure data formatters + LLM everywhere else.**
