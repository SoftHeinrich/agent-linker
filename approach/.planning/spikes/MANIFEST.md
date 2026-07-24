# Spike Manifest

## Idea

Transform `s_linker12c` from a hybrid (structural-rule gate + LLM verify) pipeline into a **fully LLM-driven** pipeline. Audit every rule/heuristic function, replace where feasible via a cite-evidence LLM pattern, and identify which primitives (if any) must remain as code.

## Requirements

- Canonical S21 remains byte-stable and is the fixed floor for new variants.
- New prompts use only generic English and runtime project data.
- Agentic recovery is bounded: a controller selects tools but cannot emit links.
- Report macro and pooled F1/F2; S24 must exceed fresh S21 F2 without lowering
  either F1 aggregate.
- New FN capabilities must own a non-overlapping evidence mode and show enough
  same-run causal reach to justify a pilot; low-value alias appeals and
  benchmark-boundary patches remain design-only or are rejected.

## Spikes

Spikes 006–009 and their superseded runtime/pilot artifacts are preserved under
`.planning/archive/s24-pre-orchestrator-260724/`. Active S24 development begins
with the retained replacement-orchestrator lineage at spike 010.

| # | Name | Type | Validates | Verdict | Tags |
|---|------|------|-----------|---------|------|
| 001 | llm-trailing-words | standard | Single LLM call replaces structural gate + LLM verify for trailing-word alias enrichment with evidence guardrail | ✓ VALIDATED | llm-only, enrichment, trailing-words |
| 002 | rules-audit | standard | Every rule/heuristic function classified REPLACEABLE / RISKY / ESSENTIAL with ranked removal plan | ✓ VALIDATED | audit, rules, static-analysis |
| 003 | llm-mention-classifier | standard | LLM enum emission replaces regex-based `_classify_mention` with byte-identical output strings and zero regex | ✓ VALIDATED | llm-only, mention-classification, prompt-schema |
| 004 | nogap-validator-ab | standard | Layered validator (Mode 5 justification + Mode 1 claim-rubric + Mode 2 trap-list, Mode 4 skeptic on coref) recovers effort-0 macro-F1 toward thinking-on without losing implicit-link recall | ◑ PARTIAL — shipped `s_linker20_union_layered` (Sonnet +1.1, gpt +3.8, zero implicit-recall cost); Modes 2 & 4 rejected | validator, no-reasoning, false-positive-filter, ab-test |
| 005 | upstream-candidate-gap | standard | Decompose spike 004's residual upstream gap into validator-recoverable vs extraction-bound; can the Mode-5 mechanism recover extraction? | ✓ COMPLETE — extraction-bound = 6.2% of gold, 68% run-variance, 44% non-verbatim inference; mechanism does NOT transfer to extraction (precision-at-gates vs recall-at-extraction asymmetry). Stop. | recall, candidate-generation, extraction, no-reasoning |
| 006 | s24-agentic-phase-tools | standard | Fixed-floor controller selects bounded Phase-1/Phase-4 and anchored recovery tools using runtime-only evidence | ✓ VALIDATED — production replay 5 TP / 0 FP, macro F1 +1.34pp; live Mediastore same-run +1 TP / 0 FP | agentic, tool-routing, fixed-floor, s24 |
| 007 | s24-dynamic-controller | standard | Sequential document/component/floor profiling and validator feedback produce adaptive workflows with lower cost than run-all | ✓ VALIDATED — 4 TP / 0 FP, macro F1 +1.14pp, four workflows, adaptive fallback, 5 vs 6 phase calls; oracle-informed | agentic, controller, dynamic-workflow, oracle-analysis, s24 |
| 008 | s24-semantic-appeal | standard | Semantic reconsideration of all S21-rejected grounded candidates without heuristic eligibility rules | ✗ INVALIDATED — 7 TP / 12 FP, then 5 TP / 6 FP after identity/ownership structuring; below dynamic S24 and wrong refine-not-replace architecture | s24, error-analysis, appeal, no-magic |
| 009 | s24-replacement-orchestrator | standard | Project-profile controller replaces S21's fixed workflow by selecting existing phase tools plus semantic coverage audit | ✓ VALIDATED performance — participation audit: macro F2 +3.92pp and pooled F2 +4.87pp vs S21; macro F1 +1.16pp; same route on all projects, so route diversity is not validated | s24, replacement, orchestration, phase-tools, no-magic |
| 010 | s24-relation-role-routing | standard | Fresh project-profile controller selects non-overlapping relation/role capability from exact evidence and produces project-specific workflows | ✓ VALIDATED — 3 workflows; macro F2 92.70→93.95, pooled F2 89.69→92.21 vs fresh S21; recall +6.67pp | s24, replacement, dynamic-workflow, relation-role, fresh-run |
| 011 | s24-f1-constrained-routing | comparison | Ownership-aligned entity and project-context handle review retain S24's F2 gain while macro and pooled F1 do not fall below fresh S21 | ✓ VALIDATED — macro F1 +0.80pp, pooled F1 +1.63pp, macro F2 +0.85pp, pooled F2 +1.89pp | s24, precision, f1, f2, ownership, no-magic |
| 012 | s24-targeted-fn-tools | comparison | Non-overlapping alias and identifier tools recover clear residual FNs while improving both F2 aggregates without regressing either F1 aggregate | ✓ VALIDATED — exact identifier tool adds 2 TP / 0 FP; macro/pooled F1 +3.92/+4.57 pp and F2 +1.81/+2.43 pp vs fresh S21 | s24, false-negative, identifiers, dynamic-workflow, no-magic |
| 013 | s24-lexical-entity-normalization | comparison | Exact unique catalog signatures augment the entity candidate set and replace the standalone identifier tool without losing its clean recoveries | ✓ VALIDATED — fresh pilot 2 TP / 0 FP; paired E2E macro/pooled F1 +4.34/+5.16 pp and F2 +2.82/+3.91 pp vs S21; one fewer BBB call than identifier-tool trace | s24, lexical-normalization, entity-ownership, identifiers, icse |
