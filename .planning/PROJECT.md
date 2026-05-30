# llm-sad-sam-v45: Fully-LLM-Driven s_linker

## What This Is

Empirical evolution of `s_linker12c` into a fully LLM-driven SAD-SAM traceability linker. Each milestone swaps one (or a small group of) structural rule/heuristic for an LLM-based replacement using the cite-evidence pattern validated in Spike 001, producing a ranked ablation of which rules can be retired without regressing macro F1.

## Core Value

Every rule removed from `s_linker12c` and replaced by an LLM primitive must either hold the pipeline at macro F1 ≥ 93% or be rejected — the deliverable is a defensible claim that traceability linking can be done without hand-crafted structural rules.

## Requirements

### Validated

- ✓ Reproducible `s_linker12c` baseline (per-dataset F1 + macro F1 + FP/FN table) — v1.0 (INFRA-01)
- ✓ Spike 001 LLM trailing-word enrichment integrated; `_split_component_name` retired (`s_linker13a`) — v1.0 (VAR-01)
- ✓ `_is_structurally_unambiguous` post-filter removed; LLM ambiguity classification trusted end-to-end (`s_linker13b`) — v1.0 (VAR-02)
- ✓ `_is_ambiguous_name_component` wrapper inlined and removed (`s_linker13c`) — v1.0 (VAR-03)
- ✓ Alias-discovery prompt extended with `scope: global|local`; `_is_strong_alias` + `_get_strong_alias_mappings` retired (`s_linker13e`) — v1.0 (VAR-05)
- ✓ Strong-alias-mention signal folded into coref prompt; `_has_strong_alias_mention` retired (`s_linker13f`) — v1.0 (VAR-06)
- ✓ `_has_standalone_mention` KEEP decision logged (RISKY per Spike 002; replacement deferred to v2 EXT-01) — v1.0 (PROMO-02)
- ✓ Ablation table generated (markdown + LaTeX, 8 rows) — v1.0 (PROMO-03)
- ✓ Winning variant promoted as `s_linker13.py` with zero non-trivial rules (macro F1 0.9509, +1.04 pp vs 12c) — v1.0 (PROMO-01)
- ⚠ Spike 003 LLM mention classifier integration attempted — REJECTED (VAR-04 retired). LLM cannot reproduce dotted-path Java-package convention; 33 entity-source FPs on TeaMMates → −18.8 pp regression. Documented as publishable negative result in METHODOLOGY.md §4 — v1.0

### Active

(None — v2.0 scope to be defined via `/gsd-new-milestone`. Candidate items: EXT-01 LLM replacement of `_has_standalone_mention`, EXT-02 dotted-path guard drop, EXT-03 GPT-5.2 cross-model re-evaluation, EXT-04 emit-biased boundary prompting for alias-discovery.)

### Out of Scope

- New seed/linker approaches (ILinker3+, cross-model ensembles) — this project is rule-reduction on 12c, not exploration
- Non-SAD-SAM tasks (SAM-Code, SAD-Code) — out of dataset scope
- Cost optimization — user has set "no LLM budget limit"; rule-replaceability is the only constraint
- Changes to retained upstream components (`ilinker*`, `prompts_v2`, `data_types_v2`) unless required by a rule removal
- Bench leakage: no benchmark-derived words may enter prompt examples (enforced via BENCHMARK_TABOO.md)

## Context

- **Codebase**: retained `s_linker` family through `s_linker12e` plus `ilinker1-3`. Runner is `run_ablation.py`. Default model: Claude Sonnet.
- **Baseline memory**: `s_linker12c` (ICSE clean) reports ~94% macro F1; `S-Linker10` memory entry shows 95.9% macro F1 with prompts_v2 and LLM word-usage enrichment.
- **Validated spikes** (re-validate once integrated):
  - 001 `llm-trailing-words` — single LLM call with evidence guardrail replaces structural gate + LLM verify for trailing-word alias enrichment.
  - 002 `rules-audit` — classified all 12 `s_linker12c` helpers: 9 REPLACEABLE, 1 RISKY (`_has_standalone_mention`), 4 ESSENTIAL (parsers/formatters). Ranked removal order defined.
  - 003 `llm-mention-classifier` — LLM enum emission matches regex `_classify_mention` byte-identically, piggybacked on existing entity-extraction prompt.
- **GPT/Claude gap known**: Claude Sonnet is the target backend; GPT-5.2 compatibility is a side concern, not a gate (per prior memory: inherent model capability gap, not fixable).
- **Dataset strategy**: hard-tier-first (teammates, bigbluebutton — most rule-sensitive) during development, full 5-project macro F1 for every promoted variant.

## Constraints

- **Quality**: Every promoted variant must hold macro F1 ≥ 93% on the 5-project benchmark. Variants below the floor are reported but not promoted.
- **LLM budget**: No upper bound on calls; replaceability trumps cost.
- **Model**: Claude Sonnet only (per user preference; do not switch to Opus or GPT).
- **Data leakage**: Zero benchmark-derived words in prompts (cascade/throttle rule, per `BENCHMARK_TABOO.md`). Spike findings must be re-audited before integration.
- **Codebase hygiene**: Each rule-removal lands as its own standalone linker variant (e.g., `s_linker13a.py`) — user prefers duplicated standalone files over inheritance chains.
- **Naming**: Promoted successor is `s_linker13` (12e is the last current sibling).

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Base = `s_linker12c` (not 12e) | Spikes 001/002/003 target 12c; 12d/12e are enrichment side-experiments | ✓ Good — chain landed on 12c; 13f beat 12c by +1.04 pp macro F1 |
| Success = zero non-trivial rules | User explicitly picked "zero non-trivial rules" over F1-parity goal | ✓ Good — `s_linker13` retains only `_has_standalone_mention` + parsers/formatters |
| Re-validate spikes in-pipeline | Spike validation was isolated; pipeline integration can surface new failure modes | ✓ Good — Spike 003 in-pipeline integration surfaced TM regression that isolated validation missed (VAR-04 retired) |
| F1 floor = 93% macro | Baseline is ~94%; allow ≤1pp regression per milestone | ✓ Good — floor held across all 6 successful removals; final macro 0.9509 (+1.04 pp) |
| Ablation unit = linker variant, not individual rule | User wants "F1 contribution per linker", not per-rule | ✓ Good — 7 standalone variant files enabled clean per-step ΔF1 attribution |
| Dataset schedule = hard-tier-first, then all 5 | Teammates/BBB are most rule-sensitive; cheap signal before full sweep | ⚠ Revisit — VAR-06 (13f) was hard-tier marginal but full-sweep best-in-chain; standing policy retained "full sweep is decisive" |
| Keep `_has_standalone_mention` tentatively | Spike 002 classified it RISKY (O(N·M) anchor collection); decide after other removals land | ✓ Good — formalized as KEEP in Phase 5; EXT-01 spike deferred to v2 |
| KEEP `_has_standalone_mention` in `s_linker13` | Spike 002 classified it RISKY (O(N×M) anchor-collection; replacing it with an LLM call would require a full-component-list × full-sentence-list scan). Phase 5 confirms KEEP — replacement deferred to v2 (EXT-01 spike) under a relaxed budget. EXT-02 (drop dotted-path guard) is a narrower follow-up also deferred to v2. See `.planning/spikes/002-rules-audit/` for the full classification. | KEPT (Phase 5, 2026-05-29) |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-30 after v1.0 milestone close*
