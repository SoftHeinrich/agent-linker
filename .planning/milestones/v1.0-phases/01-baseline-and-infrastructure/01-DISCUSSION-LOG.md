# Phase 1: Baseline and Infrastructure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-24
**Phase:** 01-baseline-and-infrastructure
**Areas discussed:** Gray-area selection, SDK scope conflict, Baseline protocol, Checkpoint + diskcache

---

## Gray Area Selection

| Option | Description | Selected |
|--------|-------------|----------|
| SDK migration + enforcement | Rip-and-replace vs parallel backend; prove temp=0.0 + caching headers | |
| Baseline protocol | Single run vs N-run median vs best-of-N | |
| Checkpoint + diskcache | `_VARIANT_NAME` namespace + diskcache role | ✓ |
| 13a scope + guardrail | Standalone copy + evidence guardrail strictness | |

**User's choice:** "Checkpoint + diskcache" selected.
**Notes (verbatim):** "no replace for anthropic sdk, current backend is fine. single run"

The notes pre-answered two non-selected areas (SDK migration → drop, baseline protocol → single run), so those were folded into the context rather than discussed further.

---

## SDK scope (conflict resolution)

| Option | Description | Selected |
|--------|-------------|----------|
| Strike from Phase 1 | Drop INFRA-02/INFRA-04 entirely; record scope change | ✓ |
| Defer to later phase | Keep requirements alive but move out of Phase 1 | |
| Keep as-is, no migration now | Skip work, don't formally strike | |
| Minimal compliance | Stay on claude -p but document limits | |

**User's choice:** Strike from Phase 1.
**Notes:** Triggered because user's "no replace for anthropic sdk" conflicts with INFRA-02 (SDK migration) and INFRA-04 (re-validate after migration). Four of the five Phase-1 success criteria depend on those.

---

## Single run (scope of single-run decision)

| Option | Description | Selected |
|--------|-------------|----------|
| Both: 12c + 13a single run each | Single run per variant on full sweep; hard-tier still runs first | ✓ |
| 12c single, 13a re-run if borderline | Conditional re-run near GATE-05 threshold | |
| 12c single, 13a twice on hard tier always | Mirrors Phase 4's 13e treatment | |

**User's choice:** Both single run.

---

## Namespace (checkpoint dir per variant)

| Option | Description | Selected |
|--------|-------------|----------|
| `_VARIANT_NAME` class attr (Recommended) | Explicit class-level constant | ✓ |
| Derive from `__name__` | Couple dir to classname | |
| Constructor arg | Pass variant_name= at call site | |

**User's choice:** `_VARIANT_NAME` class attr.

---

## diskcache (role)

| Option | Description | Selected |
|--------|-------------|----------|
| Replace custom SHA checkpoint (Recommended) | Swap llm_client.py private cache for `diskcache.Cache` | ✓ |
| New phase-output cache layer | Keep SHA + add phase-level cache | |
| Dep placeholder only | Add to pyproject; defer wiring | |
| Both: swap SHA + phase cache | Aggressive dual change | |

**User's choice:** Replace custom SHA checkpoint.

---

## Enforcement (GATE-06)

| Option | Description | Selected |
|--------|-------------|----------|
| Directory-per-variant, runtime assertion (Recommended) | Assert `_VARIANT_NAME` present in path | ✓ |
| Directory-per-variant, convention only | Review-based | |
| Disable checkpoint for variant runs | Always fresh | |

**User's choice:** Runtime assertion.

---

## Continue prompt

| Option | Description | Selected |
|--------|-------------|----------|
| Next area — 13a scope + guardrail | Discuss standalone copy + guardrail strictness | |
| More on checkpoint/diskcache | Migration of old caches, back-compat | |
| I'm ready for context | Write CONTEXT.md now | ✓ |

**User's choice:** Ready for context.

---

## Claude's Discretion

- 13a file structure (standalone copy of 12c with `_split_component_name` removed, per Spike 001 signature)
- Evidence guardrail strictness (use Spike 001 light guardrail; no structural fallback)
- Prompt constant placement in 13a (inline vs `prompts_v2.py`)
- `VARIANT_SPECS` registration ordering (append-only)

## Deferred Ideas

- Temperature=0.0 / prompt-caching header enforcement (dropped; only revisits if project adopts Anthropic SDK in a future phase)
- Back-compat migration of existing SHA checkpoint files into diskcache
- Phase-output cache layer above LLM-response cache
- `_has_standalone_mention` LLM replacement (already deferred to Phase 5 / EXT-01)
