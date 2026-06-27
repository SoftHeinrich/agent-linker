---
phase: 06
plan: 05
type: execute
subsystem: llm-prompt
requirements: [EXT-01]
tags: [llm-prompt, standalone-mention, alias-aware, gate-06, prompts_v2, ext-01]
provides:
  - "STANDALONE_MENTION_RULES_PRE_FILTERED_ALIAS_AWARE prompt constant (sub-variant pre_alias)"
  - "STANDALONE_MENTION_RULES_LLM_ONLY_ALIAS_AWARE prompt constant (sub-variant sem_alias)"
  - "STANDALONE_MENTION_RULES_PRE_FILTERED_FULL_KNOWLEDGE prompt constant (sub-variant pre_full)"
  - "STANDALONE_MENTION_RULES_LLM_ONLY_FULL_KNOWLEDGE prompt constant (sub-variant sem_full)"
  - "GATE-06 pre-clearance section for the four alias-aware constants"
requires:
  - "Plan 06-01: STANDALONE_MENTION_RULES_PRE_FILTERED and STANDALONE_MENTION_RULES_LLM_ONLY constants (byte-identical)"
  - "06-CONTEXT.md D-07 (alias map + running link map fed into standalone judge)"
  - "06-CONTEXT.md D-11 (integration shape project-agnostic, empty knowledge tolerated)"
affects:
  - "src/llm_sad_sam/linkers/experimental/prompts_v2.py (+110 lines, append-only)"
  - ".planning/phases/06-ext-01-project-agnostic-standalone-mention-llm-primitive/06-GATE-06-AUDIT.md (+55 lines, append-only)"
tech_stack:
  added: []
  patterns: ["str.replace-based placeholder substitution ({KNOWN_ALIASES_BLOCK}, {RUNNING_LINK_MAP_BLOCK})"]
key_files:
  created: []
  modified:
    - "src/llm_sad_sam/linkers/experimental/prompts_v2.py"
    - ".planning/phases/06-ext-01-project-agnostic-standalone-mention-llm-primitive/06-GATE-06-AUDIT.md"
decisions:
  - "Use literal `{KNOWN_ALIASES_BLOCK}` / `{RUNNING_LINK_MAP_BLOCK}` placeholder tokens substituted via `str.replace(...)` (not `.format(...)`) to coexist with literal JSON braces in the template."
  - "RUNNING_LINK_MAP block included ONLY in the two `*_FULL_KNOWLEDGE` constants — alias-only sub-variants (pre_alias, sem_alias) carry only the alias block."
  - "All illustrative examples drawn from BENCHMARK_TABOO safe-SE-textbook domains (Compiler / OS). Abbreviation example uses `SymTbl -> SymbolTable` — does not overlap any benchmark abbreviation."
metrics:
  duration_seconds: 163
  tasks_total: 2
  tasks_complete: 2
  files_modified: 2
  files_created: 0
  completed_date: "2026-05-30"
---

# Phase 06 Plan 05: Alias-aware Standalone-Mention Prompts Summary

**One-liner:** Authored four alias-aware standalone-mention prompt constants in `prompts_v2.py` (pre_alias / sem_alias / pre_full / sem_full) using `{KNOWN_ALIASES_BLOCK}` and `{RUNNING_LINK_MAP_BLOCK}` placeholders, and appended a GATE-06 pre-clearance audit confirming zero benchmark surface forms under word-bounded mechanical scan.

## What Was Built

Two append-only edits, both passing GATE-06 word-bounded mechanical scan:

### 1. Four new prompt constants in `src/llm_sad_sam/linkers/experimental/prompts_v2.py`

Appended under a new section header `# Tier 1 — Standalone-Mention Detection (EXT-01) — Alias-Aware (Plan 06-05)`, between the Plan 06-01 section and the existing `# Tier 2 — Seed Reference Disambiguation` section. The Plan 06-01 constants `STANDALONE_MENTION_RULES_PRE_FILTERED` and `STANDALONE_MENTION_RULES_LLM_ONLY` remain byte-identical (`git diff` shows zero deletions of their assignment lines).

| Sub-variant tag | Constant name                                              | Placeholders                               |
|-----------------|------------------------------------------------------------|--------------------------------------------|
| `pre_alias`     | `STANDALONE_MENTION_RULES_PRE_FILTERED_ALIAS_AWARE`        | `{KNOWN_ALIASES_BLOCK}` × 1                |
| `sem_alias`     | `STANDALONE_MENTION_RULES_LLM_ONLY_ALIAS_AWARE`            | `{KNOWN_ALIASES_BLOCK}` × 1                |
| `pre_full`      | `STANDALONE_MENTION_RULES_PRE_FILTERED_FULL_KNOWLEDGE`     | `{KNOWN_ALIASES_BLOCK}` × 1 + `{RUNNING_LINK_MAP_BLOCK}` × 1 |
| `sem_full`      | `STANDALONE_MENTION_RULES_LLM_ONLY_FULL_KNOWLEDGE`         | `{KNOWN_ALIASES_BLOCK}` × 1 + `{RUNNING_LINK_MAP_BLOCK}` × 1 |

Placeholders are substituted at call time by Plan 06-06 linker code using `prompt.replace("{KNOWN_ALIASES_BLOCK}", ...)` — NOT `.format(...)` — because the JSON template at the end of each prompt uses literal `{...}` braces. Empty alias/linkmap blocks become the literal lines `(none discovered)` / `(none yet)`, per D-11.

### 2. GATE-06 pre-clearance section in `06-GATE-06-AUDIT.md`

Appended a `## Plan 06-05 — Alias-aware prompt pre-clearance` section containing:

- **(a)** the exact mechanical scan command and its recorded stdout `NO HITS`,
- **(b)** a reviewer-defensibility table covering all 6 illustrative example sentences across the four new constants (Parser/lexer; SymTbl→SymbolTable; compiler.parser.ASTBuilder; Parser-style; Disk I/O→FileSystem; S12: Scheduler + "It"),
- **(c)** an integration-shape generality note (D-11) explaining the prompts are project-agnostic data sinks for project-agnostic upstream outputs,
- a **PASS** decision and open items handed to Plan 06-06.

## How It Was Verified

```
--- Verification 1: 4 assignments ---
4
--- Verification 2: 6 STANDALONE constants total ---
6
--- Verification 3: word-bounded scan on the four NEW constants ---
NO HITS
--- Verification 4: Plan 06-05 header in audit (exactly once) ---
1
```

Plus the Task-1 Python import check passed: `OK: 4 new constants present, placeholders correct` (confirms all 4 constants are module-level attributes, all 4 contain `{KNOWN_ALIASES_BLOCK}`, the 2 `*_FULL_KNOWLEDGE` constants contain `{RUNNING_LINK_MAP_BLOCK}` while the 2 `*_ALIAS_AWARE` constants do NOT).

Plan-level append-only invariant: `git diff` on both modified files shows insertions only (110 + 55 = 165 inserted lines, 0 deletions).

## Mechanical scan stdout (literal)

```
NO HITS
```

Recorded in `06-GATE-06-AUDIT.md` under `### (a) BENCHMARK_TABOO.md mechanical scan (word-bounded, operative check)`.

## Pointers for Plan 06-06

Plan 06-06 introduces `s_linker13g_{pre,sem}_{alias,full}.py`. Use this mapping:

| Sub-variant filename            | Prompt constant to import                                       |
|---------------------------------|-----------------------------------------------------------------|
| `s_linker13g_pre_alias.py`      | `STANDALONE_MENTION_RULES_PRE_FILTERED_ALIAS_AWARE`             |
| `s_linker13g_sem_alias.py`      | `STANDALONE_MENTION_RULES_LLM_ONLY_ALIAS_AWARE`                 |
| `s_linker13g_pre_full.py`       | `STANDALONE_MENTION_RULES_PRE_FILTERED_FULL_KNOWLEDGE`          |
| `s_linker13g_sem_full.py`       | `STANDALONE_MENTION_RULES_LLM_ONLY_FULL_KNOWLEDGE`              |

Substitution recipe at call time:

```python
from llm_sad_sam.linkers.experimental.prompts_v2 import (
    STANDALONE_MENTION_RULES_PRE_FILTERED_ALIAS_AWARE,
    STANDALONE_MENTION_RULES_LLM_ONLY_ALIAS_AWARE,
    STANDALONE_MENTION_RULES_PRE_FILTERED_FULL_KNOWLEDGE,
    STANDALONE_MENTION_RULES_LLM_ONLY_FULL_KNOWLEDGE,
)

# Build the blocks from upstream-discovered values (verbatim, no normalization)
alias_block = "\n".join(f"{term} -> {comp}" for term, comp in aliases.items()) or "(none discovered)"
linkmap_block = "\n".join(f"S{snum}: {comp}" for snum, comp in linkmap.items()) or "(none yet)"

# Substitute via str.replace (NOT .format — the JSON template uses literal braces)
prompt_pre_alias = STANDALONE_MENTION_RULES_PRE_FILTERED_ALIAS_AWARE.replace(
    "{KNOWN_ALIASES_BLOCK}", alias_block
)
prompt_sem_full = (
    STANDALONE_MENTION_RULES_LLM_ONLY_FULL_KNOWLEDGE
    .replace("{KNOWN_ALIASES_BLOCK}", alias_block)
    .replace("{RUNNING_LINK_MAP_BLOCK}", linkmap_block)
)
```

Plan 06-06 also owes a GATE-06 re-scan that covers the four new linker bodies (helper code is in GATE-06 scope) and confirms the alias/linkmap injection helpers echo upstream values verbatim (no normalization, no rewriting, no hand-coded values).

## Deviations from Plan

None — plan executed exactly as written. The action block was copy/pasted byte-for-byte into `prompts_v2.py`; the audit section was appended at end-of-file with `<today>` replaced by `2026-05-30`. No deviations triggered, no architectural decisions needed, no auth gates.

## Commits

| Task | Type   | Hash    | Message                                                              |
|------|--------|---------|----------------------------------------------------------------------|
| 1    | feat   | db0b56c | feat(06-05): add four alias-aware standalone-mention prompts         |
| 2    | docs   | dcf744f | docs(06-05): append GATE-06 pre-clearance for alias-aware prompts    |

## Known Stubs

None. The prompt placeholders `{KNOWN_ALIASES_BLOCK}` / `{RUNNING_LINK_MAP_BLOCK}` are intentional integration-surface markers substituted by Plan 06-06 — not unwired stubs. Plan 06-05's scope is the prompt text only; Plan 06-06 wires the substitution helpers and the linker bodies that consume them.

## Self-Check: PASSED

- `src/llm_sad_sam/linkers/experimental/prompts_v2.py` — FOUND (modified, +110 lines, all 4 constants importable)
- `.planning/phases/06-ext-01-project-agnostic-standalone-mention-llm-primitive/06-GATE-06-AUDIT.md` — FOUND (modified, +55 lines, Plan 06-05 section present)
- Commit `db0b56c` (Task 1) — FOUND in `git log`
- Commit `dcf744f` (Task 2) — FOUND in `git log`
- All 4 plan-level verifications PASS (4 assignments / 6 STANDALONE constants / NO HITS / 1 audit header)
