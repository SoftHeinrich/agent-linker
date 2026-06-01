---
phase: 10-scaffolding
plan: 02
subsystem: linkers-experimental
tags: [clean-02, helper-extraction, frozen-compat, gate-06]
requires:
  - src/llm_sad_sam/linkers/experimental/s_linker13.py (FROZEN — source of helpers)
  - src/llm_sad_sam/linkers/experimental/s_linker13d.py (FROZEN — source of mention-type helpers)
provides:
  - src/llm_sad_sam/linkers/experimental/helper_v3.py (versioned sibling, 6 public helpers + MENTION_TYPES)
affects:
  - Phase 10 Plan 03 (s_linker13_clean) — imports these helpers verbatim
  - Phase 12 PROMPT-02 — semantic trims will happen here, not at the linker level
tech-stack:
  added: []
  patterns: [versioned-sibling-module, verbatim-extraction, lift-self-to-params]
key-files:
  created:
    - src/llm_sad_sam/linkers/experimental/helper_v3.py
  modified: []
decisions:
  - "build_component_profile signature: self.model_knowledge / self.doc_knowledge lifted to explicit parameters. The body is otherwise byte-identical — the original truthy guards on both attributes already cover None, so the lifted call site matches behaviour exactly."
  - "Single file (helper_v3.py) chosen over the (a) doc-knowledge / (b) coref-alias / (c) ambiguity / (d) misc per-concern grouping. CONTEXT.md explicitly permits this when one file suffices; the extracted surface is six functions, splitting would add import overhead without clarity."
  - "MENTION_TYPES is duplicated, not re-exported from SLinker13d, to keep helper_v3 free of any dependency on the frozen variant class. The two frozensets are byte-identical (verified via test)."
  - "EXT-01 (LLM replacement of has_standalone_mention) and EXT-02 (drop dotted-path guard) remain deferred per PROJECT.md Key Decisions; the helper is extracted verbatim, RISKY-KEEP intact."
metrics:
  duration: ~12min
  completed: 2026-05-31
---

# Phase 10 Plan 02: Helper v3 Extraction (CLEAN-02) Summary

Extracted the six pure-function helpers currently inlined inside `s_linker13.py` and `s_linker13d.py` into a new versioned sibling, `src/llm_sad_sam/linkers/experimental/helper_v3.py`, byte-identical bodies, frozen files untouched, GATE-06 clean. Plan 10-03 (`s_linker13_clean`) can now `from llm_sad_sam.linkers.experimental.helper_v3 import …` instead of re-inlining.

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Create helper_v3.py with extracted pure helpers | `eae3028` | src/llm_sad_sam/linkers/experimental/helper_v3.py |

## Exported Public Surface

All exports are extracted verbatim — bodies are character-for-character copies aside from the mechanical edits noted in the "Signature change" column.

| Export | Origin (file:lines) | Signature change |
|--------|---------------------|------------------|
| `MENTION_TYPES` (frozenset) | `s_linker13d.py:95` | None — duplicated verbatim |
| `coerce_mention_type(value)` | `s_linker13d.py:97–110` (`@staticmethod _coerce_mention_type`) | Dropped leading underscore. Body identical (references the module-level `MENTION_TYPES` instead of `SLinker13d.MENTION_TYPES` — same set). |
| `format_mention_string(mention_type, alias_used=None)` | `s_linker13d.py:112–133` (`@staticmethod _format_mention_string`) | Dropped leading underscore. Body identical. |
| `build_component_profile(comp_name, model_knowledge, doc_knowledge)` | `s_linker13.py:594–615` (`def _build_component_profile(self, comp_name)`) | Dropped leading underscore. `self.model_knowledge` and `self.doc_knowledge` lifted to explicit parameters. Body identical (original guards `if self.model_knowledge` / `if self.doc_knowledge` are truthy checks that already cover None). |
| `parse_snum(val)` | `s_linker13.py:1107–1117` (`@staticmethod _parse_snum`) | Dropped leading underscore. Body identical. |
| `has_standalone_mention(comp_name, text)` | `s_linker13.py:1119–1147` (`@staticmethod _has_standalone_mention`) | Dropped leading underscore. Body identical (regex-backed Spike 002 RISKY primitive — KEEP intact; EXT-01 / EXT-02 deferred). |
| `get_comp_names(components)` | `s_linker13.py:1149–1152` (`@staticmethod _get_comp_names`) | Dropped leading underscore. Body identical. |

## Frozen-Compat Confirmation

```
$ git diff --quiet -- \
    src/llm_sad_sam/linkers/experimental/s_linker13.py \
    src/llm_sad_sam/core/data_types_v2.py \
    src/llm_sad_sam/core/document_loader_v2.py \
    src/llm_sad_sam/pcm_parser_v2.py \
    src/llm_sad_sam/linkers/experimental/prompts_v2.py
$ echo $?
0
```

`s_linker13d.py` is also untouched (only read, not modified) although the plan's check list named only the five files above.

## GATE-06 Confirmation

```
$ grep -E "Reencoding|FreeSWITCH|kurento|Recording Service|Redis PubSub|HTML5 Server" \
    src/llm_sad_sam/linkers/experimental/helper_v3.py
$ echo $?
1     # (grep exits 1 = no matches — GATE-06 clean)
```

No benchmark component names appear in code, docstring, comments, or examples.

## Verification Results

| Check | Status |
|-------|--------|
| `python -c "from llm_sad_sam.linkers.experimental import helper_v3"` exits 0 | PASS |
| All six required public names exported (`dir(helper_v3)` includes them) | PASS |
| `MENTION_TYPES == frozenset({proper_case, lowercase, dotted_path, via_alias, indirect})` | PASS |
| `format_mention_string` matches all 6 EXPECTED entries from `tests/test_s_linker13d_parity.py` | PASS (all 6 byte-identical) |
| `coerce_mention_type("garbage_enum_value")` raises ValueError matching `/Unknown mention_type/` | PASS |
| `coerce_mention_type("lowercase")` returns `"lowercase"` | PASS |
| Module docstring contains "helper_v3 — extracted helpers for v2.1 _clean variant chain (Plan 10-02)" | PASS |
| Module docstring contains "v2.0 helpers" and "are NOT modified" | PASS |
| `git diff --stat` on the 5 listed frozen files reports zero changes | PASS |
| `grep -E "Reencoding\|FreeSWITCH\|kurento\|Recording Service\|Redis PubSub\|HTML5 Server" helper_v3.py` returns nothing | PASS (GATE-06 clean) |
| `pytest tests/test_s_linker13d_parity.py -q` still passes (sanity — frozen files unchanged) | PASS (10 passed in 0.04s) |

## Behavioural Spot-Check

End-to-end behaviour sanity-checked on `build_component_profile` with both `SimpleNamespace` mocks:

- Ambiguous component with single alias → produces `"- Classification: AMBIGUOUS — …"` plus `'- Known aliases: "srv"'` (matches `s_linker13` original output character-for-character).
- Distinctive component, no aliases → `"- Classification: DISTINCTIVE — …"` plus `"- Known aliases: none"`.
- `model_knowledge=None`, `doc_knowledge=None` → falls through to DISTINCTIVE + no aliases (mirrors original truthy guards).

`has_standalone_mention`, `parse_snum`, `get_comp_names` all return the documented values on representative inputs.

## Deviations from Plan

None auto-fixed (no Rule 1/2/3 events). One clarifying choice logged here:

1. **`MENTION_TYPES` duplicated rather than re-imported from `SLinker13d`.** The plan's required imports list does not include the frozen variant class, and importing it would couple `helper_v3` to a frozen file's class symbol. The frozenset is duplicated byte-for-byte; correctness is verified by `assert h.MENTION_TYPES == frozenset({…})` in the plan's own automated verification command. Equivalent semantics, cleaner module boundary, consistent with the "extract a clean helper surface" intent of CLEAN-02.

## Authentication Gates

None. Pure source extraction — no LLM calls, no network, no `.env` reads.

## Known Stubs

None. All six helpers are full extractions with documented behaviour, not placeholders.

## Self-Check: PASSED

- File `src/llm_sad_sam/linkers/experimental/helper_v3.py`: FOUND
- Commit `eae3028`: FOUND in `git log`
- Frozen files diff-clean: CONFIRMED via `git diff --quiet`
- GATE-06 grep: CLEAN
- Parity tests (`tests/test_s_linker13d_parity.py`): 10/10 PASS
