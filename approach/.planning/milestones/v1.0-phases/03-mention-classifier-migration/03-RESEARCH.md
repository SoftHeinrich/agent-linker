# Phase 3 — Research Notes

**Gathered:** 2026-05-29
**Scope:** VAR-04 (`s_linker13d.py`) — replace static 4-regex `_classify_mention` with LLM enum emission piggybacked on `_extract_entities_enriched`, with STRICT enum coercion.

This document is a thin research companion to `03-CONTEXT.md` (which is exhaustive at 230 lines). It cites the three load-bearing artifacts the planner must read for VAR-04 and surfaces the one architectural decision the CONTEXT defers ("how to wire mention_type into the seed pipeline since seed runs in parallel with entity extraction"). For pipeline structure, gate definitions, baselines, or standing-policy questions, read `03-CONTEXT.md` first; it carries 30 locked sub-decisions and is the source of truth.

## 1. The Four Regex Branches Being Replaced

`s_linker13b.py:541-573` defines `_classify_mention(comp_name, text) -> str` with five exit strings (5, not 4 — Spike 003 §"Current" wording was off by one):

| Branch | Predicate | Returns |
|---|---|---|
| L547 | `self._has_standalone_mention(comp_name, text)` (no regex itself but uses `re` internally) | `"proper case, standalone"` |
| L552-554 | `re.search(rf'\b{re.escape(comp_lower)}\b', text)` AND in-dotted-context check via `re.finditer` | `"lowercase, inside dotted path"` |
| L552-563 | `re.search(rf'\b{re.escape(comp_lower)}\b', text)` AND not in dotted | `"lowercase mention"` |
| L566-571 | `self.doc_knowledge.aliases` scan with `re.search(rf'\b{re.escape(alias)}\b', ...)` | `f'via known alias "{alias}"'` |
| L573 | fallback | `"indirect/unclear match"` |

The 4 `re.` calls all live in this method (L552, L554, L568). `_has_standalone_mention` (L1031) is a separate primitive — out of scope for VAR-04 (EXT-01 → Phase 5).

## 2. Spike 003 Reference Contract

`.planning/spikes/003-llm-mention-classifier/spike.py` (135 lines) is the validated reference. Three load-bearing pieces 13d copies verbatim:

- `MENTION_TYPES = {"proper_case", "lowercase", "dotted_path", "via_alias", "indirect"}` (L34) — copy as a `frozenset` per D-21.
- `format_mention(mention_type, alias_used)` (L37-52) — copy as `_format_mention_string` on `SLinker13d`; produces the byte-identical 5 strings.
- 4 self-verifying tests (L69-122) — adapt as `tests/` (or inline) per CONTEXT.md §"Claude's Discretion".

What 13d does **NOT** copy from the spike: `consume_candidate()`'s lenient pattern (`if mt not in MENTION_TYPES: mt = "indirect"`). CONTEXT D-21a / D-21b are explicit — production uses **STRICT** coercion (raise `ValueError`). The spike's lenient path stays in the spike for portability.

## 3. Parity-Probe Lesson (Phase 2 D-13a)

`02-02-SUMMARY.md` documents the empirical pattern: **a code edit can be byte-identical at the classification stage and still drift F1 5-6pp on BBB** through prompt-cache-stream timing perturbation. The 13c→13b BBB delta (-0.057, full-sweep) had byte-identical `model_knowledge.ambiguous_names` across both variants — proven by the 5/5 parity probe in Plan 02-02.

**Direct implication for VAR-04:** even if the LLM emits exactly the right mention_type for every case and the formatter produces byte-identical strings (the structural protection from D-21d), BBB may still drift up to ~6pp from a one-time prompt-cache miss on the modified `_extract_entities_enriched` prompt (the prompt grows by ~80-120 tokens). The 6pp BBB carry-over (D-24) admits this band. Treat any BBB drop ≤ 6pp as a timing-stream artifact per D-27b, not a classification-coverage failure.

**What this changes for the plan:** the byte-identical-string unit test (D-21d) is necessary but **not sufficient** to predict the full-sweep outcome. It guards against classification-coverage bugs (LLM emits wrong enum → downstream prompt drifts) but cannot guard against the cache-stream channel. The full sweep is the only empirical answer.

## 4. The One Open Architectural Decision

CONTEXT.md D-22 names two callsites of `_classify_mention` and says only the planner can decide how to wire `mention_type` into the second one (`_run_seed_validation` L461). The complication: **seed extraction and entity extraction run in parallel** (`s_linker13b.py:188-196`, `_run_parallel({"seed_val", "entity", "coref"})`), so `_extract_entities_enriched`'s LLM output is **not available** to `_run_seed_validation` at the time it formats `match_ctx` (L461).

Three implementation options, in order of preference:

| Option | Adds new LLM call? | Touches files outside 13d? | Plan-CONTEXT alignment |
|---|---|---|---|
| **A. Piggyback on `_run_seed_validation`'s own per-component LLM call.** Extend the disambig prompt to emit `mention_type` per case alongside `meaning`. Format `match_ctx` from a **placeholder** (e.g., just the sentence text) on the first pass, capture the LLM's `mention_type` emissions, and use them downstream if needed. | No (extends an existing call) | No | Best fit for D-20a "no new LLM call" wording. |
| **B. Add a small dedicated piggyback classification pass inside 13d**, mirroring the entity-extraction call's prompt shape, scoped to seed sentences only. Cleaner separation but adds one LLM call (one batch per dataset, ~1 query). | Yes (one batch) | No | Violates D-20a literally but is the cleanest parallel-dict source. |
| **C. Extend `ILinker3.extract()` to emit `mention_type`.** | No (extends an existing call) | **Yes** — ILinker3 is shared with sibling linkers (`ilinker1`, `ilinker2`). | Rejected — touches files outside 13d, breaks standalone-variant discipline. |

**Recommended default for the planner: Option A.** It is the only option that satisfies (a) D-20a "no new LLM call", (b) the standalone-file discipline (no edits to ILinker3 or sibling files), and (c) the parity guarantee (the formatter consumes the captured enum directly). The planner is free to choose Option B if it surfaces a complication during implementation — the resulting LLM-call delta is one batch per dataset (sub-second cache hit on re-runs).

Note: `_run_seed_validation` (L461) currently embeds `match_ctx` *before* the LLM call (the prompt is built using `match_ctx`, then sent). Option A requires sending a placeholder string on the first pass, then on a follow-up (or by accepting the once-per-component LLM call's emission both for `mention_type` AND for `meaning`). The simplest implementation: change `match_ctx = self._classify_mention(...)` to `match_ctx = candidate.sentence_text` (no classification — let the LLM see the raw sentence and decide both `mention_type` and `meaning` in one shot), then format the downstream string via `_format_mention_string` only where `EvidenceBundle.mention_type` is read (i.e., not at the seed-disambig site, where the raw sentence is sufficient signal).

**This means the seed-disambig site MAY not need a `mention_type` at all** — the LLM has the sentence text in the case_lines, so the classification hint is redundant for the disambig decision. If that reading holds, D-22's "two consumers" reduces to "one consumer" (`_build_evidence_bundle`), and the seed-pipeline plumbing problem evaporates. **The planner should verify this reading in Task 1 by running a quick A/B on TM** (the seed-disambig dataset most sensitive to the `Mention: {match_ctx}` line per ROADMAP criterion #4 — string coupling).

## 5. References

- `.planning/phases/03-mention-classifier-migration/03-CONTEXT.md` — 30 locked decisions, especially D-20 / D-21 / D-22 / D-24 / D-27.
- `.planning/spikes/003-llm-mention-classifier/{README.md,spike.py}` — reference contract.
- `.planning/spikes/002-rules-audit/` — confirms `_classify_mention` is REPLACEABLE.
- `.planning/phases/02-ambiguity-cleanup/02-01-SUMMARY.md` — 13b shipped 0.9519 macro; parent baseline for ΔF1 sanity (D-24b).
- `.planning/phases/02-ambiguity-cleanup/02-02-SUMMARY.md` — D-13a parity-probe lesson; BBB 6pp band justification.
- `src/llm_sad_sam/linkers/experimental/s_linker13b.py:461,541-573,590,725,1031` — the lines this phase touches.
- MEMORY.md — Claude Sonnet only; standalone files; no benchmark-derived prompt examples.

---
*Phase: 03-mention-classifier-migration*
*Research notes: 2026-05-29*
