# Phase 1: Baseline and Infrastructure - Context

**Gathered:** 2026-04-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Capture a reproducible `s_linker12c` baseline, rework checkpoint storage to be per-variant (no cross-variant leak), and ship `s_linker13a` (Spike 001 LLM trailing-word enrichment, replacing `_split_component_name` + structural candidate-gate) passing the dual floor — giving a clean starting point for the 13-series ablation chain.

**In scope:** 12c baseline run, checkpoint dir namespacing via `_VARIANT_NAME`, diskcache migration, 13a variant file + registration, hard-tier then full-sweep validation.

**Out of scope (this phase):** SDK migration (INFRA-02/INFRA-04 struck — see decision D-01), prompt-caching/temperature enforcement tooling, any 13b+ variant work.

</domain>

<decisions>
## Implementation Decisions

### Scope Change — SDK Migration
- **D-01:** Strike INFRA-02 and INFRA-04 from Phase 1. Backend stays on `claude -p` subprocess (current `LLMBackend.CLAUDE` path in `llm_client.py`). User rationale: "current backend is fine". Claude CLI exposes no temperature flag and no caller-controlled cache headers, so the success criteria tied to those cannot be met via CLI; requirements are removed from this phase rather than faked. These do not move to a later phase — they are dropped.
- **D-01a:** REQUIREMENTS.md and ROADMAP.md must be updated to reflect this before Phase 1 plans are drafted (planner should surface this as a prerequisite task).

### Baseline Protocol
- **D-02:** Single run on full 5-project sweep for BOTH the 12c baseline (INFRA-01) and the 13a comparison (VAR-01) — no N-run median, no best-of-N, no averaging. GATE-05 hard-tier-first still applies: teammates + BBB run first; if 13a regresses >1pp on either, rework instead of proceeding to full sweep. No automatic re-run on borderline deltas.

### Checkpoint Namespacing (INFRA-05)
- **D-03:** Each linker class declares a module/class-level `_VARIANT_NAME = "s_linkerXX"` constant. `_checkpoint_dir` is derived from `_VARIANT_NAME` (e.g. `results/phase_cache/<_VARIANT_NAME>/`). No hardcoded `"s_linker12c"` string anywhere outside the 12c file. `s_linker13a` declares `_VARIANT_NAME = "s_linker13a"`.
- **D-04:** `run_ablation.py` passes/consumes variant identity through `_VARIANT_NAME` (or reads it off the class), never by string literal.

### diskcache Role (INFRA-03)
- **D-05:** Replace the custom SHA-256 file checkpoint in `src/llm_sad_sam/llm_client.py` (`_prompt_hash`, `_checkpoint_path`, `_load_cached_response`, `_save_cached_response`, `_query_checkpoint`) with `diskcache.Cache`. Cache key = same SHA-256 of prompt text (stable across migration). External `LLMBackend.CHECKPOINT` API unchanged — drop-in. Existing on-disk caches from prior runs are not migrated (single-run baseline means they get regenerated; no back-compat burden).
- **D-06:** `tabulate>=0.9.0` is a pyproject dep but is not exercised in Phase 1 — it is consumed in Phase 5 (PROMO-03 ablation table). Phase 1 only adds it to `pyproject.toml`.

### GATE-06 Enforcement (per-variant independent runs)
- **D-07:** Runtime assertion in the checkpoint constructor: if `self._checkpoint_dir` path does not contain `self._VARIANT_NAME`, raise immediately. Fail-fast, not a lint rule. Applies to every variant that touches the CHECKPOINT backend.

### Claude's Discretion
- 13a file structure — full standalone copy of `s_linker12c.py` with `_split_component_name` fully deleted and `_enrich_trailing_words` replaced per Spike 001's `fully_llm_driven(...)` signature. Matches user's stated standalone-file preference (per MEMORY.md).
- Evidence guardrail strictness — use the light guardrail validated in Spike 001 tests 2+3 (reject alias if word absent from cited sentence; reject if cited sentence contains full component name). No fallback to structural splitter — that's the rule being removed.
- Prompt constant placement in 13a — inline or from `prompts_v2.py`, planner decides. Taboo audit required either way (GATE-04).
- `CANONICAL_VARIANTS` and `VARIANT_SPECS` registration order in `run_ablation.py` — append at end of existing list.

### Folded Todos

None — no pending todos crossed Phase 1 scope at this time.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project spec
- `.planning/PROJECT.md` — Key Decisions (base=12c, zero non-trivial rules, ablation unit = variant), constraints, Spike summaries.
- `.planning/REQUIREMENTS.md` — INFRA-01, INFRA-03, INFRA-05, VAR-01 (phase-1 scoped); INFRA-02, INFRA-04 now struck per D-01; GATE-01..06.
- `.planning/ROADMAP.md` §Phase 1 — goal, success criteria (note: criterion #2 is void per D-01 and should be removed when ROADMAP.md is updated).

### Spikes (to integrate)
- `.planning/spikes/001-llm-trailing-words/README.md` — validated pattern for VAR-01. 4 self-verifying tests, including parity test against structural pipeline.
- `.planning/spikes/001-llm-trailing-words/spike.py` — reference implementation / signature for `fully_llm_driven(...)`.
- `.planning/spikes/002-rules-audit/` — rule classification; confirms `_split_component_name` is REPLACEABLE.

### Research context
- `.planning/research/ARCHITECTURE.md` — pipeline structure, variant layout.
- `.planning/research/STACK.md` — current deps and backend stack.
- `.planning/research/PITFALLS.md` — known failure modes (especially Claude variance).

### Codebase targets
- `src/llm_sad_sam/linkers/experimental/s_linker12c.py` — baseline variant; source for 13a copy and target of `_split_component_name` removal.
- `src/llm_sad_sam/llm_client.py` — checkpoint functions to swap to diskcache (D-05); runtime assertion lives here (D-07).
- `run_ablation.py` — `CANONICAL_VARIANTS`, `VARIANT_SPECS`, baseline invocation.
- `pyproject.toml` — add `anthropic>=0.40.0` (dep only; not imported in Phase 1 per D-01), `diskcache>=5.6.1`, `tabulate>=0.9.0`.
- `BENCHMARK_TABOO.md` — required audit input for any new prompt constant (GATE-04).

### Memory / prior art (auto-memory)
- MEMORY.md — S-Linker10 prompts_v2 pattern; Claude run-to-run variance note; standalone-file preference.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `llm_client.py` LLMBackend.CHECKPOINT — already factored as a pluggable cache-with-fallback. Migration to diskcache is a drop-in at 5 private methods; external API (`query`, `enable_checkpoint`) unchanged.
- `prompts_v2.py` — clean prompt module without dead `CONVENTION_GUIDE` / `WORD_USAGE_PROMPT` per MEMORY.md. 13a prompt for LLM trailing-word discovery can live here or inline.
- `s_linker12c.py` exists and is the documented ICSE baseline (~94% macro F1 per PROJECT.md).
- `run_ablation.py` `VARIANT_SPECS` dict pattern — standard registration point; append-only.

### Established Patterns
- Standalone variant files, duplicated code (no inheritance). 12a/b/c/d/e all follow this. 13a must match.
- Checkpoint dir is currently `results/phase_cache/` (and variants: `phase_cache_gpt`, `phase_cache_variance`, …) — per-run, name-suffixed. `_VARIANT_NAME` pattern generalizes this.
- Single-run sweeps produce `results/ablation_results/ablation_YYYYMMDD_HHMMSS.json` — existing directory; INFRA-01 drops its baseline JSON here.
- SHA-256 hash as cache key for prompts — keep this key scheme across diskcache swap so prior caches (if any) could theoretically transfer (not required, per D-05).

### Integration Points
- `diskcache.Cache(path)` replaces `_checkpoint_dir` usage; `.get(key) / .set(key, value)` replaces `_load_cached_response / _save_cached_response`. Atomic writes and size-bounded out of the box.
- Runtime assertion site: `LLMClient.enable_checkpoint` and the `CHECKPOINT` branch of `__init__` — both set `_checkpoint_dir`. Assertion must fire from whichever entry point the variant uses.

</code_context>

<specifics>
## Specific Ideas

- "no replace for anthropic sdk, current backend is fine. single run" — user instruction verbatim. Drives D-01 and D-02.
- 13a success = Spike 001 parity test (test 4) still passes once integrated in pipeline — if parity breaks, 13a is reworked, not force-promoted.

</specifics>

<deferred>
## Deferred Ideas

- Temperature=0.0 enforcement and prompt-caching header assertions — originally INFRA-02 scope; now dropped (D-01). If the project ever moves to the Anthropic SDK, these return as a new requirement, not as a resurrected INFRA-02.
- Back-compat migration of existing on-disk SHA checkpoint files into diskcache — not needed under single-run baseline. Revisit only if a future phase wants to replay old caches.
- Phase-output cache layer (above LLM-response cache) — could amortize re-runs; out of scope while ablation demands independent runs per variant (GATE-06).
- `_has_standalone_mention` LLM replacement — explicitly deferred to Phase 5 keep-decision + EXT-01.

### Reviewed Todos (not folded)
None — no pending todos surfaced for this phase.

</deferred>

---

*Phase: 01-baseline-and-infrastructure*
*Context gathered: 2026-04-24*
