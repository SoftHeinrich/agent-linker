# Phase 3: Mention Classifier Migration - Context

**Gathered:** 2026-05-29
**Status:** Ready for planning
**Mode:** `gsd-discuss-phase --auto` (recommended defaults selected by Claude; no human Q&A — every decision below is locked with a cited source)

<domain>
## Phase Boundary

Retire the static 4-branch regex `_classify_mention` from the linker chain by shipping one standalone variant:

- **VAR-04 / `s_linker13d.py`** — copy from `s_linker13b.py` (Phase 2 winner; macro F1 0.9519 on the canonical 12c reference, +0.0114 over 12c, BBB clean at -0.005). Extend `_extract_entities_enriched`'s LLM output schema to emit a `mention_type` enum field (5 values, per Spike 003) plus optional `alias_used`. Replace the consumer-side `_classify_mention(comp_name, sentence_text)` call inside `_build_evidence_bundle` (and the same call inside `_run_seed_pipeline` at L461) with a pure formatter that maps the LLM enum to the byte-identical evidence-bundle strings the old regex emitted. Delete `_classify_mention` and its 3 `re` calls (lines 552 / 554 / 568) once no callsite remains.

The variant must pass **GATE-01 dual floor** on the full 5-project sweep:
- **macro F1 ≥ 0.93**
- **no dataset more than 2pp below the 12c Plan 04 baseline (MS 0.984 / TS 0.963 / TM 0.938 / BBB 0.844 / JAB 0.973, macro 0.9405) — except BBB, which gets a 6pp tolerance carried over from the 2026-05-29 standing policy set during Phase 2 closure.**

**Parent baseline for ΔF1 (D-12 carry-over):** ΔF1 vs **13b** for ablation-row purposes (13c was admitted into Phase 2 under the same 6pp BBB loosening, but its full-sweep BBB drift was an artifact of cache-stream timing on byte-identical classification — using 13b as the structural parent for 13d keeps the ablation table reading "what code change moved F1 here"). GATE-01 itself is computed vs **12c** as for every other variant.

**Why s_linker13b (not 13c) as the parent file?** Two-pronged sanity check:
1. The 13c→13b BBB delta (-0.057 on full-sweep, byte-identical classification) is documented evidence that copying from 13c could introduce a hidden cache-stream perturbation that confounds VAR-04's own measurement (D-13a evidence in Plan 02-02 SUMMARY).
2. The cumulative-removal chain still holds: `_is_ambiguous_name_component` and `_is_structurally_unambiguous` were both removed in 13b — copying 13b into 13d preserves both removals. The only difference vs starting from 13c is that 13d inlines `_is_ambiguous_name_component`'s body itself (or alternatively just trusts the wrapper as a 4-line dict lookup if the planner finds it cleaner — see D-19c).

**In scope:** Variant file `s_linker13d.py`, prompt-schema extension on the existing `_extract_entities_enriched` LLM call (no new LLM call — Spike 003 piggyback pattern), `MentionType` enum + strict coercion, parity formatter replacing the 5 regex branch strings, registration in `run_ablation.py`, BENCHMARK_TABOO audit on the new prompt-schema text, hard-tier (TM + BBB) gate then full 5-project sweep, ablation log row for 13d.

**Out of scope (this phase):**
- `_is_strong_alias` / `_get_strong_alias_mappings` removal (VAR-05 → Phase 4)
- `_has_strong_alias_mention` (VAR-06 → Phase 4)
- `_has_standalone_mention` keep/replace decision (Phase 5 / EXT-01)
- Promotion artifact (PROMO-* → Phase 5)
- Any change to the `_classify_components` prompt body beyond the schema additions in `_extract_entities_enriched` (different LLM call, different scope)
- Any change to `_has_standalone_mention`, which is still consumed by `_classify_mention`'s `proper_case` branch — its LLM replacement is explicitly deferred to EXT-01

</domain>

<decisions>
## Implementation Decisions

### Variant File Layout (D-19)
- **D-19:** `s_linker13d.py` is a **standalone file**, full copy of `s_linker13b.py` (NOT 13c), edited in place. No inheritance, no shared helpers module. **Source:** Phase 2 D-08 ("Each variant is a standalone file, full copy of its parent, edited in place — no inheritance, no shared helpers module"); MEMORY.md ("User prefers standalone linker files (duplicate code intentionally, not inheritance chains)"); plus the Plan 02-02 SUMMARY evidence that 13c's BBB drift is a cache-stream artifact (parent-from-13b avoids inheriting that artifact's setup).
- **D-19a:** The 13d module docstring includes `REMOVED_FROM: s_linker13b` and `RULES_REMOVED: ["_classify_mention"]`. The cumulative-removal list is optional (Phase 2 D-claude's-discretion precedent). **Source:** GATE-03 + Phase 2 §"Claude's Discretion".
- **D-19b:** The 13d `__init__` print banner string identifies the variant (e.g. `"SLinker13d (13b + Spike 003 LLM mention-type enum)"`). **Source:** Phase 1 Plan 05 SUMMARY Deviation #3 (banner-not-updated bug); Phase 2 §"Claude's Discretion" (banner must be updated when copying).
- **D-19c:** Because 13d copies from 13b (not 13c), `_is_ambiguous_name_component` still exists as a 4-line wrapper in 13d. The planner may either (i) leave it untouched (still REPLACEABLE per Spike 002 but VAR-03's work — out of scope for VAR-04), or (ii) inline-and-delete it as a zero-risk piggyback on the file copy. Default recommendation: **leave it untouched** — the 13c parity probe proved inlining is byte-identical at the classification stage, so doing it here costs nothing but contaminates the ablation row's `rules_removed` list. **Source:** Phase 2 §"Specifics" (13c was mostly dead-code cleanup); Phase 2 D-08 (one rule per variant).

### Prompt-Schema Extension (D-20)
- **D-20:** Extend the existing `_extract_entities_enriched` LLM prompt (s_linker13b.py:725, output schema currently `{"references": [{"sentence": N_INTEGER, "component": "Name", "matched_text": "text"}]}` — see L692) to emit two additional fields per reference: `"mention_type"` (string enum) and `"alias_used"` (string-or-null). The 5 enum values are exactly the Spike 003 set: `proper_case | lowercase | dotted_path | via_alias | indirect`. **Source:** Spike 003 `spike.py:34` (`MENTION_TYPES = {"proper_case", "lowercase", "dotted_path", "via_alias", "indirect"}`); REQUIREMENTS.md VAR-04 ("LLM enum emission piggybacked on entity-extraction prompt").
- **D-20a:** **No new LLM call is added.** The mention-type emission piggybacks on the existing `_extract_entities_enriched` call — net LLM cost delta zero, per Spike 003 README §Results. **Source:** Spike 003 README ("No new LLM call needed — piggyback on existing `_extract_entities_enriched` prompt … Net LLM cost delta: zero").
- **D-20b:** Prompt-schema text stays **inline** in `s_linker13d.py` (no `prompts_v2.py` edits). **Source:** Phase 1 D-claude's-discretion ("Prompt constant placement in 13a — inline or from prompts_v2.py, planner decides"); Phase 2 D-09 ("Inline prompt constants stay inline. No prompts_v2 module changes in this phase").

### Exact-String Contract (D-21)
- **D-21:** Define a module-level `MENTION_TYPES = frozenset({"proper_case", "lowercase", "dotted_path", "via_alias", "indirect"})` constant (mirrors Spike 003 `spike.py:34`). The LLM-emitted `mention_type` string is coerced via membership check.
- **D-21a:** **STRICT policy on unknown enum: raise `ValueError` immediately** (do not silently fall back to `"indirect"`). Rationale: the ablation purity argument — silently degrading an out-of-enum response to `"indirect"` would hide a prompt-conformance regression as ambiguity-classification noise, exactly the failure mode D-13a / D-15 warn about. Surfacing immediately means a single-character drift in the LLM output produces a load-bearing test failure rather than a 1-2pp F1 hit that would be misread as variance. **Source:** REQUIREMENTS.md VAR-04 ("coercion assertion is present so an out-of-enum LLM response raises immediately rather than silently degrading"); Phase 2 D-15 ("LLM substitution can be inert ... cache-stream timing perturbation. If a regression nonetheless appears, treat it as a classification-coverage issue, not a cache-stream issue").
- **D-21b:** **Document the fallback option (NOT adopted):** Spike 003's reference consumer (`spike.py:62`) uses a lenient pattern: `if mt not in MENTION_TYPES: mt = "indirect"`. This is preserved in the Spike for portability but is **explicitly rejected for the production variant** under D-21a. If a future phase wants the lenient path back (e.g., for GPT cross-model gating per EXT-03), it returns as its own decision, not as a quiet behavioral default in 13d.
- **D-21c:** Parity formatter (`_format_mention_string`) lives inline on `SLinker13d` and produces the exact 5 strings the old `_classify_mention` produced — verified by structural copy from Spike 003 `spike.py:37-52`:
  - `proper_case → "proper case, standalone"`
  - `lowercase → "lowercase mention"`
  - `dotted_path → "lowercase, inside dotted path"`
  - `via_alias` + `alias_used="X" → 'via known alias "X"'` (with embedded double quotes)
  - `via_alias` + `alias_used=None → "via known alias"` (no-quote fallback — preserved verbatim from Spike 003 even though `_classify_mention` in 13b never produced this exact string; harmless because the LLM is asked to populate `alias_used` whenever it emits `via_alias`)
  - `indirect → "indirect/unclear match"`
- **D-21d:** **Byte-identical string parity is a hard acceptance criterion.** Plan 03-* includes a unit test that asserts every downstream prompt string consuming `EvidenceBundle.mention_type` reads byte-identical text for at least one synthetic case per enum branch. **Source:** ROADMAP Phase 3 success criterion #2 ("coercion assertion is present so an out-of-enum LLM response raises immediately"), criterion #4 ("Hard-tier run shows no change in seed-validation rejection rates vs 13c baseline (string coupling verified stable)"); Spike 003 README §"What to Expect" (test 3: "Output strings match `_classify_mention` output byte-for-byte").

### Consumer Migration (D-22)
- **D-22:** Two callsites of `_classify_mention` in `s_linker13b.py` are migrated to read `mention_type` from the candidate object:
  1. `_build_evidence_bundle` (L590 → `mention_type = self._classify_mention(comp_name, candidate.sentence_text)`) — replaced by `mention_type = self._format_mention_string(candidate.mention_type, candidate.alias_used)`.
  2. `_run_seed_pipeline` inner loop (L461 → `match_ctx = self._classify_mention(comp_name, sent.text)`) — this site receives a `SadSamLink` (no `mention_type` field), so the planner has two options: (a) thread `mention_type` through `SadSamLink` from the seed pass (preferred, mirrors entity pipeline), or (b) keep one path that re-derives the string from raw text **only for the seed-disambig prompt** (would re-introduce regex, violating VAR-04's "no regex branches"). **Default:** option (a) — extend `SadSamLink`'s constructor signature (or a parallel dict keyed by `(sentence_number, component_id)`) to carry the enum.
- **D-22a:** `EvidenceBundle.mention_type` is unchanged (still a `str` — the prompt-facing string). The change is purely the source of that string: regex → LLM enum → formatter. Downstream consumers (`_format_evidence` L616-628) need no edit. **Source:** Spike 003 §"How to Run" + §"What to Expect" (test 4: "Consumer functions reference zero regex").
- **D-22b:** Once `_classify_mention` has no callsite remaining, delete the method (s_linker13b.py:541-573) and the `import re` line if no other code in 13d uses `re` (check first; `_has_standalone_mention` and other helpers may still). The variant's `rules_removed` list claims `_classify_mention` exactly. **Source:** REQUIREMENTS.md VAR-04 (`_classify_mention` 4-branch regex replaced by LLM enum emission); GATE-03 (structured docstring).

### Baseline Protocol (D-23)
- **D-23:** **Single run** on full 5-project sweep for 13d. No N-run median, no best-of-N. Per-variant independent run; **do not re-run 12c**; the canonical baseline JSON is `results/ablation_results/ablation_20260528_173020.json` (Phase 1 Plan 04 — MS 0.984 / TS 0.963 / TM 0.938 / BBB 0.844 / JAB 0.973, macro 0.9405). **Source:** Phase 1 D-02 (single-run rule); Phase 2 D-10 (single-run carry-over); STATE.md "D-02 single-run baseline applies — DO NOT re-run 12c" (Phase 3 instruction).

### Standing Policy Carry-Over (D-24)
- **D-24:** **GATE-01 BBB tolerance: 6pp. Other 4 datasets: 2pp. Macro floor: 0.93.** This is the standing policy set on 2026-05-29 in Phase 2 closure (Plan 02-02 SUMMARY §"User Resolution (2026-05-29)") and **inherited by Phase 3+ without further direction**. BBB floor = 0.844 - 0.06 = **0.784**. **Source:** Plan 02-02 SUMMARY §"User Resolution"; STATE.md §"Standing Policy (Phases 3+)" (`GATE-01 BBB tolerance: 6pp`); ROADMAP Phase 2 closure entry.
- **D-24a:** **GATE-05 hard-tier policy under the standing policy:** TM + BBB run first. Auto-approve to full sweep if **both** deltas ≥ -0.01pp vs 12c; marginal (-0.01 to -0.02 on TM, -0.01 to -0.06 on BBB) → halt and flag, surface checkpoint; hard reject (delta < -0.02 on TM or < -0.06 on BBB) → no full sweep, rework. **Source:** Phase 2 D-13b adapted to the wider BBB band; STATE.md "GATE-05 hard-tier auto-approve thresholds carry over (TM ≥ -0.01, BBB ≥ -0.06 with the wider tolerance)".
- **D-24b:** **Compare 13d to BOTH 12c (for GATE-01) AND 13b (for delta-of-delta sanity).** GATE-01 enforcement is vs 12c (per ROADMAP success criterion #3); the ablation-row "ΔF1 vs parent" column is vs 13b (per D-12 carry-over: "Each variant's ΔF1 is computed vs its immediate structural parent" — and 13d's structural parent is 13b per D-19). The 13c row remains in the ablation table from Phase 2; 13d's row is a sibling, not a successor of 13c. **Source:** Orchestrator instructions; D-12 carry-over.

### Checkpoint Namespacing (D-25)
- **D-25:** `s_linker13d` declares `_VARIANT_NAME = "s_linker13d"`. The D-07 runtime assertion in `_checkpoint_dir` carries forward unchanged via the cp from 13b. **Source:** Phase 1 D-03 + D-07; Phase 2 D-11.
- **D-25a:** `run_ablation.py` consumes variant identity through the class constant. Registration is **append-only** — 13d appended after 13c in `CANONICAL_VARIANTS` and `VARIANT_SPECS`. **Source:** Phase 1 D-04; Phase 2 D-11a.

### Variance Re-Run Trigger (D-26)
- **D-26:** Run a second hard-tier pass (cache cleared) **only if** the first hard-tier run lands a marginal flag per D-24a. No variance re-run on auto-approve. No variance re-run on hard reject. If full-sweep fails GATE-01 on BBB at the **6pp** band (< 0.784), surface the failure for user decision — do NOT auto-loosen the standing policy further. **Source:** Phase 2 D-14 (precedent); Plan 02-02 SUMMARY (D-13a / D-14 evidence the marginal-band variance re-run is the recovery channel, not a tolerance-widening one).

### LLM-Substitution Inertness Risk (D-27)
- **D-27:** Apply the Phase 1 / Phase 2 lesson explicitly: an LLM-schema change can be **inert** (produce no behavioral change vs the regex) and still cause downstream FP/FN swings purely through prompt-cache-stream timing perturbation (Spike 001 mechanism, Plan 02-02 D-13a reconfirmation). **For VAR-04 the risk is materially lower than Phase 1's** because:
  - No new LLM call is added — the schema is extended on a call that already exists at the same point in the prompt-cache stream (`_extract_entities_enriched`).
  - The prompt body itself only grows by ~80-120 tokens (two new JSON output fields + their per-enum guidance). Claude's prompt-cache stream is keyed on full prompt text — there will be a one-time cache miss on the modified prompt, but the same-call ordering is preserved.
  - The downstream `mention_type` string is asserted byte-identical to the regex output (D-21d), so the LLMs consuming `EvidenceBundle.mention_type` see identical input.
- **D-27a:** **However, the prompt change WILL invalidate the cached `_extract_entities_enriched` LLM response for every dataset.** Plan 03-* must explicitly clear `results/phase_cache/s_linker13d/` before the hard-tier run (cp from 13b will NOT carry usable caches because the variant name differs — D-25 — but document the intent so it isn't accidentally re-used in a re-run). **Source:** Phase 1 Plan 05 §Issues "Stray smoke-test pickle dir leaked into 13a cache" + general D-07 hygiene.
- **D-27b:** **If 13d full-sweep BBB drifts into the [-0.04, -0.06] band:** treat it as the same D-13a cache-stream-timing artifact, not a classification-coverage failure (Plan 02-02 SUMMARY established this as the empirical pattern for byte-identical-classification edits). The 6pp BBB carry-over admits this band. **Source:** Plan 02-02 SUMMARY §"Evidence for D-13a (timing-stream hypothesis — RECONFIRMED)".

### Taboo Audit (D-28)
- **D-28:** **Real audit, not smoke-test.** The new prompt-schema text introduces multiple new tokens (enum names, per-enum guidance text). Every new prompt constant is run through the same substring-match BENCHMARK_TABOO audit Phase 1 / Phase 2 used. **Source:** GATE-04; Phase 1 Plan 05 §Issues "gui in ambiguity"; Phase 2 Plan 01 SUMMARY §"BENCHMARK_TABOO smoke-audit log" (3-layer → 3-tier carry-over).
- **D-28a:** **Specific hazards to pre-flag for the planner:** the universal-taboo list (BENCHMARK_TABOO.md §"Universal Taboo") includes `client`, `server`, `storage`, `common`, `logic`, `cache`, `auth`, `recording`, `persistence`, `facade`, `database/DB`, `registry`, `UI`, `model`, `preferences`, `conversion`, `validation`, `dedicated`, `cascade`. Any prompt example or guidance phrase ("e.g. lowercase mention of a storage component", "e.g. the cache layer") must avoid these. Use the same safe-SE-textbook placeholders Phase 1 / Phase 2 used: `TaskScheduler`, `Scheduler`, `Dispatcher`, `Broker`, `Parser`, `Lexer`. **Source:** BENCHMARK_TABOO.md; MEMORY.md ("No dataset-specific examples in prompts — data leakage. Use safe SE textbook domains (compiler, OS, e-commerce).").
- **D-28b:** **The Spike 003 example uses `TaskDispatcher` and `Dispatcher` as placeholders** (`spike.py:18`) — these are clean. Reuse the same placeholder set in 13d's prompt-schema text. **Source:** Spike 003 `spike.py` docstring example.

### Ablation Log Row (D-29)
- **D-29:** One new row in the ablation table (`PROMO-03`) for 13d. Records: per-dataset F1 (5 columns), macro F1, ΔF1 vs 12c (5 columns + macro), ΔF1 vs **13b** (1 column for macro per D-24b), `rules_removed=["_classify_mention"]`, FP-by-phase breakdown (seed / entity / coref). Markdown via `tabulate` (LaTeX output deferred to Phase 5). **Source:** Phase 2 D-17; REQUIREMENTS.md PROMO-03; ROADMAP Phase 3 success criterion #3.

### Wave Structure (D-30)
- **D-30:** **One sequential plan** in this phase (single variant). Phase 2's two-plan structure does not apply — VAR-04 is a single rule removal. The plan covers: (a) prompt-schema extension + parity formatter + consumer migration + `_classify_mention` deletion, (b) registration in `run_ablation.py`, (c) BENCHMARK_TABOO audit, (d) byte-identical-string parity unit test, (e) hard-tier (TM + BBB) gate, (f) full 5-project sweep, (g) ablation row generation. **Source:** ROADMAP Phase 3 (1 variant = 1 plan); REQUIREMENTS.md VAR-04 (single requirement, no sub-criteria); Phase 1 Plan 05 + Phase 2 Plans 02-01 / 02-02 precedent (one plan per variant).

### Claude's Discretion
- Exact docstring wording (must include `REMOVED_FROM: s_linker13b` and `RULES_REMOVED: ["_classify_mention"]`; the cumulative list is optional).
- Whether to thread `mention_type` through `SadSamLink` constructor (default per D-22 option a) or to maintain a parallel `{(snum, comp_id): mention_type}` dict in the seed pipeline — planner decides during implementation. Both are valid; the dataclass extension is cleaner but touches more files.
- Whether the parity unit test lives in `tests/` (preferred — discoverable by `pytest`) or inline as a `if __name__ == "__main__"` test at the bottom of `s_linker13d.py` (Spike 003 pattern). Default: `tests/` for discoverability; Spike pattern is acceptable if the file count must stay minimal.
- Exact prompt-schema wording for the two new fields. Constraints: (a) D-28a taboo-clean placeholders only, (b) emit the enum value verbatim (no synonyms), (c) populate `alias_used` whenever `mention_type == "via_alias"`. Planner authors the text; the schema constraint is fixed.
- Whether to clear `results/phase_cache/s_linker13d/` between hard-tier and full sweep. **Default: no clearing** (matches Phase 2 D-claude's-discretion precedent — per-dataset checkpoints are independent and full sweep extends hard-tier).
- Whether to update the `__init__` print banner exact string (must reference 13d; precise wording free).

### Folded Todos

None — STATE.md "Pending Todos" is empty.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents (`gsd-phase-researcher`, `gsd-planner`) MUST read these before planning or implementing.**

### Project specs (gate definitions, requirement IDs, standing policy)
- `.planning/PROJECT.md` — Core Value (macro F1 ≥ 93% or reject), Key Decisions (base=12c, ablation unit = variant, standalone files), constraints.
- `.planning/REQUIREMENTS.md` — **VAR-04** (Phase 3 scope: `s_linker13d.py` — Spike 003 integration with enum-contract test); **GATE-01..GATE-06** (every variant must satisfy); REQUIREMENTS.md `Traceability` row "VAR-04 | Phase 3 | Pending".
- `.planning/ROADMAP.md` §Phase 3 (lines 62-72) — goal, depends-on, success criteria #1-4.
- `.planning/STATE.md` §"Standing Policy (Phases 3+)" — 6pp BBB carry-over, 2pp others, 0.93 macro floor (lines 36-41); D-02 single-run baseline reminder.

### Phase 1 inheritance (decisions that carry forward)
- `.planning/phases/01-baseline-and-infrastructure/01-CONTEXT.md` — D-02 (single-run baseline), D-03/D-04 (`_VARIANT_NAME` discipline), D-07 (runtime assertion).
- `.planning/phases/01-baseline-and-infrastructure/01-05-SUMMARY.md` — Spike 001 lessons (LLM-substitution inertness, prompt-cache-stream timing perturbation, BBB variance), §"Gate Resolution (2026-05-28)" + §"Variance Re-Run (2026-05-16)" precedent.

### Phase 2 inheritance (decisions that carry forward)
- `.planning/phases/02-ambiguity-cleanup/02-CONTEXT.md` — D-08 (standalone file pattern), D-09 (inline prompts), D-10 (single-run sweep), D-11 (`_VARIANT_NAME` per variant), D-12 (ΔF1 vs parent for ablation; vs 12c for gate), D-13 (BBB tolerance), D-13b (GATE-05 thresholds), D-14 (variance re-run trigger), D-15 (LLM-substitution inertness risk), D-17 (ablation row schema), D-18 (sequential plans).
- `.planning/phases/02-ambiguity-cleanup/02-01-SUMMARY.md` — 13b shipped clean (macro +0.0114, BBB -0.005); evidence pure-removal does not exhibit BBB perturbation (D-13a); canary probe pattern (`layer1.pkl` inspection); BENCHMARK_TABOO `layer` substring false-positive lesson.
- `.planning/phases/02-ambiguity-cleanup/02-02-SUMMARY.md` — D-13a RECONFIRMED via 5/5 parity probe; user 6pp BBB resolution on 2026-05-29; documented evidence that byte-identical-classification edits can still drift F1 5-6pp on BBB via cache-stream perturbation.

### Spike (validated implementation pattern)
- `.planning/spikes/003-llm-mention-classifier/README.md` — verdict VALIDATED; piggyback-on-existing-LLM-call pattern; net LLM cost delta zero; 4 self-verifying tests (enum branches, unknown→fallback, byte-identical strings, zero regex references).
- `.planning/spikes/003-llm-mention-classifier/spike.py` — reference implementation: `MENTION_TYPES` constant (L34), `format_mention()` parity formatter (L37-52), `consume_candidate()` lenient pattern (L55-64; rejected for production per D-21b), all 4 unit tests (L69-122).
- `.planning/spikes/002-rules-audit/` — confirms `_classify_mention` is REPLACEABLE (not RISKY, not ESSENTIAL).
- `.planning/spikes/001-llm-trailing-words/README.md` — Phase 1 precedent for variant-shipping protocol + evidence-guardrail pattern (relevant only as a template; Spike 003 uses a strict-coercion variant of the same cite-evidence schema).

### Codebase targets (lines to read/edit)
- `src/llm_sad_sam/linkers/experimental/s_linker13b.py` — **copy source for 13d**.
  - `EvidenceBundle.mention_type` field (L46): comment notes the 5 regex-derived strings — these are the byte-identical targets for D-21d.
  - L461: `_run_seed_pipeline` callsite — `match_ctx = self._classify_mention(comp_name, sent.text)` — migration site (D-22 option a).
  - L541-573: `_classify_mention` method definition — to delete in 13d (4 regex branches: standalone, lowercase, dotted_path, alias).
  - L590: `_build_evidence_bundle` callsite — `mention_type = self._classify_mention(comp_name, candidate.sentence_text)` — primary migration site (D-22).
  - L692: existing `_extract_entities_enriched` LLM output schema `{"references": [{"sentence": N_INTEGER, "component": "Name", "matched_text": "text"}]}` — schema extension site (D-20).
  - L714: `matched = ref.get("matched_text", "")` — pattern for reading new `mention_type` / `alias_used` fields off the LLM response.
  - L725: `_extract_entities_enriched` method definition — host of the LLM call whose prompt schema is extended.
  - `_VARIANT_NAME` constant and D-07 assertion (in the `_checkpoint_dir` property carried over from 12c) — preserve as-is via the copy; only `_VARIANT_NAME` value changes to `"s_linker13d"`.
- `run_ablation.py` — append `s_linker13d` after `s_linker13c` in `CANONICAL_VARIANTS` and `VARIANT_SPECS`. Same registration shape as Phase 1 Plan 05 / Phase 2 Plans 02-01, 02-02.
- `BENCHMARK_TABOO.md` — full project list + universal-taboo list; D-28a hazard list for prompt-schema text.

### Research context (background, not action)
- `.planning/research/ARCHITECTURE.md` — pipeline structure (Tier 1 / Tier 2 / coref / boundary-filter wave); locates `_extract_entities_enriched` and `_build_evidence_bundle` in the entity wave.
- `.planning/research/PITFALLS.md` — Claude run-to-run variance documentation backing D-27.

### Memory / prior art
- MEMORY.md — standalone-file preference (D-19); Spike-001-style LLM-substitution inertness (D-27); Claude run-to-run variance pattern; "No dataset-specific examples in prompts — data leakage" (D-28a); GPT compatibility is a side concern, not a gate (out of scope for Phase 3).
- `.planning/STATE.md` — Phase 2 closure timestamp (2026-05-29), 12c full-sweep baseline JSON path (`results/ablation_results/ablation_20260528_173020.json`), standing policy section.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_VARIANT_NAME` pattern + D-07 runtime assertion (Phase 1 INFRA-05) — carries forward unchanged via the cp from 13b. 13d sets the constant to `"s_linker13d"`.
- `run_ablation.py` `CANONICAL_VARIANTS` / `VARIANT_SPECS` append-only registration — exercised in Phase 1 Plan 05 and Phase 2 Plans 02-01 / 02-02; same shape for 13d.
- 12c full-sweep baseline JSON (Phase 1 Plan 04, `results/ablation_results/ablation_20260528_173020.json`) — reuse for ΔF1 vs 12c per D-23 + D-24b. **Do not re-run 12c.**
- 13b full-sweep baseline JSON (Phase 2 Plan 02-01, `results/ablation_results/ablation_20260528_190916.json`) — reuse for ΔF1 vs 13b per D-24b. **Do not re-run 13b.**
- `tabulate` dep (Phase 1 D-06, first exercised in Phase 2 D-17) — reused for the 13d ablation row.
- `Spike 003 `format_mention()`` (`.planning/spikes/003-llm-mention-classifier/spike.py:37-52`) — drop-in parity formatter; copy verbatim into 13d as `_format_mention_string` (instance method) or `_format_mention_string` (staticmethod). Spike-side `consume_candidate()` lenient pattern is **rejected** per D-21a.

### Established Patterns
- Standalone variant files (12a/b/c/d/e + 13a/b/c) — duplicated code is the project's reproducibility artifact.
- Append-only registration in `run_ablation.py`.
- Per-variant pickle cache namespacing under `results/phase_cache/<_VARIANT_NAME>/<dataset>/`; D-07 fail-fast assertion catches namespace bugs at construct time.
- Hard-tier-first (TM + BBB) → full 5-project sweep gate sequence.
- Inline prompt constants; BENCHMARK_TABOO substring audit on every new prompt text.
- Single-run sweeps (no N-run median, no best-of-N); variance re-run only on D-26 marginal-band trigger.

### Integration Points
- `_extract_entities_enriched` (s_linker13b.py:725) is the LLM call whose output schema extends. The existing call site reads each reference's `sentence`, `component`, and `matched_text` (L714); the new code reads two additional keys (`mention_type`, `alias_used`) per reference and writes both onto whatever candidate dataclass holds the extraction output (planner identifies during implementation; the existing per-reference loop at L714 is the obvious surgical point).
- `_build_evidence_bundle` (s_linker13b.py:575) is the **single primary consumer** of `_classify_mention`. After migration its `mention_type = …` line reads from the candidate (D-22a). The downstream `_format_evidence` (L616) consumes the resulting `EvidenceBundle.mention_type` string without further change.
- `_run_seed_pipeline` (L631-516, inner loop at L461) is the **second consumer**. The seed pipeline currently builds `SadSamLink` objects (no mention-type field). Either (a) extend `SadSamLink` to carry a mention-type (planner default, mirrors entity pipeline), or (b) build a parallel `{(snum, comp_id): mention_type}` dict during seed extraction. Both paths are clean; (a) is preferred.
- `prompts_v2.py` is **not edited** in this phase (D-20b). All new prompt-schema text stays inline in `s_linker13d.py`.

### Slip-Channel / Failure Modes to Pre-Watch
- **Schema-conformance slip:** the LLM occasionally emits an unknown enum value or omits the field entirely. D-21a says raise immediately — but the planner must decide where the assertion fires (read site, vs at the `_extract_entities_enriched` JSON-parse loop). Recommendation: in the parse loop, so the failure surfaces with the offending dataset/sentence in the traceback.
- **`alias_used` field missing for `via_alias`:** the formatter D-21c falls back to `"via known alias"` (no quotes) when `alias_used` is None — this is a Spike-003-preserved branch. Document this so a planner doesn't mistakenly raise on missing `alias_used`.
- **Cache-stream timing on BBB:** D-27 lessons; the 6pp tolerance (D-24) admits the [-0.04, -0.06] band as a documented limitation if it occurs. If 13d full-sweep BBB drops > 6pp, surface for user direction per D-26.
- **Seed-pipeline string coupling:** ROADMAP Phase 3 success criterion #4 specifically calls out "no change in seed-validation rejection rates vs 13c baseline (string coupling verified stable)". The seed-disambig prompt at L471-483 embeds `Mention: {match_ctx}` — if the migrated mention strings drift even one character vs the regex output, the seed-disambig LLM's behavior may shift. The byte-identical parity test (D-21d) is the structural protection; the criterion-#4 measurement is the empirical proof.

</code_context>

<specifics>
## Specific Ideas

- **The 6pp BBB tolerance is now standing policy**, not a one-time exception. Phase 3 inherits it; Phase 4 and Phase 5 are expected to inherit it absent further direction (STATE.md §"Standing Policy (Phases 3+)").
- **13d is the cleanest test of "schema extension on an existing LLM call".** Phase 1 (Spike 001) added a new LLM call → BBB perturbation. Phase 2 13b/13c removed synchronous code → 13b clean, 13c byte-identical classification but BBB drifted (cache-stream artifact, D-13a). 13d *extends* an existing call's output schema without adding a new call — it is the first variant in the chain whose LLM cost delta is exactly zero. If 13d BBB lands within the 2pp original tolerance (i.e., the 6pp carry-over is *insurance* not exercised), that is fresh evidence the BBB perturbation pattern is sensitive specifically to call-count or call-ordering changes, not to prompt-content changes.
- **Conversely, if 13d BBB still drifts > 2pp despite no new call:** that is evidence the BBB perturbation is driven by prompt-content changes (one-time cache miss on the longer prompt), not call-ordering. Either reading is publishable evidence for the methodology writeup (PROMO-04, Phase 5).
- **STRICT enum coercion (D-21a, NOT the Spike's lenient pattern) is the right safety knob for ablation purity.** A silent fallback to `"indirect"` would convert a prompt-conformance regression into a 1-2pp F1 hit that looks like variance — exactly the failure mode Phase 1 / Phase 2 cache-stream evidence shows is hardest to debug. Failing fast is the ablation-purity discipline.

</specifics>

<deferred>
## Deferred Ideas

- **LLM-side replacement of `_has_standalone_mention`** — the `proper_case` branch of the original `_classify_mention` (s_linker13b.py:547-548) delegates to `_has_standalone_mention`. After 13d, the LLM directly emits `proper_case` and the formatter never calls `_has_standalone_mention` for mention classification. **But** `_has_standalone_mention` is still consumed elsewhere in 13b (anchor collection in `_build_evidence_bundle` L599, and in the seed pipeline). Its full LLM replacement is **EXT-01 → Phase 5 keep-decision**, not VAR-04 scope.
- **Lenient enum coercion** (Spike 003 `consume_candidate()` pattern, `if mt not in MENTION_TYPES: mt = "indirect"`) — rejected for production under D-21b; preserved in the Spike. If a future cross-model phase (EXT-03, GPT-5.2 re-evaluation) needs it back, it returns as its own decision, not as a quiet behavioral default.
- **Prompt-schema extraction into `prompts_v2.py`** — the new schema text could live in a shared module if Phase 4 / 5 want to reuse it (e.g., for the alias-scope schema in VAR-05). Phase 3 keeps it inline per D-20b; the planner for VAR-05 may extract at that point.
- **Threading `mention_type` through `SadSamLink`** vs a parallel dict — D-22's two options. The planner picks during implementation; option (a) is preferred but option (b) is a clean backstop if extending `SadSamLink`'s constructor signature ripples too far. Either way, no new public API.
- **Combining VAR-04 + VAR-05 + VAR-06 into a single variant** — explicitly rejected by ROADMAP Phase 3 vs Phase 4 split and by the "one rule removal per variant" Key Decision. Keep separate.
- **LaTeX rendering of the ablation table** — `tabulate` will emit it via `tablefmt="latex"`; this is a Phase 5 PROMO-03 deliverable. Phase 3 ships only the **row** (markdown via `tabulate`) for 13d.
- **GPT-5.2 cross-model run of 13d** — EXT-03, out of scope. Claude Sonnet only per PROJECT.md constraint and MEMORY.md ("Always use Claude Sonnet … never opus").
- **Re-spike 003 in pipeline before integration** — not needed. Spike 003's 4 self-verifying tests are byte-level (string equality, bytecode `co_names` check); the in-pipeline parity criterion (D-21d) extends the spike's test 3 to all 5 enum branches in the actual `EvidenceBundle` consumer path.

### Reviewed Todos (not folded)
None — STATE.md "Pending Todos" is empty.

</deferred>

---

*Phase: 03-mention-classifier-migration*
*Context gathered: 2026-05-29 (auto mode — recommended defaults selected from Phase 1+2 precedent + ROADMAP Phase 3 + Spike 003)*
