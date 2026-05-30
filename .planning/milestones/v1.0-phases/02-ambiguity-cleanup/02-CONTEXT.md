# Phase 2: Ambiguity Cleanup - Context

**Gathered:** 2026-05-28
**Status:** Ready for planning
**Mode:** `gsd-discuss-phase --auto` (recommended defaults selected by Claude; no human Q&A — every decision below is locked with a cited source)

<domain>
## Phase Boundary

Retire the two structural ambiguity helpers from `s_linker12c` by shipping two standalone variants in order:

1. **VAR-02 / `s_linker13b.py`** — remove `_is_structurally_unambiguous` (the static structural post-filter at `s_linker12c.py:269`). Trust the LLM `ambiguous` list emitted by `_classify_components` (`s_linker12c.py:309`) on its own. Source filter callsite to retire: `s_linker12c.py:340` (the `len(n.split()) == 1 and not self._is_structurally_unambiguous(n)` guard inside `_classify_components`).
2. **VAR-03 / `s_linker13c.py`** — inline the body of `_is_ambiguous_name_component` (`s_linker12c.py:1102`) at its two callsites (`s_linker12c.py:631`, `s_linker12c.py:825`), then remove the wrapper. With `_is_structurally_unambiguous` already gone in 13b, the wrapper degenerates to a single dict lookup against `model_knowledge.ambiguous_names`.

Both variants must pass GATE-01 dual floor on the full 5-project sweep: macro F1 ≥ 0.93 AND no dataset more than 2pp below the `s_linker12c` Plan 04 baseline — **except BBB, which gets a 4pp tolerance** carried over from the user-loosened gate set on 2026-05-28 during Phase 1 closure (see Phase 1 Summary §"Gate Resolution"). Macro floor stays at 0.93.

**In scope:** Variant files 13b + 13c, registration in `run_ablation.py`, BENCHMARK_TABOO audits on any new/changed prompt text, hard-tier-first gate (TM + BBB) then full 5-project sweep, ablation log rows for both variants.

**Out of scope (this phase):**
- `_classify_mention` regex replacement (VAR-04 → Phase 3)
- `_is_strong_alias` / `_get_strong_alias_mappings` (VAR-05 → Phase 4)
- `_has_strong_alias_mention` (VAR-06 → Phase 4)
- `_has_standalone_mention` keep/replace decision (Phase 5 / EXT-01)
- Any change to `_classify_components`'s prompt content beyond removing the structural co-filter at line 340 (the LLM is already producing the `ambiguous` list — the change is purely "trust it as-is")
- Promotion artifact (PROMO-* → Phase 5)

</domain>

<decisions>
## Implementation Decisions

### Variant File Layout (D-08)
- **D-08:** Each variant is a **standalone file**, full copy of its parent, edited in place — no inheritance, no shared helpers module. 13b is copied from `s_linker12c.py`; 13c is copied from `s_linker13b.py`. **Source:** Phase 1 D-03/D-04 precedent and MEMORY.md user preference ("standalone linker files (duplicate code intentionally, not inheritance chains)"). The cumulative-removal chain (13c inherits 13b's removal) is realized by copying from the previous variant's file, not by importing.

### Prompt Placement (D-09)
- **D-09:** Inline prompt constants stay inline. No prompts_v2 module changes in this phase. `_classify_components`'s prompt body is unchanged — only the post-filter co-guard at L340 (`and not self._is_structurally_unambiguous(n)`) is removed. **Source:** Phase 1 Claude's-Discretion precedent ("Prompt constant placement in 13a — inline or from prompts_v2.py, planner decides"); the simpler edit here is no prompt change at all.

### Baseline Protocol (D-10)
- **D-10:** **Single run** on full 5-project sweep for both 13b and 13c. No N-run median, no best-of-N. **Source:** Phase 1 D-02 ("Single run on full 5-project sweep ... no N-run median, no best-of-N, no averaging"). Variance re-runs are *only* triggered if a borderline GATE-05 result hits the marginal band (see D-13).

### Checkpoint Namespacing (D-11)
- **D-11:** Each variant declares its own `_VARIANT_NAME` class constant: `s_linker13b` declares `_VARIANT_NAME = "s_linker13b"`; `s_linker13c` declares `_VARIANT_NAME = "s_linker13c"`. The D-07 runtime assertion in `_checkpoint_dir` (already in 12c at `s_linker12c.py:1126`) carries forward unchanged via the copy. **Source:** Phase 1 D-03 + D-07.
- **D-11a:** `run_ablation.py` consumes variant identity through the class constant, never by string literal. The `CANONICAL_VARIANTS` / `VARIANT_SPECS` registration is **append-only** (13b appended after 13a; 13c appended after 13b). **Source:** Phase 1 D-04 + Plan 05 §Commands Executed.

### Parent Baseline for ΔF1 (D-12)
- **D-12:** Each variant's ΔF1 is computed **vs its immediate structural parent**, not vs 12c, for ablation-table purposes:
  - 13b ΔF1 vs 12c (baseline = Plan 04 `ablation_20260528_173020.json` 12c row, MS 0.984 / TS 0.963 / TM 0.938 / BBB 0.844 / JAB 0.973, macro 0.9405)
  - 13c ΔF1 vs 13b (planner: re-use 13b's freshly produced sweep numbers, do not re-run 12c)
  - For **GATE-01** (the pass/fail gate), the comparison is **vs 12c** for both variants — ΔF1 vs parent is for the ablation table only. **Source:** ROADMAP Phase 2 success criteria (#1 and #2 both reference "12c baseline" explicitly, not "parent"); PROMO-03 spec in REQUIREMENTS.md ("ΔF1 vs parent ... per-dataset F1").

### Gate Tolerance — BBB Carry-Over (D-13)
- **D-13:** **BBB gets a 4pp per-dataset tolerance**; the other four datasets (mediastore, teastore, teammates, jabref) use the standard 2pp tolerance; macro floor stays at 0.93. This is the **same loosened gate set** that the user authorized for Phase 1 on 2026-05-28 to admit Spike 001's timing-perturbation regression. **Source:** Phase 1 Summary §"Gate Resolution (2026-05-28, user direction)" and STATE.md `last_activity` line.
- **D-13a:** **Rationale for carrying it to Phase 2:** Spike 001's mechanism (an added LLM call perturbs Claude's prompt-cache stream and causes 2-3 FN swings on BBB HTML5 Client/Server partials in Tier-2) is a property of the pipeline running 13a's `_enrich_trailing_words`. 13b and 13c are downstream of that call — 13b copies from 12c (not 13a) per the project's "base = 12c" rule (PROJECT.md Key Decisions), so 13b and 13c do *not* include Spike 001 enrichment and the BBB timing perturbation should not recur. The 4pp tolerance is carried *as insurance*, not as an expected use; if BBB lands within the 2pp band on 13b, that is the expected outcome.
- **D-13b:** **GATE-05 hard-tier policy carry-over:** TM + BBB run first. Auto-approve to full sweep if **both** deltas ≥ -0.01pp vs 12c; marginal (-0.01 to -0.02 on TM, -0.01 to -0.04 on BBB) → halt and flag; hard reject (delta < -0.02 on TM or < -0.04 on BBB) → no full sweep, rework. **Source:** Phase 1 Plan 05 checkpoint table + 2026-05-28 BBB-tolerance update.

### Variance Re-Run Trigger (D-14)
- **D-14:** Run a second hard-tier pass (cache cleared) **only if** the first hard-tier run lands a marginal flag per D-13b. No variance re-run on auto-approve. No variance re-run on hard reject. **Source:** Phase 1 Plan 05 §"Variance Re-Run (2026-05-16)" precedent — variance re-runs were the marginal-band recovery path that eventually grounded the 2026-05-28 BBB tolerance decision.

### LLM-Substitution Inertness Risk (D-15)
- **D-15:** Plan for the Spike-001 lesson: an LLM substitution can be **inert** (produce zero new signal) and still cause downstream FP/FN swings purely through cache-stream timing perturbation. **For Phase 2 this is much less likely** because 13b/13c are pure **removals** of synchronous in-process Python functions — no new LLM call is added. The `_classify_components` LLM call already exists in 12c and is unchanged in 13b/13c; only its post-filter is removed. If a regression nonetheless appears, treat it as a **classification-coverage** issue (the LLM's `ambiguous` list missing single-word ambiguous names that the structural code previously caught), not a cache-stream issue. **Source:** Phase 1 Summary §"Failure-Mode Analysis" + MEMORY.md ("Claude run-to-run variance ... not fixable by temperature/seed").

### Taboo Audit (D-16)
- **D-16:** No new prompt constants are added in this phase (D-09). The taboo audit is therefore **a smoke-test that the existing `_classify_components` prompt and any docstring updates are still clean**, not a full re-audit. The audit script is the same substring-match script Phase 1 used; the docstring `RULES_REMOVED: ["_is_structurally_unambiguous"]` and `RULES_REMOVED: ["_is_ambiguous_name_component"]` are short tokens that should not collide with the `BENCHMARK_TABOO.md` keyword list (planner verifies). **Source:** GATE-04 + Phase 1 Plan 05 §Issues "gui in ambiguity" precedent.

### Ablation Log Rows (D-17)
- **D-17:** Two new rows in the ablation log (`PROMO-03` table) for 13b and 13c. Each row records: per-dataset F1 (5 columns), macro F1, ΔF1 vs 12c (5 columns + macro), ΔF1 vs parent (1 column for macro), rules-removed list, FP-by-phase breakdown (seed / entity / coref). Wide-format table; canonical output is markdown via `tabulate` (added to `pyproject.toml` in Phase 1 D-06 but not yet exercised — Phase 2 is its first use). **Source:** REQUIREMENTS.md PROMO-03; ROADMAP Phase 2 success criterion #4.

### Wave Structure (D-18)
- **D-18:** Two **sequential** plans, not parallel:
  - Plan 02-01: VAR-02 (`s_linker13b.py`, registration, hard-tier gate, full sweep, ablation row)
  - Plan 02-02: VAR-03 (`s_linker13c.py`, registration, hard-tier gate, full sweep, ablation row, inheriting 13b's clean removal)
  - 13c **must not start** until 13b passes GATE-01 — 13c is defined as 13b-minus-the-wrapper; if 13b fails the gate, 13c's parent baseline is undefined. **Source:** ROADMAP Phase 2 §Success Criteria (both criteria #1 and #2 are independent dual-floor passes, but #2 explicitly references "no call to `_is_ambiguous_name_component` or `_is_structurally_unambiguous`", which only holds if 13b shipped first).

### Claude's Discretion
- 13b docstring exact wording (must include `REMOVED_FROM: s_linker12c` and `RULES_REMOVED: ["_is_structurally_unambiguous"]`).
- 13c docstring exact wording (must include `REMOVED_FROM: s_linker13b` and `RULES_REMOVED: ["_is_ambiguous_name_component"]`; cumulative list optional).
- Whether to inline `_is_ambiguous_name_component`'s body identically at both callsites (L631, L825) or to introduce a tiny lambda-equivalent local — planner decides during 02-02; behavior must be byte-identical to the wrapper after `_is_structurally_unambiguous` is gone.
- Print-banner string in `__init__` (Phase 1 Plan 05 deviation #3 lesson: update the banner when copying — don't leave a 12c/13a string in a 13b/13c file).
- Whether to clear `results/phase_cache/s_linker13{b,c}/` between hard-tier and full sweep — planner default: **no clearing** for the second sweep (full sweep extends hard-tier with the remaining 3 datasets; per-dataset checkpoints are independent).

### Folded Todos

None — no pending todos in STATE.md cross this phase.

</decisions>

<specifics>
## Specific Ideas

- The user's loosened gate (BBB 4pp, others 2pp, macro 0.93) is treated as the **standing policy for the remainder of the 13-series**, not just a one-time exception. Phase 2 inherits it; Phases 3-5 are expected to inherit it absent further direction.
- 13b is the **first variant in the chain whose removal touches only a synchronous Python function** (no new LLM call). It is the cleanest possible test of whether ablation deltas in this project are dominated by (a) genuine signal loss vs (b) cache-stream timing perturbation. If 13b passes hard-tier within the 1pp auto-approve band on BBB, that is evidence the 13a BBB regression was indeed timing-stream perturbation, not signal.
- 13c is mostly a **dead-code cleanup**: once `_is_structurally_unambiguous` is gone in 13b, the wrapper's `if self._is_structurally_unambiguous(comp_name): return False` branch is a no-op against an undefined symbol. 13c removes the wrapper to make this trivially explicit. The functional change vs 13b is exactly zero; the F1 numbers should be byte-equal modulo Claude run-to-run noise. If they aren't, the **timing-stream hypothesis is reconfirmed** and the planner should log it as evidence supporting D-13.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents (`gsd-phase-researcher`, `gsd-planner`) MUST read these before planning or implementing.**

### Project specs (gate definitions, requirement IDs)
- `.planning/PROJECT.md` — Core Value (macro F1 ≥ 93% or reject), Key Decisions (base=12c, ablation unit = variant, standalone files), constraints.
- `.planning/REQUIREMENTS.md` — **VAR-02, VAR-03** (Phase 2 scope); **GATE-01..GATE-06** (every variant must satisfy); STRUCK INFRA-02/04 (not relevant here).
- `.planning/ROADMAP.md` §Phase 2 (lines 50-60) — goal, depends-on, success criteria #1-4.

### Phase 1 inheritance (decisions that carry forward)
- `.planning/phases/01-baseline-and-infrastructure/01-CONTEXT.md` — D-02 (single-run baseline), D-03/D-04 (`_VARIANT_NAME` discipline), D-07 (runtime assertion).
- `.planning/phases/01-baseline-and-infrastructure/01-05-SUMMARY.md` — Spike 001 lessons (LLM-substitution inertness, prompt-cache-stream timing perturbation, BBB variance), §"Gate Resolution (2026-05-28)" defining the 4pp BBB tolerance carried into Phase 2 per D-13.

### Spikes (validated rule classifications)
- `.planning/spikes/002-rules-audit/` — confirms `_is_structurally_unambiguous` and `_is_ambiguous_name_component` are REPLACEABLE (not RISKY, not ESSENTIAL).
- `.planning/spikes/001-llm-trailing-words/README.md` — Phase 1 precedent for variant-shipping protocol; relevant only as a template.

### Codebase targets (lines to edit)
- `src/llm_sad_sam/linkers/experimental/s_linker12c.py` — copy source for 13b.
  - L268-277: `_is_structurally_unambiguous` definition (to delete in 13b)
  - L340: callsite inside `_classify_components` (the `and not self._is_structurally_unambiguous(n)` guard — to delete in 13b)
  - L1102-1108: `_is_ambiguous_name_component` wrapper (to delete in 13c; first remove its dependence on `_is_structurally_unambiguous` in 13b by deleting the L1104 line)
  - L631, L825: `_is_ambiguous_name_component` callsites (to inline in 13c)
  - L1120-1130: `_checkpoint_dir` and D-07 assertion (preserve as-is via the copy; only `_VARIANT_NAME` changes)
- `run_ablation.py` — append 13b after the last current entry, then append 13c after 13b. Same `CANONICAL_VARIANTS` and `VARIANT_SPECS` lists used in Phase 1 Plan 05.
- `BENCHMARK_TABOO.md` — referenced by GATE-04 (smoke-test only this phase per D-16).

### Research context (background, not action)
- `.planning/research/ARCHITECTURE.md` — pipeline structure (Tier 1 / Tier 2 / coref / boundary-filter wave).
- `.planning/research/PITFALLS.md` — Claude run-to-run variance documentation backing D-15.

### Memory / prior art
- MEMORY.md — standalone-file preference (D-08); Spike-001-style LLM-substitution inertness (D-15); Claude run-to-run variance pattern.
- `.planning/STATE.md` — Phase 1 closure timestamp (2026-05-28), 12c full-sweep baseline JSON path (`results/ablation_results/ablation_20260528_173020.json`).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_VARIANT_NAME` pattern + D-07 assertion (Phase 1 INFRA-05 deliverable) — carries forward by the standard cp-and-edit. 13b changes the constant to `"s_linker13b"`, 13c to `"s_linker13c"`.
- `run_ablation.py` `CANONICAL_VARIANTS` / `VARIANT_SPECS` append-only registration — exercised in Phase 1 Plan 05; same shape for 13b/13c.
- 12c full-sweep baseline (Phase 1 Plan 04) — already produced; reuse the JSON for ΔF1 vs 12c per D-12. **Do not re-run 12c.**
- `tabulate` dep (Phase 1 D-06) — first real use in this phase for the ablation row format (D-17).

### Established Patterns
- Standalone variant files (12a/b/c/d/e + 13a) — duplicated code is the project's reproducibility artifact.
- Append-only registration in `run_ablation.py`.
- Per-variant pickle cache namespacing under `results/phase_cache/<_VARIANT_NAME>/<dataset>/`.
- Hard-tier-first (TM + BBB) → full 5-project sweep gate sequence.

### Integration Points
- `_classify_components` (12c:L309) is the only LLM-emitting function whose output is consumed by the to-be-removed structural co-filter. After 13b, its `ambiguous` list is consumed verbatim (no `len(n.split()) == 1` and no `_is_structurally_unambiguous` co-filter). Side effect: multi-word names emitted in `ambiguous` will now flow through into `model_knowledge.ambiguous_names`. Planner must verify this is acceptable — the only consumers of `model_knowledge.ambiguous_names` are the L631 and L825 callsites (both inside `_is_ambiguous_name_component`'s body in 12c, both wrapped in `_is_structurally_unambiguous` short-circuit at L1104), so under 12c semantics multi-word ambiguous names were effectively silently filtered. Under 13b, they will pass through. If this surfaces as FPs, the 13b plan needs a remediation hook (e.g., a prompt amendment to `_classify_components` to bound the LLM's `ambiguous` set to single-word names — but the spec for 13b is "trust the LLM list", so this is a documented limitation, not a code patch).

</code_context>

<deferred>
## Deferred Ideas

- **Prompt-side amendment to `_classify_components`** to enforce single-word `ambiguous` items inside the prompt itself (rather than via the now-removed Python co-filter). If 13b regresses through "multi-word names slipping into `ambiguous_names`" (the documented integration risk above), Phase 2 *as scoped* logs it and rejects 13b. A follow-up rework that touches the prompt becomes its own decision; it would be 13b' (re-spin), not Phase 3.
- **Combining 13b + 13c into a single variant** (one file removing both helpers at once) — explicitly rejected by ROADMAP Phase 2's two-criterion structure (#1 and #2) and by the "ablation unit = linker variant, one rule removal per variant" Key Decision. Keep separate.
- **LaTeX rendering of the ablation table** — `tabulate` will emit it via `tablefmt="latex"`; this is a Phase 5 PROMO-03 deliverable, not Phase 2. Phase 2 ships only the **rows** (markdown via the same `tabulate` call) for 13b and 13c.
- **Spike 002 re-classification audit** — `_is_structurally_unambiguous` was classified REPLACEABLE by Spike 002. The phase trusts that classification and does not re-spike.

### Reviewed Todos (not folded)
None — STATE.md "Pending Todos" is empty.

</deferred>

---

*Phase: 02-ambiguity-cleanup*
*Context gathered: 2026-05-28 (auto mode — recommended defaults selected from Phase 1 precedent + roadmap)*
