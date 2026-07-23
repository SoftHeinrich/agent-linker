---
phase: 06
plan: 06
subsystem: llm-linker
requirements: [EXT-01]
tags: [llm-linker, sad-sam, standalone-mention, ext-01, alias-aware, gate-06, gate-07]
dependency_graph:
  requires:
    - 06-01: STANDALONE_MENTION_RULES_PRE_FILTERED + STANDALONE_MENTION_RULES_LLM_ONLY (base prompts in prompts_v2)
    - 06-02: s_linker13g_pre.py + s_linker13g_sem.py (rejected-baseline templates kept in tree, untouched)
    - 06-05: STANDALONE_MENTION_RULES_{PRE_FILTERED,LLM_ONLY}_{ALIAS_AWARE,FULL_KNOWLEDGE} (4 alias-aware prompt constants in prompts_v2)
  provides:
    - 4 new linker classes registered in run_ablation registry, each with Tier-1b sequential standalone-mention map and alias / linkmap injection
    - Pareto inputs for Plan 06-07 (D-02 anchor-diff) and Plan 06-08 (winner promotion)
  affects:
    - run_ablation.py CANONICAL_VARIANTS + VARIANT_SPECS (4 entries appended; canonical=False)
tech-stack:
  added: []
  patterns:
    - "str.replace prompt substitution to preserve literal JSON braces"
    - "Tier-1a parallel + Tier-1b sequential DAG split"
    - "Project-agnostic block formatters that echo upstream-discovered data verbatim"
key-files:
  created:
    - src/llm_sad_sam/linkers/experimental/s_linker13g_pre_alias.py
    - src/llm_sad_sam/linkers/experimental/s_linker13g_sem_alias.py
    - src/llm_sad_sam/linkers/experimental/s_linker13g_pre_full.py
    - src/llm_sad_sam/linkers/experimental/s_linker13g_sem_full.py
  modified:
    - run_ablation.py
decisions:
  - "T-06-06-06 acyclicity: *_full variants consume raw_seed_links from Tier-1a (ILinker3 output), NOT coref antecedents. Coref runs in Tier 2 and itself reads standalone_map, so feeding coref output back into Tier-1b would form a cycle. Documented inline in each *_full docstring and in _format_linkmap_block."
  - "Alias block uses BOTH global and local scopes (vs. _extract_entities_enriched which filters to global-only). Standalone-mention is a per-sentence judgement where local-scope aliases are useful context, not a recall-leakage risk."
  - "Prompt substitution uses str.replace, not str.format — JSON return-shape templates contain literal { } that must not be clobbered."
  - "All 4 new variants registered canonical=False. Canonical s_linker13g namespace reserved for Plan 06-08 byte-copy of the winning sub-variant."
metrics:
  duration_s: 293
  completed_date: "2026-05-30"
  tasks_completed: 3
  files_created: 4
  files_modified: 1
---

# Phase 06 Plan 06: Alias-Aware and Full-Knowledge Standalone-Mention Sub-Variants Summary

Four new SAD-SAM linker sub-variants implementing D-07 alias/linkmap knowledge injection on top of the s_linker13g_{pre,sem} rejected-baseline templates, with Tier-1 DAG re-topology to feed upstream-discovered project knowledge into the standalone-mention LLM judge.

## File Line Counts

| File | Lines |
|------|-------|
| `s_linker13g_pre_alias.py` | 1296 |
| `s_linker13g_sem_alias.py` | 1248 |
| `s_linker13g_pre_full.py`  | 1320 |
| `s_linker13g_sem_full.py`  | 1273 |

All exceed plan `min_lines` (1200 / 1180). All four files contain their respective class name + `_VARIANT_NAME` matching their filename, and import the corresponding Plan 06-05 prompt constant.

## Sub-Variant -> Sub-Variant Diff Sizes (Plan 06-06 internal symmetry check)

| Comparison | Diff line count | Notes |
|------------|-----------------|-------|
| `pre_alias` -> `pre_full` | 106 | docstring expansion + import swap + `_format_linkmap_block` (~20 lines) + `self._raw_seed_links =` Tier-1b store + `_compute_standalone_mention_map` adds 2nd `.replace` call + the prompt-constant rename. |
| `sem_alias` -> `sem_full` | 105 | symmetrical change set; LLM-only branch (no pre-filter) so same delta shape. |

The deltas are tightly scoped: each `*_full` adds exactly the linkmap helper + the linkmap block substitution and stores raw_seed_links during the Tier-1a→Tier-1b transition.

## `run_ablation.py --list-variants` Snippet (filtered to `s_linker13g_*`)

```
s_linker13g_pre
s_linker13g_sem
s_linker13g_pre_alias
s_linker13g_sem_alias
s_linker13g_pre_full
s_linker13g_sem_full
```

6 entries total: 2 rejected-baseline (Plan 06-02, untouched) + 4 new (this plan). All 4 new entries have `canonical=False`; canonical `s_linker13g` is intentionally vacant and reserved for Plan 06-08.

## Mechanical Banned-Term Scan (per-file stdout)

The scan uses the word-bounded `grep -iwE` invocation from the plan's `<verify>` block.

```
== scan src/llm_sad_sam/linkers/experimental/s_linker13g_pre_alias.py ==
NO_HITS_s_linker13g_pre_alias.py
== scan src/llm_sad_sam/linkers/experimental/s_linker13g_sem_alias.py ==
NO_HITS_s_linker13g_sem_alias.py
== scan src/llm_sad_sam/linkers/experimental/s_linker13g_pre_full.py ==
NO_HITS_s_linker13g_pre_full.py
== scan src/llm_sad_sam/linkers/experimental/s_linker13g_sem_full.py ==
NO_HITS_s_linker13g_sem_full.py
```

All four files clear the BENCHMARK_TABOO mechanical scan. No benchmark-specific surface form, per-component casing table, or project-specific identifier appears in the new file bodies. The alias / linkmap substitution code iterates upstream-discovered data verbatim — there is no hand-coded normalization, synonym table, or per-component branch.

## Tier-1b Sequencing Notes

**All four new variants** restructure the Tier-1 DAG from a single 4-key parallel block (`{model, doc_knowledge, seed, standalone_map}`) into a 3-key parallel Tier-1a (`{model, doc_knowledge, seed}`) followed by a sequential Tier-1b that calls `self._compute_standalone_mention_map(...)`.

- `doc_knowledge` is fully populated **before** `_compute_standalone_mention_map` runs, so `_format_alias_block` always sees the final `self.doc_knowledge.aliases` map.
- For `*_full` variants, `self._raw_seed_links = raw_seed_links` is set during the Tier-1a→Tier-1b transition (immediately after `raw_seed_links = acq["seed"]`), so `_format_linkmap_block` always sees the final ILinker3 output.
- `grep -c 'self._raw_seed_links\|_format_linkmap_block'` on both `*_full` files returns 5 (assignment + helper read + helper def + 2 references inside helper/header) confirming wiring is symmetric across the two `*_full` variants.

## T-06-06-06 Acyclicity Decision (inline-documented)

D-07 mentions "coref antecedents" as a candidate input to the standalone-mention judge. Plan 06-06 explicitly does NOT feed coref output into Tier-1b. Coref runs in Tier 2 (`_run_coreference`) and itself consumes `self._standalone_map` (verified at `s_linker13g_pre.py:1089`). Feeding coref antecedents into the Tier-1b standalone-mention map would form a cycle: standalone_map → coref → standalone_map.

The simplest acyclic interpretation chosen for `*_full` variants is the **running link map = raw_seed_links from Tier-1a**. This is the only link source available BEFORE coref runs. The decision is documented inline in:

- The module docstring of `s_linker13g_pre_full.py` and `s_linker13g_sem_full.py` ("T-06-06-06 acyclicity note").
- The `_format_linkmap_block` docstring of both `*_full` variants.

If Plan 06-08 results show this hurts recall on coref-heavy datasets, that is an empirical finding for the ablation, NOT a wiring bug — a future plan can revisit the trade-off (e.g., a 3-tier DAG with a partial coref pass between Tier-1a and Tier-1b).

## Pre-Existing Files Untouched

`git diff --stat` against the following files returned empty after each task commit:

- `src/llm_sad_sam/linkers/experimental/s_linker13.py`
- `src/llm_sad_sam/linkers/experimental/s_linker13g_pre.py`
- `src/llm_sad_sam/linkers/experimental/s_linker13g_sem.py`
- `src/llm_sad_sam/linkers/experimental/prompts_v2.py`

The rejected-baseline pair stays in tree as the D-09 ablation row; the canonical baseline (`s_linker13.py`) is untouched; the Plan 06-05 prompt constants are untouched.

## Plan 06-07 Pointers

For the D-02 anchor-set diff stage in Plan 06-07, feed all 4 new variant names through `scripts/ext01_diff_stage.py`:

- `s_linker13g_pre_alias`
- `s_linker13g_sem_alias`
- `s_linker13g_pre_full`
- `s_linker13g_sem_full`

The 2-variant rejected-baseline pair (`s_linker13g_pre`, `s_linker13g_sem`) is the comparator column.

## Commits

| Task | Hash | Message |
|------|------|---------|
| 1 | `20ee1b0` | feat(06-06): add alias-aware standalone-mention sub-variants |
| 2 | `249b01f` | feat(06-06): add full-knowledge standalone-mention sub-variants |
| 3 | `09a58b1` | feat(06-06): register alias/full sub-variants in run_ablation registry |

## Deviations from Plan

None - plan executed exactly as written. All three tasks completed on first iteration; no Rule 1/2/3 auto-fixes triggered; no authentication gates encountered; no architectural decisions required.

## Verification Run

```
$ python -c "from llm_sad_sam.linkers.experimental.s_linker13g_pre_alias import SLinker13gPreAlias; \
            from llm_sad_sam.linkers.experimental.s_linker13g_sem_alias import SLinker13gSemAlias; \
            from llm_sad_sam.linkers.experimental.s_linker13g_pre_full import SLinker13gPreFull; \
            from llm_sad_sam.linkers.experimental.s_linker13g_sem_full import SLinker13gSemFull; \
            print('all 4 importable')"
all 4 importable

$ python -c "import run_ablation as r; specs = r.VARIANT_SPECS; \
            ks=['s_linker13g_pre_alias','s_linker13g_sem_alias','s_linker13g_pre_full','s_linker13g_sem_full']; \
            assert all(k in specs for k in ks); \
            assert all(not specs[k].get('canonical', False) for k in ks); \
            import importlib; [importlib.import_module(specs[k]['module']) for k in ks]; \
            print('OK all 4 registered, none canonical, all importable via registry')"
OK all 4 registered, none canonical, all importable via registry

$ python run_ablation.py --list-variants | grep -c 's_linker13g_'
6
```

All `<success_criteria>` from the plan are satisfied:

- [x] Four new linker files exist, importable
- [x] Each imports its matching Plan 06-05 prompt constant
- [x] Each Tier-1b-sequences `standalone_map` after `doc_knowledge` (and after `seed` for `*_full`)
- [x] Each `_compute_standalone_mention_map` substitutes the relevant blocks via `str.replace`
- [x] Each is registered in run_ablation.py with `canonical=False`
- [x] The rejected-baseline pair and the canonical baseline remain in tree untouched

## Self-Check: PASSED

All claimed artifacts verified on disk and in git history:

- 4 new linker files present in `src/llm_sad_sam/linkers/experimental/`
- SUMMARY.md present at expected phase-dir path
- All 3 task commits (`20ee1b0`, `249b01f`, `09a58b1`) found via `git log --oneline --all`
