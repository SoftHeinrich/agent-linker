---
phase: 02-ambiguity-cleanup
date: 2026-05-28
tags: [research, var-02, var-03, ambiguity, removal]
---

# Phase 02 — Research Notes (compact)

Scope is small and inheritance-heavy: every implementation question is already answered by CONTEXT.md (decisions D-08..D-18) and the Phase 1 Plan 05 precedent. This note pins down the **callsite map**, **Phase-1 lessons that carry over**, and **the one open risk** Phase 2 has to keep an eye on. No new techniques to research.

---

## 1. Callsite map (the only "research" Phase 2 needs)

Source file: `src/llm_sad_sam/linkers/experimental/s_linker12c.py` (1159 lines, single class `SLinker12c`).

### `_is_structurally_unambiguous` — VAR-02 target
- **Definition**: L268-277 (8 lines, `@staticmethod`).
- **Callsites** (2):
  1. **L340** — inside `_classify_components`. The line is the post-filter co-guard:
     ```python
     knowledge.ambiguous_names = {
         n for n in raw_ambiguous
         if len(n.split()) == 1 and not self._is_structurally_unambiguous(n)
     }
     ```
     **Action in 13b**: delete the entire `and not self._is_structurally_unambiguous(n)` clause. Keep the `len(n.split()) == 1` predicate — that single-word filter is the legacy guard that prevented multi-word LLM-emitted names from leaking into `model_knowledge.ambiguous_names`. Removing it is **not** in 13b's scope (CONTEXT.md §"Out of scope" — only the structural co-filter goes; the single-word filter is a separate decision).
     
     CONTEXT integration risk (already flagged at the bottom of CONTEXT.md `<code_context>`): with the structural co-filter gone, **multi-word ambiguous names** the LLM emits will still be excluded by the surviving `len(n.split()) == 1` predicate, so the risk is actually *narrower than CONTEXT.md suggests*. The truly exposed names are **single-word CamelCase/uppercase** names (e.g. `Authentication`, `RPC`) that the LLM emits as ambiguous — under 12c they were filtered out by `_is_structurally_unambiguous`; under 13b they will pass through into `model_knowledge.ambiguous_names` and reach the L631/L825 consumers.
  
  2. **L1104** — inside `_is_ambiguous_name_component`'s body:
     ```python
     def _is_ambiguous_name_component(self, comp_name):
         if self._is_structurally_unambiguous(comp_name):
             return False
         ...
     ```
     **Action in 13b**: delete the L1104 short-circuit line entirely. After this delete, the wrapper degenerates to the dict lookup at L1106-1108. This is intentional — it makes 13c trivial (CONTEXT.md D-18).

### `_is_ambiguous_name_component` — VAR-03 target
- **Definition**: L1102-1108 (7 lines, instance method).
- **Callsites** (2):
  1. **L631** — inside `_build_evidence_bundle` (the call returns a bool stored in `is_ambig` then placed on `EvidenceBundle.is_ambiguous`).
  2. **L825** — inside `_separate_ambiguous_candidates` (used as a gate to bucket lowercase-only mentions into the `generic_candidates` map for downstream LLM judgement).

After 13b ships, the wrapper body is literally:
```python
def _is_ambiguous_name_component(self, comp_name):
    if not self.model_knowledge or not self.model_knowledge.ambiguous_names:
        return False
    return comp_name in self.model_knowledge.ambiguous_names
```
…i.e., a single dict-set lookup. **13c inlines that lookup** at L631 and L825, removes the wrapper.

**Inlining shape** (Claude's discretion in CONTEXT.md — planner chooses byte-identical pattern at both sites):
```python
# at L631 (in 13c)
is_ambig = bool(
    self.model_knowledge
    and self.model_knowledge.ambiguous_names
    and comp_name in self.model_knowledge.ambiguous_names
)

# at L825 (in 13c) — used inside an `and` chain, parenthesize for safety
if has_lowercase and (
    self.model_knowledge
    and self.model_knowledge.ambiguous_names
    and c.component_name in self.model_knowledge.ambiguous_names
):
```

The behaviour is **byte-identical** to the post-13b wrapper. The functional change from 13b to 13c is therefore exactly zero; any F1 delta is Claude run-to-run noise (D-13a / D-15 explicitly call this out as the canary).

---

## 2. Phase 1 lessons that apply directly

| Lesson (from Plan 01-05 SUMMARY §Deviations) | Phase 2 instance |
|---|---|
| **Plan/code drift on line numbers** — the SUMMARY found `_split_component_name` was already gone in 12c. | The line numbers in this RESEARCH.md were re-verified against the current 12c on 2026-05-28 (L268-277, L309-341, L631, L825, L1102-1108). They are accurate. |
| **Taboo audit substring false positive** — `"gui"` inside `"ambiguity"`. | No new prompt constants in Phase 2 (D-09). The taboo audit is a smoke-test only (D-16); the existing `_classify_components` prompt was already audited under 12c. **However**, the new docstrings contain the literal word `_is_structurally_unambiguous` (substring `gui`) and `_is_ambiguous_name_component` (substring `gui`). Both 13b and 13c plans must audit **only the prompt body**, not the docstring — same as 12c precedent. |
| **`__init__` print-banner drift** — copy left "12c" string in 13a. | Both plans must update the banner string in `__init__`. |
| **Smoke-test pickle dir leak** — `/tmp/fake_dataset.txt` test left a directory in `phase_cache/`. | Both plans must clean up `results/phase_cache/<variant>/fake_dataset/` after the import smoke test, if it appears. |
| **GATE-05 hard reject pattern** — BBB regressed 4.8pp on 13a from a timing-perturbation channel. | **Less likely in Phase 2** — 13b/13c add no LLM calls (D-15). Both variants only **remove** synchronous Python code. The 4pp BBB tolerance (D-13) is carried as insurance, not as expected use. |

---

## 3. The one open risk

**Single-word CamelCase/uppercase names slipping into `model_knowledge.ambiguous_names` (13b)**

- **Mechanism**: under 12c, `_is_structurally_unambiguous(name)` returns `True` for any single-word name with internal capital (e.g. `iPhone`, hypothetical) or all-uppercase (e.g. `RPC`, `API`, `UI`). The L340 co-filter then *excluded* such names from `ambiguous_names`. Under 13b, the LLM's `ambiguous` list is consumed verbatim modulo the single-word filter, so an LLM emitting `"RPC"` or `"UI"` as ambiguous would land in `model_knowledge.ambiguous_names`.
- **Downstream consumers** of `ambiguous_names` post-13b: only `_is_ambiguous_name_component` (L1102), which is called at L631 and L825. At L631 the boolean ends up on `EvidenceBundle.is_ambiguous` and feeds judge prompts. At L825 it gates the `generic_candidates` bucket → triggers the LLM "is this a generic mention?" round.
- **Possible failure mode**: a CamelCase component name (e.g. a hypothetical `MediaStore` if the LLM ever emitted it as ambiguous) routed through L825 would be funneled into the generic-mention LLM round. That round is **conservative** (only rejects if LLM says "yes, generic"), so the realistic worst case is a small drop in precision when the generic-mention LLM mis-judges a real CamelCase mention as generic.
- **Why this is acceptable for Phase 2**: 
  - The LLM in `_classify_components` is **already prompted not to classify CamelCase names as ambiguous** (the few-shot `AMBIGUITY_FEW_SHOT` constant in `prompts_v2.py` carries that signal). Empirically the LLM has been **observed to emit only lowercase single-word names** as ambiguous on the 5 benchmark projects under 12c.
  - GATE-01 dual floor catches any concrete regression. If the slip channel turns out to be real on Phase 2's full sweep, the BBB-loosened gate still gives a 4pp cushion, and 13b can be reworked with a `len(n) > 1` + `name.islower()` guard re-introduced inline (no new file, no new variant).

**Recommendation for 13b plan**: include a **post-classification probe** (one-shot grep on the run log) that prints `model_knowledge.ambiguous_names` for each dataset to a log file, and flags any name that fails `name.islower()`. Non-blocking — just observability. Documented as the canary for the deferred prompt-side amendment (CONTEXT.md `<deferred>`).

**Recommendation for 13c plan**: include a **functional-parity probe** — after the hard tier, grep the run log for the printed `model_knowledge.ambiguous_names` set and confirm it is **byte-identical** to 13b's set on the same dataset. If they diverge, the timing-stream hypothesis (D-13a) is reconfirmed and gets logged.

---

## 4. Reuse pointers

- **Plan layout template**: `01-05-PLAN.md` — 4 tasks (file create + registration, hard-tier, checkpoint, full sweep). Phase 2 plans match shape 1:1; only differences are (a) different file being created and (b) different callsite-removal steps.
- **GATE-01 enforcement script**: `01-05-PLAN.md` Task 4 step 2 — the inline `GATE_EOF` python block that loads newest-run rows by variant and applies macro + per-dataset floors. Phase 2 reuses with `MACRO_FLOOR=0.93`, `PER_DS_TOLERANCE=0.02`, and **a per-dataset override for `bigbluebutton` to 0.04** (D-13).
- **Hard-tier gate script**: same plan, Task 2 — `python run_ablation.py --variants <name> --datasets teammates bigbluebutton`. Phase 2 reuses verbatim with the variant name swapped.
- **Taboo audit script**: same plan, Task 1 step 7 — but Phase 2 only runs it if a prompt body changed (D-09 says it doesn't). Phase 2 plans still keep the script as a paranoid sanity check; expected output `TABOO AUDIT CLEAN` since no prompt edits occur.
- **12c baseline JSON**: `results/ablation_results/ablation_20260528_173020.json` — used as the `f12c` source. **Do not re-run 12c** (D-02 / D-10).

---

## 5. Out-of-scope confirmations (per CONTEXT.md)

- No prompt edits anywhere in this phase (D-09). The `_classify_components` prompt body is unchanged.
- No removal of `len(n.split()) == 1` from L340 — that is a separate guard, not the structural-unambiguity filter.
- No prompts_v2 module changes (D-09).
- No `_classify_mention`, `_is_strong_alias`, `_get_strong_alias_mappings`, `_has_strong_alias_mention`, `_has_standalone_mention` touches — those are Phases 3, 4, 5 (D-18 / CONTEXT.md "Out of scope").

---

*Phase 2 research — short by design. The execution risk is dominated by Claude run-to-run variance (D-15) and the BBB timing-stream channel from Phase 1 (D-13a), both of which the gates already account for.*
