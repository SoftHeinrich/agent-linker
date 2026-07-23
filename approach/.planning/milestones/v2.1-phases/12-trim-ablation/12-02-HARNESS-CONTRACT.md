# Phase 12-02 — Single-Step Ablation Harness Contract

This is the canonical phase-to-upstream-checkpoint dependency table for
`llm_sad_sam.ablation.single_step`. Plans 12-03 / 12-04 / 12-05 cite this
document rather than re-derive the dependencies. Source authority for the
columns:

- **PHASE_ORDER + DOWNSTREAM_DEPS**: `src/llm_sad_sam/ablation/single_step.py`
  module-level constants. The harness imports them; tests pin them.
- **Phase → prompts map**: `.planning/research/PROMPT-HARNESS-SURVEY.md` §0
  (which prompts fire inside each phase of `s_linker13_clean.link()`).
- **Trim plan ownership**: 12-CONTEXT.md "Trim Steps".

## Dependency Table

| phase | upstream_checkpoint_required | downstream_phases_to_rerun | which_prompts_fire_in_this_phase | trim_plan_that_modifies_it |
|---|---|---|---|---|
| layer1 | (none — first phase) | layer2, entity_candidates, entity_decisions, final | `AMBIGUITY_FEW_SHOT`, `AMBIGUITY_RULES`, `DOC_KNOWLEDGE_EXTRACTION_RULES`, `DOC_KNOWLEDGE_JUDGE_EXAMPLES`, `DOC_KNOWLEDGE_JUDGE_RULES` | 12-03 (Step 1 — judge), 12-05 (Step 3 — runtime rubric on judge) |
| layer2 | layer1 | final | `ENTITY_EXTRACTION_RULES`, `VALIDATION_RULES`, `COREF_RULES`, `SEED_DISAMBIGUATION_RULES` | (none — Step 2 modifies entity_candidates / entity_decisions sub-phases inside the layer2 DAG; see rows below) |
| entity_candidates | layer1 | entity_decisions, final | `ENTITY_EXTRACTION_RULES` | 12-04 (Step 2 — ent+val merge) |
| entity_decisions | layer1, entity_candidates | final | `VALIDATION_RULES` | 12-04 (Step 2 — ent+val merge) |
| final | layer2, entity_candidates, entity_decisions | — | (no prompts — pure dedup) | (none) |

### Step 2 (12-04) re-runs TWO sub-phases

The ent+val merge collapses overlapping rubric across
`ENTITY_EXTRACTION_RULES` and `VALIDATION_RULES`. The harness handles this by
invoking `--phase entity_candidates` (extraction re-run) and then
`--phase entity_decisions` (validation re-run) sequentially with the new
merged prompt; the latter reads the freshly produced `entity_candidates.pkl`
from the per-run cache subdir.

### CRITICAL HARNESS CONTRACT (entity_candidates / entity_decisions reuse rule)

`layer2.pkl` is NOT a re-runnable phase — it is the synthesis pickle of the
`_run_parallel({seed_val, coref, entity})` block inside
`s_linker13_clean.link()`. The harness uses `layer2.pkl` purely as a CACHE
READ for the seed_val and coref tracks when re-running entity_candidates or
entity_decisions; the entity track is overridden surgically. Concretely:

- Load `layer1.pkl` -> restore `model_knowledge`, `doc_knowledge`.
- Load `layer2.pkl` -> reuse cached `seed_links` and `coref_links` as-is.
- Re-execute ONLY the entity sub-pipeline:
  - `phase == "entity_candidates"`: call `_extract_entities_enriched(...)` only.
  - `phase == "entity_decisions"`: load baseline `entity_candidates.pkl`, then
    call `_validate_with_evidence(...)` only.
- Reconstruct the layer2-equivalent state from
  `{seed_links: cached, coref_links: cached, validated_entity_links: fresh}`
  and feed into final dedup.

The harness MUST assert zero live LLM calls on the seed_val and coref tracks
when phase ∈ {entity_candidates, entity_decisions}. Implementation: monkey-patch
`_run_seed_validation` and `_run_coreference` to raise `AssertionError` if
ever called during the surgical re-run. Trim plans 12-04 ablations rely on
this surgical contract — without it, every Step 2 trim would pay live seed_val
+ coref cost it does not need.

## Per-Run Cache Subdir

Every `run_single_step` invocation receives a `results_dir` for the JSON
output AND uses a per-run tmp subdir for the `PHASE_CACHE_DIR` so the
variant's own `_save_phase` does not overwrite the canonical baseline cache
at `results/phase_cache/<variant>/<dataset>/`. The baseline cache is
read-only from the harness's perspective.

## Coupling Surface (Technical Debt)

The harness reaches into `s_linker13_clean` by method name:
`_run_entity_pipeline`, `_extract_entities_enriched`, `_validate_with_evidence`,
`_run_seed_validation`, `_run_coreference`, `_save_phase`, `_checkpoint_dir`,
`model_knowledge`, `doc_knowledge`, `_current_text_path`, `_ilinker3`. These
are semi-private but stable in `s_linker13_clean`. Phase 13 promotion
(`s_linker13_min`) MUST either preserve these names or update the harness
in lock-step.

## Cross-References

- 12-CONTEXT.md "Execution Method — Checkpoint-Loaded Single-Step Ablation
  (USER DIRECTIVE)" — the user-imposed constraint that forbids full-pipeline
  re-runs per trim.
- `src/llm_sad_sam/ablation/single_step.py` — the canonical constants
  PHASE_ORDER + DOWNSTREAM_DEPS that tests pin.
- `tests/test_single_step_harness.py` — pins the contract rows above.
- `tests/test_mention_type_ablation.py` — pre-existing checkpoint-load pattern
  the harness extends with phase-replay capability.
