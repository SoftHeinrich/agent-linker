"""Single-step ablation engine for the v2.1 trim chain (Phase 12, PROMPT-02).

Loads upstream checkpoints from `results/phase_cache/<variant>/<dataset>/`,
re-executes ONE target phase on a given variant, propagates the result through
the downstream phases that depend on the modified phase's output, scores
against the gold standard, and writes a per-run results JSON with delta vs the
v2.0 baseline.

Why this exists: 12-CONTEXT.md "Execution Method — Checkpoint-Loaded
Single-Step Ablation (USER DIRECTIVE)" rules out full-pipeline sweeps per
trim. The harness reuses the cached output of upstream phases instead of
recomputing them, and only re-runs the modified phase plus its downstream
descendants per `DOWNSTREAM_DEPS`.

CRITICAL HARNESS CONTRACT (entity_candidates / entity_decisions reuse rule)
--------------------------------------------------------------------------
`layer2.pkl` is NOT a re-runnable phase — it is the synthesis pickle of the
`_run_parallel({seed_val, coref, entity})` block inside
`s_linker13_clean.link()`. When the requested phase is ``entity_candidates``
or ``entity_decisions``, the harness uses ``layer2.pkl`` purely as a CACHE
READ for the seed_val + coref tracks; the entity track is overridden
surgically. The harness MUST NOT make any live LLM calls to
``_run_seed_validation`` or ``_run_coreference`` in this mode — Task 2's
acceptance test asserts this by raising if either method is invoked.

PHASE -> DOWNSTREAM RE-RUN TABLE (see 12-02-HARNESS-CONTRACT.md)
---------------------------------------------------------------
- layer1            -> layer2, entity_candidates, entity_decisions, final
- layer2            -> final
- entity_candidates -> entity_decisions, final
- entity_decisions  -> final
- final             -> (terminal, no downstream)

The harness coupling reaches into the semi-private methods
``_run_entity_pipeline``, ``_extract_entities_enriched``,
``_validate_with_evidence``, ``_run_seed_validation``,
``_run_coreference`` by name. They are stable in ``s_linker13_clean`` (Phase
10 promotion contract), but any future refactor MUST preserve them or
update this harness in lock-step. Tracked as technical debt in
12-02-SUMMARY.md.
"""

from __future__ import annotations

#: Canonical pipeline order. Indices into this tuple drive the upstream
#: checkpoint requirements (every phase before X must exist before X can
#: re-run) and the downstream re-run set (every phase after X may need to
#: re-execute when X's output changes).
PHASE_ORDER = ("layer1", "layer2", "entity_candidates", "entity_decisions", "final")


#: Per-phase downstream re-run dependency. Maps a modified phase to the
#: tuple of phases that MUST be re-run when its output changes. Sourced
#: from 12-CONTEXT.md decisions ("the executor MUST either re-run downstream
#: phases or use existing checkpoint") and the s_linker13_clean.link()
#: DAG structure: seed_val + coref + entity all read layer1 state, and
#: dedup ("final") reads everything in layer2 + entity_decisions.
DOWNSTREAM_DEPS = {
    "layer1": ("layer2", "entity_candidates", "entity_decisions", "final"),
    "layer2": ("final",),
    "entity_candidates": ("entity_decisions", "final"),
    "entity_decisions": ("final",),
    "final": (),
}
