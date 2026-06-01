# Link Provenance Data Structure — Deferral Note

**Captured:** 2026-05-31
**User directive:** "add datastructure in the final output for deeper ablation: which sub-linker produce which links; which validator killed which links to full ablation"
**Status:** Deferred to Phase 13 or v2.2 (out of scope for Phase 12 close)

## What The User Asked For

Each final `TraceLink` (and each candidate link that was *killed* before reaching the final set) should carry full provenance:

- **Producer** — which sub-pipeline emitted it (seed_extraction, coreference, entity_extraction, partial_inject, validated_entity, …)
- **Validator chain** — the ordered list of validators that touched the candidate
- **Verdict per validator** — KEPT / KILLED + the validator's reason
- **Evidence spans** — the text spans the validator cited (already partially present in `bundles`)

This enables *post-hoc deep ablation*: instead of re-running the LLM pipeline to test "what if we drop validator X", read the provenance and simulate the drop offline. Massive cost reduction for future ablation work.

## Why Defer

This is NOT a prompt-trim — it is a **codebase-wide data-structure refactor** affecting:

1. `core/data_types_v2.py` (or new `data_types_v3.py` sibling) — extend `TraceLink` with `provenance: ProvenanceRecord`.
2. `s_linker13.py` (FROZEN — cannot edit) → only the new variant `s_linker13_clean_v3` or a fresh `_v3+provenance` sibling.
3. Every pipeline stage that creates or filters a link → must record into the provenance log.
4. `ablation/single_step.py` (Phase 12 harness) → extend to surface provenance in results JSON.
5. `evaluate.py` (metrics_output) — extend to leverage provenance for deeper FP/FN attribution.
6. The 5-dataset baseline `tests/fixtures/v2_0_baseline.json` does not need re-snapshotting; provenance is *additive*.

Scope is a phase of its own — appropriate to land as either:
- **Phase 13 Plan 13-02** (alongside `s_linker13_min` promotion), so the canonical promotion ships with full provenance.
- **v2.2 dedicated milestone** ("Ablation Infrastructure"), keeping v2.1 focused on prompt trimming.

## Recommended Shape (For When It Lands)

```python
@dataclass(frozen=True)
class ValidatorTouch:
    validator_id: str          # e.g. "convention_filter", "judge_v3", "ambiguity_gate"
    verdict: Literal["KEEP", "KILL", "PROMOTE", "DEMOTE"]
    reason_code: str           # short stable id, e.g. "BENCHMARK_TABOO_MATCH", "AMBIGUOUS_NAME_UNRESOLVED"
    evidence_span: tuple[int, int] | None  # sentence offsets

@dataclass(frozen=True)
class ProvenanceRecord:
    producer: str              # "seed_extraction" | "coreference" | "entity_extraction" | "partial_inject" | ...
    producer_confidence: float
    touches: tuple[ValidatorTouch, ...]
    final_verdict: Literal["EMITTED", "KILLED"]
    killer: str | None         # validator_id of the killer (if KILLED), else None

@dataclass(frozen=True)
class TraceLinkV3:
    sentence: int
    component: str
    source: str                # KEPT for back-compat with v2.0 baseline JSON
    confidence: float
    provenance: ProvenanceRecord
```

## Result JSON Extension

`results/ablation_results/<plan>/<variant>/<dataset>/final.json` gains:

```json
{
  "links": [...existing TraceLink dicts with .provenance...],
  "killed_candidates": [
    {"sentence": N, "component": "X", "provenance": {...}}
  ],
  "validator_stats": {
    "convention_filter": {"kept": 87, "killed": 12, "kill_reasons": {"BENCHMARK_TABOO_MATCH": 8, ...}},
    "judge_v3": {...},
    "ambiguity_gate": {...}
  }
}
```

## What v2.1 Phase 12 Gets Without This

The current Phase 12 harness already records phase-level F1 deltas (`results/ablation_results/12_XX/.../verdict.json`). This is sufficient for the *single-prompt trim* ablation Phase 12 was designed for. The provenance refactor unlocks *cross-validator* and *cross-producer* ablation, which is the next layer of ablation depth — naturally a Phase 13 or v2.2 deliverable.

## Phase 13 Plan Hook

When Phase 13 plans land, ensure a `Plan 13-02 — Link Provenance for Ablation Infrastructure` (or equivalent) is on the roadmap. Wire the provenance data structure into `s_linker13_min` at promotion so the canonical artifact ships with provenance from day one. This also gives Phase 13's ABLATION-TABLE.md (REQUIREMENTS GATE-03) a richer schema to populate.
