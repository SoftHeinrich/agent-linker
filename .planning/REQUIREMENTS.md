# Requirements: llm-sad-sam-v45 Rule-to-LLM Ablation

**Defined:** 2026-04-22
**Core Value:** Every rule removed from `s_linker12c` and replaced by an LLM primitive must hold macro F1 ≥ 93% (and no single dataset > 2pp below 12c baseline) — or be rejected. Deliverable: defensible claim that traceability linking works without hand-crafted structural rules.

## v1 Requirements

Requirements for the 13-series promotion chain. Each maps to roadmap phases.

### Baseline & Infrastructure

- [ ] **INFRA-01**: Reproducible `s_linker12c` baseline captured (per-dataset + macro F1, FP/FN table, JSON result file in `results/ablation_results/`)
- [ ] **INFRA-02**: `anthropic>=0.40.0` SDK added to `pyproject.toml`; `llm_client.py` migrated from `claude -p` subprocess to direct SDK call with `temperature=0.0` and prompt caching
- [ ] **INFRA-03**: `diskcache>=5.6.1` and `tabulate>=0.9.0` added to `pyproject.toml`
- [ ] **INFRA-04**: Baseline F1 unchanged (within run-to-run variance) after SDK migration, confirmed on hard tier first, then full 5-project sweep
- [ ] **INFRA-05**: Each variant's `_checkpoint_dir` namespaced per variant (no `"s_linker12c"` hardcoded string leaking across variants)

### Variant Promotion Chain (13a → 13f)

- [ ] **VAR-01**: `s_linker13a.py` — Spike 001 integrated; `_split_component_name` removed; `_enrich_trailing_words` replaced by LLM + evidence guardrail. Hard-tier then full sweep. Dual floor met.
- [ ] **VAR-02**: `s_linker13b.py` — `_is_structurally_unambiguous` post-filter removed; trust LLM ambiguity classification from `_classify_components`. Dual floor met.
- [ ] **VAR-03**: `s_linker13c.py` — `_is_ambiguous_name_component` wrapper inlined/removed. Dual floor met.
- [ ] **VAR-04**: `s_linker13d.py` — Spike 003 integrated; `_classify_mention` 4-branch regex replaced by LLM enum emission piggybacked on entity-extraction prompt. Enum-contract test proves byte-identical strings or downstream prompts updated accordingly. Dual floor met.
- [ ] **VAR-05**: `s_linker13e.py` — alias discovery prompt extended to emit `scope: global|local`; `AliasEntry` defined inline; `_is_strong_alias` + `_get_strong_alias_mappings` removed. Run twice on hard tier before promotion. Dual floor met.
- [ ] **VAR-06**: `s_linker13f.py` — strong-alias-mention signal folded into coref prompt evidence schema; `_has_strong_alias_mention` removed. Dual floor met.

### Promotion & Ablation Artifact

- [ ] **PROMO-01**: Winning variant promoted as `s_linker13.py` with zero non-trivial rules (only `_has_standalone_mention` + parsers/formatters surviving)
- [ ] **PROMO-02**: `_has_standalone_mention` KEEP decision formally logged as Key Decision in PROJECT.md (RISKY per Spike 002, O(N·M) call sites)
- [ ] **PROMO-03**: Ablation table generated: one row per variant (12c → 13a → 13b → 13c → 13d → 13e → 13f → 13) with per-dataset F1, ΔF1 vs parent, rules-removed list, FP-by-phase (seed/entity/coref). Output: markdown + LaTeX via `tabulate`.
- [ ] **PROMO-04**: Research writeup (markdown) documenting methodology, results, and the retained-primitive rationale

### Process & Quality Gates

- [ ] **GATE-01**: Every variant passes dual floor: macro F1 ≥ 93% AND no dataset > 2pp below 12c per-dataset baseline
- [ ] **GATE-02**: Every variant registered in `CANONICAL_VARIANTS` and `VARIANT_SPECS` in `run_ablation.py`
- [ ] **GATE-03**: Every variant has structured docstring with `REMOVED_FROM: <parent>` and `RULES_REMOVED: [...]`
- [ ] **GATE-04**: Benchmark-taboo audit performed on every new prompt constant against `BENCHMARK_TABOO.md` before variant registration
- [ ] **GATE-05**: Hard-tier-first dev loop enforced: regress >1pp on BBB or TM vs parent → no full sweep, re-work variant
- [ ] **GATE-06**: Per-variant independent runs (no phase-checkpoint replay across variants; no shared phase 1 LLM outputs)

## v2 Requirements

Deferred to future milestone. Tracked but not in current roadmap.

### Extended Ablation

- **EXT-01**: Spike on replacing `_has_standalone_mention` with LLM primitive (cost-benefit; likely rejected but documented)
- **EXT-02**: Drop `_has_standalone_mention` dotted-path guard; let LLM mention classifier handle dotted-path detection
- **EXT-03**: GPT-5.2 cross-model re-evaluation of `s_linker13` (documents the Claude-vs-GPT gap on the rule-free pipeline)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| GPT backend as promotion gate | Prior work: 3.9pp gap is inherent model capability, not fixable. Claude Sonnet only. Mention as limitation in paper. |
| Cost optimization / cross-variant call batching | Phase 1 LLM variance makes shared outputs ambiguate ablation deltas. Full independent runs per variant. |
| Inheritance chains between variants | User preference against; standalone files are the reproducibility artifact. |
| Phase checkpoint replay across variants | `_has_standalone_mention` used in anchor collection; replaying 12c checkpoints into 13f invalidates ablation. |
| Automated taboo regex scan on commit | Real leakage risk is semantic, not keyword-matchable. Manual audit step only. |
| Per-dataset F1 floor thresholds (different per project) | Adds complexity; single 93% macro floor + "no dataset >2pp below 12c" already covers the worst cases. |
| New seed/linker approaches (ILinker3+, ensembles) | This milestone is rule-reduction on 12c, not seed exploration. |
| SAM-Code / SAD-Code tasks | Dataset scope is SAD-SAM only. |
| Temperature / seed tuning for reproducibility | Claude has no seed param; temperature 0.0 already in plan. Prior memory shows tuning slightly worse. |

## Traceability

Which phases cover which requirements. Filled during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 1 | Pending |
| INFRA-02 | Phase 1 | Pending |
| INFRA-03 | Phase 1 | Pending |
| INFRA-04 | Phase 1 | Pending |
| INFRA-05 | Phase 1 | Pending |
| VAR-01 | Phase 2 | Pending |
| VAR-02 | Phase 3 | Pending |
| VAR-03 | Phase 3 | Pending |
| VAR-04 | Phase 4 | Pending |
| VAR-05 | Phase 5 | Pending |
| VAR-06 | Phase 6 | Pending |
| PROMO-01 | Phase 7 | Pending |
| PROMO-02 | Phase 7 | Pending |
| PROMO-03 | Phase 7 | Pending |
| PROMO-04 | Phase 7 | Pending |
| GATE-01 | All phases | Pending |
| GATE-02 | All phases | Pending |
| GATE-03 | All phases | Pending |
| GATE-04 | All phases | Pending |
| GATE-05 | All phases | Pending |
| GATE-06 | All phases | Pending |

**Coverage:**
- v1 requirements: 22 total
- Mapped to phases: 22
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-22*
*Last updated: 2026-04-22 after initial definition*
