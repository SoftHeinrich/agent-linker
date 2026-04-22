# Feature Research

**Domain:** Empirical ablation research pipeline — rule-to-LLM replacement in a traceability linker
**Researched:** 2026-04-21
**Confidence:** HIGH (derived from direct codebase reading and spike audit documents, no inference needed)

## Feature Landscape

### Table Stakes (Users Expect These)

Features the research pipeline must have to produce a paper-worthy ablation. Missing these means the results are not reproducible or not credible.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Reproducible per-variant runs | Any ablation claim requires the same inputs, same seed, same docs each run. Without this, delta F1 between variants is noise, not signal. | LOW | `run_ablation.py` already loads the same benchmark paths deterministically; LLM variance is the residual — fixed model + fixed prompt is the right control. Temperature is not a knob here (prior memory: temperature 0.0 slightly worse). |
| Per-dataset and macro F1 reporting | The five benchmark projects have different difficulty profiles. A variant that wins macro F1 while collapsing on teammates/BBB is not promoted. | LOW | Already implemented in `run_variant()` + `print_summary()`. Extend output rather than rebuild. |
| FP/FN counts and source attribution | Ablation papers need to show which phase produced each surviving FP, and whether FN recovery is happening. Without source tagging the ablation story is unverifiable. | LOW | Already partially implemented: `fp_by_source` uses `link.source` field. Needs FP-phase bucketing added (seed / entity / coref). FN attribution already has `transarc_had` flag. |
| F1 floor enforcement with clear "rejected" status | Every variant below macro F1 ≥ 93% must be flagged rejected, not silently reported. The floor is the gate that keeps the paper's promotion chain credible. | LOW | Not yet implemented. Add a post-run check in `run_ablation.py` or as a reporting annotation. Floor = 93% per PROJECT.md. |
| Standalone linker file per variant | Each rule-removal lands as `s_linker13a.py`, `s_linker13b.py`, etc. — not an inheritance chain. User preference is explicit. | LOW | Established pattern in the codebase. Write each variant as a full file with a docstring enumerating what was removed vs baseline. |
| Variant registration in `CANONICAL_VARIANTS` and `VARIANT_SPECS` | `run_ablation.py` discovers variants through these two structures. A variant that is not registered cannot be run. | LOW | Mechanical step; include it in the "definition of done" for each variant. |
| Hard-tier-first dataset routing | Teammates and BBB are the most rule-sensitive projects (most FPs from ambiguous components, most coref complexity). Running only these two first gives cheap signal before committing to a full 5-project sweep. | LOW | `--datasets teammates bigbluebutton` already works. The feature is CLI routing discipline, not new code. Document as the required development loop: hard-tier gate first, then full 5-project sweep before promotion. |
| Spike re-validation harness | Spikes 001/002/003 were validated in isolation. Integration can surface new failure modes. Each spike integration must be re-validated inside the full pipeline on both hard-tier datasets before the variant is registered. | MEDIUM | Not yet a formal step. Needs a test entry point: run spike-replacement variant on hard tier, compare delta to baseline 12c, confirm no regression before writing the full standalone linker file. |
| Ablation diff (rules removed between adjacent variants) | Each variant's docstring and the ablation table must record which helpers were removed and which structural regex sites were retired. This is the machine-readable audit trail. | LOW | Implement as a structured docstring section `REMOVED_FROM: <parent>` + `RULES_REMOVED: [list]`. The ablation table in the paper is then generated from these docstrings. |
| Macro F1 ablation table (one row per promoted variant) | The deliverable is a table: `12c → 13a → 13b → … → 13` with per-dataset F1, delta F1, and rules-removed column. This table is what the paper presents. | LOW | Implement as a script that reads JSON result files from `results/ablation_results/` and generates the table. |

### Differentiators (Competitive Advantage)

Features that strengthen the paper's contribution beyond basic ablation numbers.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Evidence-cite prompt pattern (Spike 001 validated) | The cite-evidence pattern (LLM emits the matched span as evidence, not just a boolean) is the architectural insight that makes rule removal safe. It is the paper's core technique. Every rule-replacement variant must use this pattern consistently — not just for trailing words but for mention classification, alias scoping, and ambiguity. | MEDIUM | Pattern is proven for trailing-word enrichment. Replications for `_classify_mention` (Spike 003) and `_is_strong_alias` scope field need to follow the same evidence-guardrail structure: prompt asks for evidence, response is rejected if evidence does not appear in the input text. |
| Per-phase FP attribution (which phase produced each FP) | Reviewers will ask "where do the remaining FPs come from?" Showing FP counts by phase (seed / entity / coref) across the promotion chain demonstrates the ablation is not just moving FPs between phases. | MEDIUM | Extend `fp_by_source` bucketing: seed FPs = `source=="seed"`, entity FPs = `source=="entity"`, coref FPs = `source=="coreference"`. Already possible from existing `link.source` field. Add a per-phase FP table to the JSON output. |
| Prompt-schema versioning (coexistence of variant A and B prompts) | When two adjacent variants differ in their prompt schemas (e.g., alias discovery now emits `scope: global|local`), the prompts must be able to coexist in the repo without one overwriting the other. | LOW | Use `prompts_v2.py` as the stable base. Variant-specific prompt additions live in the linker file itself (as class-level constants or module-level strings), not in `prompts_v2.py`. This is already the pattern for `SEED_DISAMBIGUATION_RULES` in `s_linker12c.py`. |
| Regression alert on hard-tier before full sweep | If a new variant regresses BBB or teammates below its parent variant by more than 1pp, the full 5-project sweep should not run. This saves cost and prevents wasted runs on variants that are already broken. | LOW | Implement as a CLI mode: `--fast-check` runs only hard-tier datasets and prints PASS/FAIL relative to a named baseline variant. Not a complex feature — it is the development loop discipline enforced by a flag. |
| FN recovery tracking (`transarc_had` field) | The `fn_details` output already has `transarc_had: bool`. Tracking which FNs were recoverable by TransArc (but missed by the LLM pipeline) across the promotion chain shows whether rule removal changes the recall ceiling. | LOW | Already in `run_variant()`. Extend the ablation table to include a `fn_transarc_recoverable` column. This is a 5-line addition to the JSON aggregation. |
| Benchmark-taboo audit step in the integration checklist | Every time a new prompt constant is written for a rule-replacement variant, it must be audited against `BENCHMARK_TABOO.md` before the variant is registered. This is a process differentiator: it makes the paper's "no data leakage" claim defensible. | LOW | Enforce as part of the spike re-validation harness: run a grep/scan for known taboo terms against all new prompt strings. Not automated; manual review with a checklist. |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| GPT backend support / cross-model evaluation | Prior work showed GPT-5.2 has a 3.9pp gap and massive run-to-run variance (±5-12 links). It is tempting to include for completeness. | Adding GPT as a gate adds variance that cannot be controlled. Prior memory is explicit: the gap is inherent model capability, not fixable, and was already documented. Running GPT sweeps doubles cost without adding to the ablation story. | Claude Sonnet only. The paper's claim is "LLM-driven pipeline"; it does not need to be model-agnostic. Note GPT compatibility as a limitation in the paper. |
| Cost optimization / LLM call batching across variants | Sharing phase 1 results (model analysis, doc knowledge) across multiple variants in a single run would save API calls. | Phase 1 LLM outputs (ambiguous names, aliases) vary between runs due to LLM variance. If two variants share phase 1 results, any F1 difference becomes ambiguous: did the rule removal cause it, or did the shared phase 1 output? | Run each variant independently from scratch, even if slower. PROJECT.md is explicit: "no LLM budget limit; replaceability trumps cost." |
| Inheritance chain for linker variants | Tempting to make `s_linker13a` inherit from `s_linker13` and override only changed methods. Reduces duplicate code. | User preference is explicit against inheritance chains. More importantly, inheritance makes the "what was removed" audit non-obvious — you have to trace the MRO to see what changed. For a paper's supplement, standalone files are the reproducibility artifact. | Standalone files, duplicate code intentionally. Each file's docstring is its own changelog. |
| Automated prompt-leakage detection | Running a regex scan against `BENCHMARK_TABOO.md` terms on every commit sounds like a quality gate. | The taboo list is project-specific domain knowledge; automating it creates a false sense of security. The real leakage risk is semantic (a safe-sounding SE domain example that happens to mirror a benchmark component's role), not keyword-matchable. | Manual audit step in the integration checklist. The checklist is the control, not automation. |
| Phase checkpoint sharing across variants for "ablation replay" | Saving phase 1 outputs once and reusing them for all variants would make ablation faster and more controlled. Prior codebase (V30d) had this with `resume_from_phase`. | This is cost optimization in disguise. See above. Additionally, the RISKY helper `_has_standalone_mention` is used in both phase 1 (anchor collection) and phase 2 (evidence bundle construction). If a variant removes it, phase checkpoint replay from a 12c checkpoint would include 12c's anchor-collection behavior, invalidating the ablation. | Full independent runs only. |
| Dynamic F1 floor based on per-dataset variance | It might seem more principled to set a floor per dataset rather than a single macro floor. | Adds complexity with no benefit for a 5-project benchmark. The 93% macro floor already requires BBB and TM (the hardest datasets) to not collapse since they have the most room to regress. Per-dataset floors would require maintaining 5 separate thresholds and justifying them in the paper. | Single macro F1 floor = 93%. Report per-dataset breakdowns in the ablation table. |

## Feature Dependencies

```
Hard-tier-first routing
    └──requires──> Variant registration (CANONICAL_VARIANTS + VARIANT_SPECS)
                       └──requires──> Standalone linker file per variant

Ablation diff table
    └──requires──> Structured docstring (REMOVED_FROM + RULES_REMOVED per variant)
    └──requires──> JSON result files per variant run

F1 floor enforcement
    └──requires──> Per-dataset and macro F1 reporting (already exists)

Spike re-validation harness
    └──requires──> Hard-tier-first routing (run on TM+BBB, not all 5)
    └──requires──> Benchmark-taboo audit (manual check before registering)

Per-phase FP attribution
    └──requires──> link.source field (already exists in SadSamLink)
    └──enhances──> Ablation diff table (shows FP movement across variants)

Evidence-cite prompt pattern
    └──enables──> each REPLACEABLE helper becoming LLM-driven
    └──must-precede──> Standalone linker file for that removal

Prompt-schema versioning (variant-local constants)
    └──enables──> Coexistence of alias-scope schema (13a+) with pre-scope schema (12c)
    └──requires──> prompts_v2.py treated as read-only stable base
```

### Dependency Notes

- **Spike re-validation requires hard-tier routing:** The spike was validated in isolation. The only cheap way to confirm pipeline integration is to run the new variant on TM + BBB (the two datasets most sensitive to rule changes) before committing to a full 5-project sweep.
- **Ablation diff requires structured docstrings:** The ablation table is generated from result JSON + docstring metadata. If docstrings are unstructured prose, the table cannot be generated consistently.
- **Evidence-cite pattern enables all REPLACEABLE removals:** Spikes 001 and 003 both use the same structural idea — the LLM emits an evidence field that is cross-checked against the input. This pattern is the precondition for trusting each removal. Without it, rule removal is just hoping the LLM is right.
- **Prompt-schema versioning conflicts with shared `prompts_v2.py` edits:** Once alias discovery emits `scope: global|local` (for `_is_strong_alias` removal), the extraction prompt that consumes the alias list must also change. These changes must live in the variant file, not in `prompts_v2.py`, or they will silently break older variants.

## MVP Definition

### Launch With (v1 — first promoted variant, `s_linker13a`)

- [x] Standalone linker file per variant (established pattern, zero new work)
- [x] Variant registration in CANONICAL_VARIANTS + VARIANT_SPECS (mechanical, must be done)
- [ ] Spike 001 integrated: `_split_component_name` removed, trailing-word enrichment is fully LLM-driven with evidence guardrail — re-validated on hard tier
- [ ] Benchmark-taboo audit of all new prompt strings in 13a
- [ ] F1 floor check: 13a macro F1 ≥ 93% confirmed, result recorded in ablation table row
- [ ] `REMOVED_FROM: s_linker12c` + `RULES_REMOVED: [_split_component_name]` in 13a docstring

This is the minimum needed to demonstrate the promotion chain exists and the process is credible.

### Add After Validation (v1.x — during the 13b-13e removal chain)

- [ ] Hard-tier regression alert (`--fast-check` flag) — add when variant iteration becomes frequent enough that manual dataset selection is error-prone
- [ ] Per-phase FP attribution in JSON output — add when the ablation table needs the FP-source breakdown column (likely after 13b or 13c when validation rules are changed)
- [ ] Spike 003 integrated: `_classify_mention` folded into entity-extraction prompt — re-validated on hard tier before registering 13b
- [ ] `_is_structurally_unambiguous` removed (trust LLM ambiguity classification directly) — 13c
- [ ] `_is_ambiguous_name_component` wrapper inlined — 13c (trivially follows 13c)
- [ ] `scope` field added to alias discovery prompt; `_is_strong_alias` + `_get_strong_alias_mappings` retired — 13d
- [ ] `_has_strong_alias_mention` folded into coref prompt schema — 13e

### Future Consideration (v2+ — after full promotion chain lands)

- [ ] Ablation table generator script (reads JSON results, produces LaTeX/Markdown table) — defer until all variants are registered and the final results are stable
- [ ] `_has_standalone_mention` fate decision — keep as boundary primitive or prove LLM parity on anchor collection. This requires a dedicated spike (Spike 002 classified it RISKY, O(N·M) in anchor collection). Do not attempt this until all other removals are in.
- [ ] FN recovery tracking column in ablation table — add after full chain is done; the signal is only meaningful when the full progression is visible

## Feature Prioritization Matrix

| Feature | Research Value | Implementation Cost | Priority |
|---------|---------------|---------------------|----------|
| Standalone file + variant registration | HIGH (reproducibility) | LOW | P1 |
| Spike re-validation harness (hard-tier check) | HIGH (correctness gate) | LOW | P1 |
| F1 floor enforcement with rejected status | HIGH (paper credibility) | LOW | P1 |
| Ablation diff (structured docstring + table row) | HIGH (paper deliverable) | LOW | P1 |
| Evidence-cite pattern in each replacement | HIGH (core technique) | MEDIUM | P1 |
| Benchmark-taboo audit step | HIGH (leakage prevention) | LOW | P1 |
| Per-phase FP attribution | MEDIUM (reviewer question) | LOW | P2 |
| Hard-tier regression alert (`--fast-check`) | MEDIUM (iteration speed) | LOW | P2 |
| Prompt-schema versioning (variant-local constants) | MEDIUM (coexistence) | LOW | P2 |
| FN recovery tracking column | LOW (supplementary) | LOW | P3 |
| Ablation table generator script | LOW (formatting aid) | LOW | P3 |

**Priority key:**
- P1: Must have for any promoted variant to be credible
- P2: Should have, add as the promotion chain grows
- P3: Nice to have, add after the full chain is stable

## Competitor Feature Analysis

This project has no direct software competitors. The relevant comparison is against the prior S-Linker series (what the current runner already supports) and what the paper's ablation section requires.

| Feature | Existing `run_ablation.py` | What's Missing |
|---------|---------------------------|---------------|
| Per-dataset F1 | Yes — full metrics per dataset | F1 floor rejection annotation |
| FP by source | Yes — `fp_by_source` dict | Per-phase bucketing (seed/entity/coref as named tiers) |
| FN details | Yes — `transarc_had` flag | Column in the ablation table |
| Hard-tier routing | Yes — `--datasets` flag | Documentation as required development loop |
| Variant registration | Yes — CANONICAL_VARIANTS | New 13x entries (mechanical, per-variant work) |
| Checkpoint saving | Yes — pickle per phase in 12c | No sharing across variants (intentional anti-feature) |
| Ablation diff | No | Structured docstring convention + table generator |
| Spike integration gate | No | Hard-tier re-validation step before variant registration |
| Benchmark-taboo check | Manual only | Checklist item in integration process |

## Sources

- `.planning/PROJECT.md` — requirements, constraints, F1 floor, dataset schedule, out-of-scope decisions
- `.planning/spikes/002-rules-audit/AUDIT.md` — 12-helper classification, ranked removal order, RISKY/ESSENTIAL/REPLACEABLE verdicts
- `.planning/spikes/MANIFEST.md` — spike status and validated patterns
- `src/llm_sad_sam/linkers/experimental/s_linker12c.py` — phase structure, rule call sites, checkpoint/logging infrastructure
- `run_ablation.py` — existing runner capabilities, VARIANT_SPECS schema, metrics implementation
- `MEMORY.md` (project memory) — V31/V32 ablation history, GPT gap, prompt leakage incidents, heuristic ablation findings

---
*Feature research for: rule-replacement ablation research pipeline (s_linker12c → s_linker13)*
*Researched: 2026-04-21*
