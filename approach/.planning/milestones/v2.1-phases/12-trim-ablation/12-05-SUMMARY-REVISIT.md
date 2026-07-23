---
phase: 12-trim-ablation
plan: 12-05-REVISIT
title: Step 3 Inference-Time Rubric — REVISIT (overturns prior GATE-06 REJECT; new REJECT on cross-model)
status: completed
verdict: REJECT (cross-model arm; NOT leakage)
completed: 2026-05-31
requirements: [PROMPT-01, PROMPT-02]
tags: [trim, prompt-engineering, runtime-rubric, AHE, agentic-rubrics, gate-06, gate-01, revisit, methodological-correction]
dependency-graph:
  requires: [12-01, 12-02, 12-CONTEXT, 12-05 (prior REJECT — preserved)]
  provides: [revisit verdict, cross-dataset isolation methodology, refined GATE-06 operationalization]
  affects: [12-06, 13-01]
tech-stack:
  added: ["scripts/audit_12_05_revisit.py — cross-dataset isolation audit"]
  patterns: ["GATE-06 operationalized as cross-dataset rubric isolation (vs. strict static-output-purity)"]
key-files:
  created:
    - .planning/phases/12-trim-ablation/12-05-SUMMARY-REVISIT.md
    - scripts/audit_12_05_revisit.py
    - results/ablation_results/12_05_trim3_runtime_rubric/revisit_audit.json
    - results/ablation_results/12_05_trim3_runtime_rubric/claude_revisit/<3 datasets>/layer1.json
    - results/ablation_results/12_05_trim3_runtime_rubric/gpt54_revisit/<5 datasets>/layer1.json
  modified:
    - results/ablation_results/12_05_trim3_runtime_rubric/verdict.json (revisit verdict supersedes prior; prior preserved in 12-05-SUMMARY.md)
decisions:
  - "Methodological correction: GATE-06 prohibits hardcoded benchmark vocabulary in STATIC SOURCE/PROMPTS; runtime LLM analysis of input data is what GATE-06 mandates, NOT forbids."
  - "Operationalized cross-dataset isolation test (Test C): term t in dataset A's rubric is a leak iff (a) t is a PCM component of dataset B != A, AND (b) t is NOT in A's PCM, AND (c) t is NOT in A's input document text. This rejects lexical-coincidence false positives."
  - "Claude PASSES the relaxed v2.1 GATE-01 (macro 0.9396) AND cross-dataset isolation."
  - "gpt-5.4 PASSES cross-dataset isolation but FAILS GATE-01 cross-model floor (0.8855 < 0.8977, gap -1.22pp)."
  - "Final verdict: REJECT — but on a cross-model capability gap, NOT on leakage. The prior leakage-basis REJECT is OVERTURNED."
  - "Variant is NOT carried to Plan 13-01 (s_linker13_min) due to cross-model failure. Open for v2.2+ revisit if cross-model gap closes."
metrics:
  duration_min: ~50 (Claude full sweep 17min + gpt-5.4 full sweep 24min + analysis)
  tasks_completed: 4
  llm_calls_revisit: 10 rubric-builder calls (5 Claude + 5 gpt-5.4) plus downstream layer1/2 cascade
  files_modified: 4
  commits: TBD
---

# Phase 12 Plan 12-05 REVISIT — SUMMARY

**One-liner:** Re-evaluation of the runtime-rubric trim under the correct GATE-06 reading (dynamic runtime LLM analysis of input is *mandated* by the rule, not forbidden); cross-dataset isolation holds on both backends, static surface clean, relaxed Claude gate PASSES, cross-model gpt-5.4 gate FAILS by 1.22pp. Prior leakage-basis REJECT is OVERTURNED; new verdict REJECTS on a model-capability gap, not on methodology.

## Verdict

| Gate | Required | Observed | Status |
|------|----------|----------|--------|
| GATE-06 static surface | 0 taboo hits in seed example + prompt template | 0 hits each | PASS |
| GATE-06 cross-dataset rubric isolation (Claude) | no rubric for A contains another dataset's component vocabulary unless verified in A's doc | 0 violations / 1 benign lexical overlap ("UI") | PASS |
| GATE-06 cross-dataset rubric isolation (gpt-5.4) | same | 0 violations / 1 benign lexical overlap ("UI") | PASS |
| GATE-01 Claude RELAXED — macro F1 | ≥ 0.90 | **0.9396** | PASS |
| GATE-01 Claude RELAXED — BBB absolute | ≥ 0.79 | **0.8108** | PASS |
| GATE-01 Claude RELAXED — per-dataset drop | ≥ −2pp vs s_linker13_clean | all within ±2pp | PASS |
| GATE-01 cross-model (gpt-5.4) — macro F1 | ≥ 0.8977 | **0.8855** | **FAIL (−1.22pp)** |
| GATE-02 frozen-file invariant | `git diff --quiet` on 6-file set | clean | PASS |

**Overall:** REJECT — **on cross-model F1 gap, NOT on leakage.** The prior REJECT's GATE-06 leakage basis is OVERTURNED. The variant remains rejected for v2.1 promotion but for a different (model-capability) reason; the runtime-rubric pattern itself is GATE-06-compliant under the methodologically-correct operationalization.

## Methodological Correction

The prior Plan 12-05 SUMMARY (verdict_timestamp 2026-05-31T12:30:00Z, committed) applied this reading of GATE-06:

> "Any benchmark-derived term in any LLM input/output = leakage."

This reading is incorrect. CLAUDE.md's actual GATE-06 spec reads (verbatim):

> "Never hardcode word lists derived from benchmark datasets (e.g., component names, project-specific terms, synonym mappings). These constitute benchmark leakage and invalidate evaluation results."
> "All domain-specific knowledge (ambiguous component names, generic terms, aliases) must be **discovered dynamically at runtime** via LLM analysis of the input data."
> "Prompt examples must be abstract — never use names resembling benchmark components."

The first and third clauses prohibit STATIC benchmark vocabulary in source/prompts. The middle clause MANDATES dynamic runtime discovery. The runtime-generated rubric IS the dynamic-runtime mechanism the rule requires.

### Structural argument

The entire `s_linker13` pipeline produces project-specific outputs at runtime by design:

| Phase | Project-specific runtime output |
|---|---|
| Alias extraction (Pass A/B) | Emits mappings like `"Image Provider" -> ImageProvider` — verbatim project component names |
| Doc-knowledge judge | Approves/rejects aliases by name — verbatim project component names |
| Coreference | Resolves pronouns to project component names — verbatim |
| Validation | Reasons about candidate links naming project components — verbatim |
| Convention filter | Decides per-component whether to keep links — verbatim |

If runtime project-specific output were a GATE-06 violation, no LLM call in the pipeline would be compliant. The strict-reading is self-defeating.

### Operational Test C (cross-dataset rubric isolation)

The actual leakage risk in the runtime-rubric regime is **cross-dataset contamination** — the model emitting vocabulary it could only know from a *different* benchmark dataset. We formalize this as:

> Term `t` in dataset A's rubric is a cross-dataset leak iff ALL hold:
>
> (a) `t` is a PCM component name in some OTHER dataset B (B ≠ A);
> (b) `t` is NOT in dataset A's own PCM;
> AND (c) `t` is NOT in dataset A's own input document text.

Conditions (b)+(c) reject false positives from lexical coincidence — e.g. "UI" is teammates' literal PCM component name AND a 5-occurrence abbreviation in the teastore document. Under Test C, the teastore rubric mentioning "UI" is the model discovering teastore's own vocabulary; it is NOT contamination from teammates.

### Information-theoretic guarantee

The rubric builder receives exactly one document per call. It cannot output cross-dataset-specific tokens unless they exist as model priors from training. Test C is the empirical check.

## Round 2 (Claude Sonnet) — full 5-dataset sweep

### Per-dataset F1 (Claude Sonnet, s_linker13_trim3_runtime_rubric_clean, layer1 cascade)

| Dataset | F1 (trim3) | F1 (s_linker13_clean baseline) | Δ | FP | FN | Per-dataset gate |
|---|---|---|---|---|---|---|
| mediastore    | 0.9667 | 0.9836 | −0.0169 | 0 | 2 | PASS (≤ 2pp drop) |
| teastore      | 1.0000 | 1.0000 | +0.0000 | 0 | 0 | PASS |
| teammates     | 0.9474 | 0.9381 | +0.0093 | 3 | 3 | PASS (improved) |
| bigbluebutton | 0.8108 | 0.8036 | +0.0072 | 4 | 17 | PASS (≥ 0.79 absolute + improved) |
| jabref        | 0.9730 | 0.9730 | +0.0000 | 1 | 0 | PASS |
| **Macro**     | **0.9396** | 0.9397 | −0.0001 | — | — | **PASS (≥ 0.90)** |

Variant matches the s_linker13_clean Claude baseline within run-to-run variance. **No regression.** BBB improves to 0.8108 (above swattr-validated SAD-SAM floor of 0.79).

### Provenance

- `mediastore`, `teastore`: layer1.json from prior Plan 12-05 partial-halt run, `results/.../claude/s_linker13_trim3_runtime_rubric_clean/<ds>/layer1.json`.
- `teammates`, `bigbluebutton`, `jabref`: REVISIT sweep, `results/.../claude_revisit/s_linker13_trim3_runtime_rubric_clean/<ds>/layer1.json`.

## Round 3 (gpt-5.4) — full 5-dataset sweep

### Per-dataset F1

| Dataset | F1 (gpt-5.4) | F1 (Claude) | Δ (gpt-5.4 − Claude) | FP | FN |
|---|---|---|---|---|---|
| mediastore    | 0.8966 | 0.9667 | −7.0pp  | 1  | 5 |
| teastore      | 0.9811 | 1.0000 | −1.9pp  | 0  | 1 |
| teammates     | 0.8130 | 0.9474 | −13.4pp | 16 | 7 |
| bigbluebutton | 0.7636 | 0.8108 | −4.7pp  | 6  | 20 |
| jabref        | 0.9730 | 0.9730 | +0.0pp  | 1  | 0 |
| **Macro**     | **0.8855** | 0.9396 | −5.4pp | — | — |

Cross-model gate floor 0.8977 → **FAIL (gap −1.22pp)**. Delta vs anchor (s_linker13 gpt-5.4 macro 0.9077) = −2.22pp, exceeds 1.0pp tolerance.

**Drag analysis:**
- teammates 0.8130: GPT generates 16 FPs vs Claude's 3 — GPT over-applies the "approve shortened references" rubric criterion (the BBB-style pattern of "approve abbreviation X for component Y" generalizes too liberally on teammates' 8-component PCM with generic English names like Logic, Storage, Common, UI).
- bigbluebutton 0.7636: GPT under-recovers with 20 FNs — chronic GPT BBB weakness predates this trim (see MEMORY.md V32 GPT cross-model evaluation; BBB has been GPT's weakest dataset across all variants).
- mediastore 0.8966: GPT slightly over-rejects (5 FNs vs Claude's 2) — within model-variance band but contributes to the macro drag.

This pattern is consistent with MEMORY.md's documented Claude-vs-GPT capability gap (~5.7pp on V32). The trim does NOT introduce a new GPT weakness; it inherits the model-capability gap.

## Cross-dataset rubric isolation audit (Test C)

Audited 10 rubrics (5 Claude + 5 gpt-5.4):

### Claude

| Rubric for | Distinct PCM-component-name terms in rubric | All present in own document? | Cross-dataset leaks |
|---|---|---|---|
| mediastore    | (none flagged) | n/a | 0 |
| teastore      | ImageProvider (input-passed), Persistence, Registry, UI, WebUI | all 4 present in teastore.txt; ImageProvider passed as candidate input | 0 |
| teammates     | E2E, Test Driver, Common, UI, Logic, Client, Storage | all 7 in teammates.txt | 0 |
| bigbluebutton | Apps | yes (BBB.txt) | 0 |
| jabref        | model | yes (jabref.txt) | 0 |
| **Total** | — | — | **0** |

### gpt-5.4

| Rubric for | Cross-dataset leaks (Test C violations) |
|---|---|
| mediastore | 0 |
| teastore   | 0 |
| teammates  | 0 |
| bigbluebutton | 0 |
| jabref     | 0 |
| **Total** | **0** |

### Benign lexical overlaps (recorded, NOT violations)

Both Claude and gpt-5.4 teastore rubrics mention "UI". "UI" is:
- Teastore's discovered alias for WebUI (5 occurrences in teastore.txt — `in_own_doc=True`); AND
- Teammates' literal PCM component name (`appears_as_pcm_name_in=["teammates"]`).

Under Test C condition (c), this is NOT a leak — the model discovered "UI" from teastore's own document, not from prior knowledge of teammates. Lexical coincidence between an English abbreviation and a different dataset's PCM-name is exactly what (c) is designed to exclude.

### Conclusion

Under the operationalized cross-dataset isolation test, every rubric the builder emitted across 10 sweeps (2 backends × 5 datasets) is information-theoretically isolated to its own input document. No prior-knowledge leakage from other benchmarks was detected on either backend.

## Static surface re-audit

- `RUBRIC_BUILDER_SEED_EXAMPLE`: 0 taboo hits (compiler-style example, GATE-06 clean).
- `RUBRIC_BUILDER_PROMPT` template: 0 taboo hits (abstract JSON placeholder, no benchmark vocabulary).
- 7 `DOC_KNOWLEDGE_JUDGE_EXAMPLES` preserved byte-equal (V35a guard).

## Threat model resolution (revisited)

| Threat ID | Prior disposition | Revisited outcome |
|---|---|---|
| T-12-05-01 (rubric leaks benchmark vocabulary) | mitigate via audit; REJECT if hits > 0 | **Operationalization corrected.** "Hits > 0" was strict reading. Refined to: REJECT if cross-dataset leak (Test C). 0 violations on both backends → **NOT A LEAK.** |
| T-12-05-02 (frozen-file edits) | mitigate via `git diff --quiet` | PASS — unchanged. |
| T-12-05-03 (rubric builder silent fallback) | track fallback_count | Fallback NEVER triggered across 10 sweeps (5 Claude + 5 gpt-5.4). Mechanism always live. |
| T-12-05-04 (rubric provenance) | sweep.log | sweep.log captured for all 10 rubrics. |
| T-12-05-05 (extra LLM call cost) | accept | +10 extra calls for full ablation. Acceptable. |

## Files

| Created | Modified |
|---|---|
| `.planning/phases/12-trim-ablation/12-05-SUMMARY-REVISIT.md` (this file) | `results/ablation_results/12_05_trim3_runtime_rubric/verdict.json` (revisit verdict) |
| `scripts/audit_12_05_revisit.py` | — |
| `results/ablation_results/12_05_trim3_runtime_rubric/revisit_audit.json` | — |
| `results/ablation_results/12_05_trim3_runtime_rubric/claude_revisit/sweep.log` (3 rubrics) | — |
| `results/ablation_results/12_05_trim3_runtime_rubric/claude_revisit/.../{teammates,bigbluebutton,jabref}/layer1.json` | — |
| `results/ablation_results/12_05_trim3_runtime_rubric/gpt54_revisit/sweep.log` (5 rubrics) | — |
| `results/ablation_results/12_05_trim3_runtime_rubric/gpt54_revisit/.../<5 datasets>/layer1.json` | — |

Frozen files unchanged: `prompts_v2.py, s_linker13.py, s_linker13_clean.py, data_types_v2.py, document_loader_v2.py, pcm_parser_v2.py`.

The prior `12-05-SUMMARY.md` is **PRESERVED unchanged** — it documents the historical strict-reading REJECT for milestone audit traceability. This file (`12-05-SUMMARY-REVISIT.md`) supersedes it with the methodologically-correct verdict.

## Comparison to prior verdict

| Aspect | Prior verdict | Revisit verdict |
|---|---|---|
| Overall | REJECT | REJECT |
| Basis | GATE-06 leakage (3/3 audited rubrics named project components) | GATE-01 cross-model gap (gpt-5.4 macro 0.8855 < 0.8977 floor by 1.22pp) |
| Methodology status | Used strict reading that, applied consistently, invalidates the entire pipeline | Used correct CLAUDE.md reading: dynamic runtime LLM analysis is mandated; cross-dataset isolation is the testable criterion |
| Claude arm | Not evaluated (halted at 2/5 datasets) | Evaluated; PASS (macro 0.9396) |
| gpt-5.4 arm | Not evaluated (skipped per strategic mode) | Evaluated; FAIL (macro 0.8855) |
| Cross-dataset isolation | Not tested as a distinct criterion | Tested; PASS on both backends |
| Strength of rejection | Interpretive (depends on which GATE-06 reading) | Data-driven (cross-model F1 gap is empirically measured) |

## Downstream signals

- **Plan 12-06 (GATE-06 defensibility audit):** This trim is now a case study for the *correct* GATE-06 operationalization. The audit narrative is: "Phase 12 took the strongest theoretical V35-escape mechanism, faithfully implemented it per supplement Techniques 2+3, and rigorously tested two distinct GATE-06 readings — the strict-static-output reading (which is self-defeating) and the cross-dataset-isolation reading (which is the testable criterion CLAUDE.md actually mandates). The trim passes the methodologically-correct GATE-06 test on both backends. It is rejected on cross-model F1, not on leakage, demonstrating that the v2.1 milestone applies rigorous criteria that distinguish methodological failures from model-capability failures."
- **Plan 13-01 (s_linker13_min promotion):** Variant is NOT carried forward because of the cross-model failure. The cross-model gate is a v2.1 thesis claim and cannot be relaxed. The variant could be re-considered in v2.2+ with backend-adaptive prompts (deferred item ADAPTER-01) or if gpt-5.4 cross-model gap closes naturally with model improvements.

## Requirements progress

- **PROMPT-01:** v2→v3 mapping entry for `DOC_KNOWLEDGE_JUDGE_RULES` updates to: "REPLACED by inference-time rubric builder (AHE + Agentic Rubrics, supplement Techniques 2+3); under refined GATE-06 (cross-dataset isolation operationalization) the trim is **GATE-06-compliant on both backends**; rejected for v2.1 promotion solely on GATE-01 cross-model gap (gpt-5.4 macro 0.8855 < 0.8977 floor)." Plan 12-06 inherits this for the mapping-doc maintenance.
- **PROMPT-02:** Highest-risk trim fully ablated under methodologically-correct gates; verdict recorded with explicit cross-dataset isolation evidence on BOTH backends.

## Stub tracking

None — no stubs introduced. Variant is a fully wired subclass with a complete override.

## Threat flags

None — no new threat surface beyond the threat-model already enumerated. T-12-05-01 was the predicted threat; under the refined GATE-06 reading it did NOT materialize.

## Deviations from plan

- **Methodological correction (recorded as USER DIRECTIVE):** The user explicitly directed a revisit under a "methodologically-correct reading + relaxed gates" framing. We honored: (1) corrected GATE-06 operationalization to cross-dataset isolation; (2) applied relaxed v2.1 GATE-01 thresholds (macro ≥ 0.90, BBB ≥ 0.79); (3) preserved the prior REJECT's history rather than overwriting.
- **No fallback path exercised:** Across 10 sweeps the rubric builder always produced a non-empty rubric. The variant's `_trim3_fallback_count` static-rubric fallback is dead code in practice on Claude and gpt-5.4 with the current prompts.

## Self-Check: PASSED

- `.planning/phases/12-trim-ablation/12-05-SUMMARY-REVISIT.md` — EXISTS (this file).
- `scripts/audit_12_05_revisit.py` — EXISTS.
- `results/ablation_results/12_05_trim3_runtime_rubric/revisit_audit.json` — EXISTS.
- `results/ablation_results/12_05_trim3_runtime_rubric/verdict.json` — UPDATED (revisit verdict; prior preserved in 12-05-SUMMARY.md).
- All 10 layer1.json files present (5 Claude across claude/ + claude_revisit/, 5 gpt-5.4 in gpt54_revisit/).
- `results/ablation_results/12_05_trim3_runtime_rubric/claude_revisit/sweep.log` — EXISTS.
- `results/ablation_results/12_05_trim3_runtime_rubric/gpt54_revisit/sweep.log` — EXISTS.
- Frozen-file `git diff --quiet` — PASSED.
- GATE-02 baseline regression — UNAFFECTED (variant is in "missing" list from Plan 12-05 GREEN).
