---
phase: 45-audit
plan: 04
subsystem: prompt-audit
tags: [audit, dkj, benchmark-leak, doc-knowledge-judge]
dependency_graph:
  requires: [45-01]
  provides: [DKJ section of s_linker20-PROMPT-AUDIT.md]
  affects: [Phase 46 (MINIMIZE)]
tech_stack:
  added: []
  patterns: [BENCHMARK_TABOO mechanical grep, v2.1 GATE-06 cross-dataset isolation, D-04 strict few-shot interpretation, D-06 Family A/B rewording families]
key_files:
  created:
    - .planning/phases/45-audit/45-04-SUMMARY.md
  modified:
    - .planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md (DKJ section + Verdict Summary rows)
decisions:
  - "DOC_KNOWLEDGE_JUDGE_EXAMPLES verdict = benchmark-leak (CacheLayer head noun cache hits MediaStore §Components + MediaStore §Keywords + Universal Taboo) — auto-classify per D-02 per-dataset rule"
  - "DOC_KNOWLEDGE_JUDGE_RULES verdict = domain-loaded on the 'architectural tier or technology platform' clause; per D-05, no rewording proposed (Phase 46 empirical loop)"
  - "_prompt_doc_knowledge_judge opener verdict = clean (generic SE nouns, same precedent as DKX opener)"
  - "Family A synthetic-neutral names: BookManager/Mgr (Example 1 swap), MailSender (Example 2 swap) — fresh names not in BENCHMARK_TABOO and not in the Safe SE Textbook list"
  - "6 cut rows for DOC_KNOWLEDGE_JUDGE_EXAMPLES (drop-block + 3 Family A variants + 2 Family B variants) + 1 cut row for DOC_KNOWLEDGE_JUDGE_RULES domain-loaded flag = 7 CUT-DKJ-NN rows total"
metrics:
  duration: 8m
  completed: 2026-06-08
---

# Phase 45 Plan 04: DKJ Audit Summary

One-liner: Audited the Phase 1 Doc-Knowledge Judge section — confirmed `DOC_KNOWLEDGE_JUDGE_EXAMPLES` is the highest-yield benchmark-leak in the entire prompt set (via `CacheLayer` → `cache` MediaStore hit) and emitted a 7-row CUT-DKJ table covering drop-block + Family A name-swap + Family B concept-only + DOC_KNOWLEDGE_JUDGE_RULES domain-loaded flag, with all synthetic replacement names grep-cleared against the full taboo list.

## Final Verdicts

| Item | Verdict | LOC | Cut Rows |
|---|---|---|---|
| `DOC_KNOWLEDGE_JUDGE_EXAMPLES` | benchmark-leak | 7 | 6 (CUT-DKJ-01 drop + CUT-DKJ-02..04 Family A + CUT-DKJ-05..06 Family B) |
| `DOC_KNOWLEDGE_JUDGE_RULES` | domain-loaded ("architectural tier or technology platform") | 1 | 1 (CUT-DKJ-07 flag; rewording deferred to Phase 46 per D-05) |
| `_prompt_doc_knowledge_judge` (prose, line 306) | clean | 16 total (1 audit-relevant prose line) | 0 |

## Cut Row Counts by Category

| Category | Count | Cut IDs |
|---|---|---|
| Drop-block (REQ-V264-06) | 1 | CUT-DKJ-01 |
| Family A (synthetic-neutral name swap) | 3 | CUT-DKJ-02 (Ex1), CUT-DKJ-03 (Ex2 — primary leak removal), CUT-DKJ-04 (combined) |
| Family B (concept-only / name-stripped) | 2 | CUT-DKJ-05 (Ex1), CUT-DKJ-06 (Ex2) |
| Domain-loaded flag | 1 | CUT-DKJ-07 |
| **Total** | **7** | — |

Minimums per plan satisfied: ≥1 drop-block (1 emitted), ≥1 Family A (3 emitted), ≥1 Family B (2 emitted).

## Family A Synthetic-Name Grep Clearance

All proposed Family A synthetic names were grep-cleared against `BENCHMARK_TABOO.md` (5 per-dataset sections + Universal Taboo + Safe SE Textbook Examples) before inclusion:

| Synthetic Name | Whole-word hits | Substring hits on parts | Clearance |
|---|---|---|---|
| `BookManager` | 0 | `book` → 0; `manager` → 0 per-dataset, only Safe SE list (lines 63, 66 — explicitly safe) | CLEAR — fresh (not in Safe SE list as a whole word) |
| `Mgr` (alias short form) | 0 | (abbreviation, no benchmark overlap) | CLEAR |
| `MailSender` | 0 | `mail` → 0; `sender` → 0 | CLEAR — fully fresh name (not even in Safe SE list) |

Forbidden-name sentinel grep (from plan `<verify>` automated gate):
```
grep -E 'OrderProcessor|CacheManager|ClientWrapper|EventBus|ServerProxy|AdapterChain|LogicEngine|StorageHelper|ValidatorService' s_linker20-PROMPT-AUDIT.md
```
→ 0 matches. No taboo synthetic names slipped through.

## Benchmark-Leak Audit Evidence (mechanical grep)

| Query | Result | Verdict driver |
|---|---|---|
| `grep -niw 'cache' BENCHMARK_TABOO.md` | 3 hits: MediaStore §Components `Cache`, MediaStore §Keywords `cache`, Universal Taboo `cache (MediaStore)` | Confirms `CacheLayer` → `cache` substring auto-classifies as `benchmark-leak` per D-02 |
| `grep -niwE 'Handler\|RequestHandler\|CacheLayer' BENCHMARK_TABOO.md` | 0 whole-word hits | `Handler` / `RequestHandler` individually clean; `CacheLayer` triggers only via substring |
| `grep -nw 'system' BENCHMARK_TABOO.md` | 0 hits | Few-shot prose `"the system"` is clean |
| `grep -niwE 'platform\|tier\|entity\|grouping' BENCHMARK_TABOO.md` | 0 hits | DOC_KNOWLEDGE_JUDGE_RULES tier/platform clause is `domain-loaded`, not `benchmark-leak` |
| `grep -niwE 'component\|name\|mapping' BENCHMARK_TABOO.md` | only standing Universal-Taboo `component` entry | `_prompt_doc_knowledge_judge` opener is clean (generic SE noun, v2.1 GATE-06 isolation passes) |

## GATE-01 Verification

```
git diff --quiet src/llm_sad_sam/linkers/experimental/s_linker19.py \
                  src/llm_sad_sam/linkers/experimental/prompts_v5.py \
                  src/llm_sad_sam/linkers/experimental/s_linker13_min.py
```
→ exit 0. **GATE-01 byte-equal HOLDS.** Zero edits to frozen source artefacts. Only edits in this plan: the DKJ anchor region of `s_linker20-PROMPT-AUDIT.md` + the 3 DKJ rows of the Verdict Summary table (lines ~76–78).

## Plan Verification Output

Automated `<verify>` gate:
```
OK 7 cuts
OK — no forbidden names
OK GATE-01 byte-equal
```

## Deviations from Plan

None — the plan executed as written.

Notes on minor stylistic choices preserved (within Claude's Discretion per CONTEXT.md):
- Detail blocks `> **CUT-DKJ-NN detail:**` live inline under the cut table per the section-local pattern used by AMB / DKX, not in a doc-level appendix.
- Risk justifications attached inline in the `risk` cell (consistent with AMB / DKX precedent).
- Family A used two synthetic names (`BookManager`/`Mgr` + `MailSender`) drawn from disjoint domains to demonstrate name-rotation robustness without batching into a single domain.

## Self-Check: PASSED

- DKJ section between `<!-- SECTION:DKJ:START -->` / `<!-- SECTION:DKJ:END -->`: FOUND
- 3-row header table with `DOC_KNOWLEDGE_JUDGE_EXAMPLES` / `DOC_KNOWLEDGE_JUDGE_RULES` / `_prompt_doc_knowledge_judge`: FOUND
- Header verdicts: benchmark-leak / domain-loaded / clean — FOUND
- CUT-DKJ-01 drop-block row: FOUND
- ≥1 Family A row: FOUND (3 rows: CUT-DKJ-02..04)
- ≥1 Family B row: FOUND (2 rows: CUT-DKJ-05..06)
- DOC_KNOWLEDGE_JUDGE_RULES domain-loaded flag row: FOUND (CUT-DKJ-07, `after = [Phase 46 empirical loop]`)
- All cut rows gated by `tests/test_s_linker20_prompt_doc_judge.py @ phase_1_doc_judge`: FOUND
- Verdict Summary rows for the 3 DKJ items updated (no more TBD): FOUND
- GATE-01 source files byte-equal vs HEAD: PASSED (`git diff --quiet` exit 0)
- No forbidden synthetic names (OrderProcessor / CacheManager / etc.): PASSED
- Family A synthetic-name grep clearances documented inline: FOUND (in CUT-DKJ-02 / CUT-DKJ-03 detail blocks + this SUMMARY table)
