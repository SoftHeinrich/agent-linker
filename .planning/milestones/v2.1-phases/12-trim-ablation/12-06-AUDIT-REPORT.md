---
phase: 12-trim-ablation
plan: 12-06
artifact: gate06_audit_report
audited: "2026-05-31T15:10:00Z"
requirements: [PROMPT-01, PROMPT-04]
gate: GATE-06
verdict_class:
  shipped_variants:
    prompts_v3: PASS
    s_linker13_trim1_judge_clean: PASS
    s_linker13_trim9_seed_runtime_clean: PASS
    s_linker13_clean_v3: PASS
    helper_v3: PASS
  frontier_variants:
    s_linker13_trim2_entval_clean: PASS (GATE-06) but REJECTED on GATE-01 Claude
    s_linker13_trim3_runtime_rubric_clean: PASS (GATE-06) but REJECTED on GATE-01 cross-model
    s_linker13_trim4_ambiguity_runtime_clean: PASS (GATE-06) but REJECTED on GATE-01 Claude per-dataset
    s_linker13_trim5_extraction_runtime_clean: PASS (GATE-06) but REJECTED on GATE-01 Claude per-dataset
    s_linker13_trim6_judge_examples_runtime_clean: PASS (GATE-06) but REJECTED on GATE-01 cross-model
    s_linker13_trim7_entity_runtime_clean: PASS (GATE-06) but REJECTED on GATE-01 Claude per-dataset
    s_linker13_trim8_validation_runtime_clean: PASS (GATE-06) but REJECTED on GATE-01 Claude per-dataset
overall_audit_verdict: PASS
---

# Phase 12 — GATE-06 Audit Report (PROMPT-04 closure)

**Audited:** 2026-05-31T15:10:00Z
**Method:** Pure static analysis — module-level string-constant extraction via `ast` + case-insensitive whole-word regex match against the full BENCHMARK_TABOO.md surface (100 distinct terms across MediaStore / TeaStore / Teammates / BigBlueButton / JabRef components/aliases/keywords + Universal Taboo).
**Reviewer adjudication:** every hit dispositioned safe / leaked / borderline per CLAUDE.md GATE-06 spec.
**Zero LLM calls.** **Zero edits to frozen v2.0 files.**

## 1. Audit Surface

### 1.1 Shipped variants (carry to Plan 13-01)

| File | Constants scanned |
|------|-------------------|
| `src/llm_sad_sam/linkers/experimental/prompts_v3.py` | 9 (AMBIGUITY_FEW_SHOT, AMBIGUITY_RULES, DOC_KNOWLEDGE_EXTRACTION_RULES, DOC_KNOWLEDGE_JUDGE_EXAMPLES, DOC_KNOWLEDGE_JUDGE_RULES, ENTITY_EXTRACTION_RULES, VALIDATION_RULES, COREF_RULES, SEED_DISAMBIGUATION_RULES) |
| `src/llm_sad_sam/linkers/experimental/s_linker13_trim1_judge_clean.py` (Plan 12-03, **ACCEPT**) | 1 (DOC_KNOWLEDGE_JUDGE_RUBRIC_V3); 1 alias (DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3 = prompts_v2 byte-equal) |
| `src/llm_sad_sam/linkers/experimental/s_linker13_trim9_seed_runtime_clean.py` (Plan 12-12, **ACCEPT**) | 2 (SEED_RUBRIC_BUILDER_SEED_EXAMPLE, SEED_RUBRIC_BUILDER_PROMPT) |
| `src/llm_sad_sam/linkers/experimental/s_linker13_clean_v3.py` | 2 module-level constants (ALIAS_SCOPE_SCHEMA, ANTECEDENT_ALIAS_GUIDE) |
| `src/llm_sad_sam/linkers/experimental/helper_v3.py` | 0 string constants (only function bodies; the regex pattern is a constructed primitive, not a prompt body) |

### 1.2 Frontier variants (NOT shipped — Plan 13-01 excluded, kept for negative-result traceability)

| File | Plan | Original-gate verdict | Constants scanned |
|------|------|-----------------------|-------------------|
| `s_linker13_trim2_entval_clean.py` | 12-04 | REJECT (Claude macro 0.9235 < 0.93; BBB drop −6.59pp) | 3 (ENTVAL_MERGED_RUBRIC_V3, ENTITY_EXTRACTION_RULES_V3, VALIDATION_RULES_V3 — last two are headers + shared core) |
| `s_linker13_trim3_runtime_rubric_clean.py` | 12-05 (REVISIT) | REJECT (gpt-5.4 macro 0.8855 < 0.8977 floor; GATE-06 leakage REJECT OVERTURNED) | 2 (RUBRIC_BUILDER_SEED_EXAMPLE, RUBRIC_BUILDER_PROMPT) |
| `s_linker13_trim4_ambiguity_runtime_clean.py` | 12-07 | REJECT (Claude JAB −2.56pp > 2pp tolerance) | 2 (AMBIGUITY_RUBRIC_BUILDER_SEED_EXAMPLE, AMBIGUITY_RUBRIC_BUILDER_PROMPT) |
| `s_linker13_trim5_extraction_runtime_clean.py` | 12-08 | REJECT (Claude TS −3.57pp > 2pp tolerance) | 2 |
| `s_linker13_trim6_judge_examples_runtime_clean.py` | 12-09 | REJECT (gpt-5.4 cross-model gap 0.39pp) | 2 |
| `s_linker13_trim7_entity_runtime_clean.py` | 12-10 | REJECT (Claude JAB −2.56pp + TS −1.82pp) | 2 |
| `s_linker13_trim8_validation_runtime_clean.py` | 12-11 | REJECT (Claude TS −3.57pp + JAB −2.56pp) | 2 |

### 1.3 Out of scope for direct prompt-body audit (audited indirectly)

- **`prompts_v3_axiom.py`** — Voyager-built axiom prompts (Phase 12 EXTENSION pilot). NOT in any shipped or evaluated trim. Carried for future Phase 12+ research but does not gate 13-01. **Deferred from this audit per execution prompt context.** Spot-check: header explicitly cites GATE-06 with reviewer-defensibility self-audit; constants are aggressively stripped abstract principles.
- **`s_linker13_trim*` runtime rubrics emitted at inference time** — covered by Plan 12-05-REVISIT cross-dataset isolation methodology (`revisit_audit.json`); not re-audited here since the methodology is settled and inherited.

## 2. TABOO Sweep Construction

**Source:** `BENCHMARK_TABOO.md` (snapshot 2026-05-31).
**Term count:** 100 distinct case-insensitive whole-word tokens, sorted longest-first to prefer multi-word matches.
**Allow-list:** Safe SE Textbook Examples from BENCHMARK_TABOO.md §"Safe SE Textbook Examples" (Lexer, Parser, AST, CodeGenerator, Optimizer, SymbolTable, Scheduler, MemoryManager, FileSystem, ProcessTable, Dispatcher, Router, Multiplexer, PacketHandler, ShoppingCart, PaymentGateway, InvoiceHandler, InventoryTracker, Repository, CommitLog, BranchManager, MergeResolver, RenderEngine, PhysicsSimulator, InputHandler, SceneGraph, Broker, Wrapper, Connector).
**Constant extraction:** Python `ast.parse` → module-level `Assign` with single-name target → `_eval_str_node` recursively evaluates literal strings, f-string-with-literal-only-segments, `+` concatenation, `.strip()` / `.format()` (takes pre-call body), and tuple-of-strings concatenation. Comments and module docstrings are NOT scanned, per plan.
**Verification:** Audit script saved at `/tmp/gate06_audit_12_06.py`; JSON output saved at `/tmp/gate06_audit_results.json`.

Heuristic supplement: a CamelCase scan (`\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b`) was run on the three shipped variants. Names found: `AbstractSyntaxTree`, `CamelCase`, `GameRenderEngine`, `IndexManager`, `InvoiceHandler`, `MemoryManager`, `PaymentSystem`, `RenderEngine`, `SymbolTable`, `TaskDispatcher`, `CodeGenerator`. All are textbook SE allow-list examples. No benchmark-derived CamelCase tokens.

## 3. Full TABOO Sweep Results

### 3.1 Shipped variants

| File | Constant | hit_count | hit_terms | reviewer_disposition |
|------|----------|-----------|-----------|----------------------|
| `prompts_v3.py` | AMBIGUITY_FEW_SHOT | 0 | — | safe |
| `prompts_v3.py` | AMBIGUITY_RULES | 0 | — | safe |
| `prompts_v3.py` | DOC_KNOWLEDGE_EXTRACTION_RULES | 0 | — | safe |
| `prompts_v3.py` | DOC_KNOWLEDGE_JUDGE_EXAMPLES | 1 | `layer` (Universal) | safe — see §4.1.1 |
| `prompts_v3.py` | DOC_KNOWLEDGE_JUDGE_RULES | 1 | `order` (Universal) | safe — see §4.1.2 |
| `prompts_v3.py` | ENTITY_EXTRACTION_RULES | 0 | — | safe |
| `prompts_v3.py` | VALIDATION_RULES | 0 | — | safe |
| `prompts_v3.py` | COREF_RULES | 0 | — | safe |
| `prompts_v3.py` | SEED_DISAMBIGUATION_RULES | 0 | — | safe |
| `s_linker13_trim1_judge_clean.py` | DOC_KNOWLEDGE_JUDGE_RUBRIC_V3 | 0 | — | safe |
| `s_linker13_trim9_seed_runtime_clean.py` | SEED_RUBRIC_BUILDER_SEED_EXAMPLE | 0 | — | safe |
| `s_linker13_trim9_seed_runtime_clean.py` | SEED_RUBRIC_BUILDER_PROMPT | 0 | — | safe |
| `s_linker13_clean_v3.py` | ALIAS_SCOPE_SCHEMA | 0 | — | safe |
| `s_linker13_clean_v3.py` | ANTECEDENT_ALIAS_GUIDE | 0 | — | safe |
| `helper_v3.py` | (no module-level string constants) | 0 | — | n/a |

**Shipped subtotal:** 2 hits — both Universal-Taboo English vocabulary in generic SE contexts. Both are **safe** (see §4.1).

### 3.2 Frontier variants

| File | Constant | hit_count | hit_terms | reviewer_disposition |
|------|----------|-----------|-----------|----------------------|
| `s_linker13_trim2_entval_clean.py` | ENTVAL_MERGED_RUBRIC_V3 | 0 | — | safe |
| `s_linker13_trim2_entval_clean.py` | _EXTRACTION_HEADER | 0 | — | safe |
| `s_linker13_trim2_entval_clean.py` | _VALIDATION_HEADER | 0 | — | safe (`_VALIDATION_HEADER` *name* contains "VALIDATION" but identifier names are not scanned — only string bodies are) |
| `s_linker13_trim3_runtime_rubric_clean.py` | RUBRIC_BUILDER_SEED_EXAMPLE | 0 | — | safe |
| `s_linker13_trim3_runtime_rubric_clean.py` | RUBRIC_BUILDER_PROMPT | 0 | — | safe |
| `s_linker13_trim4_ambiguity_runtime_clean.py` | AMBIGUITY_RUBRIC_BUILDER_SEED_EXAMPLE | 1 | `common` (Universal) | safe — see §4.2.1 |
| `s_linker13_trim4_ambiguity_runtime_clean.py` | AMBIGUITY_RUBRIC_BUILDER_PROMPT | 0 | — | safe |
| `s_linker13_trim5_extraction_runtime_clean.py` | (both constants) | 0 | — | safe |
| `s_linker13_trim6_judge_examples_runtime_clean.py` | (both constants) | 0 | — | safe |
| `s_linker13_trim7_entity_runtime_clean.py` | (both constants) | 0 | — | safe |
| `s_linker13_trim8_validation_runtime_clean.py` | VALIDATION_RUBRIC_BUILDER_SEED_EXAMPLE | 1 | `validation` (Universal) | safe — see §4.2.2 |
| `s_linker13_trim8_validation_runtime_clean.py` | VALIDATION_RUBRIC_BUILDER_PROMPT | 0 | — | safe |

**Frontier subtotal:** 2 hits — both Universal-Taboo terms used as English in textbook-SE seed examples. Both are **safe** (see §4.2).

**Overall sweep total:** 4 hits across 17 constants in 12 files. **Zero leaked**, **zero borderline**, **4 safe** under reviewer adjudication (§4).

## 4. Reviewer Adjudication Per Hit

The hit-count threshold for adjudication is "any match against the BENCHMARK_TABOO.md term surface". Each match is judged against the CLAUDE.md GATE-06 rule: *"Could this prompt body, read by a person unfamiliar with the benchmark projects, plausibly be written for a NON-benchmark system?"* If yes → **safe**. If the match's context names a benchmark project component / alias / unique-keyword → **leaked**. Ambiguous → **borderline** (requires user adjudication).

### 4.1 Shipped — prompts_v3.py

#### 4.1.1 `DOC_KNOWLEDGE_JUDGE_EXAMPLES` — `layer`

**Context:** `'query execution layer' -> IndexManager (synonym)`
**TABOO source:** Universal Taboo (BBB — FreeSWITCH Event Socket Layer alias word).
**Reviewer rationale:** "query execution layer" is a database-textbook noun phrase referring to the architectural concept of an execution layer in a DBMS (cf. relational-algebra execution engine literature). The companion noun "IndexManager" is in the Safe SE Textbook allow-list. No benchmark project named. A person unfamiliar with the benchmarks reads this as "an example pairing a database-style descriptive phrase with its component name". **SAFE.**
**Provenance:** prompts_v3.py is byte-equal to prompts_v2.py for this constant (Plan 12-01 mapping); the phrasing was audited at v2.0 close and re-confirmed here.

#### 4.1.2 `DOC_KNOWLEDGE_JUDGE_RULES` — `order`

**Context:** `DECISION RULES (apply in order):`
**TABOO source:** Universal Taboo (TeaStore — OrderBasedRecommender component word).
**Reviewer rationale:** "apply in order" is English used in the meaning "in sequence" — a directive to apply numbered rules sequentially. No noun-phrase relationship to TeaStore's OrderBasedRecommender. **SAFE.**
**Provenance:** byte-equal to prompts_v2.py (Plan 12-01).

### 4.2 Frontier — runtime rubric-builder seed examples

#### 4.2.1 `AMBIGUITY_RUBRIC_BUILDER_SEED_EXAMPLE` (trim4) — `common`

**Context:** `CamelCase compounds and common abbreviations (API, TCP, RPC) are always architectural.`
**TABOO source:** Universal Taboo (Teammates Common component word).
**Reviewer rationale:** "common abbreviations" is English used in the meaning "frequently-encountered abbreviations". The trailing examples (API, TCP, RPC) are network-protocol acronyms — neutral SE vocabulary. No noun-phrase relationship to Teammates' Common component. **SAFE.**

#### 4.2.2 `VALIDATION_RUBRIC_BUILDER_SEED_EXAMPLE` (trim8) — `validation`

**Context:** `A good 5-item validation rubric for this example would be:`
**TABOO source:** Universal Taboo (Teammates — "input validation" keyword).
**Reviewer rationale:** "validation rubric" labels the *task* the rubric performs (the validation phase of the s_linker13 pipeline). The Teammates "validation" keyword refers to input validation in a web-form context; this is a rubric labeling phrase, not a component reference. The trim8 variant is the *validation-phase rubric builder* — using "validation" in its own name is structural, not a project leak. **SAFE.**

### 4.3 Borderline Hits

**None.** All four matches are unambiguously generic SE vocabulary in textbook contexts. No user adjudication is required for the lexical sweep.

The Task 3 checkpoint (user adjudication) is therefore **vestigial for the lexical layer**: it remains available for confirming the per-trim Final Disposition table and any policy-level interpretation, but no hit requires it for safe/leaked re-classification.

## 5. Reviewer-Defensibility Per Trim

### 5.1 Shipped: prompts_v3.py (Step 0)

- **Provenance:** Byte-equal to prompts_v2.py for all 9 kept constants (Plan 12-01).
- **Audit history:** prompts_v2.py was audited at v2.0 close. Plan 12-06 re-runs the FULL TABOO sweep (vs Plan 12-01's narrower 9-name probe) and confirms zero net change in the leakage surface.
- **Reviewer-defensibility:** Each prompt body uses textbook SE examples (compilers, OS, e-commerce middleware, game engines). Every component name mentioned is in the Safe SE Textbook allow-list.
- **GATE-06 verdict:** **PASS.**

### 5.2 Shipped: trim1_judge_clean (Plan 12-03 — ACCEPT)

- **Removed rules / restructured:** The original `DOC_KNOWLEDGE_JUDGE_RULES` (3 numbered rules + IMPORTANT closer) was distilled via Technique 3 (lossless rubric distillation) + Technique 8 (reasoning-before-conclusion ordering). The body went from 773 → 888 bytes (114.9% — within Phase 11 sizing budget 80-130%).
- **Each removed rule's justification:**
  - *Rule 1 (AUTO-APPROVE list)*: merged into prose form; all 4 sub-categories (abbreviations / trailing-word / CamelCase / multi-word-phrases) preserved verbatim as sub-shapes.
  - *Rule 2 (APPROVE clause)*: merged into the prose rubric; generic-word exclusion list preserved (system / process / utility / component / module).
  - *Rule 3 (REJECT clause)*: merged into the prose rubric; whole-system rejection preserved.
  - *IMPORTANT closer ("When in doubt, APPROVE")*: moved to **lead** the rubric body per Technique 8 (reasoning-before-conclusion ordering — `arXiv 2603.13351`-style directive ordering).
- **Examples preserved:** `DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3` is byte-equal to v2 (V35a guard — example removal regresses Claude by 2-5pp per MEMORY.md).
- **TABOO sweep:** 0 hits on DOC_KNOWLEDGE_JUDGE_RUBRIC_V3.
- **Reviewer-defensibility:** distilled prose uses only safe SE-textbook examples ("system", "process", "utility", "component", "module" — already in the v2 original; generic enough that a reviewer cannot infer benchmark provenance).
- **Verdict.json:** Claude relaxed GATE-01 PASS (macro 0.9553, BBB +2.54pp); gpt-5.4 cross-model GATE-01 PASS (macro 0.9173 ≥ 0.8977 floor); GATE-06 probe PASS (0 hits in published verdict.json).
- **GATE-06 verdict (Plan 12-06 full sweep):** **PASS.**
- **Carry to Plan 13-01:** **YES.**

### 5.3 Shipped: trim9_seed_runtime_clean (Plan 12-12 — ACCEPT)

- **Removed rule / replaced:** The static class-attribute `SEED_DISAMBIGUATION_RULES` (used inside `_run_seed_validation`) is replaced by a runtime rubric built per-document via one rubric-builder LLM call.
- **Mechanism:** The rubric builder receives (a) a generic compiler-style seed example, (b) the project document, (c) the component list; it emits a 4-6 item rubric covering COMPONENT vs OTHER cases. The rubric is built **once per document** and reused across all per-component dossiers. No static fallback (RuntimeError on empty rubric — user directive).
- **Justification for removal:** Seed disambiguation operates on a small candidate set (only raw seed links from ILinker3) with rich per-component dossier context (anchor sentences, mention-context classification). The runtime rubric adds calibration without amplifying noise; trim9 is the only Phase-12 runtime variant that PASSES both the Claude and gpt-5.4 original gates simultaneously (per FRONTIER-MAP §"Reviewer-grade verdict matrix").
- **TABOO sweep on static surface:** 0 hits on `SEED_RUBRIC_BUILDER_SEED_EXAMPLE` (compiler-style domain) + 0 hits on `SEED_RUBRIC_BUILDER_PROMPT` (only `{seed_example}` / `{document_text}` / `{component_list}` placeholders + abstract JSON template).
- **Runtime-rubric cross-dataset isolation:** Inherited from Plan 12-05-REVISIT methodology — the rubric builder sees only the dataset's own input document; cross-dataset leakage is impossible by construction. Sample rubrics from the Claude sweep (e.g. BBB rubric mentions "FreeSWITCH" as an OTHER example to exclude algorithmic/third-party references; teastore rubric mentions "Slope One" as an OTHER example to exclude algorithm names) — these are project-specific terms but are derived from the *input document the model just read*, not pre-trained benchmark knowledge. This is the GATE-06-compliant runtime-LLM-discovery pattern that CLAUDE.md mandates ("All domain-specific knowledge ... must be discovered dynamically at runtime via LLM analysis of the input data").
- **Reviewer-defensibility:** static surface is compiler-textbook; runtime mechanism is the prescribed pattern; cross-dataset isolation is the empirical guarantee.
- **Verdict.json:** Claude relaxed GATE-01 PASS (macro 0.9474, BBB +4.04pp); gpt-5.4 cross-model GATE-01 PASS (macro 0.9007 ≥ 0.8977 floor — by 0.30pp); cross-dataset isolation PASS on both arms (per `scoreboard.json` + Plan 12-12 SUMMARY).
- **GATE-06 verdict (Plan 12-06 full sweep):** **PASS.**
- **Carry to Plan 13-01:** **YES.**

### 5.4 Frontier: trim2_entval_clean (Plan 12-04 — REJECTED)

- **Mechanism:** ENTITY_EXTRACTION_RULES + VALIDATION_RULES merged via Technique 3 (14 → 10 rules; shared core + 2 role-specific headers).
- **TABOO sweep:** 0 hits on all 3 constants.
- **Reviewer-defensibility:** safe-SE-textbook examples ("observer pattern", "pipeline stage", `com.example.name` — same surface terms as prompts_v2 line 203).
- **GATE-06 verdict:** **PASS.**
- **GATE-01 verdict:** **REJECT** — Claude macro 0.9235 < 0.93; BBB delta −6.59pp > 6pp tolerance. Pre-empts cross-model evaluation.
- **Failing arm:** Claude / bigbluebutton.
- **Rejection rationale (mirrors verdict.json):** prompt-merge that erases the extraction-vs-validation boundary regresses Claude on highest-variance dataset, consistent with the V35a lesson (merging prompts loses density Claude leverages).
- **Carry to Plan 13-01:** **NO.** Kept in repo for negative-result traceability.

### 5.5 Frontier: trim3_runtime_rubric_clean (Plan 12-05 / 12-05-REVISIT — REJECTED)

- **Mechanism:** `DOC_KNOWLEDGE_JUDGE_RULES` replaced by runtime rubric builder (compiler-style seed example + document + candidate mappings → 4-6 item rubric).
- **TABOO sweep:** 0 hits on RUBRIC_BUILDER_SEED_EXAMPLE (compiler-style domain — Lexer / Parser / CodeGenerator / SymbolTable / Optimizer); 0 hits on RUBRIC_BUILDER_PROMPT (abstract JSON template).
- **Runtime cross-dataset isolation:** PASS on both Claude and gpt-5.4 (per Plan 12-05-REVISIT `revisit_audit.json` — `cross_dataset_violations: 0` on 5+5 rubrics, with 1 benign lexical overlap on "UI" in teastore.txt that is verifiably present in the input document, not cross-dataset prior knowledge).
- **GATE-06 verdict:** **PASS** (under the Plan 12-05-REVISIT operationalization, which is the methodologically-correct reading per CLAUDE.md). The prior 12-05 REJECT (which used a strict-reading of GATE-06) is **OVERTURNED** — preserved here for milestone audit traceability.
- **GATE-01 verdict:** **REJECT** — gpt-5.4 cross-model macro 0.8855 < 0.8977 floor (gap 1.22pp; delta −2.22pp vs anchor 0.9077 exceeds 1.0pp tolerance).
- **Failing arm:** gpt-5.4 / teammates (0.8130 — 16 FPs vs Claude's 3) + bigbluebutton (0.7636 — 6 FPs / 20 FNs).
- **Rejection rationale:** model-capability gap on gpt-5.4 (consistent with documented Claude-vs-GPT ~5.7pp gap), NOT a methodological flaw of the trim mechanism. trim3 is preserved as a case study showing the runtime-rubric mechanism is GATE-06-compliant under the correct CLAUDE.md operationalization.
- **Carry to Plan 13-01:** **NO.** Kept in repo for negative-result traceability + methodology case study.

### 5.6 Frontier: trim4 / trim5 / trim6 / trim7 / trim8 (Plans 12-07..12-11 — REJECTED)

Per-prompt runtime-rubric variants applied to the remaining static prompts. Each subclasses `SLinker13Clean` and inserts a per-document rubric-builder call ahead of its target phase. NO STATIC FALLBACK (RuntimeError on empty rubric — Phase 12 EXTENSION user directive).

Light-weight reviewer-defensibility audit (rubric-builder seeds + runtime prompt templates only):

| Variant | Target prompt | Static surface TABOO hits | Hit term & disposition | Original-gate failing arm | Scenario E verdict | Carry? |
|---------|---------------|----------------------------|------------------------|----------------------------|--------------------|--------|
| trim4 | AMBIGUITY_FEW_SHOT + AMBIGUITY_RULES → runtime | 1 | `common` in "common abbreviations" — **safe** (§4.2.1) | Claude JAB −2.56pp (1 FP on 19-link surface) | ACCEPT | NO (frontier-only) |
| trim5 | DOC_KNOWLEDGE_EXTRACTION_RULES → runtime | 0 | — | Claude TS −3.57pp | ACCEPT | NO (frontier-only) |
| trim6 | DOC_KNOWLEDGE_JUDGE_EXAMPLES → runtime (+ trim1 distilled rules) | 0 | — | gpt-5.4 cross-model gap 0.39pp | ACCEPT | NO (frontier-only) |
| trim7 | ENTITY_EXTRACTION_RULES → runtime | 0 | — | Claude JAB −2.56pp | ACCEPT | NO (frontier-only) |
| trim8 | VALIDATION_RULES → runtime | 1 | `validation` in "validation rubric" — **safe** (§4.2.2) | Claude TS −3.57pp + JAB −2.56pp | ACCEPT | NO (frontier-only) |

**GATE-06 verdict for all 5:** **PASS** on the static surface; runtime rubrics inherit Plan 12-05-REVISIT's cross-dataset isolation guarantee (rubric builder sees only the dataset's input document; cross-dataset leakage impossible by construction).

**Carry to Plan 13-01:** **NO** for all 5 — all REJECTED under the original v2.1 gates per `FRONTIER-MAP-SUMMARY.md`. Under Scenario E (relaxed gates), all 5 ACCEPT — documented in FRONTIER-MAP as the prompt-reduction-vs-accuracy envelope, **NOT promoted**.

## 6. Final Trim Disposition (input to Plan 13-01)

| Trim ID | Source plan | GATE-01 Claude | GATE-01 cross-model gpt-5.4 | GATE-06 lexical sweep | GATE-06 reviewer-defensibility | Carry to Plan 13-01? |
|---------|-------------|----------------|------------------------------|------------------------|--------------------------------|----------------------|
| trim1_judge_clean | 12-03 | **PASS** (macro 0.9553, BBB +2.54pp) | **PASS** (macro 0.9173 ≥ 0.8977) | **PASS** (0 hits) | **PASS** | **YES** |
| trim2_entval_clean | 12-04 | **FAIL** (macro 0.9235 < 0.93; BBB −6.59pp) | n/a (skipped) | **PASS** (0 hits) | **PASS** | **NO** |
| trim3_runtime_rubric_clean | 12-05/REVISIT | PASS (relaxed; macro 0.9396, BBB +0.72pp) | **FAIL** (macro 0.8855 < 0.8977 by 1.22pp) | **PASS** (0 static; runtime cross-dataset isolation PASS) | **PASS** | **NO** |
| trim4_ambiguity_runtime_clean | 12-07 | **FAIL** (Claude JAB −2.56pp) | PASS (macro 0.9005 ≥ 0.89 Scenario E) | **PASS** (1 hit, safe) | **PASS** | **NO** |
| trim5_extraction_runtime_clean | 12-08 | **FAIL** (Claude TS −3.57pp) | PASS (macro 0.9056 ≥ 0.89 Scenario E) | **PASS** (0 hits) | **PASS** | **NO** |
| trim6_judge_examples_runtime_clean | 12-09 | PASS | **FAIL** (cross-model 0.8938 < 0.8977 by 0.39pp) | **PASS** (0 hits) | **PASS** | **NO** |
| trim7_entity_runtime_clean | 12-10 | **FAIL** (Claude JAB −2.56pp) | PASS (macro 0.9007 ≥ 0.89 Scenario E) | **PASS** (0 hits) | **PASS** | **NO** |
| trim8_validation_runtime_clean | 12-11 | **FAIL** (Claude TS −3.57pp + JAB −2.56pp) | PASS (macro 0.9070 ≥ 0.89 Scenario E) | **PASS** (1 hit, safe) | **PASS** | **NO** |
| trim9_seed_runtime_clean | 12-12 | **PASS** (macro 0.9474, BBB +4.04pp) | **PASS** (macro 0.9007 ≥ 0.8977 by 0.30pp) | **PASS** (0 hits, static); runtime cross-dataset isolation PASS | **PASS** | **YES** |

**Carry-forward set for Plan 13-01:** {**trim1, trim9**} — the only two trims that PASS **all four** standing gates (GATE-01 Claude, GATE-01 cross-model, GATE-06 lexical, GATE-06 reviewer-defensibility) under the original v2.1 standing-gate scenario.

**Rejected-trims register:** {trim2, trim3, trim4, trim5, trim6, trim7, trim8} — 7 variants. All kept in repo for milestone reproducibility and reviewer-traceability of negative results.

## 7. Mapping Conflict Resolution

**Plans 12-03 and 12-05 both target `DOC_KNOWLEDGE_JUDGE_RULES`** (trim1 = static distillation; trim3 = runtime regeneration). Resolution:

- Plan 12-03 trim1 ACCEPTED on both Claude + gpt-5.4 arms.
- Plan 12-05/REVISIT trim3 REJECTED on gpt-5.4 cross-model arm.
- → **trim1 is the v3 status for `DOC_KNOWLEDGE_JUDGE_RULES`.** No Plan 13-01 ambiguity. The two trims do not compose (they target the same prompt class); composition is moot.

**Plans 12-09 trim6 and 12-03 trim1 jointly modify `DOC_KNOWLEDGE_JUDGE_EXAMPLES` / `DOC_KNOWLEDGE_JUDGE_RULES`:**
- trim1 distills RULES + preserves EXAMPLES byte-equal.
- trim6 regenerates EXAMPLES at runtime + inherits trim1's distilled RULES.
- → trim6 REJECTED, so the v3 surface is trim1's: distilled RULES + byte-equal EXAMPLES.

**No outstanding mapping conflicts.**

## 8. Frozen-File Compliance

```
$ git diff --quiet \
    src/llm_sad_sam/linkers/experimental/prompts_v2.py \
    src/llm_sad_sam/linkers/experimental/s_linker13.py \
    src/llm_sad_sam/linkers/experimental/s_linker13_clean.py \
    src/llm_sad_sam/core/data_types_v2.py \
    src/llm_sad_sam/core/document_loader_v2.py \
    src/llm_sad_sam/pcm_parser_v2.py
$ echo $?
0
```

Frozen v2.0 files are **unchanged** — verified before writing this report.

## 9. Audit Artifacts

| Path | Purpose |
|------|---------|
| `/tmp/gate06_audit_12_06.py` | Auditor script — `ast`-based constant extraction + 100-term TABOO regex sweep. Not committed (transient audit instrument). |
| `/tmp/gate06_audit_results.json` | Per-(file, constant) JSON findings. Not committed (raw transcript). |
| `.planning/phases/12-trim-ablation/12-06-AUDIT-REPORT.md` | This report. Committed. |
| `.planning/phases/12-trim-ablation/12-06-V2_TO_V3_MAPPING-FINAL.md` | Final per-constant mapping table superseding Plan 12-01. Committed. |
| `.planning/phases/12-trim-ablation/12-06-SUMMARY.md` | Plan-12-06 closeout summary. Committed. |
| `.planning/phases/12-trim-ablation/12-VERIFICATION.md` | Phase-12 verification artifact. Committed. |
| `results/ablation_results/12_03_trim1_judge/verdict.json` | Plan 12-03 verdict (input). Unchanged. |
| `results/ablation_results/12_04_trim2_entval/verdict.json` | Plan 12-04 verdict (input). Unchanged. |
| `results/ablation_results/12_05_trim3_runtime_rubric/verdict.json` | Plan 12-05/REVISIT verdict (input). Unchanged. |
| `results/ablation_results/12_extension_runtime_variants/scoreboard.json` | Plans 12-07..12-12 scoreboard (input). Unchanged. |

## 10. Verdict Summary

| Class | Variant count | GATE-06 PASS | GATE-06 FAIL | Carried to Plan 13-01 |
|-------|---------------|--------------|--------------|----------------------|
| Shipped | 2 (+ prompts_v3, helper_v3, clean_v3 — infrastructure) | 2 | 0 | **2 (trim1 + trim9)** |
| Frontier (rejected on GATE-01) | 7 | 7 | 0 | 0 |
| **Total Phase-12 trim variants audited** | **9** | **9** | **0** | **2** |

**GATE-06 holds for every Phase-12 trim variant** under the full BENCHMARK_TABOO sweep + reviewer-defensibility adjudication. The 7 rejected variants fail on **GATE-01**, not GATE-06. The shipped pair (trim1, trim9) is unambiguous.

**PROMPT-04 closed:** generality re-audit complete on every Phase-12 prompt body.

---

*Audit conducted by Claude (Opus 4.7 1M, deterministic) under the Plan 12-06 execution mandate. Zero LLM calls during audit. Pure static analysis on local source files + verdict aggregation from immutable Plan 12-03/04/05/12 verdict JSONs.*
