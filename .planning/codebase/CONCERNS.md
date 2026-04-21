# Concerns

## Technical Debt

### Linker Variant Proliferation

**Intentional but Unmanaged**: `src/llm_sad_sam/linkers/experimental/` contains **32 active linker variants**:
- **ILinker series** (3 files): `ilinker1.py`, `ilinker2.py`, `ilinker3.py`
- **S-Linker series** (29 files): `s_linker.py` through `s_linker12e.py`
  - Subdivided: S-Linker1-9 (major versions), S-Linker9a-9e (ablations), S-Linker10-10a, S-Linker11-11e, S-Linker12a-12e
- **Prompt modules** (2 files): `prompts.py`, `prompts_v2.py`

**Archive folder** (via symlink): 43 archived versions (`archiv01.py` through `archiv43.py`) + `ARCHIVE_STRATEGIES.md`

**CLAUDE.md states**: "User prefers standalone linker files (duplicate code intentionally, not inheritance chains)" — duplicated code is a deliberate design choice for reproducibility. However, **no lifecycle policy** enforces when variants are added vs archived.

**Risk**: Without governance, the 32 active files grow by ~1-3 per week (visible in git history). Each adds:
- ~1-2 KB of duplicate boilerplate (LLM class initialization, pipeline setup, import statements)
- Testing burden (run_ablation.py must enumerate all for benchmarking)
- Cognitive load during prompt audits for data leakage

**Current mitigation**: `archive/README.md` documents families of old variants. But **no rule prevents adding new variants to active folder**; they flow straight from experimentation.

### Dead Code in ModelKnowledge

Per MEMORY.md (Mar 18, 2026), **four dataclass fields persisted after utility was removed**:

| Field | Status | Last Used | Removal History |
|-------|--------|-----------|-----------------|
| `architectural_names` | Dead | Never used; computed by Phase 1 LLM | Removed S-Linker8+ but likely still in S-Linker7 and earlier |
| `shared_vocabulary` | Dead | Never computed or read | Still a dataclass field (no assignment) |
| `impl_to_abstract` | Dead | Computed but `get_abstract()` never called | Removed S-Linker8+ |
| `impl_indicators` | Dead (v9+) | Used only when `impl_filtering` ON | Removed in S-Linker9; still computed in S-Linker8 |

These cause **silent data bloat**. A linker variant may compute 50 KB of unused `architectural_names` on every run.

**Check**: `src/llm_sad_sam/linkers/experimental/s_linker8.py` still includes these computations. Variants from S-Linker7 and below are archaic.

### Scratch Documentation at Root Level

Root contains **14 analysis/debug markdown files** with no lifecycle:

- `CLEANUP_SUMMARY_FEB24.md` (Feb 24) — Post-leakage cleanup notes
- `JUDGE_*.md` (3 files) — Judge redesign analysis (Feb 23-24)
- `PHASE_*.md` (3 files) — Contribution analysis, ablation summary (Feb 24)
- `V26A_VS_V31_*.md` (2 files) — Version comparison
- `AGENTS.md` — Agent definitions (likely pre-S-Linker)
- `DOCUMENTATION_UPDATES.md` — Inert update log

**None are cited in CLAUDE.md or MEMORY.md** (except as loose references). No decision rule: are they:
- Throwaway spike notes (delete after summarization)?
- Reference documentation (move to `.planning/docs/`)?
- Historical artifacts (move to archive/)?

**Risk**: Root becomes a junk drawer. Readers confused about which doc is authoritative.

### Root-Level Test Scripts Mixed with Pytest Suite

- **Root-level (ad-hoc)**: `test_judge_multiword.py` (Mar 30), `test_prompt_audit.py` (Mar 28), `compare_s12c_vs_transarc.py` (Apr 17)
- **Pytest suite** (under `tests/`): Unknown structure (not scanned)

**Issue**: Root test scripts are not discoverable by `pytest` auto-discovery. They look like pytest tests but run standalone. This creates ambiguity — is the test suite in root or tests/?

## Rules & Heuristics Still in Code

Per `.planning/spikes/002-rules-audit/AUDIT.md` (Apr 21), **s_linker12c.py contains 12 helper functions** classified as:

### REPLACEABLE (6 functions → should be LLM-driven)

1. **`_is_structurally_unambiguous(name)`** — CamelCase/space/all-caps inference. Redundant post-filter; LLM already classifies in `_classify_components`. Remove and trust LLM.

2. **`_is_strong_alias(alias, comp_names)`** — Decides 'safe for global broadcast' based on regex strength. Should be emitted by alias-discovery LLM prompt with `{alias, scope: global|local}` field.

3. **`_has_strong_alias_mention(sent, comp, alias)`** — Boolean check during coref. Fold into coref prompt's evidence schema (already running LLM with context).

4. **`_is_ambiguous_name_component(comp_name)`** — Wraps LLM-classified `ambiguous_names` with structural guard. Drop guard; trust LLM directly.

5. **`_classify_mention(name, comp, coref_set)`** — Returns mention type string ('proper case, standalone' / 'lowercase' / 'via alias X' / 'in dotted path'). 4 regex branches. Spike 003 shows LLM can emit this as enum during extraction (no extra LLM call — piggyback on entity-extraction pass).

6. **`_get_strong_alias_mappings(doc_knowledge)`** — Filter over aliases based on `_is_strong_alias` result. Becomes trivial once #2 is gone; fold `scope` field into `doc_knowledge.aliases` schema.

### ESSENTIAL (4 functions → keep)

1. **`_parse_snum(s_string)`** — String-to-int parser ('S42' → 42). Deterministic, <1ms. Not a heuristic; keep as-is.

2. **`_get_comp_names(pcm_model)`** — List accessor. Not a heuristic; keep.

3. **`_build_component_profile(comp_name, ...)`** — String formatter/serializer. Not a heuristic; keep.

4. **`_has_standalone_mention(comp_lower, text)`** — Regex word-boundary match (`\b` guard). **RISKY to replace**: called O(N×M) in anchor collection during validation. LLM replacement would cost massive prompts per sentence-pair. Recommend KEEP as boundary primitive (could optionally narrow: drop dotted-path guard).

### RISKY (1 function → LLM replacement loses performance)

1. **`_has_standalone_mention`** — See ESSENTIAL section above.

### MIXED (1 function → orchestrator, becomes trivial post-replacements)

1. **`_build_evidence_bundle(...)`** — Calls `_classify_mention` (REPLACEABLE) + `_has_standalone_mention` (ESSENTIAL) + `_is_ambiguous_name_component` (REPLACEABLE). After replacements, mostly consumes LLM-emitted fields.

### Inline Regex Hotspots

**11 `re.*` call sites** in s_linker12c.py (L64, L263, L281, L296-297, L643, L645, L659, L863, L1135, L1159):
- Word-boundary guards (`\b{re.escape(...)}`)
- CamelCase detection (`[a-z][A-Z]`)
- Component name splitting (`[\s-]+`, CamelCase/ACRONYM patterns)

Most are in REPLACEABLE functions (#1, #5 above). Retention of `_has_standalone_mention` means **2-3 regex calls remain** for anchor collection.

## Data Leakage Risk (ICSE Critical)

### Prior Incidents (Feb 24-26, 2026)

**V31 & V32 Cleanup**: Found benchmark-derived examples in `CONVENTION_GUIDE` prompt:
- Removed: `cascade` (Teammates), `dedicated` (MediaStore), `preferences` (JabRef), `config` (BBB), `internal` (BBB/Teammates)
- Replaced with safe analogues: `cascade` → `throttle`, `preferences` → `bookmarks`, `Redis PubSub/kurento/FreeSWITCH` → `Kafka Broker/Nginx Proxy/Zookeeper`, `HTML5 Server/Recording Service` → `Kafka Broker/Nginx Proxy`

**BENCHMARK_TABOO.md** now enumerates **59 banned terms** across 5 benchmark projects (MediaStore, TeaStore, Teammates, BigBlueButton, JabRef).

### Leakage Pathways (Documented)

1. **Prompt examples** — Component names sneaking in as "neutral" illustrations.
   - **Mitigation**: Safe SE textbook domains (Compiler: Lexer/Parser/AST; OS: Scheduler/MemoryManager; E-commerce: ShoppingCart/PaymentGateway).

2. **`CONVENTION_GUIDE` rules** — Generic architecture patterns that happen to match benchmark conventions.
   - **Example** (Feb 26 fix): "components named after technologies" rule made sense but examples used `kurento` (BBB benchmark). Changed to `Nginx Proxy` (safe).
   - **Risk**: Very hard to distinguish universal heuristics from benchmark-specific cargo-cult.

3. **Alias discovery prompts** — If seeded with benchmark component synonyms (no longer done post-V26a).

### Audit Discipline Required

MEMORY.md documents that **small tweaks can swing macro F1 ±2pp**. With GPT-5.2 showing -5.7pp loss vs Claude (inherent model gap), every prompt change risks introducing leakage to "recover" performance. Current mitigation:

- **Manual before-commit prompt review** (checklist: any BENCHMARK_TABOO term present?)
- **Post-paper audit** (cross-reference CONVENTION_GUIDE, examples, prompt templates against BENCHMARK_TABOO.md)
- **Phrase-level caution**: "component that is a technology" ✓ vs "Kurento media server" ✗

**Gaps**: No automated detector. No pre-commit hook. Relies on human discipline.

## Security

### API Keys & Environment

- `.env` file (gitignored, ✓) holds `OPENAI_API_KEY` (currently commented out per memory — project uses Claude CLI backend).
- `.gitignore` correctly excludes `__pycache__/`, `*.pyc`, `.env`, `results/`, `build/`, `dist/`, `*.egg*`, `.idea/`, `.vscode/`.

**No user-facing web surface.** Research tool only; no authentication/authorization concerns. No database or persistent state requiring encryption.

**Risk level**: LOW. API key exposure is historical/test key risk (if someone uncomments `.env`).

### Dependency Audit

Not done. Project uses:
- `anthropic` library (Claude API)
- `openai` (optional, commented sections for GPT-5.2 testing)
- Standard ML stack (`torch`, `transformers` via ARDoCo dependencies)

No obvious code injection vectors (no `eval()`, `exec()`, or dynamic imports from user input). LLM prompts are static; models are fixed (Sonnet, GPT-5.2).

## Performance

### LLM Call Dominance

Runtime is dominated by LLM latency, not compute. Each linker variant makes **~100-200 LLM calls per dataset** across phases:
- Phase 1: Model analysis (generic-word detection, etc.)
- Phase 3: Document knowledge (abbreviation/synonym discovery)
- Phase 5-6: Extraction, validation (per-component judgments)
- Phase 8c: Boundary filtering (convention rule application)

**Parallelization**: S-Linker12c uses `_run_parallel()` static method for Tier 1 phases (fan-out model/text analysis). Helps but doesn't eliminate serial bottleneck in phases 5-9.

**Dataset runs**: 5 projects (MediaStore, TeaStore, Teammates, BBB, JabRef) × 30 variants in `run_ablation.py` = **~150 parallel runs**. Takes minutes per variant on shared lab hardware. No caching across runs.

### GPT-5.2 Variance (Inherent)

Per MEMORY.md (Feb 25-26):
- **Claude Sonnet**: 94.5% ±0% macro F1 (deterministic, same result every run)
- **GPT-5.2**: 90.6% ±5-12 links (massive run-to-run variance, same prompt/seed)

This is **not temperature-fixable** and reflects model behavior (not code). Affects ablation reproducibility: if repeating an experiment, GPT results may vary by 2pp due to model, not code changes.

## Fragile Areas

### LLM Prompt Sensitivity

**V35 experiment series (Mar 9)** tested 6 simplification proposals on Claude Sonnet. **All regressed**:
- V35 (all 6): -2.4pp
- V35a (example-driven CONVENTION_GUIDE): -2.5pp
- V35b (compact Phase 3 judge): -9.9pp on MediaStore
- V35c (concrete JSON output examples): -7.1pp (worst; biases output)

**Lesson**: V32 prompts are at local optimum. Every simplification loses calibration. Prompts encode subtle information density (6 calibration examples, detailed step reasoning) that contributes to Claude's performance.

**Fragility**: Unknown which parts of a prompt are critical. Removing a bullet point can swing 2pp. No principled way to prune.

### Judge Approval Bias (Intentional but Fragile)

V31/V32 judge uses three voices (Advocate/Prosecutor/Jury) with **intentional approval bias**:
- Advocate says "even generic words typically refer to component" (skews interpretation toward approval)
- Jury synthesizes with tie-breaking favoring "APPROVE"

Per MEMORY.md (Feb 23), `judge_prompt_analysis.md` documents this is an **acceptable design choice** but **fragile**:
- Weakening bias by 1-2 phrases → rebalance cascades through 70+ FPs that are "judge-immune" (bypass due to immunity rules)
- Strengthening → kills 14+ true-positive partial-name refs (e.g., "Server" → HTML5 Server)

**Risk**: Changing judge approval bias for any reason (GPT compatibility, new dataset) likely triggers F1 swings of ±2pp.

### Cross-Model Portability

- **Claude Sonnet 94.8%** (V32, s_linker12c Feb 26) — BASELINE
- **GPT-5.2 90.6%** (V32) — -5.7pp inherent gap
- **GPT-4o 86.1%** — Not viable
- **GPT-5.4** — Worse than 5.2 (87.7%)

The gap is **inherent model capability**, not fixable without regressing Claude. GPT interprets prompts more literally; misses nuance in coref, partial-name disambiguation, and synonym judgment.

## Research-Specific Concerns

### Benchmark Taboo Discipline

BENCHMARK_TABOO.md exists but **compliance is manual**:
- No automated prompt analyzer
- No pre-commit hook to reject code with taboo terms
- No CI gate ("fail if CONVENTION_GUIDE contains {cascade, kurento, ...}")

**Current process** (per MEMORY.md):
1. Develop linker & prompts
2. Run ablation; if ≥94.5% F1, declare success
3. Before paper: audit CLAUDE.md + prompts by hand for taboo terms
4. Iterate if found (happened Feb 24-26 with V31/V32)

**Risk**: As paper deadlines approach, review may be rushed. A term like "internal module" (appears in templates, slipped through twice) can leak easily.

### Ablation Reproducibility

**V30c+ pickle checkpoints** — Many ablation variants depend on **pickled state after each phase**:
- `test_heuristics.py`: Loads V30c checkpoint, ablates individual heuristics offline (zero LLM calls)
- `test_coref_debate_variants.py`: Loads V30c, swaps coref logic, re-scores

**Fragility**: If V30c checkpoints are **regenerated** (e.g., new machine, Python version upgrade), checkpoint byte representation may differ. Replay scripts could break silently (load "same" data but with different semantics).

**Mitigation**: Keep V30c pickle files in git (currently `.gitignore` includes `*.pyc` but not pickles). But no version control on pickles = no audit trail.

### Memory File Index Limits

MEMORY.md is **211 lines** (limit: 200 per system reminder). It works now but:
- Cannot add more than 10-15 lines without exceeding limit
- When limit hits, old historical notes must be removed
- No archive mechanism (e.g., MEMORY_ARCHIVE.md)

**Risk**: Critical findings from future experiments cannot be recorded in MEMORY.md. Institutional knowledge gets scattered across `.planning/spikes/` and loose markdown files.

## Recommended Next Steps

1. **Add data leakage pre-commit hook**: Flag any file containing BENCHMARK_TABOO terms
2. **Establish linker archival policy**: Define when active variants move to archive (e.g., >2 weeks old + superseded)
3. **Organize scratch docs**: Move root-level PHASE_*.md, JUDGE_*.md to `.planning/docs/historical/` with index
4. **Deprecate dead code**: Audit which variants still compute `architectural_names`, `shared_vocabulary`, etc. Remove or conditionally skip.
5. **Codify prompt audit**: Checklist in CLAUDE.md for pre-submission review (5 checks: CONVENTION_GUIDE, Phase 1-3 examples, prompt templates, any regex with architecture terms)
6. **Separate test scripts**: Move root-level `test_*.py` to `tests/` or `.planning/spikes/` depending on purpose (regression vs. exploration)
