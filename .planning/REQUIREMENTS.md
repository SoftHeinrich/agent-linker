# Requirements: llm-sad-sam-v45 — Milestone v2.6.4

**Defined:** 2026-06-05
**Milestone:** v2.6.4 — Per-Prompt Unit-Tested Minimization + Generality Pass on s_linker19
**Core Value:** Audit every LLM-call site in `s_linker19` with per-prompt golden-replay unit tests; ship `s_linker20.py` whose prompts are at the Pareto-best of size-cut × generality, without regressing the s17e-line macro F1 floor (gpt-5.4 92.3%, T=1.0pp → floor 91.3%). Backend: gpt-5.4 only. Zero new LLM calls for harness build.

## Active v2.6.4 Requirements

### HARNESS — Per-prompt unit-test infrastructure

- [ ] **REQ-V264-01** — Golden-replay test harness loads v2.6.3 `phase_cache` pkls (`results/phase_cache/openai/<project>/{layer1..4,final}.pkl`) and exposes `(prompt_built, llm_response, parsed_output)` triples for each of the 6 s19 prompt sites × 5 projects. Zero new LLM calls. Backend scope: gpt-5.4 only. Lives under `tests/harness/` (or equivalent shared fixture module).

- [ ] **REQ-V264-02** — Pytest + snapshot harness (syrupy or pytest-regressions) ships one test module per s19 prompt builder: `tests/test_s_linker20_prompt_{ambiguity,doc_extract,doc_judge,extraction,validation,coref}.py`. Each test rebuilds the prompt from the replay fixture, runs the replayed LLM response through the parser, and asserts snapshot equality on the **parsed structured output** (NOT raw LLM text — replayed LLM output is fixed). Initial snapshots captured from s19 byte-equal baseline; tests pass at REQ-V264-02 close.

### AUDIT — Identify generality + size cut candidates

- [ ] **REQ-V264-03** — Per-constant audit report covers each imported PROMPT CONSTANT used by `s_linker19`: `AMBIGUITY_FEW_SHOT`, `AMBIGUITY_RULES`, `DOC_KNOWLEDGE_EXTRACTION_RULES`, `ALIAS_SCOPE_RULES`, `DOC_KNOWLEDGE_JUDGE_EXAMPLES`, `DOC_KNOWLEDGE_JUDGE_RULES`, `ENTITY_EXTRACTION_RULES`, `VALIDATION_RULES`, `COREF_RULES`. One row per constant with columns: current LOC, generality verdict (`clean` / `domain-loaded` / `benchmark-leak`), size-cut candidates (line-level), drop-the-whole-block candidates.

- [ ] **REQ-V264-04** — Per-builder audit covers the 6 in-class f-string scaffolds: `_prompt_ambiguity`, `_prompt_doc_knowledge_extract`, `_prompt_doc_knowledge_judge`, `_prompt_extraction`, `_prompt_validation`, `_prompt_coref`. Same columns as REQ-V264-03 but for the scaffolding around the constants. Single combined artefact: `s_linker20-PROMPT-AUDIT.md`.

### MINIMIZE — Pareto-driven cuts

- [ ] **REQ-V264-05** — Per-prompt Pareto reduction loop: for each candidate cut identified in REQ-V264-03/04, apply the cut, run that prompt's golden tests (REQ-V264-02), and keep the cut iff **every parsed-output snapshot remains byte-equal** AND no benchmark-derived vocabulary is introduced. All decisions logged per candidate cut in `s_linker20-MINIMIZE-LOG.md` with verdict (kept / reverted / unsafe).

- [ ] **REQ-V264-06** — Few-shot block-drop: for each prompt with a few-shot block (initially `AMBIGUITY_FEW_SHOT`, `DOC_KNOWLEDGE_JUDGE_EXAMPLES`), run the golden suite with the **entire block removed**. Drop the block iff every parsed-output snapshot is byte-equal. If not byte-equal, attempt a 1–3 example synthetic-domain replacement and re-run; ship whichever is smallest while passing.

- [ ] **REQ-V264-07** — Lexical neutralization: where domain-loaded vocabulary appears (e.g., "software architecture component", "anaphoric references", "role-referential noun phrases"), attempt a neutral rewording (e.g., "entity", "pronouns and noun phrases that refer back"). Keep the rewording iff parsed-output snapshots are byte-equal. Target framing: "look general but still SAD/SAM-tuned" — behaviour stays tuned to SAD→SAM, only surface vocabulary changes.

### SHIP — New variant + regression

- [ ] **REQ-V264-08** — `src/llm_sad_sam/linkers/experimental/s_linker20.py`: standalone file (no inheritance from `s_linker19`), `experimental=True`, `canonical=False`. Inlines the minimized PROMPT CONSTANTS so the audit is self-contained per the user's "duplicated standalone files over inheritance" preference. `s_linker19.py` and any prompt constants `s_linker19` imports are preserved **byte-equal** (paper RQ1–RQ4 replay determinism). `run_ablation.py` learns `--variants s_linker20`.

- [ ] **REQ-V264-09** — End-to-end GPT-5.4 5-dataset macro F1 on `s_linker20` ≥ **91.3%** (= s17e 92.3% − T 1.0pp). Per-dataset constraint: no dataset drops more than 2pp vs s17e's per-dataset numbers (MediaStore 94.9%, TeaStore 96.3%, TeaMmates 89.8%, BigBlueButton 80.4%, JabRef 100.0%). Single sweep validates promotion. Log goes to `logs/v2.6.4_s_linker20_gpt.log`.

### CARRY-FORWARD — Standing Gates

- [ ] **GATE-01** (carried) — `s_linker13_min.py` AND `s_linker19.py` SHA-256 byte-equal at milestone close (paper baseline + canonical untouched). Verified via `git diff` against the v2.6.3 close hashes.
- [ ] **GATE-06** (re-verified) — Zero benchmark-derived vocabulary in any `s_linker20` prompt constant or f-string scaffold. Audit method: v2.1 cross-dataset vocabulary isolation methodology.
- [ ] **GATE-08** (budget) — Sweep budget cap ≤ **$20** for the macro F1 regression validation (5-dataset gpt-5.4 single run); zero LLM calls for golden-test build.

## Future Requirements (deferred)

- Cross-backend (Claude) confirmation sweep on `s_linker20` — v2.6.5 candidate if v2.6.4 promotes.
- Per-prompt minimization extended to `s_linker17e` family — only if 17e remains the published champion and reviewers ask for prompt-defensibility.
- Flex tier integration (`260601-flex-tier-integration.md`) — cost optimization, v2.7+.

## Out of Scope for v2.6.4

- Logic changes to `s_linker19` or `s_linker13_min` (canonical/paper frozen).
- Resumption of v2.7 (BBB recall closure, Phases 38–42) — FROZEN.
- v2.6 close (Phase 37 GATE-06 'Persistence' taboo fix) — DEFERRED.
- Cross-model Claude validation — gpt-5.4 only (per v2.3 standing policy).
- New benchmark datasets — 5-dataset benchmark unchanged.
- Aggressive behavior changes / new few-shots that aren't byte-equal on parsed outputs.

## Requirement Traceability

| REQ-ID | Phase |
|--------|-------|
| REQ-V264-01 | TBD (set by roadmapper) |
| REQ-V264-02 | TBD |
| REQ-V264-03 | TBD |
| REQ-V264-04 | TBD |
| REQ-V264-05 | TBD |
| REQ-V264-06 | TBD |
| REQ-V264-07 | TBD |
| REQ-V264-08 | TBD |
| REQ-V264-09 | TBD |
| GATE-01 | Throughout |
| GATE-06 | Throughout (close-gated) |
| GATE-08 | Sweep phase |
