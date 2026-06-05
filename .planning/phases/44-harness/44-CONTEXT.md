# Phase 44: HARNESS - Context

**Gathered:** 2026-06-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a pytest snapshot harness that replays cached LLM artefacts for all 6 `s_linker19` prompt sites × 5 projects (gpt-5.4 backend only), rebuilds each prompt deterministically, runs the **replayed** LLM response through the production parser, and asserts snapshot equality on the **parsed structured output**. Lives under `tests/harness/`. Zero new LLM calls. Read-only on `s_linker19.py` and any constants/modules it imports — GATE-01 byte-equality preserved.

**In scope (Phase 44):**
- Fixture-loading infrastructure under `tests/harness/` (or shared module) that exposes `(prompt_built, llm_response, parsed_output)` triples per prompt site × project.
- Six pytest test modules at the layout REQ-V264-02 specifies:
  `tests/test_s_linker20_prompt_{ambiguity,doc_extract,doc_judge,extraction,validation,coref}.py`.
- Initial snapshot capture from byte-equal s19 baseline.
- `pytest tests/harness/` (or full suite covering the 6 modules) green with exit 0 and provably zero LLM API calls.

**Out of scope (Phase 44):**
- Any prompt audit, generality verdict, or candidate-cut list (Phase 45 owns these).
- Any prompt minimization or rewording (Phase 46).
- `s_linker20.py` itself — file does not exist until Phase 47.
- Logic changes to `s_linker19.py`, `s_linker13_min.py`, `prompts_v5.py`, or any module imported by s19. GATE-01 byte-equal verified at phase close.
- Claude backend fixtures — gpt-5.4 only (v2.3 standing policy).
- Cross-backend confirmation, runner integration of s20, budget gates (deferred to later phases).

</domain>

<decisions>
## Implementation Decisions

### Fixture Data Source

- **D-01:** **Paired fixtures — `phase_cache` pkls AND `llm_logs/*_calls.json`.** Each test pulls `(prompt_built, response_text)` from `llm_logs` (the only place raw prompts and raw model responses are persisted) and uses the matching `phase_cache/<project>/{layer1..4,final}.pkl` as an INDEPENDENT cross-check that the harness-produced parsed output reproduces what the byte-equal s19 baseline actually parsed.
  - **Why both:** `phase_cache` pkls store parsed dataclasses (`DocumentKnowledge`, `framing_c`, `validated`, `coref_validated`, `final`) but NOT prompts and NOT raw LLM text. `llm_logs` JSON stores raw `(phase, prompt, response_text, …)` per LLM call. REQ-V264-02 explicitly requires "rebuild prompt → run replayed LLM response through parser → snapshot-equal parsed output", which forces a paired source.
  - **What gets snapshotted:** the parsed structured output produced by the harness, byte-equal to the snapshot captured from the s19 baseline. The pkl cross-check is a SECOND assertion ("our parsed output is consistent with the pkl that shipped"), not the snapshot itself.

- **D-02:** **`tests/harness/fixtures/MANIFEST.json` — explicit committed pairing for 5 projects.**
  - Each entry: `{project, pkl_dir, calls_json}` — relative paths from repo root. Adding/replacing a baseline run requires editing the manifest (explicit ledger).
  - Suggested per-project entries (paths verified at scout time, all five exist):
    | project | pkl_dir | calls_json |
    |---|---|---|
    | mediastore    | `results/phase_cache/s_linker19/openai/mediastore/`    | `results/llm_logs/s_linker19_openai_mediastore_20260605_134622_calls.json` |
    | teastore      | `results/phase_cache/s_linker19/openai/teastore/`      | `results/llm_logs/s_linker19_openai_teastore_20260604_065824_calls.json` |
    | teammates     | `results/phase_cache/s_linker19/openai/teammates/`     | `results/llm_logs/s_linker19_openai_teammates_20260604_070526_calls.json` |
    | bigbluebutton | `results/phase_cache/s_linker19/openai/bigbluebutton/` | `results/llm_logs/s_linker19_openai_bigbluebutton_20260604_070639_calls.json` |
    | jabref        | `results/phase_cache/s_linker19/openai/jabref/`        | `results/llm_logs/s_linker19_openai_jabref_20260605_134705_calls.json` |
  - **Why manifest, not auto-pair-by-mtime:** immune to log-directory cleanup, churn, or regeneration races; gives a single canonical place to see "what byte-equal baseline does the harness bind to?"; pkl/log mtime alignment is a coincidence today but not enforceable.
  - **Why not copy fixtures into the test tree:** ~5–20 MB of git churn and severs the harness from the artefacts the rest of the project consults. Manifest preserves the existing layout and adds a thin pointer.

### Builder → Phase-Tag Mapping (Locked from Code Scout)

- **D-03:** **The 6 s19 builders map to phase tags as follows** (verified by reading `s_linker19.py`):
  | Builder | Phase tag(s) in `llm_logs` |
  |---|---|
  | `_prompt_ambiguity` | `phase_1_model` |
  | `_prompt_doc_knowledge_extract` | `phase_1_doc_extract` |
  | `_prompt_doc_knowledge_judge` | `phase_1_doc_judge` |
  | `_prompt_extraction` | `phase_2_framing_c_pass1`, `phase_2_framing_c_pass2` |
  | `_prompt_validation` | `phase_4_twopass_p1`, `phase_4_twopass_p2`, **`phase_5_coref_validation`** |
  | `_prompt_coref` | `phase_5_coref` |
  - **Note for coref:** `phase_5_coref_validation` reuses `_prompt_validation` with `COREF_VALIDATION_FOCUS` (see `s_linker19.py:893–916`). Its fixtures go in `test_s_linker20_prompt_validation.py`, NOT `..._coref.py`. This was a hidden gotcha — REQ-V264-02's 6-module list is correct, but the validation module covers three phase tags, not two.

### Claude's Discretion

- **Manifest schema details:** JSON shape (a list vs an object keyed by project), inclusion of `expected_sha256` for drift detection, optional `description` field. Planner picks. Minimum keys: `project`, `pkl_dir`, `calls_json`.
- **Snapshot library** (syrupy vs pytest-regressions) — DEFERRED to planner. REQ-V264-02 explicitly leaves this open. Neither is currently installed in the env; either will need to land in `pyproject.toml` `[dev]` extras alongside its conventions.
- **Parser isolation strategy** — DEFERRED. Planner picks among (a) monkey-patching `s_linker19`'s LLM client to return cached `response_text` and running the original phase methods (closest to baseline, but pulls in a lot of orchestration code per test); (b) extracting per-site parsers as standalone helpers (cleanest test surface, biggest refactor risk for GATE-01); (c) calling the builder + stubbed LLM client end-to-end with a thin per-site adapter. Constraint: whichever path, `s_linker19.py` and `prompts_v5.py` must stay byte-equal at phase close.
- **Test parametrization granularity** — DEFERRED. REQ-V264-02 mandates "one test module per builder" (6 modules). Inside each module, planner may parametrize across projects, or across (project, call_index) tuples for builders with multiple calls per project. Both compatible with the locked fixture source.
- **`tests/harness/` layout** — REQ-V264-01 says `tests/harness/ (or equivalent)` for fixture infrastructure; REQ-V264-02 places test modules at `tests/test_s_linker20_prompt_*.py`. Planner reconciles (fixtures under `tests/harness/`, test modules either co-located under `tests/harness/` or at `tests/` top level per REQ wording).
- **Whether the manifest also records the v2.6.3 close SHA-256** for the pkl bundle (per GATE-01 byte-equal verification at Phase 47/49). Planner decides if this hash lives in the manifest or in a separate gate-check artefact.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope, requirements, and gates
- `.planning/ROADMAP.md` §"Phase 44: HARNESS" (lines 151–161) — phase goal and 4 success criteria.
- `.planning/REQUIREMENTS.md` §"HARNESS" — REQ-V264-01 (fixture infrastructure), REQ-V264-02 (six pytest modules, snapshot on parsed output, pass at REQ close).
- `.planning/PROJECT.md` §"Constraints" and §"Key Decisions" — GATE-01 byte-equality of `s_linker19.py` + `s_linker13_min.py`; gpt-5.4-only backend policy (v2.3 standing); BENCHMARK_TABOO compliance.
- `.planning/STATE.md` — current milestone is v2.6.4, Phase 44 is the first of six phases (44–49).

### Frozen source artefacts (read-only during this phase)
- `src/llm_sad_sam/linkers/experimental/s_linker19.py` — defines the 6 builders (lines 264–380) and the `_TracingLLMClient` wrapper that produced the `_calls.json` files (line 121). Phase tags set inside the run methods (lines 561, 573, 646, 793, 835, 894).
- `src/llm_sad_sam/linkers/experimental/prompts_v5.py` — 9 PROMPT CONSTANTS imported by s19 (AMBIGUITY_*, DOC_KNOWLEDGE_*, ALIAS_SCOPE_RULES, ENTITY_EXTRACTION_RULES, VALIDATION_RULES, COREF_RULES, COREF_VALIDATION_FOCUS, ANTECEDENT_ALIAS_RULES). All byte-equal-frozen for v2.6.4.
- `src/llm_sad_sam/core/data_types_v2.py` — dataclasses (`Link`, `ModelKnowledge`, `DocumentKnowledge`, etc.) used in the phase_cache pickles; needed on `sys.path` for `pickle.load`.
- `src/llm_sad_sam/llm_client.py` §`LLMClient.extract_json` — the JSON extractor at the entry of every site's parser path.

### Fixture artefacts (read-only inputs to the harness)
- `results/phase_cache/s_linker19/openai/<project>/{layer1,layer2,layer3,layer4,final}.pkl` — parsed-output cross-check source. 5 projects × 5 layers = 25 files.
- `results/llm_logs/s_linker19_openai_<project>_<TIMESTAMP>_calls.json` — list of LLM-call records, each with `{phase, ts, elapsed_s, timeout, max_retries, prompt, response_text, success, error, latency_ms, model}`. The canonical timestamps per project are pinned in D-02.

### Prior context (for pattern continuity)
- `.planning/milestones/v2.6.3-phases/43-replay-s-linker19-checkpoints-for-paper-rq1-rq4-eval/43-CONTEXT.md` — Phase 43 also used `phase_cache` pkls for replay (RQ1/RQ3/RQ4). Its 2-stage pipeline split convention may inform the harness shape (replay-stage logic stays in `agent-linker` because pickle deserialization needs `src/llm_sad_sam` on `sys.path`).
- `tests/conftest.py` — already wires `ROOT` and `ROOT/src` into `sys.path`; harness tests inherit this and can `pickle.load` directly.

### Standing taboo / gate references
- `.planning/BENCHMARK_TABOO.md` (if present) — GATE-06 vocabulary leak avoidance; relevant for any new test names or fixture content authored under `tests/harness/`. Phase 44 doesn't author prompt text, so leak risk is low — but new fixture file paths and test names should avoid benchmark-derived terms.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`tests/conftest.py`** (10 lines) — already prepends `ROOT` and `ROOT/src` to `sys.path`. Any new test under `tests/` or `tests/harness/` inherits this and can `pickle.load` phase_cache files without extra config.
- **`_TracingLLMClient`** in `s_linker19.py:121` — the exact wrapper that produced every `_calls.json` entry. Reading its `query()` path documents the serialization format the harness must consume.
- **`LLMClient.extract_json`** in `src/llm_sad_sam/llm_client.py` — single canonical JSON-extraction entry point for every parser path; any monkey-patched LLM client must short-circuit at `query()`, leaving `extract_json` intact.
- **Phase tags** set by `self.llm.set_phase("phase_X_Y")` at builder call sites — gives the harness a deterministic way to group `_calls.json` entries by builder (per D-03 mapping).

### Established Patterns
- **Standalone variant files, no inheritance** — user-stated preference in PROJECT.md §Constraints. Test modules and fixtures should follow the same shape: standalone, no clever base-class trickery.
- **Read-only on frozen artefacts** — Phase 43 established the convention that phases consuming `phase_cache` do not modify any frozen source file. Phase 44 carries this forward: GATE-01 verified at close.
- **Pickle-load needs `src/` on `sys.path`** — established by `tests/conftest.py`. Any script outside `tests/` (e.g., a one-shot snapshot-capture utility) must replicate this manually.

### Integration Points
- New `tests/harness/` package — fixture loader, manifest reader, per-builder adapter functions.
- New `tests/harness/fixtures/MANIFEST.json` — explicit pkl/calls-json pairing per project (D-02).
- New `tests/test_s_linker20_prompt_{ambiguity,doc_extract,doc_judge,extraction,validation,coref}.py` — six modules, layout per REQ-V264-02. Validation module covers three phase tags including `phase_5_coref_validation` (D-03).
- `pyproject.toml` `[dev]` extras — add snapshot lib (syrupy or pytest-regressions, planner's call) once chosen.

</code_context>

<specifics>
## Specific Ideas

- **Snapshot target = parsed structured output, not raw LLM text.** Explicitly called out in REQ-V264-02 and reaffirmed in discussion: "replayed LLM output is fixed; the assertion is on the parser's product." Pinning this prevents the common mistake of snapshotting LLM text and then declaring victory when nothing has actually been tested.
- **Manifest is the ledger.** When this phase produces new baselines later (e.g., regenerating a project's `phase_cache` after a fix), the change is visible in one place: a manifest diff. This was the explicit reason for preferring "Commit a manifest" over auto-pairing.

</specifics>

<deferred>
## Deferred Ideas

- **Cross-backend (Claude) harness fixtures** — Phase 44 is gpt-5.4 only. A Claude-side mirror harness might be useful for v2.6.5+ but is out of v2.6.4 scope.
- **Copy fixtures into `tests/harness/fixtures/<project>/` for full isolation from `results/`** — considered, rejected for v2.6.4 (5–20 MB git churn vs. an immune manifest). Revisit if `results/` directory layout changes substantially or if CI starts cleaning `results/` aggressively.
- **Pickle-bytes SHA-256 captured in manifest** — flagged as Claude's discretion; if the planner adds it, becomes the GATE-01 byte-equal verification hook for Phase 47 / Phase 49 without an extra artefact.
- **Audit, minimize, rewording, ship, sweep, close** — all explicitly Phases 45–49.

</deferred>

---

*Phase: 44-HARNESS*
*Context gathered: 2026-06-05*
