# Phase 44: HARNESS - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-05
**Phase:** 44-HARNESS
**Areas discussed:** Fixture data source

---

## Area selection

| Option | Description | Selected |
|--------|-------------|----------|
| Fixture data source | REQ-V264-01 says load from phase_cache pkls — but those hold parsed outputs only. Raw (prompt, response_text) live in `results/llm_logs/*_calls.json`. Pair both, treat llm_logs as canonical, or rebuild a unified fixture file? | ✓ |
| Snapshot library | REQ-V264-02 leaves it open: syrupy vs pytest-regressions. Neither installed. Pick one + commit to its idioms. | |
| Parser isolation strategy | (a) Monkey-patch s19's LLM client to return cached response_text and run the full per-phase method; (b) extract per-site parser as standalone helper; (c) call s19 builders + a stubbed LLM client end-to-end. Affects test granularity + s19 byte-equality risk. | |
| Test parametrization granularity | REQ says "one test module per builder" (6 modules). But each builder fires N times across 5 projects. Parametrize per project? Per (project, call_index)? Or one aggregated snapshot per builder per project? | |

**User's choice:** Fixture data source only. The other three deferred to planner / Claude's discretion in CONTEXT.md.

---

## Fixture Data Source

### Sub-question 1: canonical fixture source

| Option | Description | Selected |
|--------|-------------|----------|
| Pair pkls + llm_logs JSON | Load `llm_logs/*_calls.json` for (prompt, response_text). Load `phase_cache` pkls as an INDEPENDENT cross-check (parsed_output snapshot must match what's in the pkls). Most defensive; uses both artefacts as intended. | ✓ |
| llm_logs JSON only | Treat the most recent matching llm_logs run as the single source of truth. Snapshots captured from parse step at fixture-build time, frozen. phase_cache pkls ignored. | |
| Bake a new unified fixture once | One-time build step: read llm_logs + pkls, write `tests/harness/fixtures/<project>_<site>.json` with {prompt, response_text, parsed_output_expected}. Commit those JSON files; tests read from them. | |

**User's choice:** Pair pkls + llm_logs JSON.
**Notes:** The pair gives the harness both a deterministic input (prompt + response_text from `_calls.json`) and an independent ground truth (parsed pkl contents) for a second cross-check. CONTEXT.md D-01.

### Sub-question 2: how to pin the (pkl_dir, calls_json) pairing

| Option | Description | Selected |
|--------|-------------|----------|
| Commit a manifest | Write `tests/harness/fixtures/MANIFEST.json` with 5 entries: {project, pkl_dir, calls_json, expected_hash}. Tests read manifest verbatim. Immune to log-dir cleanup. Adding a new run requires editing the manifest — explicit ledger of what the harness binds to. | ✓ |
| Auto-pair by timestamp | At fixture-load time, walk `results/phase_cache/s_linker19/openai/<project>/`, read `final.pkl` mtime, find the `calls.json` whose filename timestamp matches. Lighter, zero new committed files. Breaks if anyone renames/deletes log files. | |
| Copy fixtures into the test tree | One-time: copy 5 pkl bundles + 5 calls.json into `tests/harness/fixtures/<project>/`. Tests read ONLY from `tests/harness/`. Severs harness completely from `results/`. Costs ~5–20 MB extra in git but the harness becomes self-contained. | |

**User's choice:** Commit a manifest.
**Notes:** Per-project pkl ↔ calls.json timestamp alignment verified at scout time (mediastore 13:46:22 ↔ 134622, teastore 06:58:24 ↔ 065824, teammates 07:05:26 ↔ 070526, bigbluebutton 07:06:39 ↔ 070639, jabref 13:47:05 ↔ 134705). Manifest is the explicit ledger; alignment is a coincidence today, not enforceable. CONTEXT.md D-02.

---

## Claude's Discretion

Captured in CONTEXT.md §"Claude's Discretion":
- Manifest JSON schema details (list vs object-keyed, inclusion of `expected_sha256`, optional `description`).
- Snapshot library (syrupy vs pytest-regressions) — REQ-V264-02 explicitly leaves this open.
- Parser isolation strategy (monkey-patch / per-site helper / builder+stub adapter) — constraint: `s_linker19.py` + `prompts_v5.py` stay byte-equal.
- Test parametrization granularity inside each of the 6 modules.
- `tests/harness/` layout reconciliation between REQ-V264-01 wording and REQ-V264-02 wording.
- Whether the manifest also records v2.6.3-close SHA-256 for the pkl bundle.

## Deferred Ideas

- **Cross-backend (Claude) harness fixtures** — v2.6.4 is gpt-5.4 only. Mirror harness candidate for v2.6.5+.
- **Copy fixtures into test tree (full isolation from `results/`)** — considered, rejected for v2.6.4 (git churn vs. immune manifest). Revisit if `results/` layout changes or CI starts cleaning it.
- **Pickle-bytes SHA-256 in manifest** — flagged as discretion; if added, becomes the GATE-01 byte-equal verification hook for Phase 47/49.
- **Audit / minimize / rewording / ship / sweep / close** — explicitly Phases 45–49.

## Hidden gotcha surfaced during scouting (logged in CONTEXT.md D-03, not a discussion item)

`phase_5_coref_validation` reuses `_prompt_validation` with `COREF_VALIDATION_FOCUS` — its fixtures go in `test_s_linker20_prompt_validation.py`, NOT `test_s_linker20_prompt_coref.py`. Verified in `s_linker19.py:893–916`. The REQ-V264-02 list of 6 modules is correct; the validation module covers three phase tags (`phase_4_twopass_p1`, `phase_4_twopass_p2`, `phase_5_coref_validation`), not two.
