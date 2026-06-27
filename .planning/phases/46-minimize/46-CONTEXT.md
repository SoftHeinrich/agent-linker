# Phase 46: MINIMIZE - Context

**Gathered:** 2026-06-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Apply each of the 19 cut candidates enumerated in Phase 45's `s_linker20-PROMPT-AUDIT.md` against the Phase 44 golden-replay harness. For every candidate produce a `kept` / `reverted` / `unsafe` verdict logged in `s_linker20-MINIMIZE-LOG.md`, where:

- **kept** = byte-equal across all gated snapshots AND no benchmark-derived vocabulary introduced (GATE-06 cross-dataset isolation re-check on the after-text)
- **reverted** = at least one gated snapshot diverged
- **unsafe** = passes byte-equal but introduces new benchmark vocab on the GATE-06 re-check

The kept cuts collectively define the **minimized prompt set** that Phase 47 will inline into `s_linker20.py`. GATE-01 byte-equal on `s_linker19.py` + `prompts_v5.py` + `s_linker13_min.py` must hold at phase close.

**In scope (Phase 46):**

- 17 trial-eligible cut candidates: all 19 audit rows MINUS the 2 protected tombstones (CUT-VAL-04 P1_FOCUS X.Y.Z clause; CUT-COR-05 coref conservatism instruction at s19:361). The tombstones get explicit `protected` log rows but are not trialled.
- Drop-block protocol per REQ-V264-06 for CUT-AMB-01 + CUT-DKJ-01: trial sequence is `drop → Family A → Family B`; smallest passing replacement wins.
- 5-row Family A + 2-row Family B handling under DKJ (3 Family A synthetic-neutral swaps + 2 Family B concept-only) — but only when the parent drop has been resolved per the above protocol.
- Production of `s_linker20-MINIMIZE-LOG.md` with one row per cut and a top-of-doc Pareto-frontier section: rows-kept count, LOC saved per section, GATE-06 isolation result per kept cut.
- One atomic commit per cut decision (kept commit = scratch update + LOG row; reverted commit = LOG row only) so any single cut can be reverted independently.

**Out of scope (Phase 46):**

- Any direct edit to `s_linker19.py`, `prompts_v5.py`, `s_linker13_min.py`, or any module they import. GATE-01 byte-equal must hold continuously throughout the phase (verified via `git diff --stat` after each cut commit).
- Creation of `s_linker20.py` — Phase 47 work. Phase 46 produces only the *recipe* (MINIMIZE-LOG.md + scratch reference text), not the standalone variant.
- New LLM calls. All decisions are driven by cached fixtures replayed through the Phase 44 harness. GATE-08 budget for v2.6.4 is reserved for Phase 48's sweep — Phase 46 is $0.
- Emergent cuts not in the audit (rows lacking a `CUT-{TAG}-NN` id). Per Phase 45 §integration-contract, no new cuts in Phase 46 unless flagged `EMERGENT` with a rationale in the log.
- Cross-backend Claude validation — gpt-5.4 only.

</domain>

<decisions>
## Implementation Decisions

### Test Orchestration (D-01)

- **D-01 — Scratch copies under `tests/scratch/` with harness adapter override.**
  - One-time setup creates `tests/scratch/s_linker19.py` and `tests/scratch/prompts_v5.py` as byte-for-byte copies of the frozen sources.
  - Phase 44 harness adapter (`tests/harness/adapters.py`) gains a small toggle (`SAD_SAM_LINKER_SOURCE`, default `production`) that swaps the import path to `tests.scratch.*` when set to `scratch`. The default path stays unchanged — production tests still import the frozen sources.
  - Each Phase 46 cut is applied to the scratch copies only. The harness runs with `SAD_SAM_LINKER_SOURCE=scratch`. The original `src/llm_sad_sam/linkers/experimental/s_linker19.py` and `prompts_v5.py` are never touched by Phase 46 — GATE-01 holds by construction, not by post-hoc revert.
  - Rationale: clean revert by `git checkout tests/scratch/` is local to the scratch dir; eliminates a class of "forgot to revert" failure modes; matches user's "duplicated standalone files over inheritance" preference at the dev-infra layer.

### Cut Ordering (D-02)

- **D-02 — Section-sequential AMB→DKX→DKJ→EXT→VAL→COR, risk-ascending within section.**
  - Outer loop walks the 6 D-08 sections in pipeline order (matches the audit doc reading order; matches phase-tag dependency graph).
  - Inner loop within each section sorts by risk tier: `low → low-med → med → med-high → high`. Ties broken by audit-doc row order.
  - Rationale: each section has its own gated test module — sequential section traversal keeps the harness run-set tight (no module-juggling); risk-ascending front-loads the cheap wins so MINIMIZE-LOG reads as "cuts we kept" first, "edge cases we rejected" later.

### Drop-Block Protocol (D-03)

- **D-03 — `drop → Family A → Family B`, smallest passing replacement wins (REQ-V264-06).**
  - For CUT-AMB-01 (drop AMBIGUITY_FEW_SHOT block) and CUT-DKJ-01 (drop DOC_KNOWLEDGE_JUDGE_EXAMPLES block): start by removing the entire block. If byte-equal on every gated snapshot, ship `drop`. The associated Family A / Family B rows on that block become moot — log them as `superseded` rows that point at the drop verdict.
  - If drop fails byte-equal, walk Family A rows in audit-row order (CUT-DKJ-02..04, the 3 synthetic-neutral swaps). Trial each independently against the pristine post-revert baseline (NOT cumulative). First Family A row that is byte-equal + GATE-06-clean wins. Remaining Family A rows on the same block log as `superseded-by-A`.
  - If all Family A rows fail, walk Family B rows similarly (CUT-DKJ-05..06 concept-only). First passing row wins; rest log as `superseded-by-B`.
  - If all of {drop, all A, all B} fail, log the block as `kept-original` (no cut applied) and proceed to the next audit row. The pristine baseline is the version with AMBIGUITY_FEW_SHOT / DOC_KNOWLEDGE_JUDGE_EXAMPLES unchanged.

### Commit Granularity (D-04)

- **D-04 — One atomic commit per cut decision.**
  - Each kept cut: commit = (scratch file change + MINIMIZE-LOG row + per-cut detail block). Message: `feat(46-NN): keep CUT-{TAG}-NN — {one-line summary}`.
  - Each reverted/unsafe cut: commit = (MINIMIZE-LOG row only — scratch is reverted before commit). Message: `chore(46-NN): revert CUT-{TAG}-NN — {failure reason}`.
  - Each tombstone (protected): commit = (MINIMIZE-LOG `protected` row only). Message: `docs(46-NN): protect CUT-{TAG}-NN — {protection rationale}`.
  - Each `superseded` row from D-03: folds into the parent drop/Family-A commit (no separate commit per superseded row).
  - Rationale: matches Phase 45's atomic-per-plan rhythm; lets any single cut be reverted via `git revert` without touching other cuts.

### MINIMIZE-LOG Schema (Claude's discretion within these guardrails)

The log artefact at `.planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md` carries per-cut rows. Required columns:

```
| cut_id | verdict | snapshot_delta | gate06_isolation | loc_saved | commit_sha | reasoning |
```

- `cut_id`: from audit (e.g., `CUT-AMB-01`)
- `verdict`: one of `kept` / `reverted` / `unsafe` / `protected` / `superseded-by-drop` / `superseded-by-A` / `superseded-by-B` / `kept-original`
- `snapshot_delta`: `0/N` (byte-equal) or `K/N` where K is the count of changed snapshots and N is the gated total per Phase 44 §D-03
- `gate06_isolation`: `clean` / `taboo:{section}:{term}` / `n/a` (for reverted/protected)
- `loc_saved`: integer LOC saved by this cut (negative if added, e.g. for a rewording longer than the original)
- `commit_sha`: short SHA of the per-cut commit
- `reasoning`: 1–2 sentence justification (especially for reverts: which snapshot diverged and how)

Per-cut detail blocks (longer rewordings, before/after diffs) live inline directly under the parent log row using the `> **{cut_id} detail:**` blockquote convention from the audit doc.

### Pareto Frontier Summary

The log opens with a `## Pareto Summary` section finalised at phase close:

- Total cuts trialled, total kept, total reverted, total unsafe, total protected
- LOC saved per section (AMB/DKX/DKJ/EXT/VAL/COR) and total
- "Smallest-passing" identifier per drop-block parent (CUT-AMB-01, CUT-DKJ-01)
- Cross-section batch summary for the recurring "software architecture …" opener pleonasm (CUT-AMB-02 + CUT-EXT-01 + CUT-VAL-02)

### Claude's Discretion

- Whether to commit the scratch copies themselves under `tests/scratch/` as the first plan's output (recommended: yes, so the scratch baseline is reproducible) or to generate them on demand each session (rejected: forfeits the byte-equal-of-baseline auditability).
- Whether the harness adapter override is implemented as an env var, a pytest fixture parametrization, or a `--linker-source` pytest CLI flag.
- Per-cut commit body content (e.g., whether to embed the full `git diff tests/scratch/` snippet in the commit message — recommended: yes for kept cuts, no for reverted to keep history readable).
- How to handle the recurring "software architecture …" opener pleonasm at the cross-section batch level — single combined commit vs three separate per-cut commits. Default: three separate commits per D-04, but the Pareto Summary cross-references them as one batch.
- Whether the Pareto Summary section is autogenerated by a small script (recommended: no, manual fill at the close-out plan is simpler and matches Phase 45's manual finalize step).
- Whether to add a "next-action" pointer in MINIMIZE-LOG referencing each kept row to the Phase 47 inline location it will populate.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 46 scope, requirements, gates

- `.planning/ROADMAP.md` §"Phase 46: MINIMIZE" — phase goal + 5 success criteria.
- `.planning/REQUIREMENTS.md` §"MINIMIZE" — REQ-V264-05 (per-cut Pareto loop + log), REQ-V264-06 (few-shot block-drop protocol), REQ-V264-07 (lexical neutralization).
- `.planning/PROJECT.md` §"Constraints" + §"Key Decisions" — GATE-01 byte-equal of s_linker13_min + s_linker19 + their imports; GATE-06 cross-dataset isolation methodology; gpt-5.4 only.
- `.planning/STATE.md` — current milestone v2.6.4, Phase 46 third of six.

### Phase 45 input (the audit deliverable Phase 46 consumes)

- `.planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md` — the 19 cut candidates with verdict + risk + gated_by columns. **Primary input.**
- `.planning/phases/45-audit/45-CONTEXT.md` §D-08 — cut row schema; D-06 Family A/B distinction; D-07 risk-tier and gated_by mapping (carried verbatim into MINIMIZE-LOG).
- `.planning/phases/45-audit/45-VERIFICATION.md` — confirms audit doc is complete (7/7 must-haves passed); GATE-01 record at audit close.
- `.planning/phases/45-audit/45-{02..07}-SUMMARY.md` — per-section audit summaries with detail on which cuts have detail blocks and per-cut GATE-06 grep-clearance evidence (especially DKJ Family A names).

### Phase 44 harness (the gate Phase 46 trials against)

- `.planning/phases/44-harness/44-CONTEXT.md` §D-03 — builder → phase-tag → test-module → snapshot-count mapping. The `gated_by` field on each MINIMIZE-LOG row maps through this.
- `tests/harness/adapters.py`, `tests/harness/loader.py`, `tests/harness/replay_client.py`, `tests/harness/manifest.py` — harness internals. Phase 46 adds the `SAD_SAM_LINKER_SOURCE` toggle here (or equivalent) per D-01.
- `tests/test_s_linker20_prompt_{ambiguity,doc_extract,doc_judge,extraction,validation,coref}.py` — the six golden-replay test modules. Phase 46 runs them with `SAD_SAM_LINKER_SOURCE=scratch` against the scratch copies.
- `tests/harness/MANIFEST.json` — pinned per-project fixture pairing. Unchanged in Phase 46.

### Frozen source artefacts (READ-ONLY in Phase 46)

- `src/llm_sad_sam/linkers/experimental/s_linker19.py` — the builders cuts target. NEVER edited by Phase 46.
- `src/llm_sad_sam/linkers/experimental/prompts_v5.py` — the constants cuts target. NEVER edited by Phase 46.
- `src/llm_sad_sam/linkers/experimental/s_linker13_min.py` — also GATE-01-protected, untouched.

### Standing gates

- `BENCHMARK_TABOO.md` — canonical taboo list for the GATE-06 re-isolation check on every `after`-text. Same methodology as Phase 45 D-02.
- `.planning/codebase/CONCERNS.md` — variant-proliferation hygiene preference. Phase 46 honors it: scratch copies in `tests/scratch/` are not new variants; they are throwaway trial fixtures.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- **Phase 44 harness** — `tests/harness/` package gives `(prompt_built, llm_response, parsed_output)` triples per s19 prompt site × project. 149 tests, 97 snapshots, exit 0 on baseline. Phase 46 reuses this verbatim; only the `adapters.py` source-selector needs an extension.
- **`BENCHMARK_TABOO.md`** — already curated per-dataset + Universal Taboo + Safe SE Textbook sections. Same mechanical grep used in Phase 45 D-02 is reused per kept cut for the GATE-06 re-isolation check.
- **Phase 45 audit doc** — `s_linker20-PROMPT-AUDIT.md` is the spec. Every Phase 46 log row references a `cut_id` from it; the audit's `gated_by` column maps directly to which test modules Phase 46 runs.
- **Phase 45 SUMMARY.md files** — `45-{02..07}-SUMMARY.md` already contain GATE-06 grep-clearance evidence for the DKJ Family A synthetic names (`BookManager`/`Mgr`/`MailSender`); Phase 46 can quote that evidence directly into the MINIMIZE-LOG rather than re-grepping.

### Established Patterns

- **Read-only on frozen artefacts.** Phase 43 (replay), Phase 44 (harness), Phase 45 (audit) all hold s19 / prompts_v5 / s13_min byte-equal. Phase 46 carries the pattern: cuts live in scratch space, not production.
- **Standalone, structured ledger artefacts.** Phase 44 → `MANIFEST.json`. Phase 45 → `s_linker20-PROMPT-AUDIT.md`. Phase 46 → `s_linker20-MINIMIZE-LOG.md`. Each `cut_id` is the link key across all three artefacts.
- **Per-builder isolation (no shared base classes).** Carries to scratch: `tests/scratch/s_linker19.py` is a flat copy, not an inheritance shim. When Phase 47 inlines into `s_linker20.py`, the kept-cut text comes from the scratch baseline.

### Integration Points

- **Phase 45 input.** The `cut_id` column on every audit row is the integration contract. Phase 46's MINIMIZE-LOG includes one row per `cut_id` (17 trial + 2 protected = 19 total).
- **Phase 47 ship.** The kept-cut text in `tests/scratch/{s_linker19.py, prompts_v5.py}` becomes the source-of-truth for Phase 47's inlined `s_linker20.py` constants and builders. Phase 47 reads the MINIMIZE-LOG to know which cuts to apply and the scratch files to get the after-text.
- **Phase 48 sweep.** GATE-06 re-verification on the shipped `s_linker20` reuses the per-kept-cut GATE-06 evidence captured in MINIMIZE-LOG — no fresh grep needed in Phase 48 unless Phase 47 emergent-edits the after-text.

</code_context>

<specifics>
## Specific Ideas

- **17 trial cuts, 2 protected tombstones.** Phase 45 produced 19 audit rows; the 2 tombstones (CUT-VAL-04 P1_FOCUS X.Y.Z; CUT-COR-05 coref conservatism) get explicit `protected` log entries but are NOT trialled — they exist for Phase 46 visibility and Phase 47 inline-decision auditing only.
- **DKJ is the only `benchmark-leak` row.** All 6 DKJ cuts (CUT-DKJ-01 drop + 3 Family A + 2 Family B) follow D-03's protocol. Per Phase 45 SUMMARY: Family A names `BookManager` / `Mgr` / `MailSender` are pre-cleared against all 5 dataset sections + Universal Taboo + Safe SE Textbook — Phase 46 can trust this grep and move directly to the harness check.
- **Cross-section pleonasm batch (CUT-AMB-02 + CUT-EXT-01 + CUT-VAL-02).** All three target the "software architecture …" opener pleonasm. They are sequenced per D-02 (one per section, low/low-med/low risk) but the Pareto Summary cross-references them as a single conceptual batch — the log notes their shared rationale.
- **Continuous GATE-01 verification.** After every per-cut commit, the Phase 46 protocol runs `git diff --stat src/llm_sad_sam/linkers/experimental/s_linker19.py src/llm_sad_sam/linkers/experimental/prompts_v5.py src/llm_sad_sam/linkers/experimental/s_linker13_min.py` — must be empty. Any non-empty result is a hard halt and triggers a `chore(46-NN): GATE-01 violation halt` commit before any further work.
- **Pareto Summary "smallest-passing" identifiers** for the two drop-block parents make it trivial for Phase 47 to look up "for AMBIGUITY_FEW_SHOT, ship X" without re-reading the per-cut log.

</specifics>

<deferred>
## Deferred Ideas

- **Cross-backend Claude validation of kept cuts.** Out of scope — gpt-5.4 only per v2.3 standing policy. Mirror minimize is a v2.6.5 candidate iff v2.6.4 promotes.
- **Auto-generating MINIMIZE-LOG from a script.** Considered, rejected (per Claude's Discretion above) — Phase 45 finalised manually and read cleanly; Phase 46 stays manual to avoid script drift vs the audit doc schema.
- **Emergent cuts not in the audit.** If reviewers spot a candidate during Phase 46 trial work, the protocol is to flag it `EMERGENT` with a rationale in the log but NOT trial it in this phase; emergents are deferred to a follow-on phase if reviewers escalate.
- **Per-prompt minimization extended to `s_linker17e`.** Future work per REQUIREMENTS.md "Future Requirements" — only if 17e remains the published champion.

</deferred>

---

*Phase: 46-MINIMIZE*
*Context gathered: 2026-06-08*
