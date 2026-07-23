# Phase 45: AUDIT — Pattern Map

**Mapped:** 2026-06-07
**Files analyzed:** 1 new planning artefact to be created
**Analogs found:** 4 / 4 (all role-match or exact-shape)

---

## File Classification

Phase 45 produces exactly one new file and edits zero source files.

| New / Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `s_linker20-PROMPT-AUDIT.md` (location: planner's discretion between `.planning/milestones/v2.6.4-*/` and `.planning/phases/45-audit/`) | planning ledger / audit doc | static (read-only artefact, no runtime data flow) | `tests/harness/fixtures/MANIFEST.json` (per-item pinned ledger, single source of truth) + `.planning/milestones/v2.6.3-MILESTONE-AUDIT.md` (milestone-scoped audit with per-phase structured tables) | role-match composite |

**Files referenced read-only (no pattern assignment needed — they are the audit subjects, not files being authored):**

| File | Role in Phase 45 |
|---|---|
| `src/llm_sad_sam/linkers/experimental/prompts_v5.py` | audit subject (read-only) |
| `src/llm_sad_sam/linkers/experimental/s_linker19.py` | audit subject (read-only) |
| `tests/test_s_linker20_prompt_ambiguity.py` | referenced in `gated_by` column only |
| `tests/test_s_linker20_prompt_doc_extract.py` | referenced in `gated_by` column only |
| `tests/test_s_linker20_prompt_doc_judge.py` | referenced in `gated_by` column only |
| `tests/test_s_linker20_prompt_extraction.py` | referenced in `gated_by` column only |
| `tests/test_s_linker20_prompt_validation.py` | referenced in `gated_by` column only |
| `tests/test_s_linker20_prompt_coref.py` | referenced in `gated_by` column only |
| `tests/harness/MANIFEST.json` | resolves `gated_by` fixture pairings |
| `BENCHMARK_TABOO.md` | consulted for D-02 mechanical grep |

---

## Pattern Assignments

### `s_linker20-PROMPT-AUDIT.md` (planning ledger, audit doc)

This file is a structured Markdown planning artefact, not code. The relevant patterns to copy come from four analog sources: the MANIFEST.json ledger schema, the milestone audit table structure, the per-phase D-XX decision tag convention, and the REQ-V264-XX row-ID style already in REQUIREMENTS.md.

---

#### Analog 1 — Per-item pinned ledger (closest structural analog)

**Source:** `tests/harness/fixtures/MANIFEST.json` (lines 1–32, full file)

```json
[
  {
    "project": "mediastore",
    "pkl_dir": "results/phase_cache/s_linker19/openai/mediastore/",
    "calls_json": "results/llm_logs/s_linker19_openai_mediastore_20260605_134622_calls.json",
    "description": "gpt-5.4 byte-equal baseline pinned for v2.6.4 Phase 44"
  },
  ...
]
```

**Why it matters for Phase 45:** MANIFEST.json is the Phase 44 ledger — one row per project, with keys that are the integration contract for downstream consumers. The audit doc follows the same philosophy: one row per cut candidate (`cut_id`), with keys (`file:lines`, `trigger`, `before`, `after`, `risk`, `gated_by`) that are the integration contract for Phase 46's `s_linker20-MINIMIZE-LOG.md`. The `cut_id` column is the direct analog of the `project` key — the primary foreign key that Phase 46 references back.

**Apply:** The audit's per-section cut table is the Markdown analog of this JSON ledger. Every row must have a unique `cut_id` and enough information for a downstream agent to execute the cut without re-reading this doc.

---

#### Analog 2 — Structured per-phase table with tagged rows

**Source:** `.planning/milestones/v2.6.3-MILESTONE-AUDIT.md` lines 19–38

```markdown
## Scope

| Phase | Name | Plans | Verification | Code Review |
|-------|------|-------|--------------|-------------|
| 43 | Replay s_linker19 checkpoints for paper RQ1–RQ4 eval | 5/5 | `passed` (11/11 COVERED, + 3 gap-closure commits) | 0 Critical, 4 Warning, 7 Info — 9 actionable findings fixed |

## Requirements Coverage (3-Source Cross-Reference)

| Requirement | Source plans | Status | Evidence |
|-------------|--------------|--------|----------|
| REQ-V263-01 | 43-02 | **satisfied** | `scripts/v2.6.3/replay_s19_{to_csv,rq3,rq4}.py` produce 60 CSVs … |
| REQ-V263-02 | 43-03 | **satisfied** | `writing/working/tables/metrics_sad-{sam,code}.tex` … |
```

**Why it matters for Phase 45:** This is the clearest prior example of the project's structured-table-in-Markdown style: tagged row IDs (`REQ-V263-01`), fixed columns, short-but-complete evidence cells. The `CUT-AMB-01` tag style in D-08 directly mirrors `REQ-V263-01` — prefix-tag + zero-padded counter — and Phase 46's `s_linker20-MINIMIZE-LOG.md` will reference cut IDs the same way Phase 46 references REQ IDs.

**Apply to audit doc:** Use the same `{PREFIX}-{NN}` tag style. Section tags `AMB`, `DKX`, `DKJ`, `EXT`, `VAL`, `COR` mirror the project's existing `REQ-V264-XX` and `D-XX` conventions. The header item table (verdict + LOC per constant) mirrors the Requirements Coverage table layout — fixed columns, one row per item, terse evidence cell.

---

#### Analog 3 — Per-section header table (verdict + LOC)

**Source:** D-08 in `.planning/phases/45-audit/45-CONTEXT.md` lines 83–98 (the schema is locked by the user decision, not derived from codebase — the planner must reproduce it exactly)

```markdown
| Item | Verdict | LOC |
|---|---|---|
| AMBIGUITY_FEW_SHOT | benchmark-leak | 7 |
| AMBIGUITY_RULES    | clean          | 1 |
| _prompt_ambiguity  | clean          | 18 |
```

Cut table immediately following each header:

```markdown
| cut_id | file:lines | trigger | before | after | risk | gated_by |
|---|---|---|---|---|---|---|
| CUT-AMB-01 | prompts_v5.py:30–36 | benchmark-leak (drop-block, REQ-V264-06) | AMBIGUITY_FEW_SHOT entire block | "" | high | tests/test_s_linker20_prompt_ambiguity.py @ phase_1_model |
| CUT-AMB-02 | prompts_v5.py:30–36 | benchmark-leak (Family A: synthetic-neutral swap) | "Scheduler" examples | "OrderProcessor" examples (full rewrite) | med | tests/test_s_linker20_prompt_ambiguity.py @ phase_1_model |
| CUT-AMB-03 | prompts_v5.py:30–36 | benchmark-leak (Family B: concept-only) | "Scheduler" examples | abstract rule restatement | med | tests/test_s_linker20_prompt_ambiguity.py @ phase_1_model |
```

**Apply:** Every section (AMB / DKX / DKJ / EXT / VAL / COR) must open with a header item table, then a cut table. Items with no cuts (e.g., `AMBIGUITY_RULES` — expected `clean`, 0 candidates) still appear in the header item table; they simply have no rows in the cut table.

---

#### Analog 4 — Builder → phase-tag mapping (locked D-03, verbatim reuse)

**Source:** `.planning/phases/44-harness/44-CONTEXT.md` lines 52–62 (D-03)

```markdown
| Builder | Phase tag(s) in `llm_logs` |
|---|---|
| `_prompt_ambiguity` | `phase_1_model` |
| `_prompt_doc_knowledge_extract` | `phase_1_doc_extract` |
| `_prompt_doc_knowledge_judge` | `phase_1_doc_judge` |
| `_prompt_extraction` | `phase_2_framing_c_pass1`, `phase_2_framing_c_pass2` |
| `_prompt_validation` | `phase_4_twopass_p1`, `phase_4_twopass_p2`, `phase_5_coref_validation` |
| `_prompt_coref` | `phase_5_coref` |
```

**Apply to `gated_by` column:** Copy this table verbatim into the audit doc's preamble (or per-section header) so Phase 46 has the mapping inline. The `gated_by` cell format is `tests/test_s_linker20_prompt_{module}.py @ {phase_tag}`. For validation cuts, list all three phase tags comma-separated.

---

## Shared Patterns

### Row-ID tag convention (`{TAG}-{NN}`)
**Source:** `.planning/REQUIREMENTS.md` lines 11–38 (`REQ-V264-XX`) and `.planning/phases/45-audit/45-CONTEXT.md` lines 36–116 (`D-01..D-08`)
**Apply to:** every `cut_id` in the audit doc. Tags: `CUT-AMB-NN`, `CUT-DKX-NN`, `CUT-DKJ-NN`, `CUT-EXT-NN`, `CUT-VAL-NN`, `CUT-COR-NN`. Zero-pad to 2 digits. Numbering restarts at `01` per section. Phase 46 references these IDs as foreign keys; any gap or duplicate breaks the integration contract.

### Read-only / zero-code-change discipline
**Source:** Phase 44 and Phase 43 established pattern (`.planning/phases/44-harness/44-CONTEXT.md` line 9; `.planning/phases/44-harness/44-PATTERNS.md` §"GATE-01 byte-equality preservation")
**Apply to:** the audit doc itself — it contains zero executable content and makes zero edits to `prompts_v5.py`, `s_linker19.py`, or any test. The `before` / `after` cells in the cut table are proposed text, not applied edits. GATE-01 holds at phase close.

### Standalone, self-contained artefact
**Source:** `.planning/codebase/CONCERNS.md` preference (via 44-CONTEXT.md §"Established Patterns") — standalone files over inheritance/shared layers.
**Apply to:** the audit doc must be readable as a standalone document. It should not require the reader to have CONTEXT.md open to understand what `benchmark-leak` means — a 2-sentence rubric recap at the top suffices. All `gated_by` paths must be repo-root-relative and not rely on reader memory of Phase 44.

### Truncation rule for `before` / `after` cells
**Source:** D-08 in 45-CONTEXT.md line 103 — "truncate with `…` over 80 chars per cell — the full proposed rewrite goes in a per-cut detail block below the table when the rewrite is long."
**Apply to:** every cut table row. Short single-token swaps fit inline. Multi-line Family B rewrites get a `> **CUT-AMB-03 detail:**` block immediately below the table (or in an appendix — planner's discretion per D-08 footnote).

---

## No Analog Found

None. All structural patterns needed for the audit doc have direct analogs in the planning tree or codebase.

---

## Metadata

**Analog search scope:** `tests/harness/fixtures/MANIFEST.json`, `.planning/milestones/v2.6.3-MILESTONE-AUDIT.md`, `.planning/milestones/v2.6.2-MILESTONE-AUDIT.md`, `.planning/REQUIREMENTS.md`, `.planning/phases/44-harness/44-CONTEXT.md`, `.planning/phases/44-harness/44-PATTERNS.md`, `.planning/phases/45-audit/45-CONTEXT.md`, `.planning/phases/45-audit/45-RESEARCH.md`.
**Files scanned (Read):** 8 files, 4 ranges.
**Pattern extraction date:** 2026-06-07.
