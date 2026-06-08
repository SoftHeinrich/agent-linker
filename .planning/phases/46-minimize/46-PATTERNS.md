# Phase 46: MINIMIZE — Pattern Map

**Mapped:** 2026-06-08
**Files analyzed:** scratch dir + harness adapter extensions + 1 new ledger artefact
**Analogs found:** 5 / 5 (all role-match or exact-shape, drawn from Phase 44 + Phase 45)

---

## File Classification

| New / Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `tests/scratch/__init__.py` | Python package marker | static | `tests/harness/__init__.py` | exact-shape |
| `tests/scratch/s_linker19.py` | byte-for-byte working copy of frozen source w/ one wiring edit (import-line rewrite) | mutated per cut, read by harness when `SAD_SAM_LINKER_SOURCE=scratch` | none direct — novel scratch pattern; conceptually parallels Phase 44's harness fixtures-as-pinned-ledger philosophy | role-match (novel) |
| `tests/scratch/prompts_v5.py` | byte-for-byte working copy | same as scratch s19 | same as above | role-match (novel) |
| `tests/harness/adapters.py` (extension) | adds env-var-aware `SLinker19` import selector | branch at module-load on `SAD_SAM_LINKER_SOURCE` | Phase 44 D-03 builder→phase-tag map (already in this file) | extension of existing structure |
| `tests/harness/inputs.py` (extension) | adds `ACCEPTED_PREFIXES` tuple in `reconstruct_validation_inputs` for CUT-VAL-02 opener flexibility | per-prefix fall-through; raises if none match | existing single-prefix `fixed_prefix` check in the same function | direct widening of existing pattern |
| `tests/test_s_linker20_prompt_*.py` (6 modules — extension) | adds env-var gate around the step-6 prompt-equality assertion | branch at test runtime on `SAD_SAM_LINKER_SOURCE` | Phase 44's two-stage assertion shape (step-6 + step-7) — Phase 46 toggles step-6 conditionally | extension of existing shape |
| `.planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md` | planning ledger (one row per cut decision) | static read-only artefact for Phase 47 to consume | `.planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md` (same shape, different verb tense — audit = candidates, minimize = decisions) | exact-shape |

**Files referenced read-only (no pattern assignment — they are inputs):**

| File | Role in Phase 46 |
|---|---|
| `src/llm_sad_sam/linkers/experimental/s_linker19.py` | frozen source — copied to scratch at phase open; NEVER edited (GATE-01) |
| `src/llm_sad_sam/linkers/experimental/prompts_v5.py` | frozen source — copied to scratch at phase open; NEVER edited (GATE-01) |
| `src/llm_sad_sam/linkers/experimental/s_linker13_min.py` | also GATE-01-protected, untouched |
| `tests/harness/loader.py`, `replay_client.py`, `manifest.py`, `fixtures/MANIFEST.json` | Phase 44 harness internals; consumed via standard import; unchanged in Phase 46 |
| `.planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md` | the 19 cut candidates; primary input |
| `.planning/phases/45-audit/45-{02..07}-SUMMARY.md` | per-section audit summaries; especially 45-04 for DKJ Family A name grep-clearance |
| `BENCHMARK_TABOO.md` | GATE-06 re-isolation source per kept cut |

---

## Pattern Assignments

### 1 — Harness Adapter Override (extends Phase 44 D-03 builder map)

**Source:** `tests/harness/adapters.py` (Phase 44) lines 28-66 — already does a static `from llm_sad_sam.linkers.experimental.s_linker19 import SLinker19` and exports `BUILDERS = {…}` as module-level callables.

**Phase 46 extension:** wrap the import in an env-var branch:

```python
_SOURCE = os.environ.get("SAD_SAM_LINKER_SOURCE", "production")
if _SOURCE == "scratch":
    from tests.scratch.s_linker19 import SLinker19
elif _SOURCE == "production":
    from llm_sad_sam.linkers.experimental.s_linker19 import SLinker19
else:
    raise RuntimeError(...)
```

The downstream `BUILDERS = {"_prompt_ambiguity": SLinker19._prompt_ambiguity, ...}` map stays unchanged — Phase 46 ONLY changes WHICH SLinker19 is bound, never how the builders are exposed. This is the smallest possible delta that makes scratch trials work without touching Phase 44's harness internals.

**Apply to:** Phase 46 plan 46-01 step 3. Pattern is structural — every new toggle in `adapters.py` should be expressible as "branch at module-load on an env var; default is the production path; invalid value raises clearly."

---

### 2 — Cut-Row Schema (extends Phase 45 D-08 audit row schema)

**Source:** `.planning/phases/45-audit/45-CONTEXT.md` D-08 — the audit row schema:

```
| cut_id | file:lines | trigger | before | after | risk | gated_by |
```

**Phase 46 mirror:** the MINIMIZE-LOG row schema:

```
| cut_id | verdict | snapshot_delta | gate06_isolation | loc_saved | commit_sha | reasoning |
```

The `cut_id` column is the foreign key linking the two artefacts. The audit's `trigger`/`before`/`after`/`risk` columns are replaced in the LOG by post-trial result columns. The `gated_by` mapping is inherited transitively — every Phase 46 row's `commit_sha` ties back through the row's verdict reasoning to the audit's `gated_by` test module.

**Apply to:** every per-cut row in `s_linker20-MINIMIZE-LOG.md`. Verdict vocabulary (from 46-RESEARCH §7.5) is the controlled enum for the `verdict` cell: `kept` / `reverted` / `unsafe` / `protected` / `superseded-by-drop` / `superseded-by-A` / `superseded-by-B` / `kept-original`.

---

### 3 — Section-Anchor Convention (exact-shape from Phase 45)

**Source:** `.planning/phases/45-audit/45-01-PLAN.md` step 9 — each section of the audit doc opens with paired HTML-comment anchors:

```
<!-- SECTION:AMB:START -->
<!-- TBD: filled by ... -->
<!-- SECTION:AMB:END -->
```

**Phase 46 carries the convention forward.** `s_linker20-MINIMIZE-LOG.md` opens with 6 `<!-- SECTION:{TAG}:START/END -->` pairs (AMB / DKX / DKJ / EXT / VAL / COR) in pipeline order. Wave-2 plans (46-02..07) each write between their section's pair. Wave-3 (46-08) populates three `<!-- FINAL:{name}:START/END -->` anchors (PARETO / REQ / GATE01) at the doc end.

The benefit is the same as in Phase 45: per-section plans run in parallel without write conflicts because their target anchors are disjoint.

**Apply to:** every MINIMIZE-LOG section + finalize block. The anchor names mirror the audit doc 1:1 so Phase 47 can grep one document or the other for the same `CUT-AMB-01` row without confusion.

---

### 4 — Ledger-Artefact Pattern (composite of MANIFEST.json + audit doc)

**Source 1:** `tests/harness/fixtures/MANIFEST.json` (Phase 44) — per-project pinned ledger; one row per project; `project` is the foreign key downstream consumers reference.

**Source 2:** `.planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md` (Phase 45) — per-cut audit; one row per cut candidate; `cut_id` is the foreign key Phase 46 references.

**Phase 46 ledger:** `s_linker20-MINIMIZE-LOG.md` — per-cut decision; one row per cut DECISION; `cut_id` is the foreign key Phase 47 references when inlining into `s_linker20.py`.

The pattern is: **standalone, structured, key-foreign-keyed Markdown ledger committed to git**. Three phases now use it (Phase 44 in JSON, Phase 45 + Phase 46 in Markdown). Each ledger has:
- a schema header at the top (column meanings + verdict vocab),
- per-row content in a Markdown table (or per-row blockquotes for long detail),
- a finalize block at the bottom (Pareto Summary + GATE-01 record),
- one atomic git commit per row decision (Phase 46 D-04 specifically; the audit doc and MANIFEST follow looser commit shapes but the standalone-artefact property holds).

**Apply to:** the MINIMIZE-LOG header (46-01 lays it down), the per-cut rows (46-02..07 append), the finalize anchors (46-08 populates).

---

### 5 — Drop-by-Empty Constant Preservation

**Source:** 46-RESEARCH §9 Q9 — when CUT-AMB-01 or CUT-DKJ-01 drops a constant block, the scratch `s_linker19.py` still imports the constant name. The two options are: (a) keep the constant assigned to `""` (drop-by-empty), preserving the import; (b) also delete the import line in scratch s_linker19. **Phase 46 standardises on (a).**

This is a NEW pattern (no direct prior analog). It exists because Phase 46 needs to drop constant BODIES without breaking the existing import graph in the same scratch trial. The pattern lets every CUT-{TAG}-NN-style "drop" be applied with a single Edit on the constant's RHS — no scratch s_linker19 edits, no scratch import-list rewrites.

**Apply to:** plan 46-02 task 2 (CUT-AMB-01 drop), plan 46-04 task 1 step DROP (CUT-DKJ-01 drop). Both tasks Edit the constant's string literal RHS to `""` while preserving the assignment LHS.

**Phase 47 follow-through:** when inlining into `s_linker20.py`, Phase 47 may collapse the empty constant interpolation entirely — `s_linker20.py` does NOT need to preserve the constant binding (it is a standalone file with no historical import graph to preserve). This is a Phase 47 simplification, NOT a Phase 46 concern.

---

## Shared Patterns

### Row-ID tag convention (`{TAG}-{NN}`)

**Source:** Phase 45 D-08 + Phase 46 CONTEXT D-08 schema (`CUT-AMB-NN` ... `CUT-COR-NN`). Zero-pad to 2 digits, restart at `01` per section.

**Apply to:** every row in the MINIMIZE-LOG. Phase 47 references these IDs as foreign keys.

### Read-only / GATE-01 discipline

**Source:** Phase 43 / 44 / 45 all hold `s_linker19.py` + `prompts_v5.py` + `s_linker13_min.py` byte-equal throughout.

**Apply to:** Phase 46 cuts the SCRATCH copies, never the production sources. The scratch surface is the structural mitigation; the per-commit `git diff --stat` is the operational verification. Phase 46 inherits the discipline; no new mechanism needed beyond the scratch directory + adapter override.

### Cross-section batching

**Source:** Phase 45 audit closing notes for the cross-section pleonasm batch (CUT-AMB-02 + CUT-EXT-01 + CUT-VAL-02 share replacement vocabulary) and the VAL-03 ↔ COR-01 lockstep.

**Apply to:** Phase 46 commits each cut separately per D-04, but the Pareto Summary (46-08) cross-references them as one conceptual unit. Plan 46-01 pre-decides the batch vocabulary (`components` bare) and records it in the LOG header so all three section plans target the same string. Plan 46-06's CUT-VAL-03 row records its chosen vocabulary so 46-07's CUT-COR-01 can read it.

### Atomic per-cut commit

**Source:** Phase 46 D-04 explicit decision; Phase 45 atomic-per-plan rhythm.

**Apply to:** every cut decision produces one commit. Superseded rows fold into their parent commit (D-04 superseded-rule). Tombstones get a dedicated `docs(46-NN): protect …` commit. The batched CUT-COR-03 + CUT-COR-04 trial produces ONE commit because the audit mandated lockstep (audit-doc line 348).

### Behavioral-vs-harness caveat

**Source:** 46-RESEARCH §6.3 + §4.4 — Phase 46 can verify harness compatibility (cached replays); it cannot verify behavioral safety (parsed-output snapshots are invariant under prompt cuts). Phase 48 sweep validates behavior.

**Apply to:** every `kept` verdict on a `high` or `med-high` risk audit row should carry an explicit caveat in its reasoning cell. The CUT-DKJ-01 drop and CUT-AMB-01 drop carry the caveat by default per CONTEXT specifics + 46-RESEARCH §4.4.

---

## No Analog Found

None. The scratch-dir + harness-override + drop-by-empty triple is the most novel pattern in Phase 46, but each piece extends an existing structural pattern (Phase 44's harness adapter, Phase 45's audit row schema, the read-only GATE-01 discipline from Phase 43 onwards).

---

## Metadata

**Analog search scope:** `tests/harness/` (full package), `.planning/phases/44-harness/44-CONTEXT.md`, `.planning/phases/44-harness/44-PATTERNS.md`, `.planning/phases/45-audit/45-CONTEXT.md`, `.planning/phases/45-audit/45-PATTERNS.md`, `.planning/phases/45-audit/45-{01,02,04,08}-PLAN.md`, `.planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md`, `.planning/phases/46-minimize/46-CONTEXT.md`, `.planning/phases/46-minimize/46-RESEARCH.md`.

**Files scanned (Read):** 9 files, ~6 ranges.

**Pattern extraction date:** 2026-06-08.
</content>
