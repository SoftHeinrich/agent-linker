---
phase: 46-minimize
plan: 01
subsystem: minimize-harness-bootstrap
tags:
  - bootstrap
  - tests/scratch
  - harness-toggle
  - SAD_SAM_LINKER_SOURCE
  - MINIMIZE-LOG
dependency_graph:
  requires:
    - tests/harness/ (Phase 44)
    - src/llm_sad_sam/linkers/experimental/s_linker19.py (READ-ONLY)
    - src/llm_sad_sam/linkers/experimental/prompts_v5.py (READ-ONLY)
    - .planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md (audit input)
  provides:
    - tests/scratch/{__init__.py, s_linker19.py, prompts_v5.py} (mutable scratch surface)
    - SAD_SAM_LINKER_SOURCE env-var toggle in tests/harness/adapters.py
    - ACCEPTED_PREFIXES tuple in tests/harness/inputs.py (pre-mitigates CUT-VAL-02 harness break)
    - step-6 prompt-equality gate in all 6 tests/test_s_linker20_prompt_*.py modules
    - .planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md skeleton (6 section anchors + 3 FINAL anchors + 2 protected tombstone rows)
  affects:
    - Wave-2 plans 46-02..07 (each appends rows under its section anchor)
    - Wave-3 plan 46-08 (fills FINAL:PARETO / FINAL:GATE01 / FINAL:REQ anchors)
tech-stack:
  added: []
  patterns:
    - env-var-gated module-load import swap (production | scratch)
    - opener-tuple acceptance (forward-compatible with a single planned vocabulary cut)
    - gated test-assertion pattern (production-strict, scratch-lenient)
key-files:
  created:
    - tests/scratch/__init__.py
    - tests/scratch/s_linker19.py
    - tests/scratch/prompts_v5.py
    - .planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md
    - .planning/phases/46-minimize/46-01-SUMMARY.md
  modified:
    - tests/harness/adapters.py
    - tests/harness/inputs.py
    - tests/test_s_linker20_prompt_ambiguity.py
    - tests/test_s_linker20_prompt_doc_extract.py
    - tests/test_s_linker20_prompt_doc_judge.py
    - tests/test_s_linker20_prompt_extraction.py
    - tests/test_s_linker20_prompt_validation.py
    - tests/test_s_linker20_prompt_coref.py
decisions:
  - "Scratch s_linker19 carries exactly one wiring edit (prompts_v5 import rewrite); annotation header documents it as NOT a cut row."
  - "Harness toggle implemented as module-load env-var read (D-01 Claude's-Discretion: env var, not pytest CLI flag or fixture parametrization) — simpler, no pytest plugin surface."
  - "ACCEPTED_PREFIXES tuple holds exactly two entries — the production prefix and the pre-decided CUT-VAL-02 replacement `Validate components in a document.` — per pleonasm-batch decision baked into MINIMIZE-LOG header."
  - "Step-6 gate uses `os.environ.get(...) != 'scratch'` (Option A from 46-RESEARCH §2.3) so any non-scratch value (including the default `production`) preserves the strict assertion."
  - "MINIMIZE-LOG skeleton lays down ALL anchors in one go (6 SECTION + 3 FINAL) so Wave-2 plans never re-edit the header — eliminates a coordination bug."
metrics:
  duration: "~25 min"
  completed: 2026-06-08
---

# Phase 46 Plan 01: Scratch bootstrap + harness toggle Summary

Bootstrap of Phase 46 infrastructure: created `tests/scratch/` mutable surface with byte-equal copies of `s_linker19.py` + `prompts_v5.py`, extended the Phase 44 harness with a `SAD_SAM_LINKER_SOURCE` env-var toggle, pre-mitigated the CUT-VAL-02 opener change via an `ACCEPTED_PREFIXES` tuple in `reconstruct_validation_inputs`, gated the step-6 prompt-equality assertion in all six s_linker20 prompt test modules behind that toggle, and laid down the `s_linker20-MINIMIZE-LOG.md` skeleton with section anchors, batch-decision notes, and two pre-filled protected tombstone rows.

## What was delivered

### 1. Scratch surface (tests/scratch/)

| File | Production SHA-256 | Scratch SHA-256 | Delta |
|------|--------------------|-----------------|-------|
| `prompts_v5.py` | `2f8b9968fd35e6a9c9e5e01bc16c8081b2bd80eb0efa4ab669f16975f8440689` | `2f8b9968fd35e6a9c9e5e01bc16c8081b2bd80eb0efa4ab669f16975f8440689` | byte-equal (0 lines diff) |
| `s_linker19.py` | `05c413d0f7fa38f46359c22a2207a6b05f82e50019388550f18f426eb6c9996d` | `ddc80770db7c1d50bf9488dc81019f5d1e7eaff57003e6d92c710d7912628578` | 1 wiring rewrite + 8-line annotation header (NOT a cut row) |

Wiring rewrite in scratch `s_linker19.py`:

```diff
-from llm_sad_sam.linkers.experimental.prompts_v5 import (
+from tests.scratch.prompts_v5 import (
```

This is necessary so cuts to `tests/scratch/prompts_v5.py` take effect when the harness imports `SLinker19` from `tests.scratch`. The 8-line annotation header at the top of the scratch s_linker19 documents this delta and explicitly states "This is NOT a cut row."

### 2. Adapter env-var schema (tests/harness/adapters.py)

```python
_SOURCE = os.environ.get("SAD_SAM_LINKER_SOURCE", "production")

if _SOURCE == "scratch":
    from tests.scratch.s_linker19 import SLinker19
elif _SOURCE == "production":
    from llm_sad_sam.linkers.experimental.s_linker19 import SLinker19
else:
    raise RuntimeError(
        f"SAD_SAM_LINKER_SOURCE must be 'production' or 'scratch', got: {_SOURCE!r}"
    )
```

Default `production` preserves bare-env CI. Any other value raises a clear `RuntimeError` at import time (eliminates silent fallback). `BUILDER_PHASE_TAGS`, `BUILDERS`, and the cross-key sanity-guard assertion are preserved byte-equal.

### 3. ACCEPTED_PREFIXES tuple (tests/harness/inputs.py)

```python
ACCEPTED_PREFIXES = (
    "Validate component references in a software architecture document.",
    "Validate components in a document.",  # CUT-VAL-02 replacement (per 46-01 batch decision)
)
```

A `for fixed_prefix in ACCEPTED_PREFIXES: if first_line.startswith(fixed_prefix): matched = ...` loop replaces the prior single-prefix `startswith` check. Control flow is unchanged when production prompts arrive (first entry matches); pre-mitigates CUT-VAL-02 (plan 46-06) so cuts to the scratch validation builder don't crash the harness on the very first record.

### 4. Step-6 gate annotations (all 6 test modules)

Each of `tests/test_s_linker20_prompt_{ambiguity,doc_extract,doc_judge,extraction,validation,coref}.py` now:

1. Imports `os` at module top.
2. Wraps the `assert rebuilt_prompt == record["prompt"]` block in `if os.environ.get("SAD_SAM_LINKER_SOURCE", "production") != "scratch":` so production-mode runs preserve strict prompt-equality (no CI regression) while scratch-mode runs skip the assertion.

The `doc_extract` module's pre-existing `prompt-version-drift` warning path (UserWarning for teastore/teammates/bigbluebutton) is wrapped inside the same gate — scratch mode skips both the warning and the assertion (the prompt-version-drift signal is irrelevant when the scratch builders may have diverged intentionally). For `_prompt_extraction` and `_prompt_validation`, the gate is at the lexical assertion location, inside the per-record test body (not at the parametrize loop), preserving the parametrization grid byte-equal.

The parsed-output snapshot assertion (`assert parsed == snapshot`) is preserved in all six modules and remains the meaningful Phase 46 signal.

### 5. MINIMIZE-LOG.md skeleton

`.planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md` was created with:

- YAML front-matter (`phase: 46-minimize`, `milestone: v2.6.4`, etc.).
- Two-paragraph schema header naming the verdict vocabulary and the scratch-mode harness semantics.
- Pleonasm-batch decision note: replacement vocabulary `components` (bare); CUT-VAL-02 full opener `Validate components in a document. {focus}`.
- VAL-03 ↔ COR-01 shared-lexicon note: target `noun phrase that refers back` (subject to 46-06 empirical trial).
- CUT-DKJ-07 vocabulary note: target `grouping that encompasses multiple elements`.
- Scratch-mode bootstrap note clarifying the import rewrite is NOT a cut row.
- Verdict vocabulary table (8 verdicts).
- 9 placeholder anchors:
    - `<!-- FINAL:PARETO:START -->` / `:END -->`
    - `<!-- SECTION:AMB:START -->` / `:END -->`
    - `<!-- SECTION:DKX:START -->` / `:END -->`
    - `<!-- SECTION:DKJ:START -->` / `:END -->`
    - `<!-- SECTION:EXT:START -->` / `:END -->`
    - `<!-- SECTION:VAL:START -->` / `:END -->`
    - `<!-- SECTION:COR:START -->` / `:END -->`
    - `<!-- FINAL:GATE01:START -->` / `:END -->`
    - `<!-- FINAL:REQ:START -->` / `:END -->`
- Two pre-filled protected tombstone rows: CUT-VAL-04 (P1_FOCUS X.Y.Z qualified-name clause) and CUT-COR-05 (coref conservatism instruction at s19:361). Commit SHAs deferred to 46-06 and 46-07 respectively.

## Verification evidence

| Check | Result |
|-------|--------|
| `tests/scratch/__init__.py`, `s_linker19.py`, `prompts_v5.py` exist | yes |
| `tests/scratch/prompts_v5.py` SHA-256 equals production | yes (`2f8b9968...`) |
| Scratch s19 has the prompts_v5 import rewritten to `tests.scratch.prompts_v5` | yes (1 occurrence at line 107) |
| Scratch s19 has zero `from llm_sad_sam.linkers.experimental.prompts_v5 import` left | yes (0 occurrences) |
| `SAD_SAM_LINKER_SOURCE` referenced in `tests/harness/adapters.py` | yes |
| `ACCEPTED_PREFIXES` referenced in `tests/harness/inputs.py` | yes |
| `Validate components in a document.` present in `tests/harness/inputs.py` | yes |
| All 6 test modules contain `SAD_SAM_LINKER_SOURCE` gate | yes (grep-verified) |
| MINIMIZE-LOG has `<!-- SECTION:AMB:START -->`, CUT-VAL-04, CUT-COR-05 | yes |
| Production-mode pytest (default env): 6 modules, 97 snapshots | **97 passed, 3 warnings, 0.29s** |
| Scratch-mode pytest (`SAD_SAM_LINKER_SOURCE=scratch`): 6 modules, 97 snapshots | **97 passed, 0 warnings, 0.24s** |
| GATE-01 byte-equal: `git diff --stat` on `s_linker19.py` + `prompts_v5.py` + `s_linker13_min.py` | empty (exit 0) |

## Deviations from Plan

None — plan executed exactly as written.

The plan also instructed an optional fallback: if `gsd-tools commit` was unavailable, use `git add` + `git commit -m ...`. The executor used the SDK path successfully; no fallback was needed.

## Notes for downstream Wave-2 plans (46-02..07)

- Append cut rows BETWEEN the `<!-- SECTION:{TAG}:START -->` / `:END -->` anchors using the schema `| cut_id | verdict | snapshot_delta | gate06_isolation | loc_saved | commit_sha | reasoning |`.
- Run pytest in scratch mode (`SAD_SAM_LINKER_SOURCE=scratch`) after each cut to verify the 97 snapshots still pass.
- After each per-cut commit, run `git diff --stat src/llm_sad_sam/linkers/experimental/{s_linker19.py,prompts_v5.py,s_linker13_min.py}` to re-verify GATE-01.
- CUT-AMB-02, CUT-EXT-01, CUT-VAL-02 all use the pre-decided replacement vocabulary `components` (bare) per the MINIMIZE-LOG batch note.
- CUT-VAL-03 trial (in 46-06) writes its chosen vocabulary into its MINIMIZE-LOG row reasoning cell; 46-07 reads that cell for CUT-COR-01.

## Self-Check: PENDING

Self-check verification (file existence + commit hash) recorded below after the atomic commit.
