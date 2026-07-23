---
phase: 47-ship
artifact_type: research
milestone: v2.6.4
produced_by: gsd-plan-phase --research-phase 47
consumed_by: Phase 47 planner
---

# Phase 47: SHIP — Research

**Researched:** 2026-06-09
**Domain:** Python linker implementation — standalone file construction, runner registration, GATE-01 byte-equal verification
**Confidence:** HIGH

---

## Summary

Phase 47 creates `src/llm_sad_sam/linkers/experimental/s_linker20.py` as a self-contained standalone
file. The file is constructed by taking the production `s_linker19.py` (byte-equal frozen source),
replacing its import of external prompt constants with the minimized constant values inlined directly,
renaming the class to `SLinker20`, updating `_VARIANT_NAME`, and removing the `from
...prompts_v5 import (...)` block entirely. The minimized constant values come from
`tests/scratch/prompts_v5.py` and `tests/scratch/s_linker19.py` at Phase 46 close (frozen).

The runner `run_ablation.py` receives two additions: one entry appended to `CANONICAL_VARIANTS`
(line ~118, after `"s_linker18"`) and one `dict(...)` entry appended to `VARIANT_SPECS` (line ~749,
before the closing `}`). Both mirror the s_linker18 pattern exactly.

GATE-01 (byte-equal freeze on `s_linker19.py`, `s_linker13_min.py`, and `prompts_v5.py`) must hold
throughout and after this phase. The SHA-256 values are known, verifiable in one command, and the
Phase 46 scratch-mode protocol ensures Phase 47 cannot accidentally mutate them.

**Primary recommendation:** Copy the production `s_linker19.py` verbatim into `s_linker20.py`,
inline the 11 prompt-constant values from `tests/scratch/prompts_v5.py` and the 2 builder-level
string changes from `tests/scratch/s_linker19.py`, rename class/`_VARIANT_NAME`, then register
the variant. Touch nothing else. Total edits: 1 new file + 2 additions to `run_ablation.py`.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

None — all implementation choices are at Claude's discretion.

### Claude's Discretion

All implementation choices — pure infrastructure phase. Use the ROADMAP phase goal, success
criteria, the Phase 46 MINIMIZE-LOG, and the frozen scratch artifacts as the authoritative inputs.

### Deferred Ideas (OUT OF SCOPE)

Phase 48 SWEEP (the behavioral validation that spends LLM budget) is the next phase and is
intentionally gated behind explicit user go-ahead.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REQ-V264-08 | `s_linker20.py`: standalone (no inheritance from s19), `experimental=True`, `canonical=False`, minimized prompt constants inlined; `s_linker19.py` and imported constants preserved byte-equal; `run_ablation.py` gains `--variants s_linker20` | Inlining source mapped exactly (Q1). Standalone construction verified (Q2). Runner edit sites identified with line numbers and verbatim patterns (Q3). |
| GATE-01 | `s_linker13_min.py` AND `s_linker19.py` SHA-256 byte-equal at milestone close | Verification command documented (Q4). Current SHA-256 values confirmed live. |
</phase_requirements>

---

## Project Constraints (from CLAUDE.md)

- Active runtime files listed in CLAUDE.md include `s_linker20.py` as the planned new file. The
  retained-file list must be updated when s_linker20.py is created (CLAUDE.md is NOT auto-updated;
  the planner should include a task to add `s_linker20.py` to the "Active Surface" list).
- `s_linker15` is the v2.6.1 production linker; `s_linker19` is the paper frozen variant. Neither
  may be modified.
- `experimental=True, canonical=False` is the correct flag pattern for new experimental variants.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Standalone linker file construction | Source file (Python module) | — | s_linker20.py is self-contained; all prompt constants inlined inside the module |
| Runner variant registration | run_ablation.py | — | CANONICAL_VARIANTS list + VARIANT_SPECS dict are the two registration sites |
| Byte-equal gate verification | Git working tree + sha256sum | tests/test_s_linker20_harness_invariants.py | GATE-01 is verified by git diff + sha256sum; the existing invariants test already checks this for s19/s13min |

---

## Standard Stack

### Core
| Item | Version | Purpose |
|------|---------|---------|
| `tests/scratch/s_linker19.py` | Phase 46 close | Authoritative source for the 4 builder-level string changes to inline |
| `tests/scratch/prompts_v5.py` | Phase 46 close | Authoritative source for the 11 prompt-constant values to inline |
| `src/llm_sad_sam/linkers/experimental/s_linker19.py` | frozen (05c413d0) | Template for s_linker20.py body — copy verbatim then apply inline changes |
| `run_ablation.py` | current HEAD | Two edit sites: CANONICAL_VARIANTS list, VARIANT_SPECS dict |

No external packages required. Phase 47 is a pure file-construction + registration phase.

---

## Package Legitimacy Audit

No external packages installed in this phase.

---

## Q1 — Inlining Source of Truth: Complete Map

### Prompt constants imported by s_linker19 (production source, lines 99–114)

The production `src/llm_sad_sam/linkers/experimental/s_linker19.py` imports from
`llm_sad_sam.linkers.experimental.prompts_v5` (line 99):

```python
from llm_sad_sam.linkers.experimental.prompts_v5 import (
    AMBIGUITY_FEW_SHOT, AMBIGUITY_RULES,
    DOC_KNOWLEDGE_EXTRACTION_RULES, DOC_KNOWLEDGE_JUDGE_EXAMPLES,
    DOC_KNOWLEDGE_JUDGE_RULES, ALIAS_SCOPE_RULES,
    ENTITY_EXTRACTION_RULES,
    P1_FOCUS, P2_FOCUS, VALIDATION_RULES,
    COREF_RULES, ANTECEDENT_ALIAS_RULES, COREF_VALIDATION_FOCUS,
)
```

**13 constants total.** Phase 46 changed 8 of them (the 12 `kept` cuts affect 8 distinct constants
because some constants received multiple cuts). The values below are from
`tests/scratch/prompts_v5.py` AT PHASE 46 CLOSE (the frozen minimized set).

### Constant values to inline in s_linker20.py

The import block is REMOVED entirely. Each constant is defined as a module-level variable inline.

| Constant | Phase 46 changed? | Source location | Value in tests/scratch/prompts_v5.py |
|----------|------------------|-----------------|--------------------------------------|
| `AMBIGUITY_FEW_SHOT` | YES (CUT-AMB-01: drop-by-empty) | prompts_v5.py:30 | `""` |
| `AMBIGUITY_RULES` | NO | prompts_v5.py:32 | unchanged from production |
| `DOC_KNOWLEDGE_EXTRACTION_RULES` | NO | prompts_v5.py:39 | unchanged from production |
| `DOC_KNOWLEDGE_JUDGE_EXAMPLES` | YES (CUT-DKJ-01: drop-by-empty) | prompts_v5.py:41 | `""` |
| `DOC_KNOWLEDGE_JUDGE_RULES` | YES (CUT-DKJ-07: `architectural tier or technology platform` → `grouping`) | prompts_v5.py:43 | See verbatim text below |
| `ALIAS_SCOPE_RULES` | NO | prompts_v5.py:45–48 | unchanged from production |
| `ENTITY_EXTRACTION_RULES` | NO | prompts_v5.py:55 | unchanged from production |
| `P1_FOCUS` | NO | prompts_v5.py:68–74 | unchanged from production |
| `P2_FOCUS` | NO | prompts_v5.py:76–80 | unchanged from production |
| `VALIDATION_RULES` | YES (CUT-VAL-01: `counterparts` → `matching entities`) | prompts_v5.py:82 | See verbatim text below |
| `COREF_VALIDATION_FOCUS` | YES (CUT-VAL-03: `role-referential phrase` → `noun phrase that refers back`) | prompts_v5.py:94–100 | See verbatim text below |
| `COREF_RULES` | YES (CUT-COR-01 + CUT-COR-02 combined) | prompts_v5.py:102 | See verbatim text below |
| `ANTECEDENT_ALIAS_RULES` | NO | prompts_v5.py:104–112 | unchanged from production |

**Verbatim minimized values for changed constants** (read from `tests/scratch/prompts_v5.py`):

```python
# AMBIGUITY_FEW_SHOT — CUT-AMB-01 (drop-by-empty)
AMBIGUITY_FEW_SHOT = ""

# DOC_KNOWLEDGE_JUDGE_EXAMPLES — CUT-DKJ-01 (drop-by-empty)
DOC_KNOWLEDGE_JUDGE_EXAMPLES = ""

# DOC_KNOWLEDGE_JUDGE_RULES — CUT-DKJ-07
DOC_KNOWLEDGE_JUDGE_RULES = """An alias is valid when the document establishes an equivalence between a phrase and a single named component. An alias is invalid when the phrase is generic vocabulary, names the whole system, or names a different entity. An alias is also invalid when it names a grouping that encompasses multiple elements, because it identifies a grouping rather than a single named unit. When uncertain, prefer APPROVE."""

# VALIDATION_RULES — CUT-VAL-01
VALIDATION_RULES = """Approve when the sentence treats the component as an architectural participant, including matching entities. Reject when the matching word is generic, names a different entity, or describes a technique that merely shares the component's name."""

# COREF_VALIDATION_FOCUS — CUT-VAL-03
COREF_VALIDATION_FOCUS = (
    "Check coref resolution: does the pronoun, 'it', 'they', 'the service', "
    "or similar noun phrase that refers back in this sentence actually refer to "
    "the named component as an architectural participant — performing "
    "operations, providing services, or being the grammatical topic of the "
    "sentence?"
)

# COREF_RULES — CUT-COR-01 + CUT-COR-02
COREF_RULES = """For each case, decide whether a pronoun or noun phrase that refers back in the target sentence refers back to a component named or aliased earlier in the context. Resolve when: (a) the component's name or a known alias appears in the surrounding context sentences, or (b) only one component has been introduced in the immediately preceding sentences — treat it as the topic of the surrounding section and resolve role-referential phrases ("it", "the module", "the service", "the component", "the system") to that topic even without a direct name repetition. Avoid resolving when two or more equally plausible antecedents exist. Known aliases include the terminal word(s) of a multi-word name, documented abbreviations, and alternate forms used in the document. When the antecedent sentence uses a known alias rather than the full canonical name, set antecedent_via_alias=true."""
```

### Builder-level f-string changes to inline (from tests/scratch/s_linker19.py)

Four prompt builder methods have their opener or body text changed. These are in-class `@staticmethod`
methods, unchanged by the import removal — but their string literals reflect the minimized text.

| Builder | Change | Line in tests/scratch/s_linker19.py | Before (production) | After (minimized) |
|---------|--------|-------------------------------------|---------------------|-------------------|
| `_prompt_ambiguity` | CUT-AMB-02: pleonasm opener | line 274 | `Classify these software architecture component names.` | `Classify these component names.` |
| `_prompt_extraction` | CUT-EXT-01: pleonasm opener | line 331 | `Extract ALL references to software architecture components from this document.` | `Extract ALL references to components from this document.` |
| `_prompt_validation` | CUT-VAL-02: pleonasm opener | line 347 | `Validate component references in a software architecture document. {focus}` | `Validate components in a document. {focus}` |
| `_prompt_coref` opener | CUT-COR-03: jargon removal | line 362 | `Resolve anaphoric references (pronouns and role-referential noun phrases) to architecture components.` | `Resolve references (pronouns and noun phrases that refer back) to components.` |
| `_prompt_coref` inline body | CUT-COR-04: jargon removal | lines 366–369 | `For each TARGET sentence below, identify any pronoun or role-referential noun phrase that refers back to a component listed above. If a target sentence has no anaphoric reference to a listed component, return no resolution for it. Be conservative — only include resolutions you are CERTAIN about.` | `For each TARGET sentence below, identify any pronoun or noun phrase that refers back to a component listed above. If a target sentence has no such reference to a listed component, return no resolution for it. Be conservative — only include resolutions you are CERTAIN about.` |

Note: CUT-COR-05 (`Be conservative — only include resolutions you are CERTAIN about.`) is a
PROTECTED tombstone — it must be preserved verbatim. It appears at the end of the inline restatement
(lines 368–369 in tests/scratch after the COR-03/04 rewrite).

The authoritative final state of every builder is in `tests/scratch/s_linker19.py` at Phase 46
close. The planner should read that file to get the exact post-cut text for each method body.

---

## Q2 — Standalone Construction

### s_linker19 class hierarchy and imports

`SLinker19` in `src/llm_sad_sam/linkers/experimental/s_linker19.py` is **already standalone with
no inheritance**. The class declaration is:

```python
class SLinker19:
    """Paper variant — clean unified design (no A/B framings, no ILinker4, ...)."""
    _VARIANT_NAME = "s_linker19"
```

It does NOT inherit from any other linker class. The docstring module header confirms: "No
subclassing — what you see is what runs."

### Imports in production s_linker19.py (lines 86–116)

Standard-library and project-infra imports that CAN and MUST remain as-is in s_linker20.py:

```python
from __future__ import annotations
import json, os, pickle, re, threading, time, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from enum import Enum

from llm_sad_sam.core.data_types_v2 import (SadSamLink, CandidateLink, ModelKnowledge, DocumentKnowledge)
from llm_sad_sam.core.document_loader_v2 import load_sentences, build_sent_map
from llm_sad_sam.linkers.experimental.helper_v3 import (has_standalone_mention, parse_snum, get_comp_names)
# NOTE: the prompts_v5 import block (lines 99–114) is REMOVED — replaced by inline constants
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient, LLMBackend, LLMResponse
```

### What must change in s_linker20.py vs s_linker19.py

1. **Remove** the `from llm_sad_sam.linkers.experimental.prompts_v5 import (...)` block (lines 99–114).
2. **Add** inline constant definitions immediately after the infra imports (before the class body),
   containing the 13 constant values from the minimized `tests/scratch/prompts_v5.py`.
3. **Change** `class SLinker19:` → `class SLinker20:`.
4. **Change** `_VARIANT_NAME = "s_linker19"` → `_VARIANT_NAME = "s_linker20"`.
5. **Change** the print statement in `__init__` (line 234): `"SLinker19 (paper variant — standalone; prompts in prompts_v5)"` → `"SLinker20 (minimized prompts — standalone; all constants inlined)"` (or equivalent).
6. **Apply** the 5 builder-level text changes listed in Q1 (AMB-02, EXT-01, VAL-02, COR-03, COR-04).
7. **Update** the module docstring to describe s_linker20 (minimized prompt set, no inheritance from s19).
8. **Do NOT inherit from SLinker19.** s_linker20.py must be a flat copy with edits applied.

### What stays identical

All 1000+ lines of logic: `_TracingLLMClient`, `MentionType`, `EvidenceBundle`, `AliasEntry`,
DAG infra (`_run_parallel`, `_iter_batches`, `_prev_prefix`), LLM call helper (`_ask`), all phase
methods (`_analyze_model`, `_learn_document_knowledge`, `_run_framing_c`, `_run_extraction_pass`,
`_classify_mention_typed`, `_all_occurrences_in_qualified_path`, `_build_evidence_bundle`,
`_format_evidence`, `_validate_with_evidence`, `_run_validation_pass`,
`_antecedent_supports_resolution`, `_run_coreference`, `_validate_coref_links`),
all logging/checkpointing methods, and the `link()` entry point.

---

## Q3 — Runner Registration: Exact Edit Sites

### Edit 1: CANONICAL_VARIANTS list — append after line 117

Current last entry (line 117):
```python
    "s_linker18",  # v2.6.3 18d + cleanup A (enum-based mention classification): clean unified variant; NOT canonical
]
```

New entry to append immediately before the closing `]` on line 118:
```python
    "s_linker20",  # v2.6.4 minimized-prompt standalone variant (experimental=True): all constants inlined, no inheritance from s19; NOT canonical
```

Result (lines 117–119 after edit):
```python
    "s_linker18",  # v2.6.3 18d + cleanup A (enum-based mention classification): clean unified variant; NOT canonical
    "s_linker20",  # v2.6.4 minimized-prompt standalone variant (experimental=True): all constants inlined, no inheritance from s19; NOT canonical
]
```

### Edit 2: VARIANT_SPECS dict — append before closing `}` on line 750

Current last entry ends at line 749 (closing `),` of `s_linker18` block), followed by `}` on line 750.

New entry to insert before the `}`:
```python
    "s_linker20": dict(
        aliases=(),
        module="llm_sad_sam.linkers.experimental.s_linker20",
        class_name="SLinker20",
        description=(
            "S-Linker20 — v2.6.4 minimized-prompt standalone variant "
            "(experimental=True, NOT canonical). "
            "Same logic as s_linker19; all prompt constants inlined directly "
            "(no import from prompts_v5). 12 Phase 46 cuts applied: "
            "AMBIGUITY_FEW_SHOT drop, DOC_KNOWLEDGE_JUDGE_EXAMPLES drop, "
            "5 cross-section pleonasm trims (AMB/EXT/VAL openers), "
            "4 lexical-jargon neutralizations (DKJ/VAL/COR). "
            "Target: gpt-5.4 macro F1 >= 91.3% (Phase 48 sweep)."
        ),
        canonical=False,
        experimental=True,
    ),
```

### Confirmation: s_linker19 is NOT in run_ablation.py

`grep -n "s_linker19"` on `run_ablation.py` returns zero results. `SLinker19` is the paper-variant
class but was never registered in the runner (it is accessed directly by the replay harness via
`tests/harness/adapters.py`, not via run_ablation). s_linker20 will be the FIRST s_linker1x+
generation variant registered in the runner using the new class from `s_linker20.py`.

### Required fields in VARIANT_SPECS dict

From inspection of all existing experimental entries (s_linker15, s_linker17a..e, s_linker18):
- `aliases`: tuple (empty `()` for all s_linker1x+)
- `module`: `"llm_sad_sam.linkers.experimental.<name>"`
- `class_name`: `"SLinker<N>"` matching the class defined in the module
- `description`: string describing the variant
- `canonical`: boolean (`False` for all experimental variants)
- `experimental`: boolean (`True` for all experimental variants)

---

## Q4 — GATE-01 Verification Mechanics

### Authoritative baseline hashes (Phase 46 close, MINIMIZE-LOG GATE-01 section)

The MINIMIZE-LOG records the SHA-256 values at Phase 46 close (2026-06-08).
Live verification confirms these values are STILL correct at Phase 47 open (2026-06-09):

```
05c413d0f7fa38f46359c22a2207a6b05f82e50019388550f18f426eb6c9996d  src/llm_sad_sam/linkers/experimental/s_linker19.py
2f8b9968fd35e6a9c9e5e01bc16c8081b2bd80eb0efa4ab669f16975f8440689  src/llm_sad_sam/linkers/experimental/prompts_v5.py
083d92ae39747e1f98bdb6c0f9254d3368150ef78c614385e2ea97b58a018b33  src/llm_sad_sam/linkers/experimental/s_linker13_min.py
```

Note: the v2.6.3 GATE-01 baseline (`43-GATE01-BASELINE.txt`) recorded `226291a3...` for s_linker19.py.
This changed during Phase 44 (harness build, which added the prompts_v5 import block to s_linker19).
The current value `05c413d0...` is the v2.6.4 open/frozen value. All subsequent phases in v2.6.4
use this value.

### Verification commands

**Primary verification (git-based — used in tests/test_s_linker20_harness_invariants.py):**
```bash
git -C /mnt/hostshare/ardoco-home/agent-linker diff --stat \
    src/llm_sad_sam/linkers/experimental/s_linker19.py \
    src/llm_sad_sam/linkers/experimental/prompts_v5.py \
    src/llm_sad_sam/linkers/experimental/s_linker13_min.py
# Expected output: empty (no output = PASS)
```

**Secondary verification (SHA-256 against known values):**
```bash
sha256sum \
    src/llm_sad_sam/linkers/experimental/s_linker19.py \
    src/llm_sad_sam/linkers/experimental/prompts_v5.py \
    src/llm_sad_sam/linkers/experimental/s_linker13_min.py
# Expected:
# 05c413d0f7fa38f46359c22a2207a6b05f82e50019388550f18f426eb6c9996d  ...s_linker19.py
# 2f8b9968fd35e6a9c9e5e01bc16c8081b2bd80eb0efa4ab669f16975f8440689  ...prompts_v5.py
# 083d92ae39747e1f98bdb6c0f9254d3368150ef78c614385e2ea97b58a018b33  ...s_linker13_min.py
```

**Automated test (already exists):**
`tests/test_s_linker20_harness_invariants.py::test_gate_01_byte_equality_s19_s13min_prompts_v5`
runs `git diff --stat HEAD -- <path>` for all three files. This test already passes. After Phase 47,
it must still pass (s_linker20.py is a NEW file; it does not mutate any of the three frozen paths).

### Why GATE-01 holds by construction in Phase 47

Phase 47 creates a NEW file `s_linker20.py` and adds two lines to `run_ablation.py`. Neither of
the two frozen paths (`s_linker19.py`, `s_linker13_min.py`) nor `prompts_v5.py` is touched.
The construction approach (copy s_linker19 → new file, apply edits to the copy) guarantees this.
The planner should include a GATE-01 verification step after every commit.

---

## Q5 — Dry-Run / Cached Execution (Success Criterion 2)

### LLM_BACKEND=checkpoint mode

`run_ablation.py` uses `get_backend()` which reads `LLM_BACKEND` env var. Setting
`LLM_BACKEND=checkpoint` causes the `LLMClient` to use `LLMBackend.CHECKPOINT`, which reads from
a `diskcache.Cache` at `CHECKPOINT_DIR` (default `./results/llm_checkpoint`).

However, for Phase 47 the success criterion is just that `run_ablation.py --variants s_linker20`
executes **without error** — not that it produces correct metrics. The zero-LLM constraint applies
to the Phase 46 harness, not to Phase 47's runner registration test.

### Correct dry-run invocation for Phase 47

Since s_linker20 needs to be registered and importable (not that the full 5-dataset sweep runs):

```bash
# Step 1: verify the variant is recognized (no datasets processed)
LLM_BACKEND=checkpoint python run_ablation.py --list-variants | grep s_linker20

# Step 2: verify the linker instantiates cleanly (CHECKPOINT backend avoids real LLM calls;
# if phase_cache misses occur they fall back to the checkpoint_fallback backend,
# but instantiation itself has zero LLM cost)
LLM_BACKEND=checkpoint python -c "
import run_ablation
from llm_sad_sam.llm_client import LLMBackend
linker = run_ablation.build_linker('s_linker20', backend=LLMBackend.CHECKPOINT)
print('OK:', linker.__class__.__name__, linker._VARIANT_NAME)
"
```

The `build_linker` path does: `importlib.import_module(spec['module'])` → `getattr(module, class_name)` → `cls(backend=LLMBackend.CHECKPOINT)`. The `SLinker20.__init__` creates a `LLMClient(backend=CHECKPOINT)` but makes zero LLM calls. This is the correct Phase 47 dry-run.

### Why `run_ablation.py --variants s_linker20` alone would make LLM calls

Running the full ablation on real datasets would make live LLM calls unless checkpoint cache is
pre-populated for `s_linker20`. The Phase 47 success criterion says "dry-run/cached mode sufficient;
no LLM calls required" — so instantiation + `--list-variants` verification is sufficient.

### Phase 48 note

Phase 48 (SWEEP) runs the full 5-dataset gpt-5.4 sweep. The phase_cache for s_linker20 does not
exist yet (only `s_linker19/openai/{mediastore,teastore,teammates,bigbluebutton,jabref}/` exist).
Phase 47 does NOT need to pre-populate the cache. Phase 48 creates it by running the sweep.

---

## Q6 — GATE-06: Benchmark-Taboo Verification

**BENCHMARK_TABOO.md location:** `/mnt/hostshare/ardoco-home/agent-linker/BENCHMARK_TABOO.md`

The MINIMIZE-LOG Pareto Summary confirms: all 12 kept cuts carry `gate06_isolation = clean`
(or `clean (no after-text)` for the two drop-by-empty cases). The benchmark-leak elimination
table confirms: 1/1 confirmed leaks eliminated.

### Re-grep method for the planner's verification step

For each changed constant body in s_linker20.py, extract the token set from the after-text and
re-run the GATE-06 grep. The Phase 46 reasoning cells in the MINIMIZE-LOG document the specific
grep commands used for each cut. The canonical approach is:

```bash
# Grep the inline constant bodies in s_linker20.py against BENCHMARK_TABOO.md
grep -niwE '<token_list>' BENCHMARK_TABOO.md
```

Where `<token_list>` is the set of distinctive tokens from each changed constant's after-text.
The two drop-by-empty cases (AMBIGUITY_FEW_SHOT = "", DOC_KNOWLEDGE_JUDGE_EXAMPLES = "") require
no grep — empty strings have zero tokens.

For the planner's verification task, the simplest approach is:

```bash
# Extract all non-empty prose text from s_linker20.py constant definitions
# and grep against BENCHMARK_TABOO.md — should yield zero hits except for
# the known-safe bare 'component' generic-SE-noun instances
grep -niwE 'grouping|encompasses|matching|noun|phrase|refers|back|topic|surrounding|section' \
    BENCHMARK_TABOO.md
# Expected: 0 hits (verified for each affected cut in Phase 46)
```

The Phase 46 MINIMIZE-LOG reasoning cells for each `kept` cut include the exact grep command
and confirmed-zero results. Phase 47 need only re-verify that no new tokens were introduced
during the inlining process (i.e., that the copy was faithful).

---

## Q7 — Phase 44 Golden Tests: Current State and Re-Pointing

### Existing test files (all present)

```
tests/test_s_linker20_prompt_ambiguity.py
tests/test_s_linker20_prompt_doc_extract.py
tests/test_s_linker20_prompt_doc_judge.py
tests/test_s_linker20_prompt_extraction.py
tests/test_s_linker20_prompt_validation.py
tests/test_s_linker20_prompt_coref.py
tests/test_s_linker20_harness_invariants.py
```

All 7 files exist. They currently pass in two modes controlled by `SAD_SAM_LINKER_SOURCE`:
- `production` (default): imports `SLinker19` from `llm_sad_sam.linkers.experimental.s_linker19`.
  The step-6 prompt-equality gate is ACTIVE. Tests pass against the frozen s19 baseline.
- `scratch`: imports `SLinker19` from `tests.scratch.s_linker19` (the minimized Phase 46 copy).
  The step-6 gate is GATED OFF. Tests pass with 0/N snapshot deltas.

### What Phase 47 does NOT need to do to the test files

The test files test `SLinker19._prompt_*` static methods (via `tests/harness/adapters.py`). They
do NOT need to be re-pointed to `SLinker20` for Phase 47. The golden tests are the HARNESS
validation for Phase 46 cuts — they validated the minimized prompts in scratch mode. Phase 47 just
ships those cuts as a new standalone file.

Phase 47 does NOT change `tests/harness/adapters.py`. The harness remains pointed at SLinker19
(production or scratch). The adapters file is the Phase 44 fixture — it proves the s19 baseline
works. s_linker20 correctness is validated by Phase 48 SWEEP (live LLM benchmark).

### Optional registration test for s_linker20

Following the pattern of `tests/test_s_linker14_voyager_registration.py`, the planner MAY include
a task to create `tests/test_s_linker20_registration.py` that checks:
1. `s_linker20` in `run_ablation.CANONICAL_VARIANTS`
2. `s_linker20` in `run_ablation.VARIANT_SPECS`
3. `spec.get("experimental") is True`
4. `spec.get("canonical") is False`
5. `spec["module"] == "llm_sad_sam.linkers.experimental.s_linker20"`
6. `spec["class_name"] == "SLinker20"`
7. `SLinker20._VARIANT_NAME == "s_linker20"`
8. `SLinker20` does NOT inherit from `SLinker19`

This is OPTIONAL for Phase 47 success criteria but is consistent with the project pattern and
adds GATE-01-style protection going forward.

---

## Architecture Patterns

### Recommended s_linker20.py construction approach

```
1. cp src/llm_sad_sam/linkers/experimental/s_linker19.py \
      src/llm_sad_sam/linkers/experimental/s_linker20.py
2. Edit s_linker20.py:
   a. Replace module docstring (update variant name + description)
   b. Remove import block (lines 99-114: the prompts_v5 import)
   c. Insert 13 inline constant definitions after the infra imports
   d. Rename class SLinker19 → SLinker20
   e. Change _VARIANT_NAME = "s_linker19" → _VARIANT_NAME = "s_linker20"
   f. Update __init__ print statement
   g. Apply 5 builder text changes (AMB-02, EXT-01, VAL-02, COR-03, COR-04)
3. Edit run_ablation.py:
   a. Append "s_linker20" entry to CANONICAL_VARIANTS
   b. Append "s_linker20": dict(...) entry to VARIANT_SPECS
4. Verify GATE-01: sha256sum / git diff --stat on s_linker19.py, prompts_v5.py, s_linker13_min.py
5. Run instantiation smoke test
```

### Inline constant block placement in s_linker20.py

Insert between the last `from llm_sad_sam... import` line and the module-level comment separator:

```python
# ... existing infra imports ...
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository
from llm_sad_sam.llm_client import LLMClient, LLMBackend, LLMResponse

# ─────────────────────────────────────────────────────────────────────────────
# Minimized prompt constants — inlined from Phase 46 minimization
# (replaces the `from prompts_v5 import ...` block in s_linker19)
# Phase 46 kept-cut inventory: AMB-01 (drop), AMB-02 (opener), DKJ-01 (drop),
# DKJ-07 (grouping), EXT-01 (opener), VAL-01 (matching entities), VAL-02 (opener),
# VAL-03 (noun phrase that refers back), COR-01 (noun phrase that refers back),
# COR-02 (topic of the surrounding section), COR-03 (opener), COR-04 (inline)
# ─────────────────────────────────────────────────────────────────────────────

AMBIGUITY_FEW_SHOT = ""
AMBIGUITY_RULES = """..."""  # unchanged
# ... etc. for all 13 constants
```

### Anti-Patterns to Avoid

- **Do NOT inherit from SLinker19.** Even if the method body is identical, the CONTEXT.md and
  REQUIREMENTS.md are explicit: no inheritance. The "duplicated standalone files over inheritance"
  preference is the architectural rule (REQ-V264-08 verbatim: "standalone file (no inheritance from
  s_linker19)").
- **Do NOT import from prompts_v5.** The whole point of s_linker20 is that it is self-contained.
  Any import from `prompts_v5` defeats the minimization audit's self-containment goal.
- **Do NOT modify tests/scratch/** in Phase 47. That directory is the Phase 46 artifact. Phase 47
  reads it but does not write to it.
- **Do NOT edit s_linker19.py or prompts_v5.py.** GATE-01.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Verifying frozen hashes | Custom hash-check script | `sha256sum` + `git diff --stat` (already used by the existing GATE-01 test) |
| Discovering the minimized prompt values | Re-reading audit docs | Read `tests/scratch/prompts_v5.py` and `tests/scratch/s_linker19.py` directly — those ARE the authoritative after-text |
| Building s_linker20 from scratch | Write 1000+ lines manually | Copy s_linker19.py, apply the 19 targeted edits (import removal, constant definitions, class rename, 5 builder changes, print stmt) |

---

## Common Pitfalls

### Pitfall 1: Stale constant values from the audit doc vs scratch files

**What goes wrong:** Planner copies constant text from the MINIMIZE-LOG `reasoning` cells instead
of reading `tests/scratch/prompts_v5.py` directly. The reasoning cells describe the change but
are not the final verbatim text.

**How to avoid:** Always read `tests/scratch/prompts_v5.py` for the constant body text. The MINIMIZE-LOG
`after` column is for reference, not for copy-paste.

**Warning signs:** `COREF_RULES` body looks slightly different than the reasoning cell describes
(multi-clause replacements are easy to mis-transcribe).

### Pitfall 2: Forgetting the `_VARIANT_NAME` change

**What goes wrong:** s_linker20.py has `_VARIANT_NAME = "s_linker19"` left unchanged. The
`_checkpoint_dir` method uses `_VARIANT_NAME` to build the cache path — a wrong variant name
would write Phase 48 sweep results into the s_linker19 cache directory.

**How to avoid:** Change `_VARIANT_NAME` in the same edit as the class rename.

### Pitfall 3: s_linker19 accidentally imported by s_linker20

**What goes wrong:** A `from llm_sad_sam.linkers.experimental.s_linker19 import SLinker19` line
ends up in s_linker20.py because someone tried to inherit rather than copy.

**How to avoid:** `grep -n "s_linker19" src/llm_sad_sam/linkers/experimental/s_linker20.py`
must return zero results after the file is written.

### Pitfall 4: VARIANT_SPECS dict entry missing `experimental=True`

**What goes wrong:** The s_linker20 dict entry omits `experimental=True`. The runner still works
but the registration tests fail.

**How to avoid:** Copy the s_linker18 dict entry pattern exactly; s_linker18 has both `canonical=False`
and `experimental=True` explicitly.

### Pitfall 5: Opener change not applied to `_prompt_coref`

**What goes wrong:** CUT-COR-03 and CUT-COR-04 are the last two cuts in the log. Implementer
applies the constant-body changes but forgets the builder method body changes for `_prompt_coref`.

**How to avoid:** Read `tests/scratch/s_linker19.py` lines 360–385 (the full `_prompt_coref`
method body) and copy that verbatim into s_linker20.py's `_prompt_coref`.

---

## Code Examples

### Full import block → inline constant replacement

In s_linker19.py (production), lines 99–114:
```python
from llm_sad_sam.linkers.experimental.prompts_v5 import (
    AMBIGUITY_FEW_SHOT, AMBIGUITY_RULES,
    DOC_KNOWLEDGE_EXTRACTION_RULES, DOC_KNOWLEDGE_JUDGE_EXAMPLES,
    DOC_KNOWLEDGE_JUDGE_RULES, ALIAS_SCOPE_RULES,
    ENTITY_EXTRACTION_RULES,
    P1_FOCUS, P2_FOCUS, VALIDATION_RULES,
    COREF_RULES, ANTECEDENT_ALIAS_RULES, COREF_VALIDATION_FOCUS,
)
```

In s_linker20.py, this block is REMOVED and replaced by inline definitions (see Q1 for the
full content of each constant).

### s_linker20 class declaration

```python
class SLinker20:
    """v2.6.4 minimized-prompt variant — standalone (no inheritance from s_linker19).

    Identical logic to SLinker19; all prompt constants inlined directly (Phase 46 kept-cut set
    applied: 12 cuts, 14 LOC saved, benchmark-leak eliminated). No import from prompts_v5.

    experimental=True, canonical=False.
    """

    _VARIANT_NAME = "s_linker20"
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Prompt constants in `prompts_v5.py`, imported by linker | Prompt constants inlined directly in `s_linker20.py` | Self-contained file — prompts auditable without navigating to a separate module |
| s_linker19 had never been in run_ablation.py CANONICAL_VARIANTS | s_linker20 registered in CANONICAL_VARIANTS + VARIANT_SPECS | First s_linker1x generation variant accessible via `python run_ablation.py --variants s_linker20` |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | s_linker20.py does not need to be added to `tests/harness/adapters.py` | Q7 | If the planner decides adapters.py should reference SLinker20 for Phase 48 harness, that is an additive change not blocking Phase 47 |
| A2 | The `_VARIANT_NAME` change also requires updating the `__init__` print statement | Q2 | Cosmetic only; does not affect test outcomes if left as s_linker19 text |
| A3 | `tests/test_s_linker20_registration.py` is optional (not a Phase 47 success criterion) | Q7 | If verifier requires it, it is a small addition |

---

## Open Questions

1. **CLAUDE.md update scope.** CLAUDE.md lists the "Active Surface" retained files. s_linker20.py
   should be added to this list. Is this in scope for Phase 47? (It is consistent with the
   convention — every production linker file is listed there.)
   - Recommendation: include as a task; it is 1 line addition.

2. **`prompts_v5.py` import scope in the constants block.** The 5 constants that did NOT change
   (AMBIGUITY_RULES, DOC_KNOWLEDGE_EXTRACTION_RULES, ALIAS_SCOPE_RULES, ENTITY_EXTRACTION_RULES,
   ANTECEDENT_ALIAS_RULES) plus P1_FOCUS, P2_FOCUS should be copied verbatim from the production
   `prompts_v5.py` (not the scratch copy, since they are identical). Either source is correct;
   using the production file is cleaner since those constants are already frozen.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3 | s_linker20.py | ✓ | (system python) | — |
| git | GATE-01 verification | ✓ | (system git) | sha256sum fallback |
| sha256sum | GATE-01 secondary verification | ✓ | (system coreutils) | — |
| tests/scratch/s_linker19.py | Inlining source of truth (builders) | ✓ | Phase 46 close | — |
| tests/scratch/prompts_v5.py | Inlining source of truth (constants) | ✓ | Phase 46 close | — |

---

## Sources

### Primary (HIGH confidence)
- `tests/scratch/s_linker19.py` — Phase 46 close frozen scratch copy. Read directly.
- `tests/scratch/prompts_v5.py` — Phase 46 close frozen minimized constants. Read directly.
- `src/llm_sad_sam/linkers/experimental/s_linker19.py` — Production frozen file. Read directly.
- `.planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md` — Per-cut verdicts and Phase 47 inline pointer table. Read directly.
- `run_ablation.py` — Read lines 40–750 for CANONICAL_VARIANTS list and VARIANT_SPECS dict.
- `tests/harness/adapters.py` — Read directly for SAD_SAM_LINKER_SOURCE toggle behavior.
- `tests/test_s_linker20_harness_invariants.py` — GATE-01 test implementation. Read directly.
- `tests/test_s_linker14_voyager_registration.py` — Registration test pattern. Read directly.

### Secondary (MEDIUM confidence)
- `.planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md §Phase 47 Inline Locations` — per-cut file:line pointer table. Cross-referenced against live file reads. [CITED]
- `.planning/milestones/v2.6.3-phases/43-replay.../43-GATE01-BASELINE.txt` — v2.6.3 SHA-256 reference. [CITED]

---

## Metadata

**Confidence breakdown:**
- Inlining source of truth (Q1): HIGH — read both frozen scratch files directly
- Standalone construction (Q2): HIGH — read production s_linker19.py directly; no inheritance confirmed
- Runner registration (Q3): HIGH — read run_ablation.py lines 40–750 directly; confirmed s_linker19 absent; exact edit sites documented
- GATE-01 verification (Q4): HIGH — hashes read directly from MINIMIZE-LOG and live sha256sum; test exists
- Dry-run execution (Q5): HIGH — traced through get_backend() → LLMBackend.CHECKPOINT path
- GATE-06 re-grep (Q6): HIGH — BENCHMARK_TABOO.md path confirmed; grep commands from MINIMIZE-LOG
- Phase 44 golden tests (Q7): HIGH — all 7 test files confirmed present; adapters.py source toggle confirmed

**Research date:** 2026-06-09
**Valid until:** 2026-07-09 (stable project; frozen source files will not change)
