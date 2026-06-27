# Phase 46: MINIMIZE — Research

**Researched:** 2026-06-08
**Domain:** Per-cut Pareto reduction loop on `s_linker19` prompt surface, gated by the Phase 44 golden-replay harness, against the 19 candidate cuts catalogued in `s_linker20-PROMPT-AUDIT.md`.
**Confidence:** HIGH on harness mechanics (verified by reading the actual code under `tests/harness/`); HIGH on cut row counts and dependencies (verified against the audit doc); MEDIUM-HIGH on the "byte-equality gate semantics" question (resolved below, but is the one place where the planner must explicitly choose between two possible interpretations).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01 — Scratch copies under `tests/scratch/` with harness adapter override.** Phase 46 creates byte-for-byte copies of `s_linker19.py` and `prompts_v5.py` under `tests/scratch/`; the harness gains a toggle (`SAD_SAM_LINKER_SOURCE`, default `production`) that swaps the import path to `tests.scratch.*` when set to `scratch`. The production sources are never touched.
- **D-02 — Section-sequential AMB→DKX→DKJ→EXT→VAL→COR, risk-ascending within section.** Outer loop walks sections in pipeline order; inner loop sorts by risk tier `low → low-med → med → med-high → high`, ties broken by audit-doc row order.
- **D-03 — `drop → Family A → Family B`, smallest passing replacement wins (REQ-V264-06).** For CUT-AMB-01 and CUT-DKJ-01: drop first; on revert, walk Family A rows independently against pristine baseline; on all-A-fail, walk Family B rows. First passing row wins; rest log as `superseded-by-{drop,A,B}`.
- **D-04 — One atomic commit per cut decision.** `feat(46-NN): keep …` for kept; `chore(46-NN): revert …` for reverted; `docs(46-NN): protect …` for tombstones. Superseded rows fold into the parent commit.

### Claude's Discretion

- Whether to commit `tests/scratch/` copies in the first plan (recommended: **yes**, so the scratch baseline is reproducible and the post-phase-close `tests/scratch/` diff IS the recipe Phase 47 consumes).
- Mechanism for the harness override: env var, pytest fixture, or `--linker-source` CLI flag (recommended below: **env var**).
- Per-cut commit body content: full diff for kept, brief reason-only for reverted.
- Single combined commit vs three separate commits for the cross-section pleonasm batch (default per D-04: **three separate commits**; Pareto Summary cross-references them).
- Pareto Summary autogeneration (recommended: **manual fill at close-out plan**).
- Add `next-action` pointer in MINIMIZE-LOG referencing each kept row to the Phase 47 inline location.

### Deferred Ideas (OUT OF SCOPE)

- Cross-backend Claude validation of kept cuts (gpt-5.4 only per v2.3 policy).
- Auto-generating MINIMIZE-LOG from a script.
- Emergent cuts not in the audit (flag as `EMERGENT` in log but DO NOT trial in this phase).
- Per-prompt minimization extended to `s_linker17e`.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REQ-V264-05 | Per-prompt Pareto reduction loop: for each candidate cut, apply, run golden tests, keep iff snapshot byte-equal AND no benchmark vocab introduced. Log per candidate. | §3 (per-cut trial protocol) + §8 (concrete commands) + §7 (verdict vocabulary). |
| REQ-V264-06 | Few-shot block-drop: test full-block removal first for `AMBIGUITY_FEW_SHOT` (CUT-AMB-01) and `DOC_KNOWLEDGE_JUDGE_EXAMPLES` (CUT-DKJ-01); if non-equal, attempt 1-3 example synthetic replacement; ship smallest passing. | §4 (drop-block decision tree). |
| REQ-V264-07 | Lexical neutralization: where domain-loaded vocabulary appears, attempt a neutral rewording; keep iff snapshots byte-equal. Behaviour stays SAD/SAM-tuned, vocabulary changes. | §3 (per-cut trial protocol for `domain-loaded` rows: AMB-02, EXT-01, VAL-01..03, COR-01..04, DKJ-07). |

</phase_requirements>

## Summary

Phase 46 trials 17 of 19 audit cuts (2 are protected tombstones, logged but not trialled) against a copy of `s_linker19.py` + `prompts_v5.py` placed under `tests/scratch/`, using the Phase 44 golden-replay harness with a small `SAD_SAM_LINKER_SOURCE=scratch` toggle wired into `tests/harness/adapters.py`. The frozen production sources are physically untouched throughout the phase, so GATE-01 byte-equal holds by construction.

The phase produces three artefacts:

1. `tests/scratch/{s_linker19.py, prompts_v5.py}` — the minimized recipe (the after-text Phase 47 inlines into `s_linker20.py`).
2. `.planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md` — one row per cut decision (19 total: 17 trial + 2 protected), with a Pareto Summary at top.
3. ~17 atomic commits, one per cut decision, so any single cut can be reverted via `git revert` without disturbing the rest.

**Critical harness semantics finding (§2 below):** the existing six test modules contain TWO assertions per test — a "step-6" prompt-equality sanity check (`rebuilt_prompt == record["prompt"]`) and the "step-7" parsed-output snapshot check (`parsed == snapshot`). The parsed-output snapshot is **invariant under prompt cuts** because it depends only on the cached `response_text`. The prompt-equality check, by contrast, will fail for any cut that changes prompt text — which is every meaningful cut. **The planner must explicitly decide how the scratch-mode harness handles step-6**; the recommended approach (Option A in §2.3) is to gate step-6 behind `SAD_SAM_LINKER_SOURCE != "scratch"` so scratch-mode runs ignore prompt-equality and the meaningful signal becomes "did the harness crash, did the parsed-output snapshot still pass, did GATE-06 stay clean."

**Primary recommendation:** 8 plans, 3 waves (Wave 1 = scratch bootstrap + adapter toggle; Wave 2 = six section plans AMB/DKX/DKJ/EXT/VAL/COR in parallel-ish; Wave 3 = finalize Pareto Summary + GATE-01 + phase-close).

## 1. Scratch Directory Layout

### 1.1 Recommended layout

```
tests/
├── scratch/                          # NEW — Phase 46 working copy
│   ├── __init__.py                    # empty file; makes tests.scratch a package
│   ├── s_linker19.py                  # byte-for-byte copy of frozen source at phase open
│   └── prompts_v5.py                  # byte-for-byte copy of frozen source at phase open
└── harness/
    ├── adapters.py                    # extended with SAD_SAM_LINKER_SOURCE toggle (§2.1)
    └── inputs.py                      # imports COREF_VALIDATION_FOCUS — see §2.2
```

That is the entire footprint. No subdirectories under `tests/scratch/` per cut; cuts mutate the two files in place and are recorded by the MINIMIZE-LOG row + commit history.

### 1.2 File list, with provenance

| File | Source at phase open | Authority for after-text |
|---|---|---|
| `tests/scratch/__init__.py` | new, empty | n/a (Python package marker) |
| `tests/scratch/s_linker19.py` | `cp src/llm_sad_sam/linkers/experimental/s_linker19.py tests/scratch/s_linker19.py` | mutated by Phase 46 cuts |
| `tests/scratch/prompts_v5.py` | `cp src/llm_sad_sam/linkers/experimental/prompts_v5.py tests/scratch/prompts_v5.py` | mutated by Phase 46 cuts |

**Import-path note.** When `s_linker19.py` is copied wholesale into `tests/scratch/`, its `from llm_sad_sam.linkers.experimental.prompts_v5 import …` line at the top will resolve to the PRODUCTION `prompts_v5.py`, not the scratch copy. Phase 46 plan-01 must patch the import line in the scratch copy of `s_linker19.py` to read `from tests.scratch.prompts_v5 import …` (single one-line edit, committed alongside the initial copy). This is the ONE edit to scratch `s_linker19.py` that is part of bootstrap and is NOT recorded as a cut row — it is a wiring change, not a prompt cut. The MINIMIZE-LOG opening note should call out this exception explicitly.

Alternative considered: rewrite imports dynamically via `importlib.util.spec_from_file_location` in `adapters.py`. Rejected — opaque, hard to debug; the static-import patch is one line and obvious to anyone reading the scratch file.

### 1.3 Synchronization with frozen sources

The scratch files are created ONCE in plan-01 from the byte-equal frozen sources. They are NOT re-synced mid-phase. If Phase 46 needs to re-baseline (e.g., a cut was committed but then the planner decides to revert it after later analysis), this is done via `git revert <commit-sha>` of the per-cut commit, not by re-copying from production.

GATE-01 verification (production sources unchanged) runs after every per-cut commit per CONTEXT specifics:

```bash
git diff --stat src/llm_sad_sam/linkers/experimental/s_linker19.py \
                src/llm_sad_sam/linkers/experimental/prompts_v5.py \
                src/llm_sad_sam/linkers/experimental/s_linker13_min.py
# Must return empty / exit 0
```

Any non-empty result triggers a hard halt and a `chore(46-NN): GATE-01 violation halt` commit before further work, per CONTEXT specifics.

## 2. Harness Adapter Override Mechanism

### 2.1 Override recommendation: environment variable

Three mechanisms were considered (CONTEXT Claude's Discretion):

| Mechanism | Pros | Cons |
|---|---|---|
| **Env var (`SAD_SAM_LINKER_SOURCE`)** | Process-wide; trivial to set inline (`SAD_SAM_LINKER_SOURCE=scratch pytest ...`); per-cut shell scripts are one-liners; matches the existing `LLM_BACKEND=openai` pattern in `.env` | Less testable in-pytest unless we use `monkeypatch` |
| **Pytest fixture parametrization** | Stays inside pytest; could run BOTH `production` and `scratch` in one session | Adds test-time branching to every test module; doubles test count for no Phase 46 benefit (we never want to run both modes in one CI run during 46) |
| **`--linker-source` pytest CLI flag** | Explicit; pytest-native | Requires per-test-module wiring via `pytest_addoption` + conftest plumbing; more code than the env var |

**Recommendation: env var.** It is the cheapest path, matches the user's existing backend-toggle pattern (`LLM_BACKEND=openai`), and integrates trivially with the per-cut Bash loop in §3.

### 2.2 Adapter implementation sketch

Current `tests/harness/adapters.py` (verified by reading lines 28–60) does a top-level `from llm_sad_sam.linkers.experimental.s_linker19 import SLinker19` and then exports `BUILDERS = {"_prompt_ambiguity": SLinker19._prompt_ambiguity, ...}` as module-level constants. The toggle needs to short-circuit this resolution.

**Recommended diff (illustrative — planner to refine in plan-01):**

```python
# tests/harness/adapters.py
from __future__ import annotations
import os
from typing import Callable

_SOURCE = os.environ.get("SAD_SAM_LINKER_SOURCE", "production")
if _SOURCE == "scratch":
    from tests.scratch.s_linker19 import SLinker19          # noqa: E402
elif _SOURCE == "production":
    from llm_sad_sam.linkers.experimental.s_linker19 import SLinker19  # noqa: E402
else:
    raise RuntimeError(
        f"SAD_SAM_LINKER_SOURCE must be 'production' or 'scratch', "
        f"got: {_SOURCE!r}"
    )

# (rest of file unchanged)
```

**Defaults to `production`** so all existing tests, CI runs, and ad-hoc developer runs continue to import the frozen `SLinker19`. Phase 46 explicitly opts in via `SAD_SAM_LINKER_SOURCE=scratch pytest …`.

### 2.3 The "step-6 prompt-equality gate" problem — CRITICAL DESIGN DECISION

Reading the actual six test modules (`tests/test_s_linker20_prompt_*.py`) confirms each test has TWO assertions:

```python
# Step 6: prompt-equality sanity gate
rebuilt_prompt = BUILDERS[_BUILDER](*args)
assert rebuilt_prompt == record["prompt"], ...

# Step 7-8: parsed-output snapshot
parsed = replay_parse(record["response_text"])
assert parsed == snapshot
```

The parsed-output snapshot (`.ambr` files under `tests/__snapshots__/`) stores parsed dicts derived from the cached `response_text` — verified by reading `test_s_linker20_prompt_ambiguity.ambr` (parsed `{architectural: [...], ambiguous: [...]}` per project). **Parsed output is invariant under any prompt cut** because the replayed LLM response is fixed.

This means the meaningful signal for "did the cut change anything?" lives ENTIRELY in the step-6 prompt-equality assertion. Three options for handling step-6 in scratch mode:

**Option A — Disable step-6 in scratch mode (RECOMMENDED).** The step-6 assertion is wrapped in `if os.environ.get("SAD_SAM_LINKER_SOURCE", "production") != "scratch":`. Scratch-mode runs assert only the parsed-output snapshot (which is invariant, so passes trivially) PLUS will surface any harness-breakage as a hard Python error (e.g., `inputs.py` reconstruct raises because the cut broke its prefix parser; or scratch SLinker19 references a constant that no longer exists).

The signal in scratch-mode is therefore:
- **Test crashed:** the cut is unsafe at the harness level → revert (snapshot delta `K/N` is logged as the count of tests that crashed; not a "model behavior" K/N, but a "harness compatibility" K/N).
- **Test passed:** the cut applied cleanly. Verdict candidate is `kept` pending GATE-06 re-grep.

**Option B — Per-cut expected-after baseline.** Add a "expected-after" delta recipe per cut row in MINIMIZE-LOG; the harness in scratch mode applies the delta to `record["prompt"]` and asserts equality. Heavier; requires per-cut text-transform expressions (sed/regex per cut); complicates the loop. Reject unless the planner has appetite for this.

**Option C — Skip step-6 entirely.** Easier than A; loses a class of harness-wiring signal. Reject.

**RECOMMENDATION: Option A.** It is one extra `if` in each of the six test modules (or one if-wrap inside a small `_prompt_equality_gate(rebuilt, logged)` helper imported into all six). The actual cut-validity signal collapses to:

> A cut is `kept` iff: (i) scratch SLinker19 imports without error, (ii) all 97 snapshot tests pass under `SAD_SAM_LINKER_SOURCE=scratch`, (iii) the cut's after-text re-greps clean against `BENCHMARK_TABOO.md`.

Anything that fails (i) or (ii) is `reverted`. Anything passing (i)+(ii) but failing (iii) is `unsafe`.

This is consistent with the CONTEXT D-01 definition (snapshot byte-equal AND no benchmark vocab introduced) once "snapshot" is interpreted as "the syrupy parsed-output snapshot" (the actual `.ambr` files), which is invariant by construction.

**Why Phase 46 still has signal under Option A:** The harness-compatibility signal is genuine and non-trivial. Several cuts in the audit WILL break the harness:

- **CUT-AMB-02** changes `_prompt_ambiguity` opener from `Classify these software architecture component names.` to a shorter form. `reconstruct_ambiguity_inputs` (lines 58–76 of `tests/harness/inputs.py`) only looks for `^NAMES:\s+(.+)$`, so this is safe — input reconstruction unaffected.
- **CUT-EXT-01** changes `_prompt_extraction` opener. `reconstruct_extraction_inputs` only looks for `^COMPONENTS:` and `\nDOCUMENT:\n` anchors — opener change is safe.
- **CUT-VAL-02** changes `_prompt_validation` opener. **`reconstruct_validation_inputs` HARDCODES** `fixed_prefix = "Validate component references in a software architecture document."` at `tests/harness/inputs.py:274`. Any change to the opener BREAKS the reconstructor → the harness raises `ValueError` in scratch mode → the cut is `reverted` automatically. **This is the principal compatibility risk in the phase.**
- **CUT-COR-03** changes `_prompt_coref` opener (line 354). `reconstruct_coref_inputs` does not anchor on the opener (it uses `^COMPONENTS:` and `--- Case N: SN ---` patterns) — safe.
- **CUT-DKJ-01** drops `DOC_KNOWLEDGE_JUDGE_EXAMPLES`. `reconstruct_doc_judge_inputs` looks for `\nPROPOSED MAPPINGS:\n` and `\n\n` blank-line terminator — dropping the examples block does NOT break the reconstructor because it terminates on the first blank line after PROPOSED MAPPINGS, which would still exist (the next blank line follows). Likely safe.
- **CUT-AMB-01** drops `AMBIGUITY_FEW_SHOT`. `reconstruct_ambiguity_inputs` does not reference the few-shot at all — safe.

**Planner action item:** plan-01 should include an explicit "harness compatibility map" verification step that re-greps `tests/harness/inputs.py` to confirm which cuts' opener changes break reconstruction. CUT-VAL-02 is the one definitely-breaks-it cut; the planner must decide whether to (a) extend `reconstruct_validation_inputs` to accept multiple opener prefixes (a small, well-scoped harness edit), or (b) accept that CUT-VAL-02 will always log `reverted` in scratch mode and require Phase 47 to make the rewording without harness coverage.

**Strong recommendation for planner:** option (a). Extending `reconstruct_validation_inputs` to recognize a small set of equivalent opener prefixes (the original "software architecture document" plus the chosen replacement) is a 3-line change in `inputs.py` and is justifiable as harness flexibility independent of Phase 46. It must NOT be classified as a cut; it is a harness extension. Apply it AS PART OF plan-01 alongside the scratch bootstrap. Same logic applies to any other opener-anchored reconstructor — but per the audit, validation is the only one.

## 3. Per-Cut Trial Protocol (the loop)

Each cut row in `s_linker20-PROMPT-AUDIT.md` is processed via the following sequence. Plan-NN scripts should encode this as a checklist; in practice the planner executes one cut at a time and commits after each.

### 3.1 Variables per cut

| Variable | Source | Example |
|---|---|---|
| `CUT_ID` | audit row `cut_id` column | `CUT-VAL-01` |
| `FILE` | audit row `file:lines` column, file portion | `prompts_v5.py` |
| `BEFORE_TEXT` | audit row `before` column | `Approve when the sentence treats the component as an architectural participant, including counterparts.` |
| `AFTER_TEXT` | audit row `after` column OR detail block | per Phase 46 empirical choice (for `domain-loaded` rows, the planner picks the universal-noun replacement) |
| `GATED_BY` | audit row `gated_by` column | `tests/test_s_linker20_prompt_validation.py @ phase_4_twopass_p1, phase_4_twopass_p2, phase_5_coref_validation` |
| `TEST_MODULE` | derived from GATED_BY | `tests/test_s_linker20_prompt_validation.py` |
| `RISK` | audit row `risk` tier (first token before em-dash) | `med` |

### 3.2 Sequence per cut

```bash
# Step 1: Confirm current scratch is at the pristine baseline (or post-previous-kept baseline)
git status tests/scratch/
git diff tests/scratch/ | head -20   # expect: empty for first cut, post-kept-cuts for later

# Step 2: Apply the cut to scratch (manual edit via planner)
$EDITOR tests/scratch/$FILE
# Or, for programmatic cuts (rare; most are prose rewrites that benefit from human review):
#   sed -i 's/BEFORE/AFTER/' tests/scratch/$FILE
# Verify the edit:
git diff tests/scratch/$FILE

# Step 3: Run the gated test module under scratch mode
SAD_SAM_LINKER_SOURCE=scratch pytest $TEST_MODULE -x --tb=short
# Capture exit code:
RC=$?

# Step 4: Read the snapshot delta from pytest output
#   - If RC=0: snapshot_delta = 0/N (all tests passed under scratch)
#   - If RC≠0: snapshot_delta = K/N where K = failed test count from pytest summary
# Phase 44 §D-03 totals per test module:
#   ambiguity:    5  snapshots
#   doc_extract:  5  snapshots
#   doc_judge:    5  snapshots
#   extraction:  18  snapshots (phase_2_framing_c_pass1 + pass2)
#   validation:  24  snapshots (3 phase tags)
#   coref:       40  snapshots

# Step 5: GATE-06 re-isolation grep on the after-text
#   Methodology: same v2.1 cross-dataset isolation methodology used in Phase 45
#   Grep the AFTER_TEXT verbatim against BENCHMARK_TABOO.md per-dataset sections + Universal Taboo.
grep -niwE "$(echo "$AFTER_TEXT" | grep -oE '\w+' | sort -u | paste -sd'|')" BENCHMARK_TABOO.md \
  | grep -v "^[0-9]*:## "  # ignore section headers
# If any hit is in a per-dataset section AND is not in Safe SE Textbook Examples → unsafe
# If hits are only in Universal Taboo entries that pass v2.1 GATE-06 isolation (generic SE noun used as such) → clean
# Hand-validate each hit per the same precedent used in Phase 45 (e.g., bare `component`, `the service` as anaphor placeholder)

# Step 6: Decide verdict (decision tree below)

# Step 7a: If KEPT — commit
git add tests/scratch/$FILE .planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md
git commit -m "feat(46-NN): keep $CUT_ID — <one-liner>"

# Step 7b: If REVERTED — revert scratch first, then commit LOG-only
git checkout tests/scratch/$FILE     # reverts to last committed state
git add .planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md
git commit -m "chore(46-NN): revert $CUT_ID — <failure reason>"

# Step 7c: If UNSAFE — revert scratch, then commit LOG-only with taboo hit detail
git checkout tests/scratch/$FILE
git add .planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md
git commit -m "chore(46-NN): revert $CUT_ID — unsafe: taboo hit {section}:{term}"

# Step 8: Continuous GATE-01 verification (per CONTEXT specifics)
git diff --stat src/llm_sad_sam/linkers/experimental/s_linker19.py \
                src/llm_sad_sam/linkers/experimental/prompts_v5.py \
                src/llm_sad_sam/linkers/experimental/s_linker13_min.py
# If non-empty: HARD HALT, file a chore(46-NN): GATE-01 violation halt commit
```

### 3.3 Verdict decision tree

```
After running the test module under scratch:

  IF pytest RC=0 (all gated tests passed):
    IF GATE-06 re-grep clean:
      → KEPT   (snapshot_delta = 0/N, gate06 = clean, commit kept-form)
    ELSE:
      → UNSAFE (snapshot_delta = 0/N, gate06 = taboo:{section}:{term}, commit reverted-form)
      (scratch reverted via `git checkout tests/scratch/$FILE` before commit)

  IF pytest RC≠0 (one or more gated tests failed):
    → REVERTED (snapshot_delta = K/N, gate06 = n/a, commit reverted-form,
                reasoning column names the failing test_id and the assertion message)
    (scratch reverted via `git checkout tests/scratch/$FILE` before commit)
```

### 3.4 Per-cut commit body content

Per CONTEXT discretion: full diff for kept, brief reason-only for reverted.

**Kept commit body:**
```
CUT-$ID — $one-line-summary

- Section: $section
- Before: $BEFORE_TEXT (truncated to 200ch)
- After:  $AFTER_TEXT  (truncated to 200ch)
- LOC saved: $N
- Snapshot delta: 0/$N
- GATE-06: clean
- Gated by: $TEST_MODULE @ $phase_tags

Diff (tests/scratch/$FILE):
$(git diff --staged tests/scratch/$FILE)
```

**Reverted commit body:**
```
CUT-$ID — $one-line-failure-reason

- Section: $section
- Failed test: $test_id
- Snapshot delta: $K/$N
- First failure line: $first_assertion_msg (200ch)
- Scratch reverted via `git checkout tests/scratch/$FILE` before commit.
```

## 4. Drop-Block Protocol Decision Tree

Per D-03. Applies to TWO parents: CUT-AMB-01 (drop AMBIGUITY_FEW_SHOT) and CUT-DKJ-01 (drop DOC_KNOWLEDGE_JUDGE_EXAMPLES).

### 4.1 Family attachment map

| Parent (drop-block) | Family A rows (synthetic-neutral swap) | Family B rows (concept-only) |
|---|---|---|
| **CUT-AMB-01** (`AMBIGUITY_FEW_SHOT`) | NONE — audit doc explicitly states no Family A/B rows for AMB section because the verdict was `clean` (no benchmark-leak) and per D-04/D-06 rewording families are emitted only for `benchmark-leak`. **CUT-AMB-01 has only the drop variant.** If drop fails → log `kept-original` for AMBIGUITY_FEW_SHOT. | NONE — same reason |
| **CUT-DKJ-01** (`DOC_KNOWLEDGE_JUDGE_EXAMPLES`) | CUT-DKJ-02 (Example 1 name swap: RequestHandler/Handler → BookManager/Mgr), CUT-DKJ-03 (Example 2 name swap: CacheLayer → MailSender), CUT-DKJ-04 (combined both-example swap, single coherent rewrite) | CUT-DKJ-05 (concept-only Example 1, name-stripped abstract VALID pattern), CUT-DKJ-06 (concept-only Example 2, name-stripped INVALID pattern) |

CUT-DKJ-07 (DOC_KNOWLEDGE_JUDGE_RULES "architectural tier or technology platform" clause) is NOT part of the drop-block protocol — it is a separate domain-loaded cut on a sibling constant, handled by the normal §3 loop.

### 4.2 Decision tree (CUT-DKJ-01 — the only one with full Family A+B coverage)

```
Step DROP — Apply CUT-DKJ-01: delete the entire DOC_KNOWLEDGE_JUDGE_EXAMPLES block (prompts_v5.py:47-53).
  Run: SAD_SAM_LINKER_SOURCE=scratch pytest tests/test_s_linker20_prompt_doc_judge.py -x
  GATE-06: trivially clean (no after-text to grep — the block is empty).

  IF pytest RC=0:
    → CUT-DKJ-01 verdict = kept (drop)
    → CUT-DKJ-02, -03, -04, -05, -06 → verdict = superseded-by-drop (no separate trial)
    → CUT-DKJ-07 still trialled in §3 loop normally.
    Pareto Summary "smallest-passing" for DOC_KNOWLEDGE_JUDGE_EXAMPLES = "drop"
    LOC saved = 7

  IF pytest RC≠0:
    Revert scratch (git checkout). Move to FAMILY A.

Step FAMILY A — Walk Family A rows in audit-row order against pristine baseline (NOT cumulative).
  For each of CUT-DKJ-02, CUT-DKJ-03, CUT-DKJ-04 in that order:
    Apply ONLY this cut to scratch (replace the exact Example 1 / Example 2 / both per the detail block).
    Run: SAD_SAM_LINKER_SOURCE=scratch pytest tests/test_s_linker20_prompt_doc_judge.py -x
    GATE-06: re-grep the after-text. Names BookManager / Mgr / MailSender pre-cleared per 45-04 SUMMARY.

    IF RC=0 AND GATE-06 clean:
      → THIS_CUT verdict = kept
      → REMAINING Family A rows in this loop → superseded-by-A
      → ALL Family B rows → superseded-by-A
      → CUT-DKJ-01 verdict = superseded-by-A (drop didn't work, but a Family-A replacement did)
      Pareto Summary "smallest-passing" = THIS_CUT identifier
      LOC saved = (length of new examples — 7 original lines), expected NEGATIVE for synthetic-neutral swaps (same line count, different names) → loc_saved = 0
      STOP — no more Family A trials.

    IF RC≠0 OR GATE-06 fail:
      Revert scratch. Continue to next Family A row.

  If all three Family A rows fail: revert scratch, move to FAMILY B.

Step FAMILY B — Walk Family B rows similarly.
  For each of CUT-DKJ-05, CUT-DKJ-06 in audit-row order:
    Apply ONLY this cut to scratch.
    Run pytest.
    GATE-06: re-grep the abstract rule prose (name-stripped — should be trivially clean).

    IF RC=0 AND GATE-06 clean:
      → THIS_CUT verdict = kept
      → REMAINING Family B rows → superseded-by-B
      → CUT-DKJ-01 verdict = superseded-by-B
      → ALL Family A rows → superseded-by-B (drop didn't work, no Family A worked, but Family B did)
      Pareto Summary "smallest-passing" = THIS_CUT identifier
      LOC saved = depends on rewrite (probably small positive; abstract rule may be shorter than worked example)
      STOP.

    IF RC≠0 OR GATE-06 fail:
      Revert scratch. Continue to next Family B row.

  If all 5 Family A + B rows fail:
    → CUT-DKJ-01 verdict = kept-original (no cut applied; block preserved verbatim)
    → CUT-DKJ-02..06 → all verdict = kept-original (superseded-by-no-passing-replacement)
    Pareto Summary "smallest-passing" = "kept-original (no replacement passed)"
    LOC saved = 0
```

### 4.3 Decision tree (CUT-AMB-01 — drop-only, no Family A/B)

```
Step DROP — Delete the AMBIGUITY_FEW_SHOT block (prompts_v5.py:30-36).
  Run: SAD_SAM_LINKER_SOURCE=scratch pytest tests/test_s_linker20_prompt_ambiguity.py -x
  GATE-06: trivially clean.

  IF pytest RC=0:
    → CUT-AMB-01 verdict = kept (drop)
    Pareto Summary "smallest-passing" = "drop"
    LOC saved = 7

  IF pytest RC≠0:
    Revert scratch.
    → CUT-AMB-01 verdict = kept-original
    Pareto Summary "smallest-passing" = "kept-original (no Family A/B rewordings emitted for clean-verdict block per D-06)"
    LOC saved = 0
```

### 4.4 Empirical likelihood (research-stage prior)

The likelihood that the drop-block (CUT-DKJ-01) passes is **low** in this harness. Reasoning (research-stage, not empirically tested):

- **The few-shot block carries load-bearing rationale shape.** The Phase 45 audit doc (lines 173, 225, 227) explicitly classifies DKJ as the highest-leverage prompt site in the audit and notes that the judge stage gates which proposed aliases reach extraction. The audit's risk tier on CUT-DKJ-01 is `high` for this reason.
- **But — under the recommended Option A scratch-mode harness — drop-block can only fail if dropping the block causes `inputs.py` to crash.** Reading `reconstruct_doc_judge_inputs` (lines 124–168), it terminates the PROPOSED MAPPINGS block on the first blank line. After dropping DOC_KNOWLEDGE_JUDGE_EXAMPLES, the prompt structure becomes `JUDGE: ...\n\nCOMPONENTS: A\n\nPROPOSED MAPPINGS:\n'term' -> Comp\n\n{DOC_KNOWLEDGE_JUDGE_RULES}\n\nReturn JSON:\n{...}\nJSON only:`. The blank-line terminator after PROPOSED MAPPINGS is still present. **Reconstructor likely survives.**
- **Therefore CUT-DKJ-01 is likely to pass byte-equal under Option A.** The Phase 46 verdict will likely be `kept` (drop).
- **CAVEAT:** This is a research-stage prediction. The reviewer judgment in Phase 45 (CUT-DKJ-01 risk = high) reflects MODEL behavior risk, which Phase 46 cannot observe via cached replays. Phase 48 sweep will measure actual model behavior; Phase 46 only measures harness compatibility. The MINIMIZE-LOG row for CUT-DKJ-01 should explicitly note this caveat so Phase 47 knows to treat the `kept` verdict as "harness-clean" not "behaviorally-validated."

For CUT-AMB-01 the same reasoning applies: dropping AMBIGUITY_FEW_SHOT leaves the surrounding structure intact (`NAMES:`, `NOW CLASSIFY THE NAMES ABOVE.`, JSON schema, AMBIGUITY_RULES). `reconstruct_ambiguity_inputs` only anchors on `NAMES:`. **Likely passes harness compatibility.** Same behavioral-vs-harness caveat applies.

## 5. Recommended Wave + Plan Decomposition

### 5.1 Plan count and cuts per section (verified against audit)

| Section | Cut rows | Trial-eligible (excl. tombstones) | Pareto-batched? |
|---|---|---|---|
| AMB | 2 (CUT-AMB-01, CUT-AMB-02) | 2 | CUT-AMB-02 is part of cross-section pleonasm batch |
| DKX | 0 | 0 | n/a |
| DKJ | 7 (CUT-DKJ-01..07) | 7 (no tombstone) | drop-block protocol on CUT-DKJ-01 + Family A/B |
| EXT | 1 (CUT-EXT-01) | 1 | part of cross-section pleonasm batch |
| VAL | 4 (CUT-VAL-01..04) | 3 (CUT-VAL-04 is tombstone) | CUT-VAL-02 part of pleonasm batch; CUT-VAL-03 shares lexicon with CUT-COR-01 |
| COR | 5 (CUT-COR-01..05) | 4 (CUT-COR-05 is tombstone) | CUT-COR-01 shares lexicon with CUT-VAL-03; CUT-COR-03+04 must be batched together |
| **Total** | **19** | **17** | — |

### 5.2 Recommended 8-plan, 3-wave decomposition

```
Wave 1 — Scaffolding (sequential, plan-01 only)
├── 46-01-PLAN.md — Scratch bootstrap + harness toggle
│   Actions:
│     - Create tests/scratch/__init__.py, tests/scratch/s_linker19.py, tests/scratch/prompts_v5.py (copies)
│     - Patch scratch s_linker19.py import line (from llm_sad_sam... → from tests.scratch.prompts_v5)
│     - Extend tests/harness/adapters.py with SAD_SAM_LINKER_SOURCE branch
│     - Extend tests/harness/inputs.py reconstruct_validation_inputs to accept multiple opener prefixes
│       (the production "Validate component references in a software architecture document." plus the planned
│       replacement vocabulary the planner picks for CUT-VAL-02; ideally a small regex or a tuple of accepted prefixes)
│     - Extend each of the 6 test modules with the step-6 gate (scratch-mode skips prompt-equality assertion)
│     - Initial MINIMIZE-LOG.md with schema header + empty rows table + 2 protected tombstone rows pre-filled
│     - Verify pytest with default (production) is exit 0; verify with SAD_SAM_LINKER_SOURCE=scratch is also exit 0 (no cuts yet)
│     - GATE-01 byte-equal of production sources verified
│     - Atomic commit: feat(46-01): scratch bootstrap + harness toggle

Wave 2 — Section trials (six section plans — parallel-ish; DKJ depends on plan-01 only, others independent)
├── 46-02-PLAN.md — AMB section (2 cuts: CUT-AMB-01 drop-only + CUT-AMB-02 pleonasm)
│   Cut order per D-02 risk-ascending: CUT-AMB-02 (low) → CUT-AMB-01 (high, drop-block)
│   Estimated time: 30 min (small section, 5 snapshots per cut)
│
├── 46-03-PLAN.md — DKX section (0 cuts → LOG completeness row only)
│   This plan exists for log-completeness: a row "no cuts attempted in DKX (audit verdict = clean for all 3 items)"
│   per the §7 edge case below.
│   Estimated time: 10 min (just LOG row + commit)
│
├── 46-04-PLAN.md — DKJ section (7 cuts; drop-block protocol — HIGHEST YIELD)
│   Cut order: D-03 drop-block protocol first on CUT-DKJ-01 → Family A → Family B → CUT-DKJ-07 (separate, low-priority)
│   Estimated time: 90 min (highest-yield section, possible drop+Family A walk, GATE-06 re-grep of each)
│   This plan also resolves the "DKJ cheap-then-coverage ordering" advice from 45-04 SUMMARY:
│     suggestion was CUT-DKJ-04 → 02 → 03 → 05/06 → 01 → 07, but per D-03 strict reading the
│     protocol is drop-first → Family A → Family B. The planner must reconcile; recommendation
│     is to follow D-03 strictly (drop-first) and let the audit doc's "cheap-then-coverage"
│     advice fall under "CUT-DKJ-07 trialled separately at end."
│
├── 46-05-PLAN.md — EXT section (1 cut: CUT-EXT-01)
│   Note: CUT-EXT-01 is part of the cross-section pleonasm batch (AMB-02 + EXT-01 + VAL-02). The Pareto
│   Summary cross-references these three but each gets its own commit per D-04 default.
│   Estimated time: 20 min (18 snapshots is the largest single-builder set after VAL/COR; one cut)
│
├── 46-06-PLAN.md — VAL section (4 cuts: CUT-VAL-01..04; CUT-VAL-04 tombstone)
│   Cut order: CUT-VAL-02 (low, pleonasm batch) → CUT-VAL-01 (med, counterparts) → CUT-VAL-03 (med-high,
│   role-referential — coordinated with COR-01 in 46-07) → CUT-VAL-04 (protected, tombstone row only)
│   Estimated time: 60 min (4 cuts, 24 snapshots, 3 phase tags per run)
│   **Inter-plan dependency:** if VAL-03 is trialled here, the chosen replacement vocabulary for
│   "role-referential phrase" must be carried into COR-01 in 46-07 — they share the lexical span.
│   Recommendation: VAL plan 46-06 picks the replacement and writes it to the MINIMIZE-LOG row;
│   COR plan 46-07 reads that row and uses the same vocabulary for COR-01.
│
├── 46-07-PLAN.md — COR section (5 cuts: CUT-COR-01..05; CUT-COR-05 tombstone)
│   Cut order: CUT-COR-02 (med, section-established topic) → CUT-COR-01 (med-high, role-referential noun phrase,
│   reuses VAL-03 vocabulary) → (CUT-COR-03 + CUT-COR-04) batched per audit (line 348 batching note) →
│   CUT-COR-05 (protected, tombstone row only)
│   Estimated time: 60 min (4 cuts, 40 snapshots — highest gating in audit; mandatory CUT-COR-03+04 lockstep)
│   **Inter-plan dependency:** reads MINIMIZE-LOG row from 46-06 to get the VAL-03 replacement vocabulary.

Wave 3 — Finalization (sequential, plan-08 only)
├── 46-08-PLAN.md — Pareto Summary + GATE-01 final + phase close
│   Actions:
│     - Backfill MINIMIZE-LOG ## Pareto Summary section (manual per CONTEXT Claude's Discretion):
│       totals (kept/reverted/unsafe/protected), per-section LOC totals, drop-block smallest-passing identifiers,
│       cross-section pleonasm batch cross-references, optional next-action pointers to Phase 47 inline locations
│     - Final GATE-01 byte-equal verification on production sources
│     - Verify tests/scratch/ has no uncommitted changes
│     - Verify all 17 trial cuts have committed LOG rows + (for kept) committed scratch updates
│     - Verify 2 tombstones have docs(46-NN) protect commits
│     - Verify section sums match REQ-V264-03/-04 tick-off (19 cut rows total in audit = 19 log rows)
│     - Atomic commit: docs(46-08): close Phase 46 — Pareto Summary finalized
```

### 5.3 Parallelizability assessment

Per the GSD wave model, all six Wave-2 section plans CAN be issued in parallel — they touch disjoint cuts and write to different sections of the MINIMIZE-LOG. However, there are TWO coordination constraints:

1. **VAL-03 ↔ COR-01 lexical share.** Both cuts target "role-referential phrase" or "role-referential noun phrase". If the planner wants the kept vocabulary to be consistent across both sites (which the audit recommends — see line 350 of audit doc), then 46-06 must precede 46-07. Recommended: **run sequentially** (06 → 07), but the other four (02, 03, 04, 05) can run in parallel after plan-01. In practice, the gsd planner runs Wave 2 plans serially unless explicitly told otherwise — so this constraint is enforced naturally.

2. **Cross-section pleonasm batch (AMB-02, EXT-01, VAL-02).** All three target the "software architecture …" opener. The chosen replacement vocabulary (e.g., `components`, `named elements`, `entities`) should be consistent across all three. The planner can either: (a) decide the replacement up-front (recommend: in plan-01's appendix or in a separate batch-decisions section of MINIMIZE-LOG), or (b) let the first plan that hits a pleonasm cut decide and propagate. (a) is cleaner and gives the planner a single decision point; (b) creates inter-plan dependencies. **Recommend (a):** pre-decide the pleonasm replacement vocabulary in plan-01 alongside the bootstrap, document it in MINIMIZE-LOG header notes.

### 5.4 Alternative decomposition considered

Considered: one plan per cut (17 plans). Rejected — too much overhead; the cut work is small per cut; commits are already atomic per D-04. Section-bundle plans (6 + bootstrap + close = 8) is the right granularity, matching Phase 45's wave shape (1 + 6 + 1 = 8).

Considered: collapse DKX (0 cuts) into plan-01 or plan-04. Rejected — log completeness is cleaner with a standalone DKX plan that emits its "no cuts attempted" log row. The plan is small (10 min) but its existence makes the section symmetry visible in `.planning/phases/46-minimize/` ls output.

## 6. Per-Section Effort Estimates and Risk Priors

### 6.1 Estimates

| Plan | Cuts | Tests/cut | Risk priors | Est. time |
|---|---|---|---|---|
| 46-01 (bootstrap) | 0 | n/a | LOW — pure plumbing; the `inputs.py` validation-opener extension is the trickiest part (3 lines) | 45 min |
| 46-02 (AMB) | 2 | 5 | LOW — CUT-AMB-02 is the lowest-risk row in the audit; CUT-AMB-01 likely passes drop (per §4.4 prediction) | 30 min |
| 46-03 (DKX) | 0 | n/a | n/a — log row only | 10 min |
| 46-04 (DKJ) | 7 | 5 | MEDIUM — drop-block likely passes harness compatibility (§4.4); CUT-DKJ-07 (med, separate) requires a careful Phase 46 empirical rewording of "architectural tier or technology platform" → universal noun | 90 min |
| 46-05 (EXT) | 1 | 18 | LOW-MED — opener pleonasm; harness-compatible because `reconstruct_extraction_inputs` doesn't anchor on opener; 18 snapshots is the most coverage for one cut | 20 min |
| 46-06 (VAL) | 4 (3 trialled, 1 tombstone) | 24 | **MED-HIGH** — CUT-VAL-02 will fail step-6 prompt-equality unless plan-01 extended `reconstruct_validation_inputs` to accept the new opener prefix (the principal compatibility risk in the phase); CUT-VAL-03 is a med-high lexical change shared with COR-01 | 60 min |
| 46-07 (COR) | 5 (4 trialled, 1 tombstone) | 40 | **MED-HIGH** — 40 snapshots is the highest gating in the audit; CUT-COR-03+04 batching is mandatory; vocabulary must stay consistent with 46-06's VAL-03 outcome | 60 min |
| 46-08 (close) | 0 | n/a | LOW — manual Pareto Summary fill + GATE-01 final | 30 min |
| **Total** | **19** (17 trial + 2 protect) | — | — | **~5h45m** |

### 6.2 Risk priors per cut (research-stage predictions, NOT verifications)

Phase 46 cannot verify behavioral safety (cached replays make parsed-output snapshots invariant). It CAN verify harness compatibility. Predictions below are research-stage priors only.

| Cut | Predicted verdict | Confidence | Reasoning |
|---|---|---|---|
| CUT-AMB-01 | `kept` (drop) | MED-HIGH | `reconstruct_ambiguity_inputs` doesn't reference the few-shot; structural anchors (`NAMES:`, etc.) preserved. |
| CUT-AMB-02 | `kept` | HIGH | `reconstruct_ambiguity_inputs` only anchors on `^NAMES:`; opener change is harness-safe. GATE-06: trivially clean (the opener has no benchmark tokens). |
| CUT-DKJ-01 | `kept` (drop) | MED-HIGH | Reconstructor terminates on blank line after PROPOSED MAPPINGS; dropping examples doesn't break it. **Behavioral risk NOT observable in this phase — caveat goes in LOG.** |
| CUT-DKJ-02..04 (Family A) | `superseded-by-drop` likely | MED | If CUT-DKJ-01 passes, all five are superseded. If it doesn't, Family A name swaps are harness-trivial (just text replacement). |
| CUT-DKJ-05..06 (Family B) | `superseded-by-drop` or `superseded-by-A` | MED | Same as above. |
| CUT-DKJ-07 | `kept` | MED | "architectural tier or technology platform" → universal-noun replacement (e.g., "grouping that encompasses multiple elements") — harness-safe (this is inside DOC_KNOWLEDGE_JUDGE_RULES, not the opener); GATE-06 clean. |
| CUT-EXT-01 | `kept` | HIGH | `reconstruct_extraction_inputs` anchors on `^COMPONENTS:` and `\nDOCUMENT:\n`; opener change is harness-safe. GATE-06: trivially clean. |
| CUT-VAL-01 | `kept` | HIGH | "counterparts" → "matching entities" (universal noun) inside VALIDATION_RULES, not the opener. Reconstructor unaffected. |
| **CUT-VAL-02** | **depends on plan-01 inputs.py extension** | — | If plan-01 extends `reconstruct_validation_inputs` to accept multiple openers → `kept`. If NOT → guaranteed `reverted` (harness raises). **This is the load-bearing planner decision in the phase.** |
| CUT-VAL-03 | `kept` | MED | "role-referential phrase" → "noun phrase that refers back" inside COREF_VALIDATION_FOCUS. Reconstructor unaffected (reads focus from prompt, not from constant). |
| CUT-VAL-04 | `protected` | — | Tombstone — not trialled. |
| CUT-COR-01 | `kept` | MED | "role-referential noun phrase" inside COREF_RULES. Reconstructor unaffected. Vocabulary chosen here must match CUT-VAL-03's. |
| CUT-COR-02 | `kept` | MED | "section-established topic" → "topic of the surrounding section" inside COREF_RULES. Reconstructor unaffected. |
| CUT-COR-03 + CUT-COR-04 (batched) | `kept` if reconstruct_coref_inputs is opener-agnostic | MED-HIGH | `reconstruct_coref_inputs` (lines 332–451 of inputs.py) anchors on `^COMPONENTS:` and `--- Case N: SN ---`, NOT on the opener at line 354. Opener change is harness-safe. The mandatory lockstep (line 354 + 358–360) is a discipline matter for the planner, not a harness gate. |
| CUT-COR-05 | `protected` | — | Tombstone. |

### 6.3 The DKJ "highest-yield" caveat

The audit doc (line 227) declares DKJ "the highest-yield section in the audit". This claim is about **benchmark-leak removal yield**, NOT about Phase-46 kept-cut yield in this harness. Phase 46's job is to confirm the leak removal works in the harness; the leak removal itself was established by Phase 45's grep clearance.

Phase 48's sweep is what validates behavioral safety. Phase 46's MINIMIZE-LOG MUST explicitly distinguish:

- **Harness verdict** (Phase 46 captures this): `kept`/`reverted`/`unsafe`/`protected`/`superseded-*`/`kept-original`.
- **Behavioral verdict** (Phase 48 captures this; OUT OF SCOPE for Phase 46): macro F1 regression vs s17e baseline.

The MINIMIZE-LOG schema in CONTEXT D-04 doesn't currently carry a behavioral verdict column — and shouldn't, because Phase 46 doesn't have that data. But the per-cut detail blocks SHOULD include a `behavioral_caveat: ` field flagging the cuts whose harness `kept` verdict is NOT a behavioral guarantee. At minimum: CUT-DKJ-01 (drop), CUT-AMB-01 (drop), and any cut on a `med-high` or `high` audit-row.

## 7. Edge Cases and Verdict Vocabulary

### 7.1 0-cut sections (DKX)

DKX has zero cut rows in the audit. **Recommendation:** plan-46-03 emits ONE LOG row noting:

```markdown
| - | n/a | n/a | n/a | 0 | $commit_sha | DKX section: 0 cut rows in audit (all 3 items verdict=clean per 45-03 SUMMARY). No trials attempted in Phase 46. Log row retained for section symmetry. |
```

OR a single section header line. The CONTEXT MINIMIZE-LOG schema doesn't have a clear "section completeness marker" row — recommend adding a `## DKX — No Cuts Attempted` section between the AMB and DKJ row groups in MINIMIZE-LOG, with a single line of explanation. This keeps the section-sequential ordering visible.

### 7.2 Protected tombstones (CUT-VAL-04, CUT-COR-05)

Per CONTEXT in-scope (line 19), the two tombstones get explicit `protected` log rows but are NOT trialled.

**Recommended row format:**

```markdown
| CUT-VAL-04 | protected | n/a | n/a | 0 | $commit_sha | Behaviorally-protected per prompts_v5.py docstring lines 5-22 + experiment_dotted_path_rename.py. Catches 2/3 code-path FPs on gpt-5.4 + 1/3 on Claude Sonnet, 0 collateral damage on 4-TP control set. Phase 46 MUST NOT cut per Phase 45 threat T-45-VAL-02. |
| CUT-COR-05 | protected | n/a | n/a | 0 | $commit_sha | Behavioral conservatism dial. Coref FP-sensitive stage; v2.6.2 s17e drove FP 43→14 via validation gating (CLAUDE.md milestone notes). Removing risks reintroducing the FP class. Phase 46 MUST NOT cut per Phase 45 threat T-45-COR-02. |
```

Commit messages per CONTEXT D-04: `docs(46-NN): protect CUT-{TAG}-NN — {one-line rationale}`. These are the third and fourth-to-last commits of the phase (executed in the section plan that owns them: 46-06 for VAL-04, 46-07 for COR-05).

### 7.3 Superseded rows under D-03

When drop-block CUT-DKJ-01 passes:

```markdown
| CUT-DKJ-01 | kept | 0/5 | clean | 7 | $sha | DROP-BLOCK passed: full DOC_KNOWLEDGE_JUDGE_EXAMPLES removed. Harness-compatible (PROPOSED MAPPINGS terminator preserved). 7 LOC saved. |
| CUT-DKJ-02 | superseded-by-drop | n/a | n/a | 0 | $sha | Drop-block CUT-DKJ-01 passed; this Family A variant moot. |
| CUT-DKJ-03 | superseded-by-drop | n/a | n/a | 0 | $sha | (same) |
| CUT-DKJ-04 | superseded-by-drop | n/a | n/a | 0 | $sha | (same) |
| CUT-DKJ-05 | superseded-by-drop | n/a | n/a | 0 | $sha | Drop-block CUT-DKJ-01 passed; this Family B variant moot. |
| CUT-DKJ-06 | superseded-by-drop | n/a | n/a | 0 | $sha | (same) |
```

Per D-04, "Each `superseded` row from D-03: folds into the parent drop/Family-A commit (no separate commit per superseded row)." So all six rows above share the SAME `$sha`.

If drop-block fails and CUT-DKJ-02 (Family A Example 1) passes:

```markdown
| CUT-DKJ-01 | superseded-by-A | 0/5 | n/a | 0 | $shaA | Drop-block REVERTED (failure detail below). CUT-DKJ-02 Family A passed and supersedes. |
| CUT-DKJ-02 | kept | 0/5 | clean | 0 | $shaA | Family A Ex1 swap: RequestHandler→BookManager, Handler→Mgr. BookManager/Mgr pre-cleared (45-04 SUMMARY). |
| CUT-DKJ-03 | superseded-by-A | n/a | n/a | 0 | $shaA | (same parent: CUT-DKJ-02) |
| CUT-DKJ-04 | superseded-by-A | n/a | n/a | 0 | $shaA | (same parent) |
| CUT-DKJ-05 | superseded-by-A | n/a | n/a | 0 | $shaA | (same parent — Family B never reached) |
| CUT-DKJ-06 | superseded-by-A | n/a | n/a | 0 | $shaA | (same) |
```

Note that the drop-block row CUT-DKJ-01 carries the REVERTED reasoning even though its verdict is `superseded-by-A`. Recommended: keep the drop-block FAILURE detail as a detail block under the CUT-DKJ-01 row.

### 7.4 Cross-section pleonasm batch (CUT-AMB-02 + CUT-EXT-01 + CUT-VAL-02)

Three independent trials per D-04 (one commit each). Pareto Summary cross-references them:

```markdown
## Pareto Summary

### Cross-section batch — "software architecture …" opener pleonasm

| Site | Cut | Verdict | Replacement vocabulary |
|---|---|---|---|
| `_prompt_ambiguity` line 266 | CUT-AMB-02 | kept | "Classify these component names." (chosen in plan-01 batch-decisions) |
| `_prompt_extraction` line 323 | CUT-EXT-01 | kept | "Extract ALL references to components from this document." |
| `_prompt_validation` line 339 | CUT-VAL-02 | depends | "Validate component references. {focus}" |

Shared rationale: per Phase 45 D-01 pragmatic rubric, the `software architecture` qualifier is pleonastic — each site has a COMPONENTS slot that already constrains scope. One replacement vocabulary across all three sites resolves the recurring `domain-loaded` flag with one harness run per affected gate.
```

### 7.5 Verdict vocabulary (complete list)

Per CONTEXT D-04 + this research:

| Verdict | Meaning | Commit type |
|---|---|---|
| `kept` | Cut applied to scratch; harness passes; GATE-06 clean | `feat(46-NN): keep CUT-... — ...` |
| `reverted` | Cut applied; harness failed (one or more snapshots crashed/diverged); scratch rolled back | `chore(46-NN): revert CUT-... — ...` |
| `unsafe` | Cut applied; harness passes; GATE-06 re-grep hit benchmark vocabulary; scratch rolled back | `chore(46-NN): revert CUT-... — unsafe: taboo:{section}:{term}` |
| `protected` | Tombstone — not trialled (CUT-VAL-04, CUT-COR-05) | `docs(46-NN): protect CUT-... — ...` |
| `superseded-by-drop` | Family A/B row moot because parent drop-block passed | folds into parent commit |
| `superseded-by-A` | Family B row moot because a Family A row passed; OR parent drop row + remaining Family A rows when this Family A row was chosen | folds into parent commit |
| `superseded-by-B` | Other Family B rows moot because this Family B row passed; parent drop and all Family A moot | folds into parent commit |
| `kept-original` | Drop, all Family A, all Family B rows failed; block preserved verbatim | single commit `chore(46-NN): kept-original CUT-... — no replacement passed` |
| `EMERGENT` | (per CONTEXT deferred) — cut not in audit; flagged but NOT trialled this phase | `docs(46-NN): defer EMERGENT cut — ...` (Phase 46 doesn't add new cuts) |

## 8. Concrete File Paths and Commands

### 8.1 File paths (absolute, never relative)

**Inputs (read-only during Phase 46):**

- `/mnt/hostshare/ardoco-home/agent-linker/.planning/phases/46-minimize/46-CONTEXT.md`
- `/mnt/hostshare/ardoco-home/agent-linker/.planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md`
- `/mnt/hostshare/ardoco-home/agent-linker/.planning/phases/45-audit/45-02-SUMMARY.md` (AMB)
- `/mnt/hostshare/ardoco-home/agent-linker/.planning/phases/45-audit/45-03-SUMMARY.md` (DKX)
- `/mnt/hostshare/ardoco-home/agent-linker/.planning/phases/45-audit/45-04-SUMMARY.md` (DKJ — pre-cleared Family A names)
- `/mnt/hostshare/ardoco-home/agent-linker/.planning/phases/45-audit/45-05-SUMMARY.md` (EXT)
- `/mnt/hostshare/ardoco-home/agent-linker/.planning/phases/45-audit/45-06-SUMMARY.md` (VAL)
- `/mnt/hostshare/ardoco-home/agent-linker/.planning/phases/45-audit/45-07-SUMMARY.md` (COR)
- `/mnt/hostshare/ardoco-home/agent-linker/.planning/phases/44-harness/44-CONTEXT.md` (D-03 builder→phase-tag map)
- `/mnt/hostshare/ardoco-home/agent-linker/src/llm_sad_sam/linkers/experimental/s_linker19.py` (READ-ONLY)
- `/mnt/hostshare/ardoco-home/agent-linker/src/llm_sad_sam/linkers/experimental/prompts_v5.py` (READ-ONLY)
- `/mnt/hostshare/ardoco-home/agent-linker/src/llm_sad_sam/linkers/experimental/s_linker13_min.py` (READ-ONLY)
- `/mnt/hostshare/ardoco-home/agent-linker/BENCHMARK_TABOO.md` (GATE-06 re-isolation source)
- `/mnt/hostshare/ardoco-home/agent-linker/tests/harness/adapters.py` (extended in plan-01)
- `/mnt/hostshare/ardoco-home/agent-linker/tests/harness/inputs.py` (validation-opener extension in plan-01)
- `/mnt/hostshare/ardoco-home/agent-linker/tests/harness/loader.py`
- `/mnt/hostshare/ardoco-home/agent-linker/tests/harness/replay_client.py`
- `/mnt/hostshare/ardoco-home/agent-linker/tests/harness/manifest.py`
- `/mnt/hostshare/ardoco-home/agent-linker/tests/harness/fixtures/MANIFEST.json`
- `/mnt/hostshare/ardoco-home/agent-linker/tests/test_s_linker20_prompt_ambiguity.py` (extended in plan-01 to gate step-6)
- `/mnt/hostshare/ardoco-home/agent-linker/tests/test_s_linker20_prompt_doc_extract.py` (same)
- `/mnt/hostshare/ardoco-home/agent-linker/tests/test_s_linker20_prompt_doc_judge.py` (same)
- `/mnt/hostshare/ardoco-home/agent-linker/tests/test_s_linker20_prompt_extraction.py` (same)
- `/mnt/hostshare/ardoco-home/agent-linker/tests/test_s_linker20_prompt_validation.py` (same)
- `/mnt/hostshare/ardoco-home/agent-linker/tests/test_s_linker20_prompt_coref.py` (same)
- `/mnt/hostshare/ardoco-home/agent-linker/tests/__snapshots__/test_s_linker20_prompt_*.ambr` (the 6 `.ambr` files; READ-ONLY mid-phase; pytest may rewrite them on snapshot drift, which would be a harness signal — but parsed-output is invariant under prompt cuts so no rewrite should occur)

**Outputs (created/mutated during Phase 46):**

- `/mnt/hostshare/ardoco-home/agent-linker/tests/scratch/__init__.py` (plan-01)
- `/mnt/hostshare/ardoco-home/agent-linker/tests/scratch/s_linker19.py` (plan-01 creates; plan-04..07 mutate per cut)
- `/mnt/hostshare/ardoco-home/agent-linker/tests/scratch/prompts_v5.py` (plan-01 creates; plan-04..07 mutate per cut)
- `/mnt/hostshare/ardoco-home/agent-linker/.planning/phases/46-minimize/46-01-PLAN.md` … `46-08-PLAN.md`
- `/mnt/hostshare/ardoco-home/agent-linker/.planning/phases/46-minimize/46-{01..08}-SUMMARY.md` (post-execution)
- `/mnt/hostshare/ardoco-home/agent-linker/.planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md` (the deliverable; built incrementally across plans 02–08)

### 8.2 One-time bootstrap commands (plan-01)

```bash
cd /mnt/hostshare/ardoco-home/agent-linker

# 1. Create scratch dir and copy frozen sources
mkdir -p tests/scratch
touch tests/scratch/__init__.py
cp src/llm_sad_sam/linkers/experimental/s_linker19.py tests/scratch/s_linker19.py
cp src/llm_sad_sam/linkers/experimental/prompts_v5.py tests/scratch/prompts_v5.py

# 2. Patch the import line in scratch/s_linker19.py
# Find the line: from llm_sad_sam.linkers.experimental.prompts_v5 import (
# Replace with: from tests.scratch.prompts_v5 import (
sed -i 's|from llm_sad_sam.linkers.experimental.prompts_v5 import|from tests.scratch.prompts_v5 import|' \
  tests/scratch/s_linker19.py
# Verify single match (otherwise the cut breaks):
grep -n "from tests.scratch.prompts_v5\|from llm_sad_sam.linkers.experimental.prompts_v5" tests/scratch/s_linker19.py

# 3. Extend tests/harness/adapters.py (manual edit per §2.2)

# 4. Extend tests/harness/inputs.py reconstruct_validation_inputs (manual edit per §2.3)
#    e.g., change line 274:
#      fixed_prefix = "Validate component references in a software architecture document."
#    to:
#      ACCEPTED_PREFIXES = (
#          "Validate component references in a software architecture document.",
#          "Validate component references.",  # CUT-VAL-02 replacement
#      )
#      for fixed_prefix in ACCEPTED_PREFIXES:
#          if first_line.startswith(fixed_prefix):
#              break
#      else:
#          raise ValueError(...)

# 5. Extend each of the 6 test modules with the scratch-mode step-6 gate (manual edit per §2.3)
# Wrap the rebuilt_prompt == record["prompt"] assertion in:
#   if os.environ.get("SAD_SAM_LINKER_SOURCE", "production") != "scratch":
#       assert rebuilt_prompt == record["prompt"], ...

# 6. Verify production mode still green
pytest tests/test_s_linker20_prompt_*.py -x

# 7. Verify scratch mode (no cuts yet) is also green
SAD_SAM_LINKER_SOURCE=scratch pytest tests/test_s_linker20_prompt_*.py -x

# 8. GATE-01 check
git diff --stat src/llm_sad_sam/linkers/experimental/s_linker19.py \
                src/llm_sad_sam/linkers/experimental/prompts_v5.py \
                src/llm_sad_sam/linkers/experimental/s_linker13_min.py
# Must be empty

# 9. Initialize MINIMIZE-LOG.md with schema header and the 2 protected tombstone rows
# (manual edit; see §7.2 for tombstone row format)

# 10. Atomic commit
git add tests/scratch/ tests/harness/adapters.py tests/harness/inputs.py \
        tests/test_s_linker20_prompt_*.py \
        .planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md
git commit -m "feat(46-01): scratch bootstrap + harness toggle"
```

### 8.3 Per-cut shell template (plans 02, 04, 05, 06, 07)

```bash
# Variables (set per cut)
CUT_ID="CUT-VAL-01"
FILE="tests/scratch/prompts_v5.py"
TEST_MODULE="tests/test_s_linker20_prompt_validation.py"

# 1. Pre-flight: scratch clean
test -z "$(git diff tests/scratch/)" || { echo "scratch has uncommitted changes"; exit 1; }

# 2. Apply cut (manual edit)
$EDITOR "$FILE"
git diff "$FILE"   # review

# 3. Run gated tests
SAD_SAM_LINKER_SOURCE=scratch pytest "$TEST_MODULE" -x --tb=short
RC=$?

# 4. GATE-06 re-grep (manual; see §3.2 step 5)
# AFTER_TEXT extracted from git diff; grep against BENCHMARK_TABOO.md

# 5. Decide verdict per §3.3 tree

# 6a. KEPT
git add "$FILE" .planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md
git commit -m "feat(46-NN): keep $CUT_ID — <one-liner>"

# 6b. REVERTED or UNSAFE
git checkout "$FILE"
git add .planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md
git commit -m "chore(46-NN): revert $CUT_ID — <reason>"

# 7. GATE-01 continuous check
git diff --stat src/llm_sad_sam/linkers/experimental/s_linker19.py \
                src/llm_sad_sam/linkers/experimental/prompts_v5.py \
                src/llm_sad_sam/linkers/experimental/s_linker13_min.py
```

### 8.4 Drop-block specific commands (plan-04 for CUT-DKJ-01)

```bash
# Step DROP — full block removal
cd /mnt/hostshare/ardoco-home/agent-linker
# Save the original block for revert reference
git show HEAD:tests/scratch/prompts_v5.py | sed -n '47,53p' > /tmp/dkj_orig_block.txt
# Remove lines 47–53 (the DOC_KNOWLEDGE_JUDGE_EXAMPLES block)
# Easiest: edit manually and delete the constant assignment + body.
$EDITOR tests/scratch/prompts_v5.py
git diff tests/scratch/prompts_v5.py

# Verify scratch SLinker19 still imports
python -c "import os; os.environ['SAD_SAM_LINKER_SOURCE']='scratch'; from tests.scratch.s_linker19 import SLinker19; print('OK')"
# If this fails with NameError on DOC_KNOWLEDGE_JUDGE_EXAMPLES → the scratch s_linker19's import
# list still references it. The planner must EITHER also delete the import line in scratch/s_linker19.py
# OR leave the constant assigned to "" in scratch/prompts_v5.py to preserve the import.
# Recommendation: leave the constant assigned to "" — minimal scratch s_linker19 edits.

# Run gated test module
SAD_SAM_LINKER_SOURCE=scratch pytest tests/test_s_linker20_prompt_doc_judge.py -x --tb=short
RC=$?

# If RC=0: DROP-BLOCK PASSED. Commit (see §3.2 step 7a).
# If RC≠0: Revert scratch, move to Family A loop (CUT-DKJ-02..04).
```

### 8.5 GATE-06 re-grep helper

```bash
# Extract whole-word tokens from the after-text (length >= 4 to skip prepositions/articles)
AFTER_TEXT="Approve when the sentence treats the entity as an architectural participant, including matching entities."
TOKENS=$(echo "$AFTER_TEXT" | grep -oE '\b[a-zA-Z][a-zA-Z]{3,}\b' | sort -u | paste -sd'|')

# Grep against BENCHMARK_TABOO sections
grep -niwE "$TOKENS" BENCHMARK_TABOO.md | head -30

# Manual review of each hit per v2.1 isolation methodology:
# - Per-dataset section hit → unsafe (unless in Safe SE Textbook Examples)
# - Universal Taboo hit → check second-pass: is the use a generic SE noun? (e.g., bare `component`, bare `the service`)
#   YES → clean (consistent with Phase 45 precedent)
#   NO  → unsafe
# - Safe SE Textbook hit → clean (affirmative-safe list)
```

## 9. Open Questions for Planner

The following are choices the planner must make explicitly in plan-01 and plans 02–07. Each is bounded and well-scoped; none are blocking — they have a recommended default below.

1. **Step-6 prompt-equality gate semantics** (§2.3 — the critical decision). **Recommended: Option A** (gate behind `SAD_SAM_LINKER_SOURCE != "scratch"`). Decide in plan-01.

2. **`reconstruct_validation_inputs` extension** (§2.3, §6.2). The current production code hardcodes `"Validate component references in a software architecture document."`. Extending to a tuple of accepted prefixes is a 3-line change. **Recommended: extend in plan-01** (not in plan-06), with the chosen CUT-VAL-02 replacement pre-decided. Alternative: leave it as-is, accept that CUT-VAL-02 will log `reverted`.

3. **Pleonasm batch replacement vocabulary** (§5.3). Three sites (AMB-02, EXT-01, VAL-02) share the same pattern. Candidates: `components`, `named elements`, `entities`. **Recommended: `components` alone** (collapses to the noun the COMPONENTS slot already names) — but Phase 46 should decide this in plan-01 and document in MINIMIZE-LOG header so all three section plans use the same target.

4. **VAL-03 ↔ COR-01 lexical share vocabulary** (§5.3). Both target "role-referential phrase" / "role-referential noun phrase". Candidate: `noun phrase that refers back`. **Recommended: plan-06 picks, writes to MINIMIZE-LOG, plan-07 reads.** Sequencing: 06 before 07.

5. **DKJ-07 replacement vocabulary** for "architectural tier or technology platform that encompasses multiple elements". Audit suggests `grouping that encompasses multiple elements`. **Recommended: use the audit suggestion** — it preserves the multi-element exclusion semantics without the domain qualifier.

6. **DKJ ordering inside plan-04**: D-03 mandates drop-first, but 45-04 SUMMARY suggests "cheap-then-coverage" ordering CUT-DKJ-04 → 02 → 03 → 05/06 → 01 → 07. **Recommended: follow D-03 strictly** (drop-first), and trial CUT-DKJ-07 last as a separate cut (it's not part of the drop-block protocol).

7. **Whether to add `next-action` pointer column** in MINIMIZE-LOG referencing each kept row to the Phase 47 inline location (CONTEXT Claude's Discretion). **Recommended: yes** — Phase 47 reads MINIMIZE-LOG to know which cuts to apply; explicit pointers reduce inference.

8. **Snapshot regeneration policy.** If a cut somehow causes the `.ambr` file to drift (very unlikely under Option A since parsed output is invariant), should pytest re-write the snapshot or fail loudly? **Recommended: fail loudly** — run pytest WITHOUT `--snapshot-update`. Any drift is a harness compatibility signal, not a "behavior change to bake in."

9. **What if scratch s_linker19.py's import list breaks after dropping a constant?** (e.g., CUT-DKJ-01 drops DOC_KNOWLEDGE_JUDGE_EXAMPLES from prompts_v5.py — the scratch s_linker19 still imports it). Two options:
   (a) Keep the constant in prompts_v5 assigned to `""` (drop-by-empty) so the import line still works.
   (b) Also delete the import line in scratch s_linker19.
   **Recommended: (a)** — minimal scratch s_linker19 edits, the import list stays one block, and the after-text in the prompt is identical (`{DOC_KNOWLEDGE_JUDGE_EXAMPLES}` interpolates as `""`). Phase 47 can collapse the empty interpolation when inlining into s_linker20.py.

10. **Should plan-03 (DKX, zero cuts) be merged into another plan?** Per §5.4, recommended NO — keep section symmetry. But if the planner prefers tighter plan count, plan-03 could fold into plan-01's section setup commentary. **Recommended: standalone plan-03**.

---

## Sources

### Primary (HIGH confidence — verified by reading the code)

- `/mnt/hostshare/ardoco-home/agent-linker/tests/harness/adapters.py` (lines 28-66) — confirms current BUILDERS resolution does a static `from llm_sad_sam.linkers.experimental.s_linker19 import SLinker19`; toggle must short-circuit this resolution.
- `/mnt/hostshare/ardoco-home/agent-linker/tests/harness/inputs.py` (full file) — confirms `reconstruct_validation_inputs` hardcodes the validation opener prefix; this is the principal harness-compatibility risk for CUT-VAL-02. Other reconstructors are opener-agnostic (anchor on `COMPONENTS:`, `DOCUMENT:`, `PROPOSED MAPPINGS:`, `--- Case N:` patterns).
- `/mnt/hostshare/ardoco-home/agent-linker/tests/harness/replay_client.py` (full file) — confirms parsed output depends only on `response_text`, not on prompt; parsed-output snapshots are invariant under prompt cuts.
- `/mnt/hostshare/ardoco-home/agent-linker/tests/test_s_linker20_prompt_ambiguity.py` and `…_validation.py` — confirm two-stage assertion per test (step-6 prompt-equality + step-7 parsed snapshot); informs Option A recommendation.
- `/mnt/hostshare/ardoco-home/agent-linker/tests/__snapshots__/test_s_linker20_prompt_ambiguity.ambr` — confirms snapshot content is parsed dict, not prompt.
- `/mnt/hostshare/ardoco-home/agent-linker/src/llm_sad_sam/linkers/experimental/s_linker19.py` (lines 100-105, 260-378) — confirms import list of prompts_v5 constants and the 6 builder bodies; informs §9 Q9 (import-line drop policy).
- `/mnt/hostshare/ardoco-home/agent-linker/src/llm_sad_sam/linkers/experimental/prompts_v5.py` (lines 1-125) — confirms constant definitions and the `prompts_v5` module docstring lines 5-22 carrying the X.Y.Z behavioral-protection record for CUT-VAL-04.
- `/mnt/hostshare/ardoco-home/agent-linker/.planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md` — primary input. All 19 cut rows verified.
- `/mnt/hostshare/ardoco-home/agent-linker/.planning/phases/46-minimize/46-CONTEXT.md` — locked decisions D-01..D-04 + scope boundary.
- `/mnt/hostshare/ardoco-home/agent-linker/.planning/phases/44-harness/44-CONTEXT.md` (§D-03) — builder→phase-tag→test-module→snapshot-count mapping.
- `/mnt/hostshare/ardoco-home/agent-linker/.planning/REQUIREMENTS.md` — REQ-V264-05/06/07 + GATE-01/06/08.
- `/mnt/hostshare/ardoco-home/agent-linker/.planning/ROADMAP.md` (lines 199-213) — Phase 46 success criteria.
- `/mnt/hostshare/ardoco-home/agent-linker/BENCHMARK_TABOO.md` (structure verified by grep on `^## `) — GATE-06 re-isolation source.

### Secondary (verified by SUMMARY files)

- `45-02-SUMMARY.md` through `45-07-SUMMARY.md` — per-section audit summaries; especially `45-04-SUMMARY.md` for DKJ Family A grep clearance (BookManager/Mgr/MailSender).

## Confidence breakdown

- **Cut row inventory and verdict vocabulary:** HIGH — direct quote from audit doc and CONTEXT.
- **Scratch directory layout:** HIGH — purely a packaging choice; CONTEXT D-01 locks the location.
- **Harness override mechanism:** HIGH — env var is a 5-line change; alternatives considered and rejected with rationale.
- **The step-6 gate semantics finding:** MEDIUM-HIGH — direct reading of code confirms the two assertions; the inference about parsed-output invariance is logically tight given that `replay_parse(response_text)` is purely a function of `response_text`. The recommendation (Option A) is sound but the planner should explicitly confirm and document this in plan-01.
- **Per-cut harness-compatibility predictions (§6.2):** MEDIUM — based on reading each reverse-extractor function; the CUT-VAL-02 prediction is HIGH-confidence because the prefix is hardcoded; other predictions are based on which anchors the reconstructors use.
- **Drop-block decision tree:** HIGH — direct encoding of D-03.
- **Wave/plan decomposition:** HIGH — 8 plans / 3 waves matches Phase 45's shape; no novel scheduling.
- **Effort estimates:** MEDIUM — research-stage priors based on cut count × snapshot count; will calibrate to reality in execution.

**Research date:** 2026-06-08
**Valid until:** Phase 46 close (estimated ~5h45m of execution time across 8 plans).
