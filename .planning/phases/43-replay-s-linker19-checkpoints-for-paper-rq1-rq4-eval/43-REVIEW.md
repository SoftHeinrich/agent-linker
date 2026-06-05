---
phase: 43-replay-s-linker19-checkpoints-for-paper-rq1-rq4-eval
reviewed: 2026-06-05T00:00:00Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - scripts/v2.6.3/__init__.py
  - scripts/v2.6.3/replay_common.py
  - scripts/v2.6.3/replay_s19_to_csv.py
  - scripts/v2.6.3/replay_s19_rq3.py
  - scripts/v2.6.3/replay_s19_rq4.py
  - ../transarc-emp/src/paper/rq1_table.py
  - ../transarc-emp/src/paper/rq3_table.py
  - ../transarc-emp/src/paper/rq4_table.py
findings:
  critical: 0
  warning: 4
  info: 7
  total: 11
status: fixed
fixed_at: 2026-06-05T00:00:00Z
fixed_summary:
  warnings_fixed: 4
  info_fixed: 5
  info_skipped_no_action: 2  # IN-05 (empty __init__.py), IN-06 (time import) — review marked as non-defects
---

# Phase 43: Code Review Report

**Reviewed:** 2026-06-05
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Phase 43 ships replay/format pipeline for s_linker19 phase-cache CSVs and TeX
table generators. All four phase invariants hold:

1. **Zero LLM calls verified.** `replay_common.assert_no_llm_env()` checks
   `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `LLM_BACKEND` (rejects anything
   other than unset/empty/`checkpoint`). All three replay scripts call it at
   the top of `main()`.
2. **transarc-emp stdlib-only verified.** `rq1_table.py`, `rq3_table.py`,
   `rq4_table.py` import only stdlib (`argparse`, `csv`, `os`, `shutil`,
   `sys`, `tempfile`, `pathlib`, `typing`) plus in-repo siblings
   (`metrics_api`, `transarc_error_analysis`, `generate_tables`).
3. **GATE-01 holds.** The replay scripts only `pickle.load()` and never
   `pickle.dump()` or write under `src/llm_sad_sam/`. No mutation of the
   phase cache. `git status` confirms no changes under `src/llm_sad_sam/`.
4. **CSV schema contract holds.** Headers in `replay_s19_rq3.py` / `replay_s19_rq4.py`
   match the README pinned contract exactly:
   - `rq3.csv`: `variant,tp,fp,fn,precision,recall,f1` (4 rows in `Full,
     NoEntityValid, NoCitation, NoValidator` order). Reader expects the same.
   - `rq3_audit.csv`: `validator,killed_gold,killed_spurious,kept_gold,kept_spurious`
     (2 rows `entity,coref`). Reader keys on the same.
   - `rq4.csv`: `linker,tps_caught,unique_tps,fps,delta_f1_if_removed`
     (2 rows `Entity,Coref`). Reader keys on the same.
   - `rq4_upset.csv`: `cell,count` (3 rows `only_E,both,only_C`). Reader uses
     these exact keys.
   - `sad-sam.csv`: `modelElementID,sentence,source`. `rq1_table.py` bridge
     copies as-is and `csv.DictReader` ignores the extra `source` column.
   - `sad-code.csv`: `sentence,codeID`. `rq1_table.py` bridge renames to the
     legacy `modelElementID,codeId` schema (case difference preserved correctly).

Defect summary: 4 Warnings and 7 Info-level findings. Highest-impact issue is
WR-01 (the documented semantics of `delta_f1_if_removed` is a literal set-diff
rather than a true linker-ablation, so the metric over-estimates `dF1` when E
and C share TPs). Schema and code agree, so this is not a Critical schema
violation, but the paper number labelled "dF1 if removed" does not match the
ablation a reader would expect.

## Warnings

### WR-01: `delta_f1_if_removed` is set-difference, not true ablation

**File:** `scripts/v2.6.3/replay_s19_rq4.py:101-102`
**Issue:** The replay computes `f1_without_E = _f1(union - E, G)` and
`f1_without_C = _f1(union - C, G)`. Since `union = E | C`, the expression
`union - E` evaluates to `C - E`, not `C`. A true linker-ablation ("remove
Entity from the pipeline") should yield F1 of `C` alone (Coref still catches
TPs that happen to overlap with Entity's predictions). The literal set-diff
drops every shared TP, so:

* When linkers share TPs, `delta_f1_if_removed` is **inflated** vs the true
  ablation delta. The published number under-states each linker's
  replaceability and overstates its uniqueness.
* The README schema literally documents this set-diff form
  (`f1(E ∪ C) - f1((E ∪ C) - linker)`), so writer and reader agree —
  this is a *semantic* mismatch with CONTEXT D-05's ablation framing,
  not a CSV contract violation.

Recommend either renaming the column to `delta_f1_if_subtracted` (and
updating the caption) or computing the true ablation:

**Fix (true ablation):**
```python
f1_only_C = _f1(C, G)  # F1 when Entity is removed → only Coref survives
f1_only_E = _f1(E, G)
per_linker = {
    "Entity": (..., round(f1_union - f1_only_C, 6)),
    "Coref":  (..., round(f1_union - f1_only_E, 6)),
}
```

If the set-diff is intentional, document it explicitly in the README and the
LaTeX caption in `rq4_table.py:render_rq4_table` so the reviewer understands
it is not a standard ablation.

### WR-02: README contract mis-attributes `rq4_upset.csv` consumer

**File:** `scripts/v2.6.3/README.md:180`
**Issue:** The README pins `rq4_upset.csv` as "Consumed by:
`transarc-emp/src/paper/rq4_upset.py`", but no such file exists. The actual
consumer is `transarc-emp/src/paper/rq4_table.py` (function
`render_rq4_upset`), which reads `rq4_upset.csv` via
`aggregate_backend_upset`. Future maintainers grepping for `rq4_upset.py`
will land nowhere; the executor contract should match reality.
**Fix:** Update the README cell to:
`**Consumed by:** transarc-emp/src/paper/rq4_table.py:render_rq4_upset (Plan 04, UpSet figure).`

### WR-03: `aggregate_backend_upset` silently accepts unknown cell labels

**File:** `transarc-emp/src/paper/rq4_table.py:89-97`
**Issue:** The aggregator pre-seeds `by_cell = {"only_E": 0, "both": 0, "only_C": 0}`
but the loop body uses `by_cell.get(cell, 0) + int(row["count"])` and assigns
back via `by_cell[cell] = ...`. If `rq4_upset.csv` ever contains an
unexpected cell label (e.g., typo `Only_E`), it is silently absorbed into
the aggregate dict but never rendered (downstream only reads the three
expected keys). The contract pins exactly three rows — a hard check
matches the README better and surfaces upstream regressions.
**Fix:**
```python
ALLOWED_CELLS = {"only_E", "both", "only_C"}
for row in _read_csv(path):
    cell = row["cell"]
    if cell not in ALLOWED_CELLS:
        raise ValueError(f"unexpected cell label {cell!r} in {path}")
    by_cell[cell] += int(row["count"])
```

### WR-04: `compute_validator_audit_counts` silently coerces missing decisions to "rejected"

**File:** `scripts/v2.6.3/replay_s19_rq3.py:112-129`
**Issue:** `dec = decisions.get(key); approved = bool(dec and dec.get("approved", False))`.
If a candidate in `layer3.candidates` has no entry in `layer3.decisions`,
the audit treats it as **rejected**. In v2.6.2 s_linker19 always writes a
decision dict for every candidate, but the silent fallback hides any
schema regression (a future change that yields decisions only for the
validated subset would silently corrupt audit counts). Should hard-fail
or warn, not silently treat missing-decision as "killed".
**Fix:**
```python
dec = decisions.get(key)
if dec is None:
    raise KeyError(
        f"missing validator decision for candidate {key} in layer3.decisions; "
        f"s_linker19 contract expects a decision dict per candidate"
    )
approved = bool(dec.get("approved", False))
```

## Info

### IN-01: Unused stdlib import (`os`) in rq1_table.py

**File:** `transarc-emp/src/paper/rq1_table.py:48`
**Issue:** `import os` is never used (search confirms only the literal
`os` in `import os` line). Doesn't affect correctness; removes a tiny bit
of noise.
**Fix:** Drop the `import os` line.

### IN-02: Unused typing import (`Tuple`) in rq3_table.py

**File:** `transarc-emp/src/paper/rq3_table.py:28`
**Issue:** `from typing import Dict, List, Tuple` — `Tuple` is never used.
**Fix:** `from typing import Dict, List`.

### IN-03: Redundant `--all` flag duplicates default behavior

**File:** `scripts/v2.6.3/replay_s19_to_csv.py:141-142` (and the same in
`replay_s19_rq3.py:177-178`, `replay_s19_rq4.py:162-163`)
**Issue:** All three CLIs already default `--backend` and `--project` to
`"all"`, so running with no flags is identical to running with `--all`.
The `--all` shortcut is documented in the README (`--all` examples) but is
functionally a no-op. Either drop the flag or document it as a no-op
convenience alias.
**Fix:** Either remove `--all` or in `main()` print a one-line note when
both are unset that "all" is the default.

### IN-04: `BACKEND_DISPLAY` exported but unused locally

**File:** `scripts/v2.6.3/replay_common.py:48, 169`
**Issue:** `BACKEND_DISPLAY` is defined in `replay_common.py` and listed in
`__all__`, but no replay script in this repo imports it. The transarc-emp
formatters duplicate the same mapping locally (`rq1_table.py:72`,
`rq3_table.py:34`, `rq4_table.py:33`). Either drop from `replay_common`
or have downstream consumers import the canonical copy.
**Fix:** Remove `BACKEND_DISPLAY` from `replay_common.py` or refactor
downstream to import it (the latter conflicts with the stdlib-only
constraint of transarc-emp; the former is simpler).

### IN-05: Empty `__init__.py`

**File:** `scripts/v2.6.3/__init__.py` (0 bytes)
**Issue:** The file exists but is empty. Replay scripts use
`sys.path.insert(0, str(Path(__file__).resolve().parent))` and then
`from replay_common import ...` instead of treating `scripts.v2.6.3`
as a package — so `__init__.py` doesn't actually make the directory
importable as a package. Not a defect, just dead.
**Fix:** Either delete the file or use proper package imports
(`from scripts.v2_6_3.replay_common import ...`).

### IN-06: `time` import is only used for end-of-run elapsed message

**File:** `scripts/v2.6.3/replay_s19_to_csv.py:28`,
`scripts/v2.6.3/replay_s19_rq3.py:30`, `scripts/v2.6.3/replay_s19_rq4.py:36`
**Issue:** The elapsed-time printout is fine but slightly noisy in CI logs
(the scripts are deterministic; replay time is not a metric). Not a
defect — just an observation that `time.time()` is wall-clock and the
printout adds a small non-determinism to stdout (different runs print
different `dt`). Doesn't affect CSV output.
**Fix:** No action required; if pure determinism in CI logs is desired,
gate the `[replay-rqN] wrote ... in ...s` line behind a `--quiet`/`--verbose`
flag.

### IN-07: Hardcoded absolute path in `replay_common.py`

**File:** `scripts/v2.6.3/replay_common.py:63-72`
**Issue:** `_BENCHMARK = Path("/mnt/hostshare/ardoco-home/ardoco/core/tests-base/...")`
is an absolute container-relative path. If the repository is checked out
on a different host or in a different layout, `load_gold_links` will fail
with `FileNotFoundError`. The path is the canonical location per the
project's docker mount, but a `BENCHMARK_ROOT` env-var override (with the
current value as default) would make these scripts portable for a
reviewer cloning the repo.
**Fix:**
```python
_BENCHMARK = Path(
    os.environ.get(
        "ARDOCO_BENCHMARK_ROOT",
        "/mnt/hostshare/ardoco-home/ardoco/core/tests-base/src/main/resources/benchmark",
    )
)
```

---

## Inline summary

| Severity | Count |
| -------- | ----- |
| Critical | 0     |
| Warning  | 4     |
| Info     | 7     |
| **Total**| **11**|

Highest-impact: **WR-01** (`delta_f1_if_removed` is a literal set-diff, not
a true linker ablation; this matches the README schema but conflicts with
the natural reading of CONTEXT D-05; recommend renaming the column or
switching to true ablation before the paper draft locks numbers).

All four phase invariants (zero LLM, stdlib-only in transarc-emp, GATE-01
byte-equality, CSV schema match) hold.

---

_Reviewed: 2026-06-05_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
