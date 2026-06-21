---
phase: 50-extract
reviewed: 2026-06-21T00:00:00Z
depth: standard
files_reviewed: 1
files_reviewed_list:
  - scripts/extract_s20union_caches.py
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: issues_found
---

# Phase 50: Code Review Report — `scripts/extract_s20union_caches.py`

## Summary

A standalone, read-only pickle→neutral-JSON extractor for the 30 frozen `s_linker20_union` caches. Verified live: deterministic (byte-identical re-run), 30/30 faithfulness PASS, exit 0, every output JSON loads under clean stdlib `json` with no linker import, and the dup-(s,c) coref cell is preserved as lists. The `pickle.load` over trusted frozen caches is by design (per threat model) and is **not** flagged. Field mappings for `CandidateLink`/`SadSamLink`/`AliasEntry`/`EvidenceBundle` are complete and produce only JSON-native values. **No BLOCKER found** — the script is read-only, has no untrusted input surface, and meets its stated gates.

The findings below are robustness/quality defects concentrated in the *verification* layer: the oracle that is supposed to be the safety net is narrower and more failure-tolerant than the phase claims ("this JSON is the contract for every later phase").

## Warnings

### WR-01: Faithfulness oracle verifies only `final.links` — most of the schema is unchecked

**File:** `scripts/extract_s20union_caches.py:249-258`, `306-330`
**Issue:** The primary gate compares only `{(s,c,source)}` of `final.links` against the CSV (lines 249-256), and `rederive_final` (306-330) additionally exercises `entity.validated` + `coref.validated`. Everything else in the emitted JSON — `entity.candidates`, `entity.decisions`, `entity.evidence_bundles`, `knowledge.model_knowledge.ambiguous_names`, `knowledge.doc_knowledge.aliases`, `coref.raw`, `coref.metadata`, `final.provenance`, `audit.phase_metrics` — is serialized with **zero verification**. A field-mapping or strip bug in those blocks (e.g. a wrong key in `_candidate_link_to_record`, an aliases term/component swap, an over-eager `raw_resolution` strip) would ship silently while the script still prints "30/30 PASS". Note also the CSV oracle and the extract's `final.links` both derive from the *same* in-memory `final` list at run time, so the primary gate mostly proves "the `final` pickle round-trips," not "the extract is faithful."
**Fix:** At minimum add cheap structural invariants that are independent of `final`: assert `len(entity.candidates) == len(layer3["candidates"])`, `len(coref.raw) == len(layer4["coref_raw"])`, and that every `final.links` (s,c) with `from_coref=False` has a matching `entity.decisions` entry with `approved=True`. Document explicitly in the module docstring that only `final.links` is oracle-verified so downstream phases don't over-trust the rest.

### WR-02: Primary CSV oracle silently drops rows with fewer than 5 fields

**File:** `scripts/extract_s20union_caches.py:256`
**Issue:** `csv_set = {(int(row[0]), row[1], row[4]) for row in csv_rows if len(row) >= 5}`. The `if len(row) >= 5` guard silently discards any oracle row that is short — including a *truncated* row that would represent a genuine missing/corrupt link. Since the oracle is the safety net, silently shrinking it can turn a real divergence into a false PASS. This also contradicts the fail-loud philosophy used for the JSON writer (`allow_nan=False`, no `default=`).
**Fix:** Don't silently filter data rows. Skip only genuinely blank rows and raise on malformed ones:
```python
csv_set = set()
for row in csv_rows:
    if not row:
        continue
    if len(row) < 5:
        raise ValueError(f"malformed oracle row in {csv_path}: {row!r}")
    csv_set.add((int(row[0]), row[1], row[4]))
```

### WR-03: Success exit code is decoupled from the advertised 30/30 coverage (magic `30`)

**File:** `scripts/extract_s20union_caches.py:434-439` (and prints `434-435`)
**Issue:** `main()` returns `1` only when `any_missing or n_fail > 0` and never asserts `n_extracted == 30` or `n_pass == 30`. The literal `30` in the printed coverage/summary lines is hardcoded, not derived from `len(MATRIX) * len(RUNS) * len(PROJECTS)`. Today the matrix multiplies to exactly 30 so `n_extracted < 30 ⟺ any_missing` and the exit code is correct. But if the matrix lists are ever edited so they no longer total 30, the script prints a stale "`N/30`" and can still exit **0** with sub-30 coverage — a silent under-coverage that a downstream consumer keying on exit code would not detect.
**Fix:** Derive the expected count and gate on it:
```python
EXPECTED = len(MATRIX) * len(RUNS) * len(PROJECTS)
print(f"\n{n_extracted}/{EXPECTED} cells extracted", flush=True)
print(f"{n_pass}/{EXPECTED} PASS", flush=True)
if any_missing or n_fail > 0 or n_pass != EXPECTED:
    return 1
return 0
```

### WR-04: Secondary ablation cross-check silently degrades to a no-op; fragile glob/first-key assumptions

**File:** `scripts/extract_s20union_caches.py:262-295` (esp. `294-295`, `293`), `363-366`
**Issue:** The advisory cross-check swallows *all* exceptions (`except Exception as exc:` at 294) and only records a string into `secondary_detail`, leaving `secondary_ok=True`. If `proj_data` is ever not a dict, or the ablation JSON's first key is not the project, the check becomes a silent no-op while still reporting `secondary_ok`. It also relies on `glob.glob(...)[0]` (363-366) whose ordering is OS-dependent — currently safe only because exactly one `ablation_*.json` exists per cell (verified) — and `break  # Only first project key` (293) assumes a single-project file. None of this affects output determinism (ablation data isn't serialized), but the cross-check provides weaker assurance than it appears to.
**Fix:** Sort the glob and pick deterministically (or assert exactly one match); narrow the `except` to `(json.JSONDecodeError, OSError, KeyError, TypeError)`; and look up the project by the cell's known `project` key rather than iterating + `break`. If a cross-check can't run, surface it (set `secondary_ok=False` or print a warning) instead of swallowing it.

## Info

### IN-01: Module-level import-time side effects make the module unsafe to import

**File:** `scripts/extract_s20union_caches.py:25, 28-29, 34`
**Issue:** Merely importing the module runs `sys.stdout.reconfigure(line_buffering=True)` (25), `sys.path.insert` (28), `os.chdir(_ROOT)` (29), and imports the linker (34). `os.chdir` mutates global process state, and the plan's own verify snippets `import extract_s20union_caches as ex`, inheriting all of these. `reconfigure` would also raise `AttributeError` if `sys.stdout` is a stream lacking it (e.g. some capture wrappers).
**Fix:** Move `reconfigure`/`chdir` into `main()` (guard `reconfigure` with `hasattr`); keep only `sys.path.insert` + the registration import at module scope.

### IN-02: `layer2` loaded then discarded; redundant `cell["layer1"]` lookups

**File:** `scripts/extract_s20union_caches.py:67`, `402-404`
**Issue:** `load_cell` reads `layer2.pkl` (67) but `to_neutral` never references it (intentional per plan, but it's a dead load worth a one-line comment). In `main()`, `meta` uses `cell["layer1"].get("elapsed_s")` (402) and `cell["layer2"]…` (403) even though local `l1`/`lf` already alias `cell["layer1"]`/`cell["final"]` — minor inconsistency.
**Fix:** Add a comment noting layer2 is loaded only for the existence/coverage gate; use the local `l1` alias for `elapsed_s` consistency.

### IN-03: `keyed_to_records` lets a value-dict key shadow the tuple key

**File:** `scripts/extract_s20union_caches.py:82`
**Issue:** `{"s": s, "c": c, **v}` — because `**v` is spread last, a value dict containing `"s"` or `"c"` would silently overwrite the tuple-derived key. Verified currently safe (decisions, evidence_bundles, coref_decisions value dicts have no `s`/`c` fields), but it's an unguarded latent footgun for future pickle schema changes.
**Fix:** Either spread first (`{**v, "s": s, "c": c}`) so the structural key always wins, or assert `"s" not in v and "c" not in v`.

---
_Reviewer: Claude (gsd-code-reviewer) · Depth: standard_
_Reviewed file: `scripts/extract_s20union_caches.py`_
