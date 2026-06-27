---
phase: 50-extract
verified: 2026-06-21T00:00:00Z
status: passed
score: 6/6 must-haves verified
overrides_applied: 0
---

# Phase 50: EXTRACT Verification Report

**Phase Goal:** Convert every frozen s_linker20_union per-run phase_cache into neutral, stdlib-loadable JSON so the downstream bundle never needs the linker classes or pickle.
**Verified:** 2026-06-21
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Running scripts/extract_s20union_caches.py writes 30 neutral JSON files (2 backends x 3 runs x 5 projects) with zero missing cells | VERIFIED | Script executed live: exit 0, printed "30/30 cells extracted"; `find results/v2.6.6_extracts -name '*.json' | wc -l` = 30 |
| 2 | Each output JSON has top-level blocks entity{candidates,validated,decisions}, coref{raw,validated,decisions}, knowledge{model_knowledge,doc_knowledge}, final{links,provenance} | VERIFIED | python3 schema check on gpt/run1/mediastore.json: all required top-level and sub-block keys present; entity keys: candidates/decisions/evidence_bundles/validated; coref keys: decisions/metadata/raw/validated; final keys: links/provenance |
| 3 | For all 30 cells the final-link set in the JSON equals that run's own *_links.csv on (sentence,component_id,source); script prints 30/30 PASS | VERIFIED | Script live run: "30/30 PASS" on stdout, exit code 0; dup-key regression cells gpt/run2/teammates (links=58) and gpt/run3/teammates (links=58) both PASS |
| 4 | Every emitted JSON loads via clean stdlib json.load in an interpreter that imports NO llm_sad_sam module (neutral / stdlib-loadable) | VERIFIED | python3 -c "import json; data=json.load(open(p))" with 'llm_sad_sam' not in sys.modules assertion passed for mediastore.json |
| 5 | Re-running the script produces byte-identical JSON (deterministic) | VERIFIED | cp -r results/v2.6.6_extracts /tmp/x6_check_verif; second run; diff -r produced empty output |
| 6 | No linker source file is modified (GATE-01): s_linker20_union.py and data_types_v2.py byte-stable | VERIFIED | feat commit e0d6acf touches only scripts/extract_s20union_caches.py (1 file, 443 insertions); git diff --name-only -- src/llm_sad_sam/ is empty in working tree; s_linker20_union.py last modified in commit a684c2c (phase 48, unrelated) |

**Score:** 6/6 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/extract_s20union_caches.py` | pickle->neutral-JSON extractor with built-in faithfulness + coverage gate; min 150 lines | VERIFIED | 443 lines; contains load_cell, to_neutral, write_json, faithfulness, rederive_final, main() |
| `results/v2.6.6_extracts/` | 30 neutral JSON cell extracts at gpt+sonnet/run1-3/project.json; contains gpt/run1/mediastore.json | VERIFIED | Exactly 30 JSON files on disk (gitignored by /results/* rule); gpt/run1/mediastore.json present and loads cleanly |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| scripts/extract_s20union_caches.py | llm_sad_sam.linkers.experimental.s_linker20_union | module import that registers AliasEntry for unpickling | VERIFIED | Line 34: `import llm_sad_sam.linkers.experimental.s_linker20_union  # noqa: F401` |
| scripts/extract_s20union_caches.py | results/v2.6.5_s20union(_sonnet)/.../phase_cache | pickle.load over the two asymmetric cache roots | VERIFIED | Line 70: `cell[name] = pickle.load(f)` inside load_cell(); MATRIX declares both asymmetric roots correctly |
| results/v2.6.6_extracts/<backend>/<run>/<project>.json final.links | s_linker20_union_<project>_links.csv | set-equality faithfulness oracle on (sentence,component_id,source) | VERIFIED | faithfulness() at lines 233-303; 30/30 PASS confirmed on live run including dup-key cells |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| results/v2.6.6_extracts/*/run*/project.json | final.links | layer4 coref_raw/coref_validated lists + final pkl final list | Yes — read directly from frozen per-run phase_cache pickles; gpt/run1/mediastore has 29 final links, gpt/run2/teammates has 58, all non-empty | FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Script exits 0, prints 30/30 PASS | `python3 scripts/extract_s20union_caches.py` | exit 0; "30/30 cells extracted"; "30/30 PASS" | PASS |
| Determinism: second run byte-identical | `diff -r results/v2.6.6_extracts /tmp/x6_check_verif` after re-run | empty diff | PASS |
| Neutrality: sample JSON loads with no llm_sad_sam import | `python3 -c "import json,sys; json.load(open(p)); assert 'llm_sad_sam' not in sys.modules"` | NEUTRALITY_OK | PASS |
| Schema: all required keys present | python3 key-set assertions on gpt/run1/mediastore.json | top_keys: audit/coref/entity/final/knowledge/meta; all sub-keys present | PASS |
| Aliases as list-of-records with term/component/scope | python3 isinstance + key check | 5 alias records; first keys: component/scope/term | PASS |
| ambiguous_names is sorted list | python3 assert amb == sorted(amb) | 4 items, sorted | PASS |
| coref.raw / coref.validated are lists (not dicts) | isinstance assertions | coref.raw=11 list, coref.validated=9 list for mediastore | PASS |
| Dup-key regression cells preserve list form | gpt/run2/teammates and gpt/run3/teammates checks | run2: raw=49/val=31/links=58; run3: raw=49/val=34/links=58 | PASS |
| json.dumps without default= succeeds | `json.dumps(data, allow_nan=False, sort_keys=True)` on sonnet/run3/teastore.json | len=53723, no exception | PASS |
| GATE-01: no linker file modified | git show --name-only e0d6acf; git diff --name-only -- src/llm_sad_sam/ | Only scripts/extract_s20union_caches.py in commit; working tree diff empty | PASS |
| GATE-06: no benchmark vocabulary | grep for component/alias gold strings | NO_GOLD_VOCAB_FOUND; only PROJECTS dir identifiers | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| EXTRACT-01 | 50-01-PLAN.md | Extraction script dumps every s_linker20_union per-run phase_cache (layer1-4 + final) into neutral, stdlib-loadable JSON — both backends, all N runs, all 5 projects | SATISFIED | scripts/extract_s20union_caches.py: 443 lines, all 5 layers loaded; 30 JSONs produced; neutrality verified by clean json.load without llm_sad_sam import |
| EXTRACT-02 | 50-01-PLAN.md | Extracted JSON captures every ablation-relevant field: entity candidates/validated/decisions (incl. p1/p2), coref coref_raw/coref_validated/coref_decisions, knowledge layer (model_knowledge + doc_knowledge), final links with per-link source/provenance | SATISFIED | Schema verified on live JSON: entity{candidates,validated,decisions,evidence_bundles}; coref{raw,validated,decisions,metadata}; knowledge{model_knowledge.ambiguous_names,doc_knowledge.aliases as list-of-records}; final{links with source field, provenance} |
| EXTRACT-03 | 50-01-PLAN.md | Extraction faithfulness verified — final-link set re-derived from each extract equals that run's own *_links.csv / ablation_*.json, per project x run x backend | SATISFIED | faithfulness() function at lines 233-303; rederive_final() guard at 306-330; 30/30 PASS confirmed live including dup-key regression cells (gpt/run2+run3/teammates) |

All three requirements declared for Phase 50 are SATISFIED. No orphaned requirements: EXTRACT-01/02/03 are the only requirements mapped to Phase 50 in REQUIREMENTS.md.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| scripts/extract_s20union_caches.py | — | TBD/FIXME/XXX | None found | No blocker debt markers |
| scripts/extract_s20union_caches.py | — | TODO/HACK/PLACEHOLDER | None found | — |

No blocker anti-patterns. Four advisory warnings exist (WR-01 through WR-04) per REVIEW.md; all are robustness/quality items in the verification layer with zero functional impact on the phase goal:

- **WR-01** (WARNING): Faithfulness oracle only verifies final.links; other schema blocks (candidates, evidence_bundles, etc.) have no cross-check. Does not affect the 30/30 PASS result or the downstream contract.
- **WR-02** (WARNING): Silent `if len(row) >= 5` guard in CSV oracle. All 30 cells PASS and no truncated rows exist in the frozen data; no functional failure observed.
- **WR-03** (WARNING): Hardcoded `30` in print lines; exit condition does not assert `n_pass == 30`. Functionally correct today; cosmetic fragility if MATRIX ever changes.
- **WR-04** (WARNING): Secondary ablation cross-check swallows exceptions. Advisory check only; primary CSV gate is authoritative.

---

### Human Verification Required

None. This phase is fully machine-verifiable offline. All must-haves are confirmed by automated checks above.

---

### Gaps Summary

No gaps. All 6 observable truths are verified, all artifacts exist and are substantive and wired, all 3 requirements are satisfied, and no blocker anti-patterns were found. The phase goal is achieved.

---

_Verified: 2026-06-21T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
