---
phase: 10-scaffolding
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - tests/fixtures/v2_0_baseline.json
  - tests/test_v20_baseline_regression.py
autonomous: true
requirements:
  - GATE-02
user_setup: []

must_haves:
  truths:
    - "A pinned v2.0 baseline JSON fixture exists at tests/fixtures/v2_0_baseline.json containing the macro F1 (and per-dataset F1) for every entry in CANONICAL_VARIANTS as of v2.0 close"
    - "pytest tests/test_v20_baseline_regression.py exits 0 against current s_linker13"
    - "The regression test compares every CANONICAL_VARIANTS entry's per-dataset F1 (precision/recall/F1) against the pinned baseline JSON within a documented float tolerance (abs diff < 1e-4)"
    - "Test docstring states the test must be run before any promotion (GATE-02 contract)"
  artifacts:
    - path: "tests/fixtures/v2_0_baseline.json"
      provides: "Pinned v2.0 baseline F1 per (variant, dataset) for regression comparison"
      contains: "s_linker13"
    - path: "tests/test_v20_baseline_regression.py"
      provides: "GATE-02 frozen-compat regression test for CANONICAL_VARIANTS"
      contains: "def test_"
  key_links:
    - from: "tests/test_v20_baseline_regression.py"
      to: "tests/fixtures/v2_0_baseline.json"
      via: "json.load against a pathlib-relative path"
      pattern: "v2_0_baseline\\.json"
    - from: "tests/test_v20_baseline_regression.py"
      to: "run_ablation.py CANONICAL_VARIANTS"
      via: "import of CANONICAL_VARIANTS (or hardcoded mirror with a sync assertion)"
      pattern: "CANONICAL_VARIANTS"
---

<objective>
Establish the GATE-02 frozen-compat regression safeguard for v2.1. Snapshot the v2.0 macro-F1
baseline for every variant in `CANONICAL_VARIANTS` into a stable fixture file, then ship a
pytest regression test that asserts every entry stays equivalent to that fixture within
harness float tolerance.

Purpose: every subsequent v2.1 trim / cleanup / promotion phase (11-13) must run this test
before any change ships. It is the hard contract that nothing currently runnable breaks.

Output: `tests/fixtures/v2_0_baseline.json` (pinned snapshot) + `tests/test_v20_baseline_regression.py`
(comparator that exits 0 for the unchanged tree and fails loudly the moment any
CANONICAL_VARIANTS entry diverges).
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/REQUIREMENTS.md
@.planning/phases/10-scaffolding/10-CONTEXT.md
@tests/test_s_linker13d_parity.py
@run_ablation.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Snapshot v2.0 baseline F1 fixture</name>
  <files>tests/fixtures/v2_0_baseline.json</files>
  <read_first>
    - run_ablation.py (read the CANONICAL_VARIANTS list at lines 40-86 — the test must cover every entry verbatim)
    - .planning/REQUIREMENTS.md (GATE-02 wording — "every variant in CANONICAL_VARIANTS produces F1 identical to the v2.0 baseline JSON")
    - .planning/phases/10-scaffolding/10-CONTEXT.md (specifics block: "snapshot it now under a stable path (e.g. tests/fixtures/v2_0_baseline.json)")
    - results/ablation_results/ablation_20260531_063446.json (most recent full s_linker13 sweep — sample to understand the per-(dataset, variant) schema: keys are dataset names mediastore/teastore/teammates/bigbluebutton/jabref, each containing variant -> {P, R, F1, tp, fp, fn, ...})
    - .planning/milestones/v2.0-MILESTONE-AUDIT.md (confirms v2.0 final macro F1 = 0.9509 and identifies what closed v2.0)
  </read_first>
  <action>
    Locate the canonical v2.0 baseline ablation JSON(s) under results/ablation_results/ and produce a single consolidated fixture at tests/fixtures/v2_0_baseline.json.

    Steps:
    1. List CANONICAL_VARIANTS from run_ablation.py (current count: 43 entries including i1/i2/i3 + the full s_linker* chain and the s_linker13g_* rejected baselines).
    2. For each entry in CANONICAL_VARIANTS, locate the most recent ablation_*.json in results/ablation_results/ that contains a measurement for that variant on each of the 5 datasets (mediastore, teastore, teammates, bigbluebutton, jabref). The canonical s_linker13 baseline is the v2.0-close sweep (audit lists ablation_20260529_215932.json as the s_linker13 final sweep with macro F1 0.9509 — use grep on the file to confirm before pinning). For the EXT-01 s_linker13g_* family use ablation_ext01_*.json files.
    3. Build a consolidated JSON fixture with this exact schema:
       {
         "schema_version": "1.0",
         "frozen_at": "v2.0-close 2026-05-31",
         "tolerance_abs_f1": 1e-4,
         "datasets": ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"],
         "variants": {
           "<variant_canonical_name>": {
             "source_file": "results/ablation_results/<filename>.json",
             "per_dataset": {
               "mediastore": {"P": <float>, "R": <float>, "F1": <float>},
               "teastore": {...},
               "teammates": {...},
               "bigbluebutton": {...},
               "jabref": {...}
             },
             "macro_f1": <float computed as mean of the 5 F1 values>
           },
           ...
         }
       }
    4. If any CANONICAL_VARIANTS entry has no sweep covering all 5 datasets in results/ablation_results/, list it under a top-level "missing" array in the fixture with explicit per-dataset null markers and an inline note. Do NOT fabricate numbers — the test must skip those entries with an explicit xfail marker rather than asserting against invented values.
    5. Write the fixture using the Write tool (NOT a heredoc).

    Concrete value pins (cross-check against the produced fixture; do not hardcode in the test):
      - s_linker13 macro F1 = 0.9509 (per v2.0 audit). Round-trip must reproduce this from the per_dataset values.
      - cross-model baseline for GATE-01 cross-model = 0.9077 (gpt-5.4) — this lives in PROJECT.md, NOT in this Claude Sonnet fixture (per requirement: GATE-02 fixture pins the Claude Sonnet baseline only).

    Zero benchmark leakage: the fixture stores only numeric F1 values and dataset names — no component names, no prompt phrasing. GATE-06 compliant.
  </action>
  <verify>
    <automated>python -c "import json,pathlib; d=json.loads(pathlib.Path('tests/fixtures/v2_0_baseline.json').read_text()); assert d['schema_version']=='1.0'; assert set(d['datasets'])=={'mediastore','teastore','teammates','bigbluebutton','jabref'}; assert 's_linker13' in d['variants']; v=d['variants']['s_linker13']; assert abs(v['macro_f1']-0.9509)<5e-3, v['macro_f1']; print('OK', len(d['variants']),'variants pinned')"</automated>
  </verify>
  <acceptance_criteria>
    - File tests/fixtures/v2_0_baseline.json exists and is valid JSON
    - schema_version == "1.0"
    - "datasets" list equals exactly ["mediastore","teastore","teammates","bigbluebutton","jabref"]
    - "variants" dict contains an entry for every name in run_ablation.CANONICAL_VARIANTS (either under "variants" with numeric F1s or under "missing" with explicit nulls; combined coverage = 100% of CANONICAL_VARIANTS)
    - variants["s_linker13"]["macro_f1"] reproduces 0.9509 within 5e-3 (allows for rounding from per-dataset stored F1s)
    - Each per-dataset entry has the three keys "P","R","F1" all of type float
    - File contains zero benchmark component names (greppable: grep -E "Reencoding|FreeSWITCH|kurento|Redis|Recording" tests/fixtures/v2_0_baseline.json returns nothing — these may legitimately appear inside source-file ablation JSONs but must NOT be copied into our fixture)
  </acceptance_criteria>
  <done>
    Fixture file committed; macro F1 for s_linker13 round-trips to 0.9509 ± 5e-3; every CANONICAL_VARIANTS entry accounted for (pinned or explicitly missing).
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Write GATE-02 regression test</name>
  <files>tests/test_v20_baseline_regression.py</files>
  <read_first>
    - tests/fixtures/v2_0_baseline.json (the fixture produced in Task 1 — its schema drives the test)
    - tests/test_s_linker13d_parity.py (existing test pattern — pytest parametrize style, module-level imports from llm_sad_sam.*)
    - run_ablation.py lines 40-86 (CANONICAL_VARIANTS) and 88-360 (VARIANT_SPECS — needed to know each variant's module + class for any future "live run" assertion; this test does NOT run the linkers, it asserts the fixture matches the current declared CANONICAL_VARIANTS surface)
    - .planning/REQUIREMENTS.md (GATE-02 exact wording)
    - .planning/STATE.md Standing Gates section
  </read_first>
  <behavior>
    - Test 1: imports CANONICAL_VARIANTS from run_ablation, loads the fixture, asserts set(CANONICAL_VARIANTS) == set(fixture["variants"].keys()) | set(fixture.get("missing", [])). Drift between the registry and the fixture is treated as a GATE-02 failure with an actionable error message ("Variant X added to CANONICAL_VARIANTS but missing from tests/fixtures/v2_0_baseline.json — snapshot it before promotion.").
    - Test 2: fixture-internal consistency — every pinned variant has all 5 datasets, all three keys (P, R, F1), all floats in [0.0, 1.0]; macro_f1 stored value matches mean(F1 across the 5 datasets) within 1e-6.
    - Test 3: s_linker13 anchor — variants["s_linker13"]["macro_f1"] ∈ [0.9509 - 5e-3, 0.9509 + 5e-3] (anchors to the audited v2.0-close value).
    - Test 4: tolerance contract — fixture["tolerance_abs_f1"] == 1e-4; this is the float tolerance future live-run assertions will use.
    - Test 5 (xfail-allowed): for each entry in fixture.get("missing", []), pytest.xfail with a message naming the variant and what evidence is missing — this is a placeholder slot for a future fixture refresh, not a hard fail.
    - Test 6 (GATE-02 contract docstring): the module docstring must contain the literal strings "GATE-02" AND "frozen-compat" AND "CANONICAL_VARIANTS" AND "v2.0 baseline JSON" so grep-based audits can find it.
  </behavior>
  <action>
    Write tests/test_v20_baseline_regression.py implementing the behavior above.

    Required imports:
      - import json, pathlib, statistics
      - import pytest
      - from run_ablation import CANONICAL_VARIANTS

    Required module constants:
      - FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "v2_0_baseline.json"
      - DATASETS = ("mediastore", "teastore", "teammates", "bigbluebutton", "jabref")

    Module docstring must include this exact paragraph (so the GATE-02 contract is self-documenting and grep-discoverable):

      "GATE-02 frozen-compat regression test. Asserts every variant in CANONICAL_VARIANTS
      stays equivalent to the pinned v2.0 baseline JSON at tests/fixtures/v2_0_baseline.json.
      This test must pass before any v2.1 promotion. See REQUIREMENTS.md GATE-02 and
      STATE.md Standing Gates."

    Implementation notes:
      - Use a module-scoped pytest fixture that loads the JSON once.
      - For Test 2, use @pytest.mark.parametrize over the list of pinned variant names.
      - Test 3 must use math.isclose(actual, 0.9509, abs_tol=5e-3) (NOT == 0.9509).
      - The missing-variants xfail loop must call pytest.xfail (not pytest.skip) so it shows up as XFAIL in the report — making missing snapshots visible without breaking CI.
      - Do NOT execute any linker code in this test — running 43 variants × 5 datasets would take hours. This is a static fixture-vs-registry consistency test. Live-run comparison is wired in Phase 13 (PROMPT-03).
      - Do NOT import any linker module (importing linker modules drags in LLMClient and triggers .env loading — keep this test pure and fast).
      - No fenced code blocks anywhere in the file body except inside docstrings if needed for examples.

    Performance target: full suite < 1 second wall-clock.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 && python -m pytest tests/test_v20_baseline_regression.py -v 2>&1 | tail -20 && python -c "import re,pathlib; t=pathlib.Path('tests/test_v20_baseline_regression.py').read_text(); assert 'GATE-02' in t and 'frozen-compat' in t and 'CANONICAL_VARIANTS' in t and 'v2.0 baseline JSON' in t, 'GATE-02 contract docstring missing required tokens'; print('docstring OK')"</automated>
  </verify>
  <acceptance_criteria>
    - File tests/test_v20_baseline_regression.py exists
    - `python -m pytest tests/test_v20_baseline_regression.py` exits 0 (XFAIL entries acceptable, FAIL / ERROR is not)
    - Module docstring contains the literal substrings: "GATE-02", "frozen-compat", "CANONICAL_VARIANTS", "v2.0 baseline JSON"
    - Test imports CANONICAL_VARIANTS from run_ablation (not a hardcoded copy)
    - Test does not import any module under llm_sad_sam.linkers.* (grep -c "from llm_sad_sam.linkers" tests/test_v20_baseline_regression.py returns 0)
    - Full pytest run for this file completes in under 5 seconds
    - At least one test parametrize covers s_linker13 anchor F1 = 0.9509 ± 5e-3
    - Adding a fake variant "fake_variant_xyz" to CANONICAL_VARIANTS (test this by patching in a throwaway branch and discarding) causes Test 1 to fail with a message containing "fake_variant_xyz" (manual smoke check — document in summary, do not commit the patch)
  </acceptance_criteria>
  <done>
    Test file exists, full pytest run passes (XFAILs allowed for missing-variant slots), GATE-02 contract docstring verifiable by grep, total runtime under 5 seconds.
  </done>
</task>

</tasks>

<verification>
1. `python -m pytest tests/test_v20_baseline_regression.py -v` exits 0
2. `cat tests/fixtures/v2_0_baseline.json | python -m json.tool > /dev/null` exits 0 (valid JSON)
3. `python -c "from run_ablation import CANONICAL_VARIANTS; import json,pathlib; d=json.loads(pathlib.Path('tests/fixtures/v2_0_baseline.json').read_text()); missing = set(CANONICAL_VARIANTS) - (set(d['variants']) | set(d.get('missing',[]))); assert not missing, missing; print('coverage 100%')"` exits 0
4. `grep -c "GATE-02" tests/test_v20_baseline_regression.py` returns >= 1
</verification>

<success_criteria>
- Pinned baseline fixture covers 100% of CANONICAL_VARIANTS entries (pinned or explicitly marked missing)
- Regression test enforces fixture-vs-registry consistency and anchors s_linker13 macro F1 to v2.0-close 0.9509 ± 5e-3
- Test runs in under 5 seconds and imports zero linker modules
- GATE-02 contract is self-documented in the test module docstring (grep-discoverable)
- Zero benchmark component names leaked into the fixture file
</success_criteria>

<output>
After completion, create `.planning/phases/10-scaffolding/10-01-SUMMARY.md` recording:
- Path to the pinned fixture
- CANONICAL_VARIANTS coverage count (pinned vs missing)
- s_linker13 macro_f1 stored value (must be ≈ 0.9509)
- Source ablation JSON files used per variant
- Any variants forced into the "missing" list and why
</output>
