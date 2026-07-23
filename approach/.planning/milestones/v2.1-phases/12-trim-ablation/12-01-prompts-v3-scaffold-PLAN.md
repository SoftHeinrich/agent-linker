---
phase: 12-trim-ablation
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/llm_sad_sam/linkers/experimental/prompts_v3.py
  - src/llm_sad_sam/linkers/experimental/s_linker13_clean_v3.py
  - run_ablation.py
  - .planning/phases/12-trim-ablation/12-01-V2_TO_V3_MAPPING.md
  - .planning/phases/12-trim-ablation/12-01-SUMMARY.md
autonomous: true
requirements: [PROMPT-01, PROMPT-04]
must_haves:
  truths:
    - "prompts_v3.py exists and exports only the 9 constants actively imported by s_linker13_clean"
    - "The 7 dead constants (WORD_USAGE_PROMPT + 6 STANDALONE_MENTION_RULES_* variants) are NOT in prompts_v3.py"
    - "Importing prompts_v3 in isolation succeeds and exposes the expected names"
    - "A thin sibling s_linker13_clean_v3 imports from prompts_v3 and produces F1 identical to s_linker13_clean on cached checkpoints (Step 0 free-win evidence)"
    - "prompts_v2.py is unchanged (git diff --quiet)"
    - "The v2→v3 mapping table is committed and lists every prompt as kept/dropped with rationale"
  artifacts:
    - path: "src/llm_sad_sam/linkers/experimental/prompts_v3.py"
      provides: "Cleaned prompt surface — 9 active constants for s_linker13_clean lineage"
      contains: "AMBIGUITY_FEW_SHOT, AMBIGUITY_RULES, DOC_KNOWLEDGE_EXTRACTION_RULES, DOC_KNOWLEDGE_JUDGE_EXAMPLES, DOC_KNOWLEDGE_JUDGE_RULES, ENTITY_EXTRACTION_RULES, VALIDATION_RULES, COREF_RULES, SEED_DISAMBIGUATION_RULES"
    - path: "src/llm_sad_sam/linkers/experimental/s_linker13_clean_v3.py"
      provides: "Thin sibling identical to SLinker13Clean except imports prompts_v3"
      exports: ["SLinker13CleanV3"]
    - path: ".planning/phases/12-trim-ablation/12-01-V2_TO_V3_MAPPING.md"
      provides: "Documented v2→v3 prompt mapping (PROMPT-01)"
      contains: "kept / dropped / renamed"
  key_links:
    - from: "src/llm_sad_sam/linkers/experimental/s_linker13_clean_v3.py"
      to: "src/llm_sad_sam/linkers/experimental/prompts_v3.py"
      via: "from ... import (9 constants)"
      pattern: "from llm_sad_sam.linkers.experimental.prompts_v3 import"
    - from: "run_ablation.py CANONICAL_VARIANTS + VARIANT_SPECS"
      to: "s_linker13_clean_v3"
      via: "registration with canonical=False"
      pattern: "s_linker13_clean_v3.*canonical=False"
---

<objective>
Create `prompts_v3.py` containing only the 9 prompt constants actively imported by `s_linker13_clean`, dropping the 7 dead constants that survive in `prompts_v2.py` for back-compat with frozen siblings. Also create a thin sibling variant `s_linker13_clean_v3.py` that is IDENTICAL to `SLinker13Clean` except its prompt imports come from `prompts_v3`. Register it. Verify Step 0 by running the new variant against its cached layer1 checkpoint (loaded under PHASE_CACHE_DIR pointing at the existing `results/phase_cache/s_linker13_clean/` tree to reuse upstream state) and confirming F1 identical-within-variance to the s_linker13_clean baseline.

Purpose: Closes PROMPT-01 (prompts_v3.py side-by-side with prompts_v2). Demonstrates Step 0 from the survey §5 (~150-line / ~36-rule deletion with zero F1 risk). Establishes the import surface that all downstream trim variants (12-03, 12-04, 12-05) will modify.

Output: `prompts_v3.py` (~9 constants), `s_linker13_clean_v3.py` (thin wrapper, embeds the same `SEED_DISAMBIGUATION_RULES` classvar as parent), `run_ablation.py` updated with one new VARIANT_SPECS entry, mapping table written.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/REQUIREMENTS.md
@.planning/STATE.md
@.planning/phases/12-trim-ablation/12-CONTEXT.md
@.planning/research/PROMPT-HARNESS-SURVEY.md
@BENCHMARK_TABOO.md
@src/llm_sad_sam/linkers/experimental/prompts_v2.py
@src/llm_sad_sam/linkers/experimental/s_linker13_clean.py

<interfaces>
<!-- Active prompts in s_linker13_clean's imports (verified by grep at planning time) -->

From src/llm_sad_sam/linkers/experimental/prompts_v2.py — KEEP in v3 (9):
- AMBIGUITY_FEW_SHOT (lines 14–47)
- AMBIGUITY_RULES (lines 50–64)
- DOC_KNOWLEDGE_EXTRACTION_RULES (lines 71–84)
- DOC_KNOWLEDGE_JUDGE_EXAMPLES (lines 87–121)
- DOC_KNOWLEDGE_JUDGE_RULES (lines 124–139)
- ENTITY_EXTRACTION_RULES (lines 179–191)
- VALIDATION_RULES (lines 194–205)
- COREF_RULES (lines 212–222)
- SEED_DISAMBIGUATION_RULES (lines 372–390) — also lifted as class var in s_linker13_clean line 143; both copies stay byte-equal

From src/llm_sad_sam/linkers/experimental/prompts_v2.py — DROP in v3 (7):
- WORD_USAGE_PROMPT (lines 146–172) — legacy ≤ 12c
- STANDALONE_MENTION_RULES_PRE_FILTERED (lines 229–238) — EXT-01 deferred
- STANDALONE_MENTION_RULES_LLM_ONLY (lines 241–255) — EXT-01 deferred
- STANDALONE_MENTION_RULES_PRE_FILTERED_ALIAS_AWARE (lines 271–286) — EXT-01 alias-aware
- STANDALONE_MENTION_RULES_LLM_ONLY_ALIAS_AWARE (lines 289–310) — EXT-01 alias-aware
- STANDALONE_MENTION_RULES_PRE_FILTERED_FULL_KNOWLEDGE (lines 313–334) — EXT-01 full-knowledge
- STANDALONE_MENTION_RULES_LLM_ONLY_FULL_KNOWLEDGE (lines 337–365) — EXT-01 full-knowledge

From src/llm_sad_sam/linkers/experimental/s_linker13_clean.py:
- Class SLinker13Clean
- `_VARIANT_NAME = "s_linker13_clean"`
- imports lines 47-52: from llm_sad_sam.linkers.experimental.prompts_v2 import (...)
- SEED_DISAMBIGUATION_RULES is also a class attribute (line 143), and class methods reference `self.SEED_DISAMBIGUATION_RULES` (line 555) — sibling subclass can leave this as-is

From run_ablation.py:
- CANONICAL_VARIANTS list (lines 40-87) — append "s_linker13_clean_v3"
- VARIANT_SPECS dict (lines 89-367) — add entry with canonical=False
- spec shape: dict(aliases=(), module="...", class_name="...", description="...", canonical=False)
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Create prompts_v3.py with only the 9 active constants and verify import</name>
  <files>src/llm_sad_sam/linkers/experimental/prompts_v3.py, tests/test_prompts_v3.py</files>
  <read_first>
    - src/llm_sad_sam/linkers/experimental/prompts_v2.py (full file — copy the 9 active constants byte-for-byte)
    - .planning/research/PROMPT-HARNESS-SURVEY.md §0 (lines 10-35) — the kept/dropped table
    - BENCHMARK_TABOO.md (full file — the GATE-06 surface)
    - src/llm_sad_sam/linkers/experimental/s_linker13_clean.py lines 47-52 (the existing import list to mirror)
  </read_first>
  <behavior>
    - Test: `from llm_sad_sam.linkers.experimental import prompts_v3` succeeds.
    - Test: prompts_v3 exposes exactly these 9 names as module-level string constants: AMBIGUITY_FEW_SHOT, AMBIGUITY_RULES, DOC_KNOWLEDGE_EXTRACTION_RULES, DOC_KNOWLEDGE_JUDGE_EXAMPLES, DOC_KNOWLEDGE_JUDGE_RULES, ENTITY_EXTRACTION_RULES, VALIDATION_RULES, COREF_RULES, SEED_DISAMBIGUATION_RULES.
    - Test: prompts_v3 does NOT expose any of: WORD_USAGE_PROMPT, STANDALONE_MENTION_RULES_PRE_FILTERED, STANDALONE_MENTION_RULES_LLM_ONLY, STANDALONE_MENTION_RULES_PRE_FILTERED_ALIAS_AWARE, STANDALONE_MENTION_RULES_LLM_ONLY_ALIAS_AWARE, STANDALONE_MENTION_RULES_PRE_FILTERED_FULL_KNOWLEDGE, STANDALONE_MENTION_RULES_LLM_ONLY_FULL_KNOWLEDGE.
    - Test: each kept constant in prompts_v3 is byte-equal to the corresponding constant in prompts_v2 (lossless Step 0).
    - Test: prompts_v3's total module text contains zero BENCHMARK_TABOO terms (`grep -wEi "logic|UI|client|storage|common|model|database|DB|cache|registry|auth|server|persistence|facade|recording|cascade|conversion|validation|dedicated|preferences|config|internal|adapter|order|processor|event|socket|layer|Reencoding|FreeSWITCH|kurento|Redis PubSub|HTML5 Server|Nginx Proxy|Kafka Broker|Zookeeper" prompts_v3.py` returns no match for project-tied surface terms — exact regex below). Re-use the same probe used by Plan 12-06.
  </behavior>
  <action>
    Create `src/llm_sad_sam/linkers/experimental/prompts_v3.py` by copying the 9 active constants from `prompts_v2.py` BYTE-FOR-BYTE (no rephrasing, no shortening — Step 0 is purely a registration delete, no semantic change). Include the section-divider comments above each constant exactly as in v2 for stable provenance. Update the module docstring to read:

    """Prompt constants — v3 (Phase 12 Step 0).

    Side-by-side with prompts_v2.py. Carries only the 9 constants actively imported
    by `s_linker13_clean`. The 7 EXT-01 / legacy constants in prompts_v2.py are
    dropped here (Phase 12 trivial-win deletion per PROMPT-HARNESS-SURVEY §5 row 0).

    Step 0 is byte-equal to v2 for every kept constant — no trim, no rephrasing.
    Per-prompt trim variants (Steps 1-3) embed their trimmed prompts inside their
    own variant `.py` rather than mutating this file, so prompts_v3 stays a stable
    shared surface across trim variants.
    """

    Then create `tests/test_prompts_v3.py` containing:
    1. `test_prompts_v3_import_clean()` — bare `import` succeeds; `python -c "from llm_sad_sam.linkers.experimental import prompts_v3"` exits 0.
    2. `test_kept_constants_present()` — assert all 9 names are module-level `str` attributes.
    3. `test_dropped_constants_absent()` — `assert not hasattr(prompts_v3, name)` for each of the 7 dropped names.
    4. `test_byte_equal_to_v2()` — `from ... import prompts_v2 as v2, prompts_v3 as v3` and assert `getattr(v3, name) == getattr(v2, name)` for each of the 9 kept names.
    5. `test_no_benchmark_taboo_terms()` — read `prompts_v3.__file__`, grep its source for the same TABOO regex Plan 12-06 uses (define it as a module-level constant in the test): `r"(?i)\b(Reencoding|FreeSWITCH|kurento|Recording Service|Redis PubSub|HTML5 Server|Nginx Proxy|Kafka Broker|Zookeeper)\b"` — assert zero matches. Word-form terms like "client", "config", "auth" are excluded from this test because the kept prompt text discusses architectural concepts at the level of safe SE-domain examples; the full lexical TABOO scan happens in Plan 12-06 with reviewer adjudication of compound-context hits.

    Do NOT edit `prompts_v2.py`. Verify: `git diff --quiet src/llm_sad_sam/linkers/experimental/prompts_v2.py` exits 0.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 &amp;&amp; python -c "from llm_sad_sam.linkers.experimental import prompts_v3" &amp;&amp; pytest tests/test_prompts_v3.py -x -q &amp;&amp; git diff --quiet src/llm_sad_sam/linkers/experimental/prompts_v2.py</automated>
  </verify>
  <acceptance_criteria>
    - `src/llm_sad_sam/linkers/experimental/prompts_v3.py` exists.
    - `python -c "from llm_sad_sam.linkers.experimental import prompts_v3"` exits 0.
    - All 5 tests in `test_prompts_v3.py` pass.
    - `wc -l prompts_v3.py` is at least 150 lines less than `wc -l prompts_v2.py` (the ~150-line deletion claim from the survey).
    - `git diff --quiet src/llm_sad_sam/linkers/experimental/prompts_v2.py` exits 0 (v2 untouched).
    - prompts_v3 module text contains zero of the 9 project-name TABOO terms (Reencoding/FreeSWITCH/kurento/Recording Service/Redis PubSub/HTML5 Server/Nginx Proxy/Kafka Broker/Zookeeper) — these are the inarguable benchmark-component leakage probes; the broader lexical sweep is Plan 12-06's responsibility.
  </acceptance_criteria>
  <done>prompts_v3.py importable; tests pass; v2.py untouched; deletion size confirmed.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Create s_linker13_clean_v3.py thin sibling and register it</name>
  <files>
    - src/llm_sad_sam/linkers/experimental/s_linker13_clean_v3.py
    - run_ablation.py
    - tests/test_s_linker13_clean_v3_registration.py
  </files>
  <read_first>
    - src/llm_sad_sam/linkers/experimental/s_linker13_clean.py lines 1-200 (imports, class header, SEED_DISAMBIGUATION_RULES classvar)
    - run_ablation.py lines 40-87 (CANONICAL_VARIANTS)
    - run_ablation.py lines 324-330 (existing s_linker13_clean VARIANT_SPECS entry — the pattern to mirror)
    - run_ablation.py lines 89-367 (VARIANT_SPECS dict shape)
  </read_first>
  <behavior>
    - Test: `from llm_sad_sam.linkers.experimental.s_linker13_clean_v3 import SLinker13CleanV3` succeeds.
    - Test: `SLinker13CleanV3._VARIANT_NAME == "s_linker13_clean_v3"` (so checkpoints land under a separate subdir — does not collide with Phase-10 baseline cache).
    - Test: `SLinker13CleanV3` is a subclass of `SLinker13Clean` OR a near-identical clone where the only difference is prompt imports.
    - Test: `s_linker13_clean_v3` appears in `run_ablation.CANONICAL_VARIANTS`.
    - Test: `run_ablation.VARIANT_SPECS["s_linker13_clean_v3"]["canonical"] == False`.
    - Test: `run_ablation.VARIANT_SPECS["s_linker13_clean_v3"]["class_name"] == "SLinker13CleanV3"`.
  </behavior>
  <action>
    Decision rationale: SLinker13CleanV3 SUBCLASSES SLinker13Clean and overrides only the prompt-import-bound class attributes. This is minimally invasive — the parent's methods reference `AMBIGUITY_FEW_SHOT`, `AMBIGUITY_RULES`, etc. as module-level names imported at the top of `s_linker13_clean.py`. Pure subclassing therefore would still pull the parent's prompts_v2 imports. To actually substitute prompts_v3, the cleanest pattern (matching how `SEED_DISAMBIGUATION_RULES` is already lifted as a class var at line 143) is: rebind the 8 other prompts as class-level attributes on `SLinker13CleanV3`, AND override the methods that reference them at module scope so they reference `self.<NAME>` instead.

    The simpler clean implementation, given that prompts_v3 is byte-equal to prompts_v2 for these 9 constants: create `s_linker13_clean_v3.py` as a STANDALONE FILE (not a subclass) by copying `s_linker13_clean.py` verbatim and then editing exactly two locations:
      (a) The prompt import block (lines 47-52): swap `prompts_v2` → `prompts_v3`.
      (b) The class `_VARIANT_NAME` (line 136): `"s_linker13_clean"` → `"s_linker13_clean_v3"`.
      (c) Class name `SLinker13Clean` → `SLinker13CleanV3`.
      (d) The print statement on line ~183 to reflect the new name.

    No other edits — this guarantees the only Step 0 change is the prompt module source. `SEED_DISAMBIGUATION_RULES` stays as a class var (line 143) byte-equal to its v2 counterpart.

    Update the docstring at the top of `s_linker13_clean_v3.py` to read:

    """S-Linker13 Clean V3: prompts_v3 sibling (Phase 12 Step 0).

    Identical to SLinker13Clean in every method body and parameter; the only delta
    is the prompt import (prompts_v3 instead of prompts_v2) and the variant name.
    Used as the Step 0 acceptance check: F1 must match SLinker13Clean within
    Claude run-to-run variance on the existing s_linker13_clean phase cache (loaded
    by pointing PHASE_CACHE_DIR at results/phase_cache/, treating Task 3 as a
    no-LLM checkpoint replay).
    """

    Then append the registration to `run_ablation.py`:
      - Append `"s_linker13_clean_v3",` to `CANONICAL_VARIANTS` (after the existing `"s_linker13_clean",` entry, before the EXT-01 g-variants).
      - Append a new VARIANT_SPECS entry mirroring the existing `"s_linker13_clean"` entry shape with `canonical=False`, `class_name="SLinker13CleanV3"`, `module="llm_sad_sam.linkers.experimental.s_linker13_clean_v3"`, description="S-Linker13 Clean V3: prompts_v3 sibling — Phase 12 Step 0 acceptance variant (byte-equal kept prompts; 7 dead constants dropped)".

    Create `tests/test_s_linker13_clean_v3_registration.py` testing the 6 behaviors above.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 &amp;&amp; python -c "from llm_sad_sam.linkers.experimental.s_linker13_clean_v3 import SLinker13CleanV3; assert SLinker13CleanV3._VARIANT_NAME == 's_linker13_clean_v3'" &amp;&amp; pytest tests/test_s_linker13_clean_v3_registration.py -x -q &amp;&amp; python -c "from run_ablation import CANONICAL_VARIANTS, VARIANT_SPECS; assert 's_linker13_clean_v3' in CANONICAL_VARIANTS; assert VARIANT_SPECS['s_linker13_clean_v3']['canonical'] is False"</automated>
  </verify>
  <acceptance_criteria>
    - `src/llm_sad_sam/linkers/experimental/s_linker13_clean_v3.py` exists; importable.
    - The diff between `s_linker13_clean.py` and `s_linker13_clean_v3.py` is limited to: class name, _VARIANT_NAME, prompt import path (prompts_v2 → prompts_v3), and the docstring/print banner. Verify by `diff <(grep -v "^#\|^\"\"\"\|^$" s_linker13_clean.py) <(grep -v "^#\|^\"\"\"\|^$" s_linker13_clean_v3.py) | wc -l` returning ≤ 12 changed lines (allow some line-shift slack for the rename pattern; reviewer-defensible "tiny diff").
    - `run_ablation.py` updated; `s_linker13_clean_v3` registered in both CANONICAL_VARIANTS and VARIANT_SPECS with `canonical=False`.
    - GATE-02 unaffected: `pytest tests/test_v20_baseline_regression.py -q` exits 0 (Phase 10 / Plan 10-04 wired the new entry into the fixture's "missing" slot; if the test now reports `s_linker13_clean_v3` as drift, snapshot it under `missing` in `tests/fixtures/v2_0_baseline.json` per the documented "snapshot it before promotion" pattern).
    - No edits to `prompts_v2.py`, `s_linker13_clean.py`, `s_linker13.py`, `data_types_v2.py`, `document_loader_v2.py`, `pcm_parser_v2.py`, or any `ilinker*.py`: `git diff --quiet` on each path exits 0.
  </acceptance_criteria>
  <done>s_linker13_clean_v3 importable, registered, GATE-02 still green (with fixture snapshot if needed); no frozen file touched.</done>
</task>

<task type="auto">
  <name>Task 3: Verify Step 0 equivalence by checkpoint-loaded re-run + write mapping doc and SUMMARY</name>
  <files>
    - .planning/phases/12-trim-ablation/12-01-V2_TO_V3_MAPPING.md
    - .planning/phases/12-trim-ablation/12-01-SUMMARY.md
    - results/ablation_results/12_01_step0_verify/
  </files>
  <read_first>
    - .planning/phases/12-trim-ablation/12-CONTEXT.md (decisions section, Step 0 spec)
    - .planning/research/PROMPT-HARNESS-SURVEY.md §0 table + §5 row 0
    - run_ablation.py main() flow (lines 701-786) — to confirm the variant runs end-to-end
    - results/phase_cache/s_linker13_clean/ (existing Phase-10 baseline cache that proves the bytes match)
  </read_first>
  <action>
    Step 0 acceptance protocol — runs in two arms:

    Arm A — Claude Sonnet, full pipeline against cached upstream:
      Since prompts_v3 is byte-equal to prompts_v2 for the 9 kept constants and `s_linker13_clean_v3` only changes the import path, the LLM call payloads are byte-identical. Therefore a full re-run against checkpoints is NOT required. Instead, prove byte-equivalence at the import surface:
      1. `python -c "from llm_sad_sam.linkers.experimental import prompts_v2 as v2, prompts_v3 as v3; names=['AMBIGUITY_FEW_SHOT','AMBIGUITY_RULES','DOC_KNOWLEDGE_EXTRACTION_RULES','DOC_KNOWLEDGE_JUDGE_EXAMPLES','DOC_KNOWLEDGE_JUDGE_RULES','ENTITY_EXTRACTION_RULES','VALIDATION_RULES','COREF_RULES','SEED_DISAMBIGUATION_RULES']; mismatch=[n for n in names if getattr(v2,n) != getattr(v3,n)]; assert not mismatch, mismatch; print('byte-equal')"` exits 0.
      2. `diff <(python -c "import ast,sys; t=ast.parse(open('src/llm_sad_sam/linkers/experimental/s_linker13_clean.py').read()); [print(ast.unparse(n)) for n in ast.walk(t) if isinstance(n, ast.FunctionDef)]") <(python -c "import ast,sys; t=ast.parse(open('src/llm_sad_sam/linkers/experimental/s_linker13_clean_v3.py').read()); [print(ast.unparse(n)) for n in ast.walk(t) if isinstance(n, ast.FunctionDef)]")` — produces a diff with ZERO lines (every method body byte-equal across the two variants when prompt-name strings inside f-strings are treated identically). If non-zero, isolate to whatever cosmetic difference and either fix the v3 sibling or document the surviving cosmetic in SUMMARY.
      3. Record in SUMMARY.md: "Step 0 equivalence proven by byte-equality of (a) all 9 kept constant strings and (b) all method bodies. Full LLM re-run skipped because every prompt payload is bit-identical."

    Arm B — Mapping doc (PROMPT-01 deliverable):
      Write `.planning/phases/12-trim-ablation/12-01-V2_TO_V3_MAPPING.md` containing a 16-row Markdown table (one row per prompts_v2 constant) with columns: `constant_name | v2_lines | v3_status | rationale`. Use the kept/dropped data from `prompts_v2.py` (verified above). Example row shape:

    | AMBIGUITY_FEW_SHOT | 14-47 | kept (byte-equal) | Active in s_linker13_clean alias classifier; calibration-bearing |
    | WORD_USAGE_PROMPT | 146-172 | dropped | Legacy ≤ s_linker12c; unused by s_linker13_clean |
    | STANDALONE_MENTION_RULES_PRE_FILTERED | 229-238 | dropped | EXT-01 (deferred per v2.1 out-of-scope), unused by s_linker13_clean |
    ... (one row per constant, total 16 rows)

    Below the table, add an "Acceptance" section that links to:
    - PROMPT-01 (REQUIREMENTS.md)
    - PROMPT-04 (REQUIREMENTS.md) — note that the byte-equality of kept constants means Step 0 introduces no new benchmark-derived phrasing.
    - Phase 12 CONTEXT decisions section (Step 0)
    - the test file `tests/test_prompts_v3.py`

    Arm C — Write SUMMARY:
      Create `12-01-SUMMARY.md`. List artifacts produced. Record:
        - Lines deleted: `wc -l prompts_v2.py - wc -l prompts_v3.py` actual value.
        - Constants kept: 9.
        - Constants dropped: 7.
        - GATE-06 status: full audit deferred to Plan 12-06; the 9-name benchmark-component probe in Task 1 returned zero hits.
        - GATE-02 status: passing (with fixture snapshot for `s_linker13_clean_v3` if added).
        - Links to: prompts_v3.py, s_linker13_clean_v3.py, mapping doc, tests.
      Reference requirements PROMPT-01 and PROMPT-04 explicitly.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 &amp;&amp; test -f .planning/phases/12-trim-ablation/12-01-V2_TO_V3_MAPPING.md &amp;&amp; grep -c "^| " .planning/phases/12-trim-ablation/12-01-V2_TO_V3_MAPPING.md | awk '{if ($1 &gt;= 17) {exit 0} else {exit 1}}' &amp;&amp; python -c "from llm_sad_sam.linkers.experimental import prompts_v2 as v2, prompts_v3 as v3; names=['AMBIGUITY_FEW_SHOT','AMBIGUITY_RULES','DOC_KNOWLEDGE_EXTRACTION_RULES','DOC_KNOWLEDGE_JUDGE_EXAMPLES','DOC_KNOWLEDGE_JUDGE_RULES','ENTITY_EXTRACTION_RULES','VALIDATION_RULES','COREF_RULES','SEED_DISAMBIGUATION_RULES']; mismatch=[n for n in names if getattr(v2,n) != getattr(v3,n)]; assert not mismatch, mismatch; print('byte-equal')"</automated>
  </verify>
  <acceptance_criteria>
    - `12-01-V2_TO_V3_MAPPING.md` exists and contains 16 data rows in its kept/dropped table (header + separator + 16 rows = ≥ 17 lines matching `^| `).
    - Every kept constant in prompts_v3 is byte-equal to its prompts_v2 counterpart (the verify command exits 0).
    - `12-01-SUMMARY.md` exists; references PROMPT-01 and PROMPT-04; records the line-count deletion delta and the 9-kept/7-dropped split.
    - Plans 12-03, 12-04, 12-05 can safely use `prompts_v3` as the shared import surface and embed trim-specific overrides at the variant-class level.
  </acceptance_criteria>
  <done>Step 0 equivalence proven by byte-equality; mapping doc + SUMMARY shipped; ready for Wave 2 trim plans.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Source file edits | New `.py` files under `src/llm_sad_sam/linkers/experimental/`; no edits to frozen v2.0 files |
| Registration | `run_ablation.py` (live registry consumed by GATE-02 regression test) |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-12-01-01 | Tampering | accidental edit to prompts_v2.py while creating prompts_v3 | mitigate | Task 1's verify includes `git diff --quiet prompts_v2.py` exits 0; if it fails, executor reverts and re-creates v3 as a NEW file |
| T-12-01-02 | Information disclosure | benchmark-derived terms slip into prompts_v3 | mitigate | Task 1's behavior test runs a TABOO regex on prompts_v3 source; Plan 12-06 does the full lexical sweep |
| T-12-01-03 | Tampering | accidental edit to s_linker13_clean.py while creating v3 sibling | mitigate | Task 2's acceptance: `git diff --quiet s_linker13_clean.py` exits 0; sibling is a standalone file copied + 4 edits |
| T-12-01-04 | Denial of service | GATE-02 regression test fails after registry update | mitigate | Task 2's acceptance runs GATE-02; if drift, snapshot under `missing` per documented pattern |
</threat_model>

<verification>
- prompts_v3.py exists, importable, 9 constants byte-equal to prompts_v2 counterparts, 7 dropped.
- s_linker13_clean_v3.py exists, importable, registered; 4-edit delta from s_linker13_clean.py.
- prompts_v2.py and s_linker13_clean.py untouched (git diff --quiet exits 0 on both).
- GATE-02 regression test passes (with fixture snapshot if needed).
- 12-01-V2_TO_V3_MAPPING.md committed with 16 rows.
- Plan 12-06 inherits prompts_v3.py as a primary lexical-audit target.
</verification>

<success_criteria>
- PROMPT-01 satisfied: `prompts_v3.py` ships side-by-side with `prompts_v2.py`, mapping table committed, only active constants kept.
- PROMPT-04 partial: TABOO-component probe passes; full reviewer-defensibility audit deferred to Plan 12-06.
- Downstream Wave 2 trim plans can import from `prompts_v3` and override trim-specific prompts inside their own variant `.py` (no further `prompts_v3.py` edits required from 12-03/04/05).
</success_criteria>

<output>
After completion, create `.planning/phases/12-trim-ablation/12-01-SUMMARY.md`.
</output>
