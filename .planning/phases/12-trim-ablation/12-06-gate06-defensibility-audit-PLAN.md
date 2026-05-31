---
phase: 12-trim-ablation
plan: 06
type: execute
wave: 3
depends_on: [01, 03, 04, 05]
files_modified:
  - .planning/phases/12-trim-ablation/12-06-AUDIT-REPORT.md
  - .planning/phases/12-trim-ablation/12-06-V2_TO_V3_MAPPING-FINAL.md
  - .planning/phases/12-trim-ablation/12-06-SUMMARY.md
  - src/llm_sad_sam/linkers/experimental/s_linker13_trim1_judge_clean.py  # docstring updates only — rationale + audit ref
  - src/llm_sad_sam/linkers/experimental/s_linker13_trim2_entval_clean.py # docstring updates only
  - src/llm_sad_sam/linkers/experimental/s_linker13_trim3_runtime_rubric_clean.py # docstring updates only
autonomous: false
requirements: [PROMPT-01, PROMPT-04]
must_haves:
  truths:
    - "Every Phase-12 retained trim variant has been audited against the FULL BENCHMARK_TABOO.md surface (not just the 9-name project-component probe used during Wave 2)"
    - "Every retained trim has a reviewer-defensibility note documenting which original rule was removed/merged/replaced and the justification (covered by another rule / model handles natively / dead — never fired)"
    - "The v2→v3 mapping table FINAL revision is committed, updating Plan 12-01's initial mapping with the trim outcomes (status per constant: kept verbatim / dropped / merged / replaced-by-runtime-rubric / rejected-trim-reverted)"
    - "Rejected trims (any with overall_verdict == REJECT from Plans 12-03/04/05) are listed in 12-06-SUMMARY with explicit failing arm and dataset(s) — these go into the milestone summary's rejected-trims register and DO NOT propagate to Plan 13-01"
    - "Each retained variant's docstring is updated with a reference to the audit report and the specific rationale for the trim"
  artifacts:
    - path: ".planning/phases/12-trim-ablation/12-06-AUDIT-REPORT.md"
      provides: "Per-trim GATE-06 audit results + reviewer-defensibility narrative"
      contains: "per-trim subsection with: full TABOO scan output, defensibility note per removed/merged/replaced rule, accept/reject decision (mirrors trim-plan verdict.json), Plan 13-01 carry signal"
    - path: ".planning/phases/12-trim-ablation/12-06-V2_TO_V3_MAPPING-FINAL.md"
      provides: "Final v2→v3 prompt mapping table after Phase 12 trim outcomes — supersedes Plan 12-01's initial mapping"
      contains: "per-constant: v2_lines, v3_status (kept/dropped/merged/replaced/rejected), trim_plan_that_modified_it, reviewer_defensibility_note"
  key_links:
    - from: ".planning/phases/12-trim-ablation/12-06-AUDIT-REPORT.md"
      to: "results/ablation_results/12_0{3,4,5}_trim*/verdict.json"
      via: "audit report aggregates verdict.json overall_verdict signals + adds the full-TABOO defensibility check"
      pattern: "verdict.json"
    - from: ".planning/phases/12-trim-ablation/12-06-V2_TO_V3_MAPPING-FINAL.md"
      to: ".planning/phases/12-trim-ablation/12-01-V2_TO_V3_MAPPING.md"
      via: "supersedes — final revision after trim outcomes"
      pattern: "supersedes 12-01-V2_TO_V3_MAPPING.md"
---

<objective>
Run the FULL `BENCHMARK_TABOO.md` lexical sweep + reviewer-defensibility audit against every Phase-12 retained trim variant (`s_linker13_trim1_judge_clean`, `s_linker13_trim2_entval_clean`, `s_linker13_trim3_runtime_rubric_clean`) and against `prompts_v3.py`. The Wave 2 trim plans used a 9-name project-component probe as a fast defense-in-depth check during ablation; this plan runs the wider sweep using the Universal Taboo + per-project Components/Aliases/Keywords lists from BENCHMARK_TABOO.md. For each finding, the auditor (Claude executor) must adjudicate whether the term is leaked or appears in a generic SE-textbook context (e.g., "logic" in "control logic" — universal-taboo flagged, but reviewer-defensible if the prompt was written about a compiler/parser example, not a benchmark project). The user is the final arbiter via the Task 3 checkpoint.

Output: a single audit report aggregating per-trim findings, a FINAL v2→v3 mapping table superseding Plan 12-01's initial mapping, docstring updates on each retained variant linking to the audit, and the milestone-level rejected-trims register.

Purpose: closes PROMPT-04 (generality re-audit on every retained trim) and finalizes PROMPT-01 (the mapping table reflects actual trim outcomes, not just Step 0's initial split). Provides the canonical accept/reject signal for Plan 13-01.
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
@.planning/phases/12-trim-ablation/12-01-V2_TO_V3_MAPPING.md
@BENCHMARK_TABOO.md
@src/llm_sad_sam/linkers/experimental/prompts_v3.py
@src/llm_sad_sam/linkers/experimental/s_linker13_trim1_judge_clean.py
@src/llm_sad_sam/linkers/experimental/s_linker13_trim2_entval_clean.py
@src/llm_sad_sam/linkers/experimental/s_linker13_trim3_runtime_rubric_clean.py

<interfaces>
<!-- Audit surface: every new prompt body authored in Phase 12 + the existing prompts_v3 -->

From BENCHMARK_TABOO.md — sections:
  - MediaStore: Components + Aliases + Keywords
  - TeaStore: Components + Aliases + Keywords
  - Teammates: Components + Aliases + Keywords
  - BigBlueButton: Components + Aliases + Keywords
  - JabRef: Components + Aliases + Keywords
  - Universal Taboo (cross-project) — the largest match surface; many terms (logic, UI, client, storage, model, etc.) are common SE words AND benchmark component names → reviewer adjudication required per hit
  - Safe SE Textbook Examples — the allow-list (parser, scheduler, dispatcher, etc.)

From Plan 12-01: `prompts_v3.py` is byte-equal to prompts_v2's 9 kept constants. The Plan 12-01 audit was a 9-name probe; this plan runs the full Universal Taboo against the same surface to verify v2.0's pre-existing prompts also pass (regression check — prompts_v2 was audited at v2.0 close; reconfirm here).

From Plans 12-03/04/05: each retained trim variant contains new prompt bodies (the rubrics, merged constants, rubric-builder template + seed example).

From verdict.json files at:
  - results/ablation_results/12_03_trim1_judge/verdict.json
  - results/ablation_results/12_04_trim2_entval/verdict.json
  - results/ablation_results/12_05_trim3_runtime_rubric/verdict.json
Each has overall_verdict ∈ {ACCEPT, REJECT} and per-arm gate_pass flags. This plan inherits and FINALIZES those decisions.
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Run the full BENCHMARK_TABOO sweep on every Phase-12 prompt body</name>
  <files>
    - .planning/phases/12-trim-ablation/12-06-AUDIT-REPORT.md
  </files>
  <read_first>
    - BENCHMARK_TABOO.md (full file — extract every term in Components / Aliases / Keywords sections for all 5 projects + the Universal Taboo section)
    - src/llm_sad_sam/linkers/experimental/prompts_v3.py (the Step 0 surface — full body)
    - src/llm_sad_sam/linkers/experimental/s_linker13_trim1_judge_clean.py (the DOC_KNOWLEDGE_JUDGE_RUBRIC_V3 body)
    - src/llm_sad_sam/linkers/experimental/s_linker13_trim2_entval_clean.py (the ENTVAL_MERGED_RUBRIC_V3 body)
    - src/llm_sad_sam/linkers/experimental/s_linker13_trim3_runtime_rubric_clean.py (the RUBRIC_BUILDER_PROMPT + RUBRIC_BUILDER_SEED_EXAMPLE bodies)
    - results/ablation_results/12_03_trim1_judge/verdict.json
    - results/ablation_results/12_04_trim2_entval/verdict.json
    - results/ablation_results/12_05_trim3_runtime_rubric/verdict.json
    - results/ablation_results/12_05_trim3_runtime_rubric/{claude,gpt54}/sweep.log (the generated rubrics — input to the Plan 12-05 audit already aggregated in its verdict; cross-confirm here)
  </read_first>
  <action>
    Step 1 — Build a single comprehensive TABOO regex from BENCHMARK_TABOO.md. Concatenate all distinct terms (case-insensitive, word-boundary anchored) from:
      - Each project's Components list
      - Each project's Aliases list
      - Each project's Keywords list
      - The Universal Taboo section
    Skip terms in the "Safe SE Textbook Examples" allow-list.

    Construct as a single Python re module-level constant in a temporary audit script:
      FULL_TABOO = re.compile(r"\b(<term1>|<term2>|...|<termN>)\b", re.IGNORECASE)

    Step 2 — For each audit target file, scan its module-level string constants only (not module docstrings, not comments). Targets:
      a. src/llm_sad_sam/linkers/experimental/prompts_v3.py — all 9 active constants
      b. src/llm_sad_sam/linkers/experimental/s_linker13_trim1_judge_clean.py — DOC_KNOWLEDGE_JUDGE_RUBRIC_V3 (the body, not the docstring), DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3 (byte-equal to v2 — should match v2's audit history)
      c. src/llm_sad_sam/linkers/experimental/s_linker13_trim2_entval_clean.py — ENTVAL_MERGED_RUBRIC_V3, ENTITY_EXTRACTION_RULES_V3, VALIDATION_RULES_V3
      d. src/llm_sad_sam/linkers/experimental/s_linker13_trim3_runtime_rubric_clean.py — RUBRIC_BUILDER_PROMPT, RUBRIC_BUILDER_SEED_EXAMPLE
      e. The body of `_learn_document_knowledge_enriched` if forked (Plan 12-05 forks it; Plans 12-03/04 monkey-patch and do not fork) — scan the variant's overridden method body for f-string literal content.

    For each target file × each constant, record:
      - hit_count: number of FULL_TABOO matches
      - hit_terms: distinct matched terms
      - context_snippet: ±40 chars around each match (so the auditor can judge generic-SE vs benchmark-leakage)

    Step 3 — Adjudicate each hit. Many Universal Taboo terms (logic, UI, client, storage, model, config, common, server, layer, etc.) are ordinary SE vocabulary AND benchmark component names. The reviewer-defensibility test is: "Could this prompt body, read by a person unfamiliar with the benchmark projects, plausibly be written for a NON-benchmark system?" If yes — the hit is reviewer-defensible despite TABOO match; if no — the hit IS leakage and the trim/prompt is REJECTED.

    For each hit, assign disposition:
      - "safe" — generic SE vocabulary in textbook context (e.g., "control flow", "the server validates", "logic layer of an OS"). Document the rationale.
      - "leaked" — context explicitly references a benchmark project's component / alias / unique-keyword (e.g., "Recording Service handles", "kurento media", "BBB apps"). REJECTS the prompt.
      - "borderline" — ambiguous; surfaces to Task 3 user checkpoint for adjudication.

    Step 4 — Write `.planning/phases/12-trim-ablation/12-06-AUDIT-REPORT.md` with this structure:
      ```
      # Phase 12 — GATE-06 Audit Report (PROMPT-04)
      **Audited:** <ISO timestamp>

      ## Audit Surface
      - prompts_v3.py (9 constants)
      - s_linker13_trim1_judge_clean.py (DOC_KNOWLEDGE_JUDGE_RUBRIC_V3, DOC_KNOWLEDGE_JUDGE_EXAMPLES_V3)
      - s_linker13_trim2_entval_clean.py (3 constants)
      - s_linker13_trim3_runtime_rubric_clean.py (2 constants + forked method body)

      ## Full TABOO Sweep Results
      | File | Constant | hit_count | hit_terms | reviewer_disposition |
      |------|----------|-----------|-----------|----------------------|
      | ... | ... | ... | ... | safe / leaked / borderline |

      ## Reviewer-Defensibility Per Trim
      ### Trim 1 (Judge) — Plan 12-03
      - Verdict from 12-03/verdict.json: ACCEPT or REJECT
      - Removed rules: <list which rules from DOC_KNOWLEDGE_JUDGE_RULES were collapsed>
      - Each removed rule's justification: ...
      - Full TABOO sweep result: ...
      - GATE-06 final verdict: PASS / FAIL

      ### Trim 2 (Ent+Val) — Plan 12-04
      ... same structure ...

      ### Trim 3 (Runtime Rubric) — Plan 12-05
      ... same structure plus: aggregated generated-rubric audit from verdict.json ...

      ## Final Trim Disposition (input to Plan 13-01)
      | Trim ID | GATE-01 Claude | GATE-01 cross-model | GATE-06 lexical | GATE-06 reviewer | Carry to Plan 13-01? |
      |---------|----------------|---------------------|------------------|------------------|----------------------|
      | trim1_judge | PASS/FAIL | PASS/FAIL | PASS/FAIL | PASS/FAIL | yes/no |
      | trim2_entval | ... | ... | ... | ... | ... |
      | trim3_runtime_rubric | ... | ... | ... | ... | ... |

      ## Borderline Hits (require Task 3 user adjudication)
      <list each borderline hit with its full context for user review>
      ```
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 &amp;&amp; test -f .planning/phases/12-trim-ablation/12-06-AUDIT-REPORT.md &amp;&amp; grep -q "## Full TABOO Sweep Results" .planning/phases/12-trim-ablation/12-06-AUDIT-REPORT.md &amp;&amp; grep -q "## Reviewer-Defensibility Per Trim" .planning/phases/12-trim-ablation/12-06-AUDIT-REPORT.md &amp;&amp; grep -q "## Final Trim Disposition" .planning/phases/12-trim-ablation/12-06-AUDIT-REPORT.md</automated>
  </verify>
  <acceptance_criteria>
    - `12-06-AUDIT-REPORT.md` exists with the four required sections (Audit Surface, Full TABOO Sweep Results, Reviewer-Defensibility Per Trim, Final Trim Disposition).
    - Every Phase-12 retained trim variant + prompts_v3.py is in the sweep table.
    - Every hit has a disposition assigned (safe / leaked / borderline).
    - Borderline hits (if any) are explicitly listed for Task 3 adjudication.
    - The Final Trim Disposition table cross-references the verdict.json files from Plans 12-03/04/05.
    - Zero edits to v2.0 frozen files: `git diff --quiet src/llm_sad_sam/linkers/experimental/prompts_v2.py src/llm_sad_sam/linkers/experimental/s_linker13.py src/llm_sad_sam/linkers/experimental/s_linker13_clean.py src/llm_sad_sam/core/data_types_v2.py src/llm_sad_sam/core/document_loader_v2.py src/llm_sad_sam/pcm_parser_v2.py` exits 0.
  </acceptance_criteria>
  <done>Full TABOO sweep + reviewer-defensibility notes drafted; borderline hits surfaced to checkpoint.</done>
</task>

<task type="auto">
  <name>Task 2: Write FINAL v2→v3 mapping table + update retained-variant docstrings</name>
  <files>
    - .planning/phases/12-trim-ablation/12-06-V2_TO_V3_MAPPING-FINAL.md
    - src/llm_sad_sam/linkers/experimental/s_linker13_trim1_judge_clean.py  # docstring only
    - src/llm_sad_sam/linkers/experimental/s_linker13_trim2_entval_clean.py # docstring only
    - src/llm_sad_sam/linkers/experimental/s_linker13_trim3_runtime_rubric_clean.py # docstring only
  </files>
  <read_first>
    - .planning/phases/12-trim-ablation/12-01-V2_TO_V3_MAPPING.md (the initial 16-row mapping)
    - .planning/phases/12-trim-ablation/12-06-AUDIT-REPORT.md (Task 1 output)
    - results/ablation_results/12_03_trim1_judge/verdict.json
    - results/ablation_results/12_04_trim2_entval/verdict.json
    - results/ablation_results/12_05_trim3_runtime_rubric/verdict.json
  </read_first>
  <action>
    Step 1 — Write `12-06-V2_TO_V3_MAPPING-FINAL.md` as the final revision superseding Plan 12-01's initial table. Same 16-row layout, with v3_status column extended to include the trim-outcome verbs:
      - "kept (byte-equal)" — same as 12-01 for the 9 unchanged constants
      - "dropped" — same as 12-01 for the 7 EXT-01/legacy constants
      - "merged" — for ENTITY_EXTRACTION_RULES + VALIDATION_RULES if Plan 12-04 ACCEPTED
      - "replaced by inference-time rubric builder" — for DOC_KNOWLEDGE_JUDGE_RULES if Plan 12-05 ACCEPTED
      - "distilled (Technique 3 + 8)" — for DOC_KNOWLEDGE_JUDGE_RULES if Plan 12-03 ACCEPTED
      - "trim attempted, rejected — reverted to v2 form in v3" — for any rule whose trim REJECTED

    Conflict resolution: Plans 12-03 and 12-05 BOTH target DOC_KNOWLEDGE_JUDGE_RULES. Possible outcomes:
      - Both accept → Plan 13-01 must choose one (mapping table records both as "candidate", Plan 13-01 adjudicates)
      - Only one accepts → mapping records that one as the v3 status
      - Both reject → mapping records DOC_KNOWLEDGE_JUDGE_RULES as "trim attempted (Plans 12-03 + 12-05), both rejected — v3 keeps v2's body verbatim"

    Each row links to the relevant verdict.json + 12-06-AUDIT-REPORT.md disposition for traceability. Bottom of file, add an "Acceptance" section listing PROMPT-01 + PROMPT-04 requirements and the linked verdict files.

    Step 2 — Update each retained variant's docstring (no other edits to the variant code). For each of the three trim variants:
      - Append to the class docstring a "## GATE-06 Audit (Phase 12 Plan 12-06)" section that includes:
        - Audit timestamp
        - TABOO sweep result (PASS / FAIL with hit count)
        - Reviewer-defensibility verdict (PASS / FAIL)
        - Carry-to-Plan-13-01 signal (yes / no)
        - Link reference: ".planning/phases/12-trim-ablation/12-06-AUDIT-REPORT.md"
      - For variants where the trim was REJECTED at any gate (GATE-01 Claude / GATE-01 cross-model / GATE-06), additionally mark the class with a module-level comment at the top: `# REJECTED at Phase 12 — kept registered for reviewer-traceability of negative results; NOT carried to Plan 13-01.`
      - Do NOT delete rejected variant files — they remain in the repo for the milestone summary's rejected-trims register and for reviewer reproducibility.

    Step 3 — Re-run the registration tests for all three variants to confirm the docstring edits did not break registration:
      - `pytest tests/test_s_linker13_trim1_judge_registration.py tests/test_s_linker13_trim2_entval_registration.py tests/test_s_linker13_trim3_runtime_rubric_registration.py -x -q`
      - All must pass; docstring edits should not affect class behavior, attribute names, or prompt-body constants (these are tested by the registration suite).
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 &amp;&amp; test -f .planning/phases/12-trim-ablation/12-06-V2_TO_V3_MAPPING-FINAL.md &amp;&amp; grep -c "^| " .planning/phases/12-trim-ablation/12-06-V2_TO_V3_MAPPING-FINAL.md | awk '{if ($1 &gt;= 17) {exit 0} else {exit 1}}' &amp;&amp; pytest tests/test_s_linker13_trim1_judge_registration.py tests/test_s_linker13_trim2_entval_registration.py tests/test_s_linker13_trim3_runtime_rubric_registration.py -x -q &amp;&amp; git diff --quiet src/llm_sad_sam/linkers/experimental/prompts_v2.py src/llm_sad_sam/linkers/experimental/s_linker13.py src/llm_sad_sam/linkers/experimental/s_linker13_clean.py src/llm_sad_sam/core/data_types_v2.py src/llm_sad_sam/core/document_loader_v2.py src/llm_sad_sam/pcm_parser_v2.py</automated>
  </verify>
  <acceptance_criteria>
    - `12-06-V2_TO_V3_MAPPING-FINAL.md` exists; has ≥ 17 table rows (header + separator + 16 data rows).
    - Each row has v3_status reflecting Phase 12 outcomes (kept / dropped / merged / replaced / distilled / rejected).
    - Conflict on DOC_KNOWLEDGE_JUDGE_RULES (Plans 12-03 + 12-05 both target it) is explicitly resolved in the table or recorded as "Plan 13-01 adjudicates" if both accept.
    - All three retained-variant registration tests pass after docstring edits.
    - Variant docstrings updated with GATE-06 audit section.
    - Rejected variants additionally marked with module-level rejection comment.
    - Zero edits to v2.0 frozen files or to s_linker13_clean.py.
  </acceptance_criteria>
  <done>FINAL mapping table committed; docstrings updated; registration tests still green.</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 3: User adjudicates borderline TABOO hits + finalizes Plan 13-01 carry signal</name>
  <what-built>Audit report + FINAL mapping table + variant-docstring updates. This checkpoint resolves any borderline TABOO hits (Universal Taboo terms in generic-SE contexts) and finalizes the carry-to-Plan-13-01 disposition for each trim.</what-built>
  <read_first>
    - .planning/phases/12-trim-ablation/12-06-AUDIT-REPORT.md "Borderline Hits" section
    - .planning/phases/12-trim-ablation/12-06-V2_TO_V3_MAPPING-FINAL.md
    - results/ablation_results/12_0{3,4,5}_trim*/verdict.json
    - BENCHMARK_TABOO.md "Universal Taboo" + "Safe SE Textbook Examples" sections
  </read_first>
  <how-to-verify>
    The user (this checkpoint) reviews the audit report and adjudicates each borderline hit:

    1. **Borderline TABOO hits** — for each hit listed in the "Borderline Hits" section of the audit report, the user states:
       - "safe — defensible as generic SE context" (auditor records the rationale in the FINAL mapping table)
       - "leaked — reject the containing trim" (auditor flips the disposition to FAIL and updates the variant's docstring + module-level rejection comment)
       - "rephrase needed — borderline but fixable" (defers to a Phase 12 revision plan-phase; the trim is held NOT carried to Plan 13-01 until rephrased)

    2. **Per-trim final disposition** — the user confirms the Final Trim Disposition table in 12-06-AUDIT-REPORT.md is correct:
       - Trim 1 (judge): carry to Plan 13-01 yes/no
       - Trim 2 (ent+val): carry to Plan 13-01 yes/no
       - Trim 3 (runtime rubric): carry to Plan 13-01 yes/no
       Reject any disposition that does not match the user's reading of the verdict.json + audit report.

    3. **Rejected-trims register** — the user confirms that every REJECT trim has:
       - Its variant file kept in the repo (not deleted)
       - Module-level "# REJECTED at Phase 12" comment present
       - An entry in the milestone summary's rejected-trims register section (the auditor adds this to `12-06-SUMMARY.md` in Task 4)

    4. **Mapping conflict resolution** — if Plans 12-03 + 12-05 both ACCEPTED (both target DOC_KNOWLEDGE_JUDGE_RULES), the user selects which to carry:
       - Plan 12-03 (static distillation) — lower risk, smaller LLM call budget
       - Plan 12-05 (runtime rubric) — higher risk, novel mechanism, +1 LLM call per dataset
       - Both — Plan 13-01 produces TWO candidate `s_linker13_min` variants
       The choice is recorded in the FINAL mapping table.
  </how-to-verify>
  <resume-signal>Type "approved" with disposition decisions (per borderline hit + per-trim carry signal); or "rephrase trim X" with details; or "reject trim X" with rationale.</resume-signal>
  <acceptance_criteria>
    - Every borderline hit has a user-adjudicated disposition (safe / leaked / rephrase) recorded in 12-06-AUDIT-REPORT.md.
    - Final Trim Disposition table in the audit report has user-confirmed values for every trim.
    - If any trim was reclassified from ACCEPT → REJECT at this checkpoint (due to borderline-hit adjudication), the variant's docstring + module-level rejection comment are updated.
    - DOC_KNOWLEDGE_JUDGE_RULES conflict (if applicable) is resolved with the user's choice recorded in the FINAL mapping table.
    - 12-06-SUMMARY.md (Task 4) inherits these user decisions.
  </acceptance_criteria>
  <done>User has adjudicated all borderline hits and confirmed carry signals; rejected trims documented; mapping conflicts resolved.</done>
</task>

<task type="auto">
  <name>Task 4: Write 12-06-SUMMARY.md + milestone rejected-trims register section</name>
  <files>
    - .planning/phases/12-trim-ablation/12-06-SUMMARY.md
  </files>
  <read_first>
    - .planning/phases/12-trim-ablation/12-06-AUDIT-REPORT.md (post-user-adjudication state)
    - .planning/phases/12-trim-ablation/12-06-V2_TO_V3_MAPPING-FINAL.md
    - results/ablation_results/12_0{3,4,5}_trim*/verdict.json
    - .planning/REQUIREMENTS.md PROMPT-01 + PROMPT-04 rows
  </read_first>
  <action>
    Write `.planning/phases/12-trim-ablation/12-06-SUMMARY.md` with these sections:

    1. **Header**: Phase 12 Plan 06 — GATE-06 Audit + Final Trim Disposition. References PROMPT-01 + PROMPT-04.

    2. **Outcomes** — bullet list:
       - Number of trims audited: 3 (Plan 12-03 judge / Plan 12-04 ent+val merge / Plan 12-05 runtime rubric)
       - Number of trims accepted (carry to Plan 13-01): N
       - Number of trims rejected: 3 − N
       - Mapping table: FINAL revision committed at 12-06-V2_TO_V3_MAPPING-FINAL.md

    3. **Per-trim Final Disposition** — short subsection per trim citing:
       - Plan ID
       - GATE-01 Claude verdict (from verdict.json)
       - GATE-01 cross-model verdict (from verdict.json)
       - GATE-06 lexical sweep verdict (from audit report)
       - GATE-06 reviewer-defensibility verdict (from audit report + user adjudication)
       - Final carry-to-Plan-13-01 signal: yes/no
       - If rejected: failing arm + datasets

    4. **Rejected Trims Register** (milestone-level) — table listing every Phase-12 trim that did not pass all gates. Plan 13-01 + the v2.1 milestone summary will reference this table directly. Columns: trim_id | failing_gate | failing_arm | datasets | mitigation_signal (carry as negative result / rephrase candidate / drop entirely).

    5. **Artifact Index** — paths to:
       - 12-06-AUDIT-REPORT.md
       - 12-06-V2_TO_V3_MAPPING-FINAL.md
       - results/ablation_results/12_0{3,4,5}_trim*/verdict.json
       - Each of the 3 trim variant files
       - prompts_v3.py

    6. **Plan 13-01 Hand-off** — explicit list:
       - Accepted trims (variants) to be carried into s_linker13_min: <list>
       - Rejected trims (NOT to be carried): <list>
       - Mapping conflict resolution (if Plans 12-03 + 12-05 both accepted DOC_KNOWLEDGE_JUDGE_RULES trims): <user choice>
       - Any rephrase-candidates to revisit in a future Phase 12 revision: <list, may be empty>

    Reference PROMPT-01 + PROMPT-04 explicitly. End with a one-line completion stamp citing the ISO timestamp and the user's adjudication signal.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 &amp;&amp; test -f .planning/phases/12-trim-ablation/12-06-SUMMARY.md &amp;&amp; grep -q "Rejected Trims Register" .planning/phases/12-trim-ablation/12-06-SUMMARY.md &amp;&amp; grep -q "Plan 13-01 Hand-off" .planning/phases/12-trim-ablation/12-06-SUMMARY.md &amp;&amp; grep -q "PROMPT-01" .planning/phases/12-trim-ablation/12-06-SUMMARY.md &amp;&amp; grep -q "PROMPT-04" .planning/phases/12-trim-ablation/12-06-SUMMARY.md &amp;&amp; git diff --quiet src/llm_sad_sam/linkers/experimental/prompts_v2.py src/llm_sad_sam/linkers/experimental/s_linker13.py src/llm_sad_sam/linkers/experimental/s_linker13_clean.py src/llm_sad_sam/core/data_types_v2.py src/llm_sad_sam/core/document_loader_v2.py src/llm_sad_sam/pcm_parser_v2.py</automated>
  </verify>
  <acceptance_criteria>
    - `12-06-SUMMARY.md` exists with all six required sections.
    - PROMPT-01 + PROMPT-04 referenced.
    - Rejected Trims Register table present (may be empty if all 3 trims accepted).
    - Plan 13-01 Hand-off section explicit and actionable.
    - Zero edits to v2.0 frozen files.
    - GATE-02 unaffected: `pytest tests/test_v20_baseline_regression.py -q` exits 0.
  </acceptance_criteria>
  <done>Phase 12 closed: every trim has been audited, accepted-or-rejected, and the hand-off to Plan 13-01 is explicit.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| TABOO sweep regex → prompt body strings | Pure lexical check on local source files; no LLM, no external services |
| Audit report → milestone summary | The Plan 12-06 SUMMARY is the canonical input for Plan 13-01 promotion decisions; bad data here propagates to the milestone wrap |
| Docstring edits → variant code | Restricted to docstring + module-level comment; no functional code change |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-12-06-01 | Information disclosure | TABOO sweep misses a leaked term not in BENCHMARK_TABOO.md (e.g., a new alias added to a benchmark dataset since BENCHMARK_TABOO was written) | accept | BENCHMARK_TABOO.md is the project's canonical leakage surface — if it's incomplete, that's a milestone-level issue separate from Plan 12-06 |
| T-12-06-02 | Tampering | docstring edits accidentally modify variant constants or methods | mitigate | Task 2's verify re-runs the registration tests for all three variants; any prompt-body or attribute change is caught |
| T-12-06-03 | Repudiation | which Plan-12-06 audit version produced which carry signal | mitigate | Audit report timestamps every section; SUMMARY ends with completion stamp; verdict.json files (immutable from Plans 12-03/04/05) are the unambiguous evidence trail |
| T-12-06-04 | Denial of service | borderline-hit adjudication blocks indefinitely | mitigate | Task 3 is a checkpoint — user must respond; if all 3 trims clearly PASS or FAIL at the lexical layer (no borderline hits), Task 3 is fast |
</threat_model>

<verification>
- Full BENCHMARK_TABOO sweep completed on prompts_v3.py + 3 trim variants.
- Every hit dispositioned (safe / leaked / borderline + user adjudication).
- FINAL v2→v3 mapping table committed with trim outcomes reflected.
- Variant docstrings updated; rejected variants marked.
- Registration tests still green.
- 12-06-SUMMARY.md committed with rejected-trims register + Plan 13-01 hand-off.
- Zero edits to v2.0 frozen files.
- GATE-02 unaffected.
</verification>

<success_criteria>
- PROMPT-04 closed: generality re-audit complete on every Phase-12 prompt body.
- PROMPT-01 finalized: v2→v3 mapping table reflects actual trim outcomes (kept / dropped / merged / replaced / rejected).
- Plan 13-01 receives an unambiguous accept/reject signal per trim and an unambiguous list of variants to carry into `s_linker13_min`.
- Milestone summary inherits the rejected-trims register for the v2.1 publication narrative.
</success_criteria>

<output>
After completion, create `.planning/phases/12-trim-ablation/12-06-SUMMARY.md`.
</output>
</content>
