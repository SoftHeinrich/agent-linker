---
phase: 10-scaffolding
plan: 04
type: execute
wave: 1
depends_on: []
files_modified:
  - .planning/PROJECT.md
  - .planning/STATE.md
autonomous: true
requirements:
  - GATE-01
user_setup: []

must_haves:
  truths:
    - "PROJECT.md Key Decisions table contains a v2.1 row codifying the cross-model gate with the exact tolerance T = 1.0pp"
    - "STATE.md Standing Gates section contains the literal string 'gpt-5.4 macro ≥ 0.9077 within ≤ 1.0pp tolerance' (committed concrete tolerance, not '≤ 1pp')"
    - "Both files agree on baseline (0.9077), model (gpt-5.4), tolerance (1.0pp), and the reference to v2.0 CROSS evidence"
    - "Nothing else in PROJECT.md or STATE.md is modified (additive edits only)"
  artifacts:
    - path: ".planning/PROJECT.md"
      provides: "Codified v2.1 cross-model gate in Key Decisions table"
      contains: "1.0pp"
    - path: ".planning/STATE.md"
      provides: "Concrete tolerance committed in Standing Gates section"
      contains: "1.0pp tolerance"
  key_links:
    - from: ".planning/PROJECT.md Key Decisions"
      to: ".planning/STATE.md Standing Gates"
      via: "shared tolerance value 1.0pp and baseline 0.9077"
      pattern: "1\\.0pp"
    - from: ".planning/STATE.md Standing Gates"
      to: ".planning/REQUIREMENTS.md GATE-01"
      via: "the v2.1 GATE-01 cross-model row references the same tolerance"
      pattern: "GATE-01"
---

<objective>
Codify the v2.1 cross-model gate (GATE-01 cross-model) as a standing project decision. The
requirement (REQUIREMENTS.md GATE-01) leaves the tolerance T loosely defined ("≤ 1pp
regression"); this plan pins T to the concrete numeric value 1.0pp and writes it into the
two canonical locations every subsequent phase consults: PROJECT.md Key Decisions table
and STATE.md Standing Gates section.

Purpose: Phase 12 (PROMPT-02 per-prompt rule-trim ablation) and Phase 13 (PROMPT-03 final
sweep) both depend on a numerically pinned tolerance. Without a concrete T, "passes" vs
"fails" cannot be decided deterministically.

Output: additive edits to PROJECT.md and STATE.md, no other content touched.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/STATE.md
@.planning/REQUIREMENTS.md
@.planning/ROADMAP.md
@.planning/phases/10-scaffolding/10-CONTEXT.md
@.planning/milestones/v2.0-MILESTONE-AUDIT.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Codify GATE-01 cross-model in PROJECT.md Key Decisions</name>
  <files>.planning/PROJECT.md</files>
  <read_first>
    - .planning/PROJECT.md (specifically the "## Key Decisions" section starting at line 111 — the table runs from line 113 onward; the additive row must land at the end of that table BEFORE the "## Evolution" section starts at line 126)
    - .planning/REQUIREMENTS.md (GATE-01 wording around lines 26-27 — "gpt-5.4 macro ≥ 0.9077 within tolerance T (T defined in milestone, e.g. ≤ 1pp regression)")
    - .planning/STATE.md (Standing Gates section starting at line 49 — read existing GATE-01 cross-model row to understand the current loose phrasing this plan replaces)
    - .planning/phases/10-scaffolding/10-CONTEXT.md (Decisions block: "Cross-model tolerance T = 1pp (matches REQUIREMENTS ≤ 1pp). Logged in PROJECT.md Key Decisions and STATE.md Standing Gates" — explicit instruction)
    - .planning/phases/10-scaffolding/10-CONTEXT.md <specifics>: "Tolerance value committed as exactly 1.0pp (not ≤ 1pp — needs a concrete number for the gate check)"
    - .planning/milestones/v2.0-MILESTONE-AUDIT.md (cross-reference for the 0.9077 v2.0 CROSS baseline provenance — search for "0.9077" and "gpt-5.4")
  </read_first>
  <action>
    Insert a new row at the END of the Markdown table in .planning/PROJECT.md "## Key Decisions" section (the table that begins at line 113 with header `| Decision | Rationale | Outcome |`). Do NOT modify, reorder, or delete any existing row.

    The new row, exact content (preserve pipe spacing of surrounding rows):

      | GATE-01 cross-model tolerance T = 1.0pp (v2.1) | Pins the loose REQUIREMENTS GATE-01 phrasing "≤ 1pp regression" to a concrete numeric tolerance so Phase 12 trim acceptance and Phase 13 promotion sweeps can be evaluated deterministically. Baseline 0.9077 is the v2.0 CROSS evidence on gpt-5.4 (see v2.0-MILESTONE-AUDIT.md "09-CROSS-REPORT.md §GATE-01"). T = 1.0pp means a variant passes iff gpt-5.4 macro F1 ≥ 0.9077 − 0.01 = 0.8977 absolute on the full 5-dataset sweep. | Codified 2026-05-31 (Phase 10, Plan 10-04) |

    Notes:
      - The decision name field uses an em-dash style consistent with the existing "KEEP _has_standalone_mention in s_linker13" row.
      - Tolerance arithmetic explicitly stated (0.9077 − 0.01 = 0.8977) so reviewers and future Claude/agents do not have to reconstruct it.
      - Outcome column matches the existing pattern "Codified <date> (<phase ref>)".

    Do NOT:
      - Add anything outside the Key Decisions table
      - Reorder existing rows
      - Modify the "## Evolution" section, "Constraints", or any other section
      - Use the loose phrasing "≤ 1pp" — must be the concrete "1.0pp" with the absolute floor stated
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 && python -c "import pathlib,re; t=pathlib.Path('.planning/PROJECT.md').read_text(); assert 'GATE-01 cross-model tolerance T = 1.0pp' in t, 'decision row missing'; assert '0.9077' in t and '0.8977' in t, 'baseline + computed absolute floor must both appear'; assert 'Phase 10, Plan 10-04' in t, 'provenance missing'; assert '## Key Decisions' in t and '## Evolution' in t, 'sections preserved'; ki=t.index('## Key Decisions'); ei=t.index('## Evolution'); assert ki < t.index('GATE-01 cross-model tolerance T = 1.0pp') < ei, 'row not inside Key Decisions table'; print('PROJECT.md OK')"</automated>
  </verify>
  <acceptance_criteria>
    - File .planning/PROJECT.md contains the literal substring "GATE-01 cross-model tolerance T = 1.0pp"
    - The same file contains both "0.9077" and "0.8977" (baseline and explicit absolute floor)
    - The new row appears between "## Key Decisions" and "## Evolution" section headers
    - "## Evolution" section is byte-identical to its pre-edit state
    - "## Constraints" section is byte-identical to its pre-edit state
    - `git diff .planning/PROJECT.md` shows additions only — no deletions of existing Key Decisions rows
    - The new row references "Phase 10, Plan 10-04" provenance
  </acceptance_criteria>
  <done>
    PROJECT.md Key Decisions table has the additive GATE-01 cross-model row with concrete tolerance 1.0pp and absolute F1 floor 0.8977.
  </done>
</task>

<task type="auto">
  <name>Task 2: Replace loose tolerance in STATE.md Standing Gates with concrete T = 1.0pp</name>
  <files>.planning/STATE.md</files>
  <read_first>
    - .planning/STATE.md (Standing Gates section at lines 49-55; the current GATE-01 cross-model line at line 52 reads: `- GATE-01 cross-model (v2.1 NEW): gpt-5.4 macro ≥ 0.9077 within ≤ 1pp tolerance (T to be committed in Phase 10)` — this is the exact line to replace; do not change any other line)
    - .planning/REQUIREMENTS.md GATE-01 (source of the original loose phrasing)
    - .planning/phases/10-scaffolding/10-CONTEXT.md (instructs the concrete 1.0pp commitment)
    - The edit produced by Task 1 in .planning/PROJECT.md (to keep wording consistent across the two locations — same baseline 0.9077, same tolerance 1.0pp, same absolute floor 0.8977)
  </read_first>
  <action>
    Edit .planning/STATE.md Standing Gates section. Replace the existing single line (currently at line 52):

      `- GATE-01 cross-model (v2.1 NEW): gpt-5.4 macro ≥ 0.9077 within ≤ 1pp tolerance (T to be committed in Phase 10)`

    with the concrete-tolerance version:

      `- GATE-01 cross-model (v2.1): gpt-5.4 macro ≥ 0.9077 within ≤ 1.0pp tolerance — i.e. variant passes iff gpt-5.4 macro F1 ≥ 0.8977 on full 5-dataset sweep (T = 1.0pp committed Phase 10 Plan 10-04; baseline 0.9077 from v2.0 CROSS evidence; see PROJECT.md Key Decisions row "GATE-01 cross-model tolerance T = 1.0pp (v2.1)")`

    Do NOT:
      - Modify or remove any other Standing Gates bullet (GATE-01 Claude row, GATE-02, GATE-06, GATE-07 must stay byte-identical)
      - Modify the YAML frontmatter
      - Modify any other section (Current Position, Performance Metrics, Accumulated Context, etc.)
      - Use the loose phrasing "T to be committed" — that string must disappear from the file

    Compatibility note: the replacement line MUST contain the literal substring "gpt-5.4 macro ≥ 0.9077 within ≤ 1.0pp tolerance" so the GATE-02 regression test (Plan 10-01) and any future grep-based audit can find it with a single pattern.
  </action>
  <verify>
    <automated>cd /mnt/hostshare/ardoco-home/llm-sad-sam-v45 && python -c "import pathlib; t=pathlib.Path('.planning/STATE.md').read_text(); assert 'gpt-5.4 macro ≥ 0.9077 within ≤ 1.0pp tolerance' in t, 'committed concrete tolerance line missing'; assert '0.8977' in t, 'absolute floor missing'; assert 'T = 1.0pp committed Phase 10 Plan 10-04' in t, 'provenance missing'; assert 'T to be committed in Phase 10' not in t, 'loose phrasing must be removed'; assert '## Standing Gates' in t, 'section header preserved'; for g in ['GATE-02 (v2.1 NEW)','GATE-06','GATE-07']: assert g in t, f'sibling gate {g} broken'; print('STATE.md OK')"</automated>
  </verify>
  <acceptance_criteria>
    - File .planning/STATE.md contains the literal substring "gpt-5.4 macro ≥ 0.9077 within ≤ 1.0pp tolerance"
    - File contains the absolute floor "0.8977"
    - File contains the provenance string "T = 1.0pp committed Phase 10 Plan 10-04"
    - File does NOT contain the obsolete loose phrasing "T to be committed in Phase 10" anywhere
    - GATE-02 v2.1 NEW row, GATE-06 row, GATE-07 row in Standing Gates section all remain byte-identical
    - GATE-01 Claude row (line 51 currently) remains byte-identical
    - YAML frontmatter unmodified
    - `git diff --stat .planning/STATE.md` shows exactly one file changed with additions ≤ 3 lines and deletions ≤ 2 lines (precise single-line replacement)
  </acceptance_criteria>
  <done>
    STATE.md Standing Gates contains the concrete tolerance commitment with absolute floor and provenance pointer; all sibling gates untouched.
  </done>
</task>

</tasks>

<verification>
1. `grep -F "GATE-01 cross-model tolerance T = 1.0pp" .planning/PROJECT.md` returns 1 match
2. `grep -F "gpt-5.4 macro ≥ 0.9077 within ≤ 1.0pp tolerance" .planning/STATE.md` returns 1 match
3. `grep -F "0.8977" .planning/PROJECT.md .planning/STATE.md` returns 2 matches (one per file)
4. `grep -F "T to be committed in Phase 10" .planning/STATE.md` returns 0 matches
5. Both files identify the provenance as Phase 10 Plan 10-04
</verification>

<success_criteria>
- Concrete tolerance T = 1.0pp pinned in both PROJECT.md Key Decisions and STATE.md Standing Gates
- Absolute F1 floor 0.8977 stated explicitly in both files (= 0.9077 − 0.01)
- Provenance "Phase 10, Plan 10-04" recorded in both files
- All sibling gates (GATE-01 Claude, GATE-02, GATE-06, GATE-07) untouched
- All other sections of both files unchanged
</success_criteria>

<output>
After completion, create `.planning/phases/10-scaffolding/10-04-SUMMARY.md` recording:
- Exact diff (insertion/replacement) applied to PROJECT.md
- Exact diff (single-line replacement) applied to STATE.md
- The committed values: baseline 0.9077, tolerance 1.0pp, absolute floor 0.8977
- Confirmation no other section was modified in either file
</output>
