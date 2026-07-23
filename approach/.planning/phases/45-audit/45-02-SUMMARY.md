---
phase: 45-audit
plan: 02
subsystem: prompt-audit
section: AMB
tags: [prompt-audit, ambiguity, phase-1, gate-01, byte-equal]
requires:
  - 45-01 (AMB anchors and Verdict Summary skeleton)
provides:
  - AMB section of s_linker20-PROMPT-AUDIT.md (header table + cut table + reviewer judgment)
  - CUT-AMB-01 (drop-block, high risk)
  - CUT-AMB-02 (domain-loaded flag on `_prompt_ambiguity` line 266, low risk)
affects:
  - .planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md (AMB anchors + Verdict Summary rows 1-3 only)
tech_stack:
  added: []
  patterns: [mechanical-taboo-grep, v2.1-GATE-06-isolation, D-01-pragmatic-rubric, D-04-strict-fewshot, REQ-V264-06-drop-block]
key_files:
  created:
    - .planning/phases/45-audit/45-02-SUMMARY.md
  modified:
    - .planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md
decisions:
  - "AMBIGUITY_FEW_SHOT verdict = clean: every token grepped against all 5 dataset sections + Universal Taboo with zero per-dataset hits; `Scheduler` is on the Safe SE Textbook Examples list (BENCHMARK_TABOO.md:63)."
  - "AMBIGUITY_RULES verdict = clean: the only Universal-Taboo overlap (`component`) is the generic SE noun and passes the v2.1 GATE-06 cross-dataset isolation check; no cut row emitted."
  - "_prompt_ambiguity verdict = domain-loaded on line 266 only (`software architecture component names`); D-01 pragmatic test holds because the NAMES: slot at line 268 already constrains scope."
  - "CUT-AMB-01 drop-block emitted per REQ-V264-06 regardless of verdict, with trigger = `drop-block (REQ-V264-06, not benchmark-leak)`; risk = high (5 snapshots, removing the only few-shot is the maximum semantic delta available)."
  - "No Family A / Family B rewording cuts emitted: per D-06 these are gated on a benchmark-leak verdict, which AMBIGUITY_FEW_SHOT did not earn."
metrics:
  duration_minutes: 8
  completed_date: 2026-06-08
  tasks_completed: 1
  files_modified: 1
  files_created: 1
  cut_rows_emitted: 2
  family_a_rows: 0
  family_b_rows: 0
---

# Phase 45 Plan 02: AMB section audit — AMBIGUITY_FEW_SHOT / AMBIGUITY_RULES / `_prompt_ambiguity`

Filled the AMB section of `s_linker20-PROMPT-AUDIT.md` between `<!-- SECTION:AMB:START -->` and `<!-- SECTION:AMB:END -->` with a 3-row header table (verdicts + LOC + notes) and a 2-row cut table (CUT-AMB-01 drop-block per REQ-V264-06 + CUT-AMB-02 domain-loaded flag), and updated the top-of-doc Verdict Summary rows 1–3 to replace TBD placeholders.

## What was audited

| Item | LOC | Verdict | Cut rows | Driver |
|---|---|---|---|---|
| `AMBIGUITY_FEW_SHOT` (prompts_v5.py:30-36) | 7 | clean | 1 (CUT-AMB-01, drop-block) | REQ-V264-06 mandates drop-block row regardless of verdict |
| `AMBIGUITY_RULES` (prompts_v5.py:38) | 1 | clean | 0 | Single sentence, zero per-dataset taboo hits, `component` passes GATE-06 |
| `_prompt_ambiguity` prose (s_linker19.py:264-282) | 19 (3 prose: 266/268/272) | domain-loaded | 1 (CUT-AMB-02, line 266 opener) | D-01 pragmatic — NAMES: slot at line 268 already constrains scope |

## Mechanical-grep results (Step 1)

Ran `grep -nw` against `BENCHMARK_TABOO.md` for every load-bearing token in the two AMB constants:

- `Scheduler` → hits ONLY at line 63 (Safe SE Textbook Examples). No per-dataset section.
- `queues`, `worker`, `worker threads`, `dispatches`, `dispatching`, `nodes`, `scheduler-based` → zero hits in any section.
- AMBIGUITY_RULES sentence tokens against Universal Taboo: only `component` overlaps, and only as the generic SE noun within "naming a specific component". v2.1 GATE-06 cross-dataset isolation check: does not identify any one project. PASS.

A1 from 45-RESEARCH.md §Assumptions Log is confirmed.

## Cut rows emitted

```
| CUT-AMB-01 | prompts_v5.py:30-36 | drop-block (REQ-V264-06, not benchmark-leak) | full AMBIGUITY_FEW_SHOT block | "" | high | tests/test_s_linker20_prompt_ambiguity.py @ phase_1_model |
| CUT-AMB-02 | s_linker19.py:266   | domain-loaded ("software architecture component names") | Classify these software architecture component names. | [Phase 46 empirical loop] | low | tests/test_s_linker20_prompt_ambiguity.py @ phase_1_model |
```

Family-A and Family-B rewording slots are intentionally empty (D-06 gates them on `benchmark-leak`, which neither constant earned).

## Deviations from Plan

None — plan executed exactly as written. The conditional path in Step 3 ("if Step 1 confirms no per-dataset taboo hits, verdict is `clean` BUT the constant still gets a drop-block CUT-AMB-01") is the path taken. The conditional Family A / Family B emission in Step 5 row B was not triggered because the verdict did not escalate. No Rule 1/2/3 auto-fixes were needed.

## GATE-01 verification

```
$ git diff --quiet src/llm_sad_sam/linkers/experimental/s_linker19.py \
                  src/llm_sad_sam/linkers/experimental/prompts_v5.py \
                  src/llm_sad_sam/linkers/experimental/s_linker13_min.py
$ echo $?
0
```

All three source files byte-equal vs HEAD. GATE-01 holds.

## Automated `<verify>` block

```
$ python3 -c "import re; t=open(...).read(); m=re.search(...); body=m.group(1); ... print('OK', len(cuts), 'rows')"
OK 2 rows
```

All assertions pass:

- header table includes AMBIGUITY_FEW_SHOT, AMBIGUITY_RULES, `_prompt_ambiguity`
- ≥1 `CUT-AMB-NN` row (2 emitted)
- `CUT-AMB-01` drop-block row present
- `phase_1_model` referenced in `gated_by`

## Scope check

- Only content between `<!-- SECTION:AMB:START -->` and `<!-- SECTION:AMB:END -->` and the AMB rows of the top-of-doc Verdict Summary table were modified in `s_linker20-PROMPT-AUDIT.md` (the Verdict Summary update was explicitly mandated by the dispatcher prompt).
- Zero edits to any file under `src/llm_sad_sam/`, `tests/`, or any other path.
- `git diff --stat` on the audit doc shows 27 insertions, 4 deletions — the 4 deletions are the 3 TBD-cell replacements in the Verdict Summary + the placeholder comment inside the AMB anchors.

## Self-Check: PASSED

- File exists: `.planning/phases/45-audit/s_linker20-PROMPT-AUDIT.md` ✓
- AMB anchors populated, both contain the expected tables and reviewer-judgment blockquote ✓
- Verdict Summary AMB rows updated from TBD → clean / clean / domain-loaded with cut counts 1 / 0 / 1 ✓
- GATE-01 byte-equal verified on all three frozen source files ✓
- Per-task commit produced (single atomic commit covering audit doc + this summary) — hash recorded below ✓
