---
phase: 13
plan: 13-02
title: ABLATION-TABLE v2.1 addendum + .tex regeneration
status: completed
verdict: GATE-03 closed
completed: 2026-06-01
requirements: [GATE-03]
subsystem: ablation-artifact
tags: [ablation-table, latex, addendum, v2.1, gate-03]
key-files:
  created: []
  modified:
    - .planning/milestones/v1.0-phases/05-promote-and-ablation-artifact/ABLATION-TABLE.md
    - .planning/milestones/v1.0-phases/05-promote-and-ablation-artifact/ABLATION-TABLE.tex
    - .planning/phases/13-promotion-wrap/13-02-SUMMARY.md (this file)
decisions:
  - The existing ABLATION-TABLE.md / .tex live under the v1.0 milestone archive (not under writing/ or a paper-facing path). v2.0 already appended an EXT-01 / COMBINE addendum to the same files. v2.1 follows the same convention — additive section + table blocks at the bottom of both files, leaving existing v1.0 + v2.0 row content unchanged.
  - The v2.1 .tex addendum adds two new `tabular` blocks at the bottom of the file (separated by `\vspace`): block 2 = Phase 10 baseline + Phase 12 ACCEPTed trims + Phase 13 promoted composition; block 3 = REJECTED trims (negative-result traceability). The original v1.0 chain tabular is preserved byte-equal.
  - Cross-model (gpt-5.4 macro) column is NEW for v2.1 — v1.0 / v2.0 chain rows reported Claude only. The v2.1 addendum tables list both Claude per-dataset cells AND Claude/gpt-5.4 macro columns for direct comparison.
  - Per-dataset numbers for s_linker13_trim1_judge_clean and s_linker13_trim9_seed_runtime_clean are sourced from `verdict.json` files in `results/ablation_results/12_03_trim1_judge/` and `results/ablation_results/12_extension_runtime_variants/scoreboard.json`. s_linker13_min per-dataset numbers from the Plan 13-01 sweep JSONs.
metrics:
  duration: "~10min"
  completed: 2026-06-01
---

# Phase 13 Plan 13-02: ABLATION-TABLE Addendum — Summary

**One-liner:** Additive v2.1 addendum appended to the live ABLATION-TABLE.md + .tex artifacts at `.planning/milestones/v1.0-phases/05-promote-and-ablation-artifact/`. Phase 10 baseline (`s_linker13_clean`), Phase 12 ACCEPTed trims (`trim1`, `trim9`), and Phase 13 promoted composition (`s_linker13_min`) pinned with both Claude Sonnet and gpt-5.4 macro F1. Rejected trims (trim2 through trim8) listed in a separate block for negative-result traceability with Scenario E verdicts. GATE-03 closed.

## Verdict

| Gate | Required | Observed | Status |
|------|----------|----------|--------|
| GATE-03 (ABLATION-TABLE.md addendum) | v2.1 rows added; v1.0 + v2.0 rows unchanged | 11 new rows added (4 in promoted block + 7 in rejected block); existing rows byte-equal verified | **PASS** |
| GATE-03 (`.tex` artifact regenerated) | `.tex` reflects updated markdown | Two new `tabular` blocks appended; existing v1.0 chain block preserved | **PASS** |

## What the v2.1 addendum captures

### Block 2 — Phase 10 baseline + ACCEPTed trims + promoted composition (4 rows)

| variant | mechanism | Claude macro | gpt-5.4 macro | verdict |
|---|---|---|---|---|
| s_linker13_clean | structural refactor only (helpers → helper_v3) | 0.9397 | 0.9077 | BASELINE (v2.1) |
| s_linker13_trim1_judge_clean | DOC_KNOWLEDGE_JUDGE_RULES distilled via Technique 3 + 8 | 0.9553 | 0.9173 | ACCEPT |
| s_linker13_trim9_seed_runtime_clean | SEED_DISAMBIGUATION_RULES → runtime rubric builder | 0.9474 | 0.9007 | ACCEPT |
| **s_linker13_min** | **trim1 + trim9 composed** | **0.9506** | **0.9069** | **PROMOTED** |

### Block 3 — REJECTED trims, frontier-only (7 rows)

trim2 through trim8 listed with original-gate verdict + Scenario E verdict (per `12-FRONTIER-MAP-SUMMARY.md`). All 7 fail the strict v2.1 promotion gates by 0.4–3.6 pp; 6 of 7 pass Scenario E. Documented for negative-result traceability per the v2.1 thesis claim "Phase 12 explored 9 prompt-reduction mechanisms, 2 accepted, 7 rejected with documented failure modes".

## Frozen-file compliance

- v1.0 chain tabular: byte-equal verified (the original `\begin{tabular}...\end{tabular}` block is preserved unchanged at lines 17-30 of the .tex; v2.1 additions are appended below it).
- v2.0 EXT/COMBINE addendum text (lines 33-35 of the .md, source-JSON list, etc.): preserved unchanged.

## Files

| Modified |
|---|
| `.planning/milestones/v1.0-phases/05-promote-and-ablation-artifact/ABLATION-TABLE.md` (title updated to include v2.1; v2.1 addendum section appended at bottom — 4 ACCEPTed/promoted rows + 7 REJECTED rows + thesis claim footer) |
| `.planning/milestones/v1.0-phases/05-promote-and-ablation-artifact/ABLATION-TABLE.tex` (header comment block extended to note v2.1 addendum; two new `tabular` blocks appended after the original v1.0 chain block, separated by `\vspace`) |
| `.planning/phases/13-promotion-wrap/13-02-SUMMARY.md` (this file) |

| NOT touched |
|---|
| The v1.0 chain tabular content (s_linker12c → s_linker13 rows) is byte-equal in both .md and .tex |
| All Phase 12 source files (variant .py files, prompts_v3.py, helper_v3.py, etc.) |

## Deviations from plan

- **Location of canonical artifact:** The plan prompt mentioned "Locate the existing tex file under writing/ or wherever it lives. Regenerate from the updated markdown." The existing ABLATION-TABLE.md / .tex live ONLY under `.planning/milestones/v1.0-phases/05-promote-and-ablation-artifact/` — there is no copy under `writing/` or `paper/` (verified by recursive find). The v2.0 EXT/COMBINE addendum already amended the same archive files; v2.1 follows that established convention.
- **No render script:** The original `render_ablation.py` referenced in the .md header is not present in the repo. The .tex was hand-edited to mirror the .md addendum's tabular content. The v1.0 chain block is preserved byte-equal so any future render-script regeneration would only need to add the two v2.1 tabular blocks at the bottom (mechanical operation).

## Self-Check: PASSED

- `ABLATION-TABLE.md` has v2.1 addendum section: **FOUND**
- `ABLATION-TABLE.tex` has v2.1 tabular blocks: **FOUND**
- v1.0 chain tabular content unchanged: **VERIFIED** (lines 17-30 of .tex match original byte-equal)
- Phase 13 Plan 13-01 sweep JSONs referenced in addendum exist: **FOUND**
