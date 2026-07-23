---
phase: 46-minimize
plan: "07"
subsystem: prompt-minimization
section_anchor: COR
tags: [minimize, cor, scratch-trial, gate-01, gate-06, batched-trial, lexicon-handoff, tombstone]
dependency_graph:
  requires:
    - 46-01 (scratch bootstrap + tombstone pre-fill)
    - 46-06 (CUT-VAL-03 — VAL-03 -> COR-01 lexicon handoff `noun phrase that refers back`, sha 8c195bc)
  provides:
    - COR section of MINIMIZE-LOG populated (4 trialled rows + tombstone backfill + closing blockquote)
    - VAL-03 -> COR-01 lexicon handoff CLOSED across 3 sites (COREF_VALIDATION_FOCUS, COREF_RULES, _prompt_coref opener+inline)
    - CUT-COR-05 tombstone protected (sha 7b153fa)
    - Largest-LOC-saved harness signal in the audit (40 snapshots × 4 kept cuts = strongest empirical pressure in Phase 46)
  affects:
    - 46-08 (Pareto Summary rolls COR section verdicts + REQ-V264-05/06/07 tick-off + final GATE-01 record)
    - 47 (SHIP) — kept-cut after-text inlined from tests/scratch/{s_linker19.py, prompts_v5.py}
tech_stack:
  added: []
  patterns:
    - "Risk-ascending trial within section (D-02): CUT-COR-02 (med) -> CUT-COR-01 (med-high, paired w/ VAL-03) -> CUT-COR-03+04 batched (med-high) -> CUT-COR-05 (protected)"
    - "Universal-noun replacement extended (`role-referential noun phrase` / `anaphoric reference` -> `noun phrase that refers back` / `such reference`)"
    - "Audit-mandated batched-trial lockstep (CUT-COR-03 + CUT-COR-04 share commit_sha per D-04 batched-trial rule + audit line 348)"
    - "VAL-03 -> COR-01 lexicon handoff (cross-plan vocabulary read from MINIMIZE-LOG)"
    - "Allow-empty docs commit for tombstone protection + separate bookkeeping commit for SHA backfill (no --amend)"
key_files:
  created:
    - .planning/phases/46-minimize/46-07-SUMMARY.md
  modified:
    - tests/scratch/prompts_v5.py (line 102 COREF_RULES — CUT-COR-02 + CUT-COR-01, two cumulative edits)
    - tests/scratch/s_linker19.py (lines 362 + 366-369 _prompt_coref opener + inline — CUT-COR-03 + CUT-COR-04 batched)
    - .planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md (COR section table 4 rows + tombstone SHA backfill + closing blockquote)
decisions:
  - "CUT-COR-02 kept: COREF_RULES body content (not opener) — reconstructor unaffected; 40/40 pass; section-scope semantics + no-direct-name-repetition exemption preserved"
  - "CUT-COR-01 kept: VAL-03 lexicon `noun phrase that refers back` applied verbatim to COREF_RULES line 102 first clause; 40/40 pass; other (later) `role-referential phrases` quoted-list site intentionally out of scope"
  - "CUT-COR-03 + CUT-COR-04 batched-trial kept: shared commit_sha f8f873f; both verdicts identical (kept); single Edit-tool invocation; opener at line 362 + inline at lines 366-369 lockstep-rewritten per audit line 348 mandate"
  - "CUT-COR-05 protected: behavioral conservatism dial empirically validated load-bearing per CLAUDE.md v2.6.2 milestone (s17e FP 43->14 via validation gating); allow-empty docs commit + separate bookkeeping backfill commit"
  - "VAL-03 -> COR-01 lexicon handoff CLOSED across 3 sites: COREF_VALIDATION_FOCUS (VAL-03), COREF_RULES (COR-01), and _prompt_coref opener+inline (COR-03+04)"
metrics:
  duration: "~10 min wall-clock"
  completed_date: "2026-06-08"
  trials_attempted: 3
  trials_kept: 3
  trials_reverted: 0
  trials_unsafe: 0
  cuts_attempted: 4
  cuts_kept: 4
  tombstones_protected: 1
  snapshots_passed: "40/40 per trial (×3 trials = 120 total snapshot passes)"
  loc_saved_section: 0
  commits_emitted: 6
---

# Phase 46 Plan 07: COR Section Trial Cuts — Summary

COR section of `s_linker20-MINIMIZE-LOG.md` populated with 5 rows in D-02 risk-ascending order (CUT-COR-02 → CUT-COR-01 → CUT-COR-03+04 batched-trialled; CUT-COR-05 tombstone protected). All four trialled cuts **kept** with 40/40 snapshots passing under `SAD_SAM_LINKER_SOURCE=scratch` and clean GATE-06 re-grep against `BENCHMARK_TABOO.md`. CUT-COR-03 + CUT-COR-04 share commit `f8f873f` per the audit-mandated batched-trial rule (audit line 348). VAL-03 → COR-01 lexicon handoff CLOSED: the replacement vocabulary `noun phrase that refers back` from 46-06's CUT-VAL-03 (sha 8c195bc) is now applied across THREE sites — `COREF_VALIDATION_FOCUS` (VAL-03), `COREF_RULES` (COR-01), and `_prompt_coref` opener+inline (COR-03+04).

## Per-cut verdicts

| cut_id | verdict | snapshot_delta | gate06 | loc_saved | commit_sha | note |
|---|---|---|---|---|---|---|
| CUT-COR-02 | kept | 0/40 | clean | 0 | `55561dc` | `section-established topic` → `topic of the surrounding section` (COREF_RULES body) |
| CUT-COR-01 | kept | 0/40 | clean | 0 | `d320c03` | `role-referential noun phrase` → `noun phrase that refers back` (COREF_RULES line 102 first clause); VAL-03 lexicon honored |
| CUT-COR-03 | kept | 0/40 | clean | 0 | `f8f873f` | `_prompt_coref` opener (line 362) batched with CUT-COR-04; shared SHA per D-04 batched-trial |
| CUT-COR-04 | kept | 0/40 | clean | 0 | `f8f873f` | `_prompt_coref` inline restatement (lines 366-369) batched with CUT-COR-03; shared SHA |
| CUT-COR-05 | protected | n/a | n/a | 0 | `7b153fa` | Conservatism dial tombstone — behaviorally protected, NOT trialled |

## Commits emitted (6 total)

1. `55561dc` — `feat(46-07): keep CUT-COR-02 — section-established topic -> topic of the surrounding section`
2. `d320c03` — `feat(46-07): keep CUT-COR-01 — role-referential noun phrase -> noun phrase that refers back (lockstep with VAL-03)`
3. `f8f873f` — `feat(46-07): keep CUT-COR-03 + CUT-COR-04 — anaphoric references batched rewrite (lockstep per audit line 348)` (BATCHED commit — both CUT-COR-03 and CUT-COR-04 LOG rows reference this SHA)
4. `7b153fa` — `docs(46-07): protect CUT-COR-05 — coref conservatism dial behaviorally protected` (allow-empty)
5. `ea638f4` — `docs(46-07): backfill CUT-COR-05 tombstone commit_sha`
6. `c166d6e` — `docs(46-07): COR section closing note + VAL-03 cross-site outcome`

## CUT-COR-03 + CUT-COR-04 batched-trial outcome (LOCKSTEP HONORED)

| field | value |
|---|---|
| audit mandate | line 348 — "Phase 46 MUST batch the two cuts as a single semantic unit … a Phase 46 rewrite that touches only one of the two sites will produce a self-contradictory prompt" |
| commit_sha shared | `f8f873f` (both rows reference identical SHA) |
| edit invocation | single Edit-tool call rewriting both line-362 opener + lines-366-369 inline atomically |
| verdict identity | both `kept` (per D-04 batched-trial rule: verdict applies to BOTH cuts — kept means both, reverted means both) |
| snapshot delta | 0/40 on a single `SAD_SAM_LINKER_SOURCE=scratch pytest tests/test_s_linker20_prompt_coref.py` invocation |
| GATE-06 evidence | single merged token set: `resolve|references|pronouns|noun|phrases|refer|back|components|target|sentence|identify|pronoun|phrase|listed|above|such|reference|return|resolution` — all 0 hits except `components` (5 hits = per-dataset `Components:` schema column headers, cleared per Phase 45 v2.1 isolation precedent / cross-section pleonasm batch reasoning) |

## VAL-03 → COR-01 lexicon handoff verification (CLOSED across 3 sites)

The shared replacement vocabulary **`noun phrase that refers back`** was chosen by CUT-VAL-03 (sha `8c195bc`, 46-06) and read from:
1. The CUT-VAL-03 row's `reasoning` cell in `s_linker20-MINIMIZE-LOG.md` ("Replacement vocabulary `noun phrase that refers back` chosen and committed in this row …"), AND
2. The VAL section closing blockquote in `s_linker20-MINIMIZE-LOG.md` (46-06 commit `80784b6`), AND
3. The `46-06-SUMMARY.md` VAL-03 → COR-01 lexicon handoff section.

This plan applied the SAME wording to three sites:

| site | constant / location | cut_id | sha | clause |
|---|---|---|---|---|
| 1 | `COREF_VALIDATION_FOCUS` (prompts_v5.py:96) | CUT-VAL-03 | `8c195bc` (46-06) | `or similar noun phrase that refers back in this sentence actually refer to…` |
| 2 | `COREF_RULES` (prompts_v5.py:102 first clause) | CUT-COR-01 | `d320c03` (this plan) | `whether a pronoun or noun phrase that refers back in the target sentence refers back to a component…` |
| 3a | `_prompt_coref` opener (s_linker19.py:362) | CUT-COR-03 | `f8f873f` (this plan) | `Resolve references (pronouns and noun phrases that refer back) to components.` |
| 3b | `_prompt_coref` inline (s_linker19.py:366-369) | CUT-COR-04 | `f8f873f` (this plan) | `identify any pronoun or noun phrase that refers back to a component listed above…` |

The vocabulary is now lexically consistent across the entire COREF surface. The OTHER occurrence of `role-referential phrases` later in `COREF_RULES` (the quoted list site `resolve role-referential phrases ("it", "the module", "the service", "the component", "the system")`) is intentionally **untouched** — it's a different clause within the same constant outside CUT-COR-01's audit-defined scope (row 336).

## Total LOC saved in COR section

**0** — all four trialled cuts are substring rewordings within single dense lines / multi-line f-string literals. No whole-line removal. (CUT-COR-04's inline restatement reflowed from 4 lines to 4 lines due to natural f-string line wrap; net = 0 line delta.) Tombstone counts as 0 per Phase 46 convention. The Pareto Summary (46-08, Wave 3) will report COR section LOC saved = 0.

## Behavioral-vs-harness caveat reminder (46-RESEARCH §6.3)

COR is the **FP-sensitive stage** of the SAD-SAM pipeline per CLAUDE.md v2.6.2 milestone notes: `s_linker17e` drove FP 43→14 (a 67% reduction) via single-pass validation gating before Phase 6 coref. The prompt-side conservatism instruction at `s_linker19.py:361` (`Be conservative — only include resolutions you are CERTAIN about.`, now wrapped to lines 368-369 after CUT-COR-04's reflow but content byte-identical) is the prompt-side counterpart of the validation-side gating closed by the v2.6.2 breakthrough.

Harness-kept verdicts in COR are the **strongest possible Phase-46 signal** because the gating is 40 snapshots — more than 1.6× VAL's 24 and more than 2× any other section in the audit. **But they remain HARNESS verdicts ONLY**: cached-replay snapshots are invariant under prompt cuts because replay parsing depends only on cached `response_text` (per 46-RESEARCH §4.4 + Phase 44 §D-03), and so the 40-snapshot pass validates harness compatibility + GATE-06 vocabulary cleanliness, NOT model behavior on live LLM calls. **Phase 48 sweep is authoritative for behavioral safety** — particularly for COR cuts, where the FP class closed by v2.6.2 s17e validation gating is most at risk from prompt rewording.

## Per-cut isolation evidence

### CUT-COR-02 — section-established topic → topic of the surrounding section
- **Before:** `…treat it as the section-established topic and resolve role-referential phrases ("it", "the module", …) to that topic…` (tests/scratch/prompts_v5.py:102, COREF_RULES body)
- **After:** `…treat it as the topic of the surrounding section and resolve role-referential phrases ("it", "the module", …) to that topic…`
- **Section-scope semantics preserved**; no-direct-name-repetition exemption + quoted role-referential placeholder list untouched
- **Harness compatibility:** COREF_RULES is body content (not opener); `reconstruct_coref_inputs` anchors on `^COMPONENTS:` + `--- Case N: ---` patterns — unaffected
- **40/40 snapshots passed**
- **GATE-06 re-grep:** `topic` / `surrounding` / `section` → 0 hits in BENCHMARK_TABOO.md

### CUT-COR-01 — role-referential noun phrase → noun phrase that refers back (lockstep with VAL-03)
- **Before:** `For each case, decide whether a pronoun or role-referential noun phrase in the target sentence refers back to a component…` (tests/scratch/prompts_v5.py:102, COREF_RULES body, first clause)
- **After:** `For each case, decide whether a pronoun or noun phrase that refers back in the target sentence refers back to a component…`
- **Lexicon lockstep:** replacement string read from CUT-VAL-03 row (sha 8c195bc, 46-06) — same wording applied to both `COREF_VALIDATION_FOCUS` and `COREF_RULES`
- **Out-of-scope sibling:** the OTHER `role-referential phrases` occurrence in COREF_RULES (the quoted-list site) is a different clause and intentionally untouched
- **40/40 snapshots passed**
- **GATE-06 re-grep:** `noun`/`phrase`/`refers`/`back`/`target`/`sentence` all 0 hits; `component` hits are bare generic SE noun anaphor (cleared per Phase 45 v2.1 / CUT-AMB-02 / CUT-VAL-03 precedent)

### CUT-COR-03 + CUT-COR-04 — anaphoric references batched rewrite (lockstep per audit line 348)
- **CUT-COR-03 (opener, line 362):**
  - Before: `Resolve anaphoric references (pronouns and role-referential noun phrases) to architecture components.`
  - After:  `Resolve references (pronouns and noun phrases that refer back) to components.`
- **CUT-COR-04 (inline, lines 366-369):**
  - Before: `For each TARGET sentence below, identify any pronoun or role-referential noun phrase that refers back to a component listed above. If a target sentence has no anaphoric reference to a listed component, return no resolution for it.`
  - After:  `For each TARGET sentence below, identify any pronoun or noun phrase that refers back to a component listed above. If a target sentence has no such reference to a listed component, return no resolution for it.`
- **Universal-noun vocabulary:** `pronoun or noun phrase that refers back` (shared with VAL-03 / COR-01); `anaphoric reference` → `such reference` (antecedent carries anaphora semantics); `architecture` dropped per cross-section batch (`components` bare, per CUT-AMB-02 / CUT-EXT-01 / CUT-VAL-02 precedent)
- **CUT-COR-05 conservatism dial preserved verbatim** — line-wrapped to lines 368-369 after natural f-string reflow but content byte-identical
- **Harness compatibility:** `reconstruct_coref_inputs` anchors on `^COMPONENTS:` + `--- Case N: ---` — both edits harness-safe per 46-RESEARCH §6.2
- **40/40 snapshots passed** on a single pytest invocation (batched-trial)
- **GATE-06 re-grep:** merged token set — all 0 hits except `components` (5 hits = `Components:` schema column headers, cleared per Phase 45 v2.1 / cross-section pleonasm batch precedent)

### CUT-COR-05 — conservatism dial tombstone (protected, NOT trialled)
- **Protected clause:** `Be conservative — only include resolutions you are CERTAIN about.` (tests/scratch/s_linker19.py, originally line 361, now wrapped to lines 368-369 after CUT-COR-04 reflow — content byte-identical)
- **Empirical evidence (CLAUDE.md v2.6.2 milestone notes):** s_linker17e drove FP 43→14 (a 67% reduction) via single-pass validation gating; the prompt-side conservatism instruction is the prompt-side counterpart of the validation-side gating
- **Threat:** Phase 45 T-45-COR-02 — Phase 46 MUST NOT cut
- **No scratch edits attempted.** Allow-empty docs commit (`7b153fa`) emitted with body quoting before-text + empirical evidence + threat reference + post-batched-trial verification. Separate bookkeeping commit (`ea638f4`) backfills the `(assigned by 46-07)` placeholder in the Protected Tombstones section row.

## GATE-01 verification

`git diff --stat src/llm_sad_sam/linkers/experimental/s_linker19.py src/llm_sad_sam/linkers/experimental/prompts_v5.py src/llm_sad_sam/linkers/experimental/s_linker13_min.py` returns empty after every commit (continuous GATE-01 hold across all 6 commits in this plan). The frozen sources are byte-equal to HEAD throughout — all cut work lives in `tests/scratch/` mirrors.

## Deviations from plan

None. Plan executed exactly as written:
- D-02 risk-ascending order honored within COR (CUT-COR-02 → CUT-COR-01 → CUT-COR-03+04 batched → CUT-COR-05 tombstone).
- 6 commits emitted (matches plan upper-bound spec: 1 for CUT-COR-02, 1 for CUT-COR-01, 1 for CUT-COR-03+04 batched, 1 for CUT-COR-05 protect, 1 for tombstone backfill, 1 for closing note).
- All four trialled cuts kept on first attempt; no reverts, no unsafe verdicts.
- CUT-COR-05 tombstone never edited in scratch (per CONTEXT in-scope §).
- CUT-COR-03 + CUT-COR-04 batched-trial lockstep honored per audit line 348 (single Edit-tool invocation, single pytest run, identical verdict, shared commit_sha).
- VAL-03 → COR-01 lexicon handoff applied verbatim per 46-06's CUT-VAL-03 outcome.

## Self-Check: PASSED

| Item | Result |
|------|--------|
| `tests/scratch/prompts_v5.py` line 102 COREF_RULES contains `topic of the surrounding section` | FOUND |
| `tests/scratch/prompts_v5.py` line 102 COREF_RULES first clause contains `pronoun or noun phrase that refers back in the target sentence refers back to a component` | FOUND |
| `tests/scratch/s_linker19.py` line 362 opener = `Resolve references (pronouns and noun phrases that refer back) to components.` | FOUND |
| `tests/scratch/s_linker19.py` lines 366-369 inline restatement contains `any pronoun or noun phrase that refers back to a component listed above` + `has no such reference` | FOUND |
| `tests/scratch/s_linker19.py` conservatism dial (`Be conservative — only include resolutions you are CERTAIN about.`) preserved (wrapped to lines 368-369 after CUT-COR-04 reflow; content byte-identical) | FOUND |
| `.planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md` COR section table has 4 trialled rows (CUT-COR-02, CUT-COR-01, CUT-COR-03, CUT-COR-04) | FOUND |
| `.planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md` CUT-COR-03 and CUT-COR-04 rows share commit_sha `f8f873f` | FOUND |
| `.planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md` COR closing blockquote present | FOUND |
| `.planning/phases/46-minimize/s_linker20-MINIMIZE-LOG.md` CUT-COR-05 tombstone row commit_sha = `7b153fa` (placeholder removed) | FOUND |
| Commit `55561dc` (CUT-COR-02 kept) in `git log` | FOUND |
| Commit `d320c03` (CUT-COR-01 kept) in `git log` | FOUND |
| Commit `f8f873f` (CUT-COR-03 + CUT-COR-04 batched) in `git log` | FOUND |
| Commit `7b153fa` (CUT-COR-05 protect, allow-empty docs) in `git log` | FOUND |
| Commit `ea638f4` (CUT-COR-05 SHA backfill bookkeeping) in `git log` | FOUND |
| Commit `c166d6e` (COR closing note) in `git log` | FOUND |
| GATE-01 final: `git diff --stat` on three frozen sources empty | PASS (zero output) |
| Post-commit deletion check (`git diff --diff-filter=D HEAD~6 HEAD`) | empty (zero deletions) |
