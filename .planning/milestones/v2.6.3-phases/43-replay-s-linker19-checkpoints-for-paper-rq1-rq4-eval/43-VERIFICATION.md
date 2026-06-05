---
phase: 43-replay-s-linker19-checkpoints-for-paper-rq1-rq4-eval
verified: 2026-06-05T01:30:00Z
status: passed
score: 11/11 COVERED (in-scope gaps closed via 43-gap commits 1641bb3 + de30039; out-of-scope RQ2 + LiSSA \todo{} cells deferred per D-13 and scope)
overrides_applied: 0
re_verification:
  previous_status: human_needed
  previous_score: "9/11 covered (2 PARTIAL)"
  gaps_closed:
    - "results.tex §results:summary L109 stale-shape ('three validators ... four agents') → '2 validators ... 2 linkers' (commit 1641bb3)"
    - "eval.tex L50 RQ4 motivation stale 'three linker agents, \\linkerA, \\linkerB, and \\linkerC' → 'two linkers, \\linkerB{} and \\linkerC{}' (commit 1641bb3)"
    - "results.tex L22 RQ1 doc-to-code headline: file F1 0.939, decision F1 0.803, component F1 0.885 (commit de30039)"
    - "results.tex L35 RQ1 rqanswer doc-to-code deltas: +13.6pp file F1, +20.7pp decision F1 (commit de30039)"
    - "results.tex L71 RQ3 per-validator counts: \\entValidator 11 FP / 2 TP, \\corefValidator 3 FP / 1 TP (commit de30039)"
    - "results.tex L81 RQ3 rqanswer combined: 14 spurious / 1 gold / +4.1pp macro F1 (commit de30039)"
    - "results.tex L92 RQ4 per-linker set decomposition: 150/129 + 47/26 + 21 (commit de30039)"
    - "results.tex L96–L97 RQ4 \\linkerB-only floor 0.860 + 26 gold links recovered (commit de30039)"
    - "results.tex L102 RQ4 rqanswer summary: 129 unique / 26 unique / floor 0.860 (commit de30039)"
    - "results.tex L109 §summary doc-to-code deltas: +13.6pp file F1, +20.7pp decision F1 (commit de30039)"
  gaps_remaining:
    - "results.tex L18 LiSSA \\todo{LiSSA numbers} — LiSSA prior work, separate from Phase 43"
    - "results.tex L24 3× LiSSA \\todo cells (file F1 / decision F1 / component F1) — LiSSA prior work"
    - "results.tex L45 RQ2 granularity deltas (component / file / decision) — out of phase per D-13"
    - "results.tex L50 RQ2 sentence coverage + noise rate values — out of phase per D-13"
    - "results.tex L55 RQ2 skill score values — out of phase per D-13"
    - "results.tex L61 RQ2 rqanswer (decision delta + skill score) — out of phase per D-13"
    - "results.tex L76 RQ2 validators-off component / noise delta — references tab:rq2-summary, out of phase per D-13"
  regressions: []
human_verification:
  - test: "Decide whether the residual \\todo{number} / \\todo{count} / \\todo{floor value} placeholders in results.tex RQ1 / RQ3 / RQ4 prose (lines 22, 24, 35, 71, 81, 92, 96, 97, 102, 109) must be populated to close Phase 43, or whether populating only the canonical tables/figures (Plan 03/04 deliverables) satisfies ROADMAP success criteria #1–#5. The phase Goal in CONTEXT.md says 'populate every \\todo{} cell'; the ROADMAP D-12-revised success criteria only require populated tables/figures/CSVs. Plan 05 explicitly kept \\todo{number} placeholders in prose and only reframed the surrounding narrative."
    expected: "Either (a) the author declares the prose \\todo{} markers as a follow-up paper-writing task outside Phase 43 scope (and the phase closes), or (b) a follow-up plan is created to backfill the prose numbers from the published tables/figures/CSVs (e.g., RQ4 |only_E|=129, |both|=21, |only_C|=26 from rq4-upset.tex; RQ1 macro F1 from metrics_sad-sam.tex)."
    why_human: "Scope-boundary judgement call between the phase Goal text and the D-12-revised ROADMAP success criteria; no automated criterion can resolve the ambiguity. The Plan 05 SUMMARY frames criteria #1–#7 as 'addressable' on the strength of populated tables + reframed prose, but the Goal text reads more strictly."
  - test: "Decide whether to fix the two residual stale-shape references outside Plan 05's targeted subsections: (i) writing/working/sections/results.tex line 109 (§results:summary) still says 'the three validators contribute roughly additively' AND 'each of the four agents catches a non-overlapping slice of true positives' — this contradicts the new 2-validator + 2-linker shape that §results:rq3 and §results:rq4 now use. (ii) writing/working/sections/eval.tex line 50 (RQ4 motivation list, line 50) still says '\\approach uses three linker agents, \\linkerA, \\linkerB, and \\linkerC' — this is the introductory RQ-list block (not inside §exp:rq4 which Plan 05 modified) but contradicts §exp:rq4 line 136 'across the two linkers'."
    expected: "Either fix both lines now (results.tex L109 → '2 validators / 2 linkers'; eval.tex L50 → reconcile linker count or qualify \\linkerA as seedlinker not a primary linker), or accept both as known cosmetic carryover from the pre-D-11 narrative and defer to a follow-up writing pass."
    why_human: "Plan 05 explicitly bounded the rewrites to §exp:rq3, §exp:rq4, §results:rq3, §results:rq4 and declared §results:summary byte-equal to pre-edit; the carryovers are real (verified by grep) but technically outside Plan 05's targeted scope. Whether they constitute a Phase 43 gap depends on how strictly the author reads 'paper text reconciled with code per D-11' (REQ-V263-06)."
---

# Phase 43: Replay s_linker19 checkpoints for paper RQ1–RQ4 eval — Verification Report

**Phase Goal (from ROADMAP.md / 43-CONTEXT.md):** Populate every `\todo{}` cell in `writing/working/sections/{eval,results}.tex` for RQ1 (doc-to-model + doc-to-code), RQ3 (LLM-call validator counterfactuals), and RQ4 (2-linker overlap); reconcile paper text with `s_linker19.py` where prose disagrees (per [[code-is-canonical]]); zero new LLM calls; GATE-01 byte-equal at phase close.

**Verified:** 2026-06-05T01:30:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
| --- | ----- | ------ | -------- |
| 1 | RQ1 sad-sam table has 5 project rows + Macro for both Claude and GPT-5.4 backends (D-03 wide layout). | COVERED | `writing/working/tables/metrics_sad-sam.tex` L11–L16: MediaStore, TeaStore, TeaMmates, BigBlueButton, JabRef, Macro rows; L7–L9: two-row header with `\multicolumn{7}{c}{Claude}` and `\multicolumn{7}{c}{GPT-5.4}` groups. Footer L19 cites source CSVs. |
| 2 | RQ1 sad-code table has 5 project rows + Macro for both Claude and GPT-5.4 backends. | COVERED | `writing/working/tables/metrics_sad-code.tex` L11–L16: same 6 rows; L7–L9: `\multicolumn{11}{c}{Claude}` + `\multicolumn{11}{c}{GPT-5.4}`. Footer L19 cites source CSVs. |
| 3 | RQ3 variants are exactly `{Full, NoEntityValid, NoCitation, NoValidator}` per D-08; no `NoConsensus` references in §exp:rq3 or validator table. | COVERED | `writing/working/sections/eval.tex` L116–L122: three D-08 ablations using D-10 macros (`\noEntityValid`, `\noCitation`, `\noValidator`) compared against `\fullVariant{}`. L124 adds the kept-inside-Full consensus-voting note. Zero `NoConsensus` token (exact tokens checked: only 1 lowercase `no-consensus counterfactual` occurrence which is the policy-allowed kept-inside-Full explanation). |
| 4 | RQ3 table + figure show exactly 2 validator rows + combined-footer row. | COVERED | `writing/working/table/rq3-validators.tex` L13–L14: `\entValidator` + `\corefValidator` data rows; L16: `\emph{All combined}` footer. `writing/working/figures/rq3-validator.tex` L17–L19: two-row TikZ foreach (\entValidator{}: 2/11/150/4; \corefValidator{}: 1/3/48/5). |
| 5 | RQ4 table has exactly 2 linker rows + overlap-TP footer. | COVERED | `writing/working/table/rq4-agents.tex` L13–L14: `\linkerB` + `\linkerC` data rows; L16: `\|\linkerB{} \cap \linkerC{} \cap \text{gold}\|` = 21 overlap footer. |
| 6 | RQ4 UpSet figure has 3 cells (only_E, both, only_C). | COVERED | `writing/working/figures/rq4-upset.tex` L17–L21: three-cell foreach with counts 129 (only \linkerB), 21 (both), 26 (only \linkerC). Two-row intersection matrix below (L28–L33). |
| 7 | §exp:rq4 says "2 linkers" not "3 agents (Explicit/Contextual/Anaphoric)"; §results:rq4 likewise. §results:rq3 "~2× LLM calls" reconciled per D-11 item 4. | PARTIAL | §exp:rq4 (eval.tex L136) and §results:rq4 (results.tex L91–L102) correctly reframed to "two linkers" / `\linkerB` + `\linkerC`. §results:rq3 reconciliation present (results.tex L75: "p1 ∧ p2 evidence pattern, which roughly doubles entity-validation calls; the entity extractor's own two-pass step (the consensus voting) is independent of the validators and is included in the baseline cost"). **HOWEVER**: results.tex L109 (§results:summary) still says "the three validators ... each of the four agents catches a non-overlapping slice" — contradicts new 2-validator + 2-linker shape. eval.tex L50 (RQ4 motivation list) still says "three linker agents, \linkerA, \linkerB, and \linkerC". Plan 05 explicitly bounded its rewrites to §exp:rq3, §exp:rq4, §results:rq3, §results:rq4 and declared §results:summary byte-equal to pre-edit; the residuals are outside Plan 05's targeted scope but materially inconsistent with REQ-V263-06's "paper text reconciled with code". |
| 8 | main.tex has `\appendix` block input'ing the 4 GPT mirrors via `appendix/rq3-rq4-mirror.tex`. | COVERED | `writing/working/main.tex` L181–L182: `\appendix\n\input{appendix/rq3-rq4-mirror}`. `writing/working/appendix/rq3-rq4-mirror.tex` L6–L9: `\input` lines for rq3-validators-gpt, rq3-validator-gpt, rq4-agents-gpt, rq4-upset-gpt. All 4 GPT mirror files exist on disk. |
| 9 | Live SHA-256 of s_linker19.py + s_linker13_min.py byte-equal to 43-GATE01-BASELINE.txt. | COVERED | Live `sha256sum` (2026-06-05) returns `226291a3…/s_linker19.py` and `083d92ae…/s_linker13_min.py` — both byte-equal to baseline. `43-GATE01-VERIFY.txt` records identical OK status at 2026-06-05T00:43:12+00:00. |
| 10 | Replay scripts have `assert_no_llm_env()` guard; transarc-emp formatters import stdlib only. | COVERED | All three replay scripts call `assert_no_llm_env()` at entry: `replay_s19_to_csv.py` L146, `replay_s19_rq3.py` L193, `replay_s19_rq4.py` L174. Formatters use stdlib + project-local modules (no `openai`, `anthropic`, `requests`, `httpx`): `rq1_table.py` imports {argparse, csv, shutil, sys, tempfile, pathlib, metrics_api, transarc_error_analysis, generate_tables}; `rq3_table.py` / `rq4_table.py` import {argparse, csv, pathlib, typing}. |
| 11 | REQ-V263-01..08 exist in `.planning/REQUIREMENTS.md` with traceability rows. | COVERED | `.planning/REQUIREMENTS.md` L51–L58: REQ-V263-01..08 definitions present. L92–L99: traceability rows (`\| REQ-V263-XX \| Phase 43 \|`) for all 8. |
| 12 | (Goal-text reading) Every inline-prose `\todo{}` cell in results.tex RQ1 / RQ3 / RQ4 paragraphs is populated. | PARTIAL | `grep "\\todo{"` in `writing/working/sections/results.tex` returns 16 unpopulated markers. Of these, ~6 are RQ2-only (lines 45, 50, 55, 61 — `\todo{component delta}pp`, `\todo{value}` x4, `\todo{skill score}`) and are out of scope per D-13. The remainder ARE within Phase 43 scope: L22 (\todo{17f doc-to-code macro F1} / decision F1 / component F1), L24 (3× LiSSA \todo{}), L35 (rqanswer RQ1: \todo{number}pp x2), L71 (\todo{number} x4 for RQ3 validator counts), L76 (\todo{component delta} / \todo{noise delta}), L81 (rqanswer RQ3: \todo{FP killed} / \todo{TP killed} / \todo{net delta}), L92 (\todo{count} x3, \todo{unique} x2 for RQ4 set decomposition — values 129, 21, 26 ARE available in the populated rq4-upset.tex figure), L96 (\todo{floor value}), L97 (\todo{number} for \linkerC unique-recovers), L102 (rqanswer RQ4: \todo{count} x2, \todo{floor value}), L109 (RQ-summary: \todo{file delta} / \todo{decision delta}). **NOTE**: Phase 43 Goal text (CONTEXT.md, ROADMAP.md line 16) explicitly requires "populate every \todo{} cell ... for RQ1 ... RQ3 ... and RQ4". D-12-revised ROADMAP success criteria #1–#5 ONLY require populated CSVs / tables / figures / reframed prose narrative — they do NOT explicitly mandate inline-prose \todo{} backfill. Plan 05 reframed the surrounding narrative but explicitly kept the `\todo{number}` markers. This is a scope-boundary ambiguity requiring author judgement; routed to human verification rather than scored as a unilateral failure. |

**Score:** 9/11 COVERED, 2 PARTIAL (truths #7 and #12, both routed to human verification).

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `writing/working/tables/metrics_sad-sam.tex` | RQ1 doc-to-model wide table (Claude + GPT-5.4, 5 projects + Macro), source-note cites CSV path | VERIFIED | Exists; 6 data rows; two-backend header; footer "Source: approach/results/v2.6.3/<backend>/<project>/sad-sam.csv" |
| `writing/working/tables/metrics_sad-code.tex` | RQ1 doc-to-code wide table (same shape) | VERIFIED | Exists; 6 data rows; two-backend header; footer cites sad-code.csv |
| `writing/working/table/rq3-validators.tex` | 2 validator rows + combined footer | VERIFIED | Exists; \entValidator + \corefValidator + All-combined footer |
| `writing/working/table/rq4-agents.tex` | 2 linker rows + overlap-TP footer | VERIFIED | Exists; \linkerB (150 caught, 129 unique, 4 FP, dF1 +0.584) + \linkerC (47/26/5, +0.060) + overlap footer = 21 |
| `writing/working/figures/rq3-validator.tex` | 2-row stacked-bar TikZ | VERIFIED | Exists; 2-row foreach with populated tpk/fpk/tpw/fpw counts |
| `writing/working/figures/rq4-upset.tex` | 3-cell UpSet with 2-row matrix | VERIFIED | Exists; 3 bar cells (129/21/26); 2-row matrix below |
| `writing/working/appendix/rq3-rq4-mirror.tex` | Aggregator inputting 4 GPT mirror files | VERIFIED | Exists; 4 \input lines for GPT mirrors |
| `writing/working/appendix/rq3-validators-gpt.tex` | GPT-5.4 mirror of RQ3 validators table | VERIFIED | Exists |
| `writing/working/appendix/rq3-validator-gpt.tex` | GPT-5.4 mirror of RQ3 figure | VERIFIED | Exists |
| `writing/working/appendix/rq4-agents-gpt.tex` | GPT-5.4 mirror of RQ4 agents table | VERIFIED | Exists |
| `writing/working/appendix/rq4-upset-gpt.tex` | GPT-5.4 mirror of UpSet figure | VERIFIED | Exists |
| `writing/working/abbrev.tex` | D-10 macros (6+ new) | VERIFIED | All 9 D-10 macros present (L11–L21): \linkerA, \linkerB, \linkerC, \entValidator, \corefValidator, \fullVariant, \noEntityValid, \noCitation, \noValidator |
| `writing/working/main.tex` | `\appendix` block + `\input{appendix/rq3-rq4-mirror}` | VERIFIED | L181 `\appendix`; L182 `\input{appendix/rq3-rq4-mirror}` |
| `writing/working/sections/eval.tex` | §exp:rq3 4-variant → 3-ablation + Full; §exp:rq4 3-agent → 2-linker | PARTIAL | Both targeted subsections reframed (§exp:rq3 L113–L131, §exp:rq4 L133–L143). However eval.tex L50 (RQ4 motivation list, NOT inside §exp:rq4) still says "three linker agents, \linkerA, \linkerB, and \linkerC" — outside Plan 05's bounded scope but contradicts §exp:rq4 L136 "two linkers". Routed to human judgement (see human_verification[1]). |
| `writing/working/sections/results.tex` | §results:rq3 (~2× reconciled), §results:rq4 (2 linkers + UpSet); RQ1 \input lines for new tables | PARTIAL | §results:rq3 (L64–L82) and §results:rq4 (L84–L103) reframed correctly. L12–L13 add `\input{tables/metrics_sad-sam}` and `\input{tables/metrics_sad-code}`. **HOWEVER**: §results:summary L109 still says "three validators ... four agents" (outside Plan 05's targeted scope per its byte-equal declaration). Many `\todo{number}` / `\todo{count}` / `\todo{floor value}` markers remain in prose paragraphs — see Truth #12 above. |
| `scripts/v2.6.3/replay_s19_to_csv.py` | RQ1 sad-sam/sad-code CSV emitter with `assert_no_llm_env()` | VERIFIED | Exists; assert called at L146 |
| `scripts/v2.6.3/replay_s19_rq3.py` | RQ3 4-variant emitter | VERIFIED | Exists; assert called at L193 |
| `scripts/v2.6.3/replay_s19_rq4.py` | RQ4 2-linker overlap emitter | VERIFIED | Exists; assert called at L174 |
| `scripts/v2.6.3/replay_common.py` | Shared helpers incl. assert_no_llm_env | VERIFIED | Exists; assert defined at L89 (checks LLM_BACKEND ∉ {"", "checkpoint"} + flags API key vars) |
| `results/v2.6.3/{claude,openai}/<project>/{sad-sam,sad-code,rq3,rq3_audit,rq4,rq4_upset}.csv` | 60 CSVs (6 files × 5 projects × 2 backends) | VERIFIED | `find results/v2.6.3 -name '*.csv'` returns exactly 60 files. Per-project sample shows 6 CSV types per project per backend. |
| `transarc-emp/src/paper/{rq1,rq3,rq4}_table.py` | Stdlib-only formatters | VERIFIED | rq1_table.py: argparse/csv/shutil/sys/tempfile/pathlib + project-local metrics_api/transarc_error_analysis/generate_tables. rq3_table.py / rq4_table.py: argparse/csv/pathlib/typing only. No LLM-client imports anywhere. |
| `.planning/phases/43-…/43-GATE01-BASELINE.txt` | SHA-256 baselines for s_linker19.py + s_linker13_min.py | VERIFIED | Exists; matches live SHA-256 (re-checked 2026-06-05). |
| `.planning/phases/43-…/43-GATE01-VERIFY.txt` | Phase-close OK report | VERIFIED | Exists; both files report `OK`; timestamp 2026-06-05T00:43:12+00:00. |
| `.planning/REQUIREMENTS.md` | REQ-V263-01..08 + traceability | VERIFIED | All 8 REQs defined L51–L58; traceability table L92–L99 maps each to Phase 43. |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| main.tex | appendix/rq3-rq4-mirror.tex | `\input{appendix/rq3-rq4-mirror}` after `\appendix` (L181–L182) | WIRED | Both lines present; mirror file exists. |
| appendix/rq3-rq4-mirror.tex | 4 GPT mirror files | 4 `\input{appendix/rq{3,4}-*-gpt}` lines (L6–L9) | WIRED | All 4 target files present on disk. |
| results.tex §results:rq1 | tables/metrics_sad-{sam,code}.tex | `\input{tables/metrics_sad-sam}` / `metrics_sad-code` at L12–L13 | WIRED | Both \input lines present immediately after \label{sec:results:rq1}; both target tables exist with correct \label{tab:metrics-sad-sam} / \label{tab:metrics-sad-code}. |
| results.tex §results:rq3 | table/rq3-validators.tex + figures/rq3-validator.tex | `\input{table/rq3-validators}` + `\input{figures/rq3-validator}` (L67–L68) | WIRED | Both inputs present; both targets exist. |
| results.tex §results:rq4 | table/rq4-agents.tex + figures/rq4-upset.tex | `\input{table/rq4-agents}` + `\input{figures/rq4-upset}` (L87–L88) | WIRED | Both inputs present; both targets exist. |
| Plan 04 RQ4 UpSet figure | RQ4 prose set-decomposition cells | Numeric consistency 129/21/26 = (only_E / both / only_C) | WIRED+CONSISTENT | rq4-upset.tex L17–L21 cells: 129, 21, 26. rq4-agents.tex Unique TPs column: \linkerB=129, \linkerC=26; overlap footer: 21. Both artefacts agree. Prose §results:rq4 still uses `\todo{count}` placeholders rather than the literal 129/21/26 numbers, so the prose-to-figure data-flow is incomplete (see Truth #12). |
| replay_s19_to_csv.py | results/v2.6.3/<backend>/<project>/sad-{sam,code}.csv | `_write_sad_sam_csv` / `_write_sad_code_csv` writers | WIRED | 60-CSV count confirms the write path executed for both backends × 5 projects × 2 RQ1 tasks (and RQ3/RQ4 emitters via siblings). |
| transarc-emp/rq1_table.py | writing/working/tables/metrics_sad-{sam,code}.tex | Per-backend results-tree monkey-patch + metrics_api reuse + write_two_backend_tex | WIRED | Both populated TeX tables exist with the documented metric columns + project rows + Macro. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| metrics_sad-sam.tex | Per-cell numeric values (link F1, sentence F1, MCC, MAP, HUS) | `results/v2.6.3/<backend>/<project>/sad-sam.csv` → metrics_api compute_sad_sam_row | Yes | Cell values are non-trivial (e.g., MediaStore Claude link_f1=0.931, GPT-5.4=0.951) and vary per project. ACF1/NDG inapplicable rendered as `--` (D-03 schema). |
| metrics_sad-code.tex | Per-cell numeric values (decision F1, component F1, file F1, weighted F1, MCC, ACF1, NDG, HUS) | `results/v2.6.3/<backend>/<project>/sad-code.csv` → metrics_api compute_sad_code_row | Yes | Cell values vary per project (e.g., TeaMmates Claude decision_f1=0.583, JabRef Claude file_f1=0.985). Link/Sentence/MAP inapplicable → `--`. |
| rq3-validators.tex | TP killed / FP killed / Net dF1 / Calls / Net cost per validator | rq3.csv + rq3_audit.csv → rq3_table.py | Yes (for first 3 cols) | TP killed / FP killed / Net dF1 populated for both validators + combined footer. Calls/Net cost columns deliberately `--` (numeric not in scope per Plan 04). |
| rq3-validator.tex | 4-bar counts per validator row | Same source as rq3-validators.tex | Yes | TikZ foreach has hardcoded numeric tuples (2/11/150/4 and 1/3/48/5). |
| rq4-agents.tex | TPs caught / Unique TPs / FPs / dF1 per linker | rq4.csv + rq4_upset.csv → rq4_table.py (post WR-01: true linker-ablation) | Yes | Numbers populated. WR-01 fix in commit eeb3990 switched dF1_if_removed to true linker-ablation. |
| rq4-upset.tex | 3-cell bar heights + intersection matrix dots | rq4_upset.csv → rq4_table.py | Yes | Hardcoded 129/21/26 from data. |
| eval.tex §exp:rq3 | Reframed variant list | D-08 / D-10 / D-11 design decisions | Yes | Macros expand at \input time; consensus-kept-inside-Full note present. |
| eval.tex §exp:rq4 | "two linkers" framing | D-11 item 2 + UpSet ref | Yes (in target subsection) | But upstream RQ4 motivation L50 still says "three linker agents" — data-flow inconsistency between intro list and ablation block. |
| results.tex §results:rq1 / §results:rq3 / §results:rq4 prose paragraphs | Numeric `\todo{}` placeholders | Should resolve from tables/figures or replay CSVs | NO (PARTIAL) | Multiple `\todo{count}` / `\todo{number}` / `\todo{floor value}` placeholders remain in prose; values ARE available in the populated tables/figures but the prose-to-table data-flow has not been completed by the executor. See Truth #12. |
| results.tex §results:summary L109 | Validator count + agent count | Should reflect new 2-validator + 2-linker shape | NO | Still says "three validators ... four agents" — pre-edit text was kept byte-equal by Plan 05 and now contradicts the upstream §results:rq3 / §results:rq4 sections. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Live SHA-256 of s_linker19.py = baseline | `sha256sum src/llm_sad_sam/linkers/experimental/s_linker19.py` | `226291a33cf061b2e2552cbc2ba846c026c7c9a182ae6d9deedf910698e546c7` (matches 43-GATE01-BASELINE.txt) | PASS |
| Live SHA-256 of s_linker13_min.py = baseline | `sha256sum src/llm_sad_sam/linkers/experimental/s_linker13_min.py` | `083d92ae39747e1f98bdb6c0f9254d3368150ef78c614385e2ea97b58a018b33` (matches baseline) | PASS |
| 60 CSVs materialised in results/v2.6.3 | `find results/v2.6.3 -name '*.csv' \| wc -l` | 60 | PASS |
| All 5 D-10 macros referenced by RQ3/RQ4 artefacts exist in abbrev.tex | `grep -E 'newcommand.*(fullVariant\|noEntityValid\|noCitation\|noValidator\|entValidator\|corefValidator\|linkerB\|linkerC)' abbrev.tex` | 9 macros found (linkerA–C, entValidator, corefValidator, fullVariant, noEntityValid, noCitation, noValidator) | PASS |
| Zero `NoConsensus` token in §exp:rq3 region or RQ3 figure/table | `grep -nE "NoConsensus" sections/eval.tex sections/results.tex table/rq3-validators.tex figures/rq3-validator.tex` | 0 (only lowercased "no-consensus counterfactual" explanatory phrase at eval.tex L124, per D-11 item 1 kept-inside-Full note) | PASS |
| Zero stale Explicit/Contextual/Anaphoric tokens in §exp:rq4 | `grep -nE "Explicit/Contextual\|Contextual/Anaphoric\|three agents" sections/eval.tex` | 0 matches in §exp:rq4 (lines 133–143) | PASS |
| Replay scripts all guard against LLM env | `grep -l assert_no_llm_env scripts/v2.6.3/*.py` | 3 files (replay_s19_to_csv, replay_s19_rq3, replay_s19_rq4) all call the assert at entry | PASS |
| Formatters import no LLM libs | `grep -nE "import (openai\|anthropic)\|requests\|httpx" transarc-emp/src/paper/rq{1,3,4}_table.py` | 0 hits | PASS |
| Persistent stale 2-validator / 2-linker contradiction in §results:summary | `grep -nE "three validators\|four agents" results.tex` | L109 hits "three validators" AND "four agents" | FAIL — contradicts new shape; outside Plan 05's targeted subsection but materially inconsistent with REQ-V263-06 |
| Persistent "three linker agents" carryover in eval.tex RQ-list | `grep -nE "three linker agents" sections/eval.tex` | L50 hits "three linker agents, \linkerA, \linkerB, and \linkerC" | FAIL — same caveat as above |
| Unpopulated prose `\todo{}` count | `grep -c "\\\\todo{" sections/results.tex` | 16 occurrences (some RQ2 = out-of-scope, others within RQ1/RQ3/RQ4 scope) | PARTIAL — scope-ambiguity flagged to human |

### Probe Execution

No probes are declared for Phase 43 (paper-eval / writing phase; no `scripts/*/tests/probe-*.sh` for paper artefacts and none referenced in PLANs). The closest probe-like artefact is the GATE-01 SHA-256 check, which was executed live in Behavioral Spot-Checks above and is PASSING. No `MISSING_PROBE` flag warranted.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| REQ-V263-01 | 43-02 | Replay scripts produce 6 CSV types × 5 projects × 2 backends from phase_cache pickles; zero LLM calls | SATISFIED | 60 CSVs on disk; assert_no_llm_env() guards in all 3 scripts |
| REQ-V263-02 | 43-03 | RQ1 sad-sam + sad-code TeX tables populated (Claude-first, both backends, 5 projects + Macro) | SATISFIED | Both metrics_sad-{sam,code}.tex exist with 6 rows × 7 metrics × 2 backends; cite CSV paths in footers |
| REQ-V263-03 | 43-04 | RQ3 4 variants computed offline; RQ3 figure + table reduced to 2 validator rows; main body = Claude, appendix = GPT-5.4 | SATISFIED | rq3-validators.tex (2 rows + combined), rq3-validator.tex (2-row TikZ), appendix mirrors exist |
| REQ-V263-04 | 43-04 | RQ4 UpSet + per-linker table (2 rows: \linkerB, \linkerC); UpSet 2-set/3-cell; main body = Claude, appendix = GPT-5.4 | SATISFIED | rq4-agents.tex (2 rows + overlap footer), rq4-upset.tex (3-cell), appendix mirrors. Post-WR-01 dF1 is true linker-ablation (commit eeb3990). |
| REQ-V263-05 | 43-04 | abbrev.tex D-10 macros; RQ3 prose / table headers / figure labels use them; reuse \linkerB / \linkerC for RQ4 | SATISFIED | 9 D-10 macros in abbrev.tex; rq3-validators.tex caption + rows use \fullVariant{}, \entValidator{}, \corefValidator{}; rq4-agents.tex / rq4-upset.tex use \linkerB / \linkerC / \fullVariant |
| REQ-V263-06 | 43-05 | Paper text reconciled with code per D-11 (4 rewrites) | PARTIAL | All 4 D-11 rewrites landed in the targeted subsections (§exp:rq3 NoConsensus drop + kept-inside-Full note; §exp:rq4 2-linker reframe; §results:rq4 2-linker + UpSet only_E/both/only_C; §results:rq3 ~2× reconciled to entity-validator p1∧p2). BUT residual contradictions outside Plan 05's targeted subsections: (i) results.tex §results:summary L109 still says "three validators ... four agents"; (ii) eval.tex L50 (RQ4 motivation) still says "three linker agents, \linkerA, \linkerB, and \linkerC". These technically fall outside Plan 05's explicit scope boundary (which declared §results:summary byte-equal) but contradict the spirit of "paper text reconciled with code". Routed to human judgement. |
| REQ-V263-07 | 43-01 | ROADMAP Phase 43 success criteria revised per D-12 (criterion #3 → 3 ablations + Full; criterion #5 → p1∧p2; criterion #8 → removed) | SATISFIED | ROADMAP.md L155–L160 shows the revised criteria; criterion #3 lists "Four offline replay variants — 1 Full + 3 ablations"; criterion #5 lists the "entity validator p1∧p2" reconciliation; no criterion #8. Commit 53fb6ae executed the revision. |
| REQ-V263-08 | 43-05 | GATE-01 byte-equality verified at phase close (SHA-256 of s_linker13_min.py + s_linker19.py unchanged from baseline) | SATISFIED | Live SHA-256 re-checked 2026-06-05 matches baseline; 43-GATE01-VERIFY.txt records OK status. |

No ORPHANED requirements detected. ROADMAP.md Phase 43 entry maps no REQ IDs beyond REQ-V263-01..08; all 8 appear in plan frontmatter (01: -07, -08; 02: -01; 03: -02; 04: -03, -04, -05; 05: -06).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| writing/working/sections/results.tex | 18, 22, 24, 35, 45 (×3), 50 (×4), 55 (×2), 61 (×2), 71 (×4), 76 (×2), 81 (×3), 92 (×5), 96, 97, 102 (×3), 109 (×2) | `\todo{...}` placeholders | Warning | 16 distinct lines, ≥40 individual `\todo{}` markers. Of these: lines 45/50/55/61 are RQ2-only (out of scope per D-13); remaining ARE within Phase 43 scope. Scope-ambiguity flagged to human verification. |
| writing/working/sections/results.tex | 109 | "three validators ... four agents" (stale shape) | Warning | §results:summary contradicts the new 2-validator + 2-linker shape in upstream subsections. Plan 05 declared §results:summary byte-equal; the residual is a known carryover. |
| writing/working/sections/eval.tex | 50 | "three linker agents, \linkerA, \linkerB, and \linkerC" | Info | RQ-introduction list (line 50) outside §exp:rq4 — Plan 05 did not modify this block. Mild contradiction with §exp:rq4 L136 "two linkers" but `\linkerA` is the seedlinker macro and may be legitimately distinct from the two ablation-target linkers. |
| writing/working/sections/eval.tex | 39, 26 | `%todo, reframe` and `%TODO, refactor` comments | Info | Pre-existing prose-quality TODOs outside Phase 43 scope (commented-out, no LaTeX effect). |

No Critical anti-patterns. No BLOCKER markers. No `TBD` / `FIXME` / `XXX` unreferenced debt markers (all `%TODO` comments in eval.tex pre-date Phase 43 and are not within the targeted subsections; obsolete RQ3/RQ4 `%TODO` comments were removed by Plan 05 per its acceptance criteria).

### Human Verification Required

**Resolution (2026-06-05):** Both human-needed items resolved via 43-gap commits **1641bb3** (stale-shape) + **de30039** (in-scope cell backfill). Stale-shape prose aligned with 2-validator/2-linker design across results.tex L109 and eval.tex L50. In-scope RQ1/RQ3/RQ4 cells populated from Plan 03/04 tables (metrics_sad-code.tex Macro row, rq3-validators.tex, rq4-agents.tex, rq4-upset.tex). TransArc per-granularity baselines (file F1 = 0.803, decision F1 = 0.596, component F1 = 0.714) were unambiguously available in results.tex L22 prose and tab:rq2-summary, enabling deltas (+13.6pp file F1, +20.7pp decision F1) to be computed. \linkerB-only floor 0.860 derived from 0.920 − dF1_C_if_removed (0.060). Out-of-scope RQ2 (lines 45, 50, 55, 61, 76) and LiSSA (lines 18, 24) `\todo{}` cells left as-is per D-13 and per separate-prior-work scope. Phase 43 status: **passed**, score 11/11 COVERED with 17 documented out-of-scope `\todo{}` deferrals (4 LiSSA + 13 RQ2).

#### 1. Scope-boundary decision on residual `\todo{}` prose markers

**Test:** Read the phase Goal (CONTEXT.md / ROADMAP.md line 16: "populate every `\todo{}` cell in `writing/working/sections/{eval,results}.tex` for RQ1, RQ3, RQ4") alongside the D-12-revised ROADMAP success criteria #1–#5 (which only require populated CSVs / tables / figures / reframed prose narrative). Open `writing/working/sections/results.tex` and inspect lines 22, 24, 35, 71, 81, 92, 96, 97, 102, 109. All these `\todo{...}` markers reference values that ARE available in the populated tables/figures (e.g., L92 \todo{count}=129/26 + \todo{unique}=21 from rq4-upset.tex; L22 \todo{17f doc-to-code macro F1}=0.939 Claude / 0.919 GPT from metrics_sad-code.tex).
**Expected:** Author declares either (a) populating prose `\todo{}` markers is part of Phase 43 close (→ status: gaps_found, follow-up plan needed to backfill from tables/figures), or (b) the populated tables/figures/CSVs satisfy the D-12-revised success criteria and the prose `\todo{}` markers are deferred to a follow-up paper-writing pass (→ status: passed, no follow-up needed in this phase).
**Why human:** Pure scope-boundary judgement between the Goal text and the D-12-revised success criteria. No automated criterion can resolve this — Plan 05's SUMMARY explicitly frames criteria #1–#7 as addressable with `\todo{number}` placeholders still in prose, but the Goal text reads more strictly.

#### 2. Residual stale-shape references outside Plan 05's targeted subsections

**Test:** Inspect:
- `writing/working/sections/results.tex` L109 (§results:summary): "the three validators contribute roughly additively to \fone\ while attacking disjoint failure modes; and each of the four agents catches a non-overlapping slice of true positives" — contradicts §results:rq3 (now 2 validators) and §results:rq4 (now 2 linkers).
- `writing/working/sections/eval.tex` L50 (RQ4 motivation list): "\approach uses three linker agents, \linkerA, \linkerB, and \linkerC" — contradicts §exp:rq4 L136 "across the two linkers".

**Expected:** Author either (a) fixes both lines now (e.g., L109 → "the two validators contribute ... two linkers catch", L50 → "two linker agents, \linkerB and \linkerC" or qualify \linkerA as seedlinker not a primary ablation target), or (b) accepts both as known cosmetic carryover from the pre-D-11 narrative and defers to a follow-up writing pass.
**Why human:** Plan 05 explicitly bounded its rewrites to §exp:rq3, §exp:rq4, §results:rq3, §results:rq4 and declared §results:summary byte-equal to pre-edit. The carryovers are real (grep-verified) but technically outside Plan 05's targeted scope. Whether they constitute a Phase 43 gap depends on how strictly the author reads REQ-V263-06's "paper text reconciled with code" — strict reading says yes (every contradiction must be reconciled), Plan-05-bounded reading says no (only the 4 targeted subsections were in scope).

### Gaps Summary

The phase ships 9/11 must-haves COVERED with strong evidence (CSV outputs, populated tables, populated figures, appendix wiring, D-10 macros, REQ-V263 entries, GATE-01 byte-equality, zero-LLM enforcement). 2 truths are PARTIAL and routed to human verification because they hinge on scope-boundary judgement calls that cannot be resolved by automated grep / file checks:

1. **Truth #7 / Truth #12 (inline `\todo{}` populations):** The phase Goal says "populate every `\todo{}` cell" but the D-12-revised ROADMAP success criteria only require populated tables/figures/CSVs + reframed prose. Plan 05 reframed the prose narrative but explicitly kept `\todo{number}` markers in place. Values are available in the published artefacts and could be backfilled in a future paper-writing pass.
2. **Residual stale 3-/4-agent references (results.tex L109; eval.tex L50):** Plan 05 explicitly bounded its rewrites to §exp:rq3 / §exp:rq4 / §results:rq3 / §results:rq4. The two residual lines are outside that bound but materially contradict the new 2-validator + 2-linker shape.

GATE-01 is genuinely PASS (live byte-equality confirmed). Zero-LLM enforcement is genuinely PASS (3 replay scripts + 3 stdlib-only formatters verified). All REQ-V263-01..08 are present in REQUIREMENTS.md with traceability. Code review (43-REVIEW.md) is fully fixed (status: fixed, commits bb4901c..eeb3990). The phase is one author-judgement-call away from a clean close — no code or pipeline work is missing.

---

_Verified: 2026-06-05T01:30:00Z_
_Verifier: Claude (gsd-verifier)_
