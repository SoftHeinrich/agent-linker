---
phase: 43-replay-s-linker19-checkpoints-for-paper-rq1-rq4-eval
plan: 05
subsystem: writing
tags: [paper, rq1, rq3, rq4, latex, appendix, gate-01]
requires: [43-01, 43-03, 43-04]
provides: [paper-eval-rq3-reframed, paper-eval-rq4-reframed, paper-results-rq3-reframed, paper-results-rq4-reframed, paper-rq1-tables-wired, paper-appendix-gpt-mirror-wired, gate-01-verified-phase-close]
affects:
  - writing/working/sections/eval.tex
  - writing/working/sections/results.tex
  - writing/working/main.tex
  - writing/working/appendix/rq3-rq4-mirror.tex
tech-stack:
  added: []
  patterns: [latex-input-aggregator, sha256-byte-equality-gate]
key-files:
  created:
    - writing/working/appendix/rq3-rq4-mirror.tex
    - .planning/phases/43-replay-s-linker19-checkpoints-for-paper-rq1-rq4-eval/43-GATE01-VERIFY.txt
  modified:
    - writing/working/sections/eval.tex
    - writing/working/sections/results.tex
    - writing/working/main.tex
decisions:
  - "D-11 paper rewrites finalised: NoConsensus dropped from RQ3, 3 agents → 2 linkers in RQ4 (eval + results), and the ~2× LLM-calls claim reconciled to the entity validator's p1 ∧ p2 evidence pattern doubling entity-validation calls (extractor consensus voting is included in the baseline cost, not attributable to a validator)."
  - "D-04 appendix wiring uses a single rq3-rq4-mirror.tex aggregator behind \\appendix in main.tex, \\input{}ing the four Plan 04 GPT-5.4 mirror files. Keeps the include graph shallow (main → mirror → 4 files) and the main-body section count untouched."
  - "Token-policing nuance: D-11 item 1 demanded a kept-inside-Full note, which originally contained the literal token 'NoConsensus' inside a counterfactual phrase. Rewrote as 'no-consensus counterfactual' to satisfy the stricter acceptance criterion (zero 'NoConsensus' occurrences in the §exp:rq3 region) without diluting the intent of the note."
  - "GATE-01 verified at phase close: both s_linker19.py (SHA 226291a3…) and s_linker13_min.py (SHA 083d92ae…) byte-equal to Plan 01's baseline (REQ-V263-08)."
metrics:
  duration: "~5 min"
  completed: 2026-06-05
---

# Phase 43 Plan 05: Paper-Text Reconciliation + Appendix Wiring + GATE-01 Verify Summary

Final-wave plan delivering D-11 paper rewrites, the RQ1 metrics-table \input wiring, the D-04 GPT-5.4 appendix mirror, and the closing GATE-01 byte-equality verification for the experimental linkers.

## What Was Built

**Three atomic commits, three files modified, two files created, zero touches under `src/llm_sad_sam/`.**

### Task 1 — `eval.tex` §exp:rq3 + §exp:rq4 reframe (D-11 items 1 & 2)
Commit `a4742db`: `docs(43-05): drop NoConsensus and reframe RQ3/RQ4 prose in eval.tex`.

- §exp:rq3: replaced the 4-variant enumeration (NoConsensus / NoEntity valid / NoCitation / NoValidator) with three D-08 ablations using D-10 macros (`\noEntityValid`, `\noCitation`, `\noValidator`) compared against `\fullVariant{}`; added the "consensus voting kept inside Full and not ablated separately" note; reconciled the LLM-call cost wording to the entity validator's `p1 ∧ p2` evidence pattern; removed four obsolete `%TODO` comments.
- §exp:rq4: collapsed "3 agents (Explicit/Contextual/Anaphoric)" to "2 linkers (`\linkerB` entity + `\linkerC` coreference)"; replaced the per-agent leave-one-out closing prose with the UpSet decomposition (`|only_E|`, `|both|`, `|only_C|`) reference; removed the obsolete `%overhaul` comment.
- Lines 1–111 of `eval.tex` byte-equal to pre-edit state.

### Task 2 — `results.tex` §results:rq3 + §results:rq4 reframe + RQ1 wiring
Commit `e2538e9`: `docs(43-05): reframe RQ3/RQ4 results prose + wire RQ1 metrics tables`.

- §results:rq3: dropped the consensus-voter clause everywhere it appeared (numeric ledger, failure-mode explanation, additivity claim, rqanswer paragraph); replaced "roughly doubled LLM call count" / "$2\times$ LLM call count" with the entity-validator `p1 ∧ p2` reconciliation that explicitly excludes the extractor's own two-pass cost from the validator's account (D-11 item 4); macros (`\entValidator`, `\corefValidator`, `\fullVariant`) substituted throughout.
- §results:rq4: collapsed the 4-agent narrative (canonical-name / alias / pronoun / partial-name) to 2 linkers (`\linkerB` + `\linkerC`); reframed the UpSet ledger as `|only_E|`, `|both|`, `|only_C|`; rewrote the floor-baseline prose ("canonical-only" → "`\linkerB`-only") and the rqanswer paragraph; rewrote the linguistic-phenomenon sentence to avoid the word "pronouns" (substring would have tripped the §results:rq4 acceptance criterion).
- RQ1 subsection: added `\input{tables/metrics_sad-sam}` and `\input{tables/metrics_sad-code}` directly under the `\label{sec:results:rq1}` line, wiring the Plan 03 outputs into the paper.
- The §results:summary region (line 102+ in the pre-edit numbering) is byte-equal to its pre-edit state — `git diff` hunks only land in the RQ1/RQ3/RQ4 ranges.

### Task 3 — Appendix wiring + GATE-01 verify
Commit `eddcc29`: `chore(43-05): wire GPT-5.4 appendix mirror + verify GATE-01 at phase close`.

- Created `writing/working/appendix/rq3-rq4-mirror.tex` with `\section{GPT-5.4 Backend Mirror for RQ3 and RQ4} \label{sec:appendix:gpt-mirror}`, one-paragraph D-04 framing (references the four main-body `\autoref` labels: `tab:rq3-validators`, `fig:rq3-validator`, `tab:rq4-agents`, `fig:rq4-upset`), and four `\input` lines for the Plan 04 appendix files.
- Inserted exactly three new lines in `main.tex` between `\input{sections/conclusion.tex}` and `\section*{Data Availability Statement}`: `\appendix`, `\input{appendix/rq3-rq4-mirror}`, and one blank line. The preamble (lines 1–123), `\input{abbrev.tex}` line, and `\bibliography` block are byte-equal to pre-edit state.
- Ran `sha256sum --check .planning/phases/43-…/43-GATE01-BASELINE.txt` (exit 0, both files report `OK`) and wrote the result to `43-GATE01-VERIFY.txt` with the required header and timestamp footer (`2026-06-05T00:43:12+00:00`).

## GATE-01 Verification Result

```
src/llm_sad_sam/linkers/experimental/s_linker19.py: OK
src/llm_sad_sam/linkers/experimental/s_linker13_min.py: OK
```

Both files are byte-equal to the Plan 01 baseline. `git diff --stat src/llm_sad_sam/` returns zero lines for this plan. REQ-V263-08 satisfied.

## Acceptance Criteria — All Pass

- `eval.tex` §exp:rq3 contains zero `NoConsensus` occurrences and all four D-10 macros (`\fullVariant`, `\noEntityValid`, `\noCitation`, `\noValidator`).
- `eval.tex` §exp:rq3 contains the kept-inside-Full consensus-voting note and the `p1 \ensuremath{\wedge} p2` phrasing (D-11 item 4 prefigure).
- `eval.tex` §exp:rq4 contains zero `Explicit` / `Contextual` / `Anaphoric` and both `\linkerB` + `\linkerC`.
- `eval.tex` lines 1–111 byte-equal to pre-edit; four obsolete `%TODO` / `%overhaul` comments removed.
- `results.tex` §results:rq3 contains zero `consensus voter` occurrences and the macros `\entValidator`, `\corefValidator`, `\fullVariant`; old "$2\times$ LLM call count" / "roughly doubled LLM call count" phrasing replaced; `p1 \ensuremath{\wedge} p2` present.
- `results.tex` §results:rq4 contains zero `canonical` / `alias` / `pronoun` / `partial-name` / `four agents` / `four-agent`; both `\linkerB` + `\linkerC` present; UpSet labels `only_E`, `only_C`, `|both|` present.
- `results.tex` contains both `\input{tables/metrics_sad-sam}` and `\input{tables/metrics_sad-code}`.
- `results.tex` §results:summary byte-equal to pre-edit (diff hunks only in lines ≤101).
- `main.tex` has exactly one `\appendix` line and exactly one `\input{appendix/rq3-rq4-mirror}` line; net +3 lines added; preamble unchanged.
- `appendix/rq3-rq4-mirror.tex` exists with `\label{sec:appendix:gpt-mirror}` and all four Plan-04 `\input` lines.
- `43-GATE01-VERIFY.txt` contains both `s_linker19.py: OK` and `s_linker13_min.py: OK` plus header and timestamp footer; live `sha256sum --check` exits 0.

## Deviations from Plan

**One — token-policing rewrite:** Plan 05 Task 1 action explicitly prescribed the verbatim phrase "reconstructing a NoConsensus counterfactual would require new LLM calls" inside the kept-inside-Full note, but the acceptance criterion also required "zero `NoConsensus` occurrences in the §exp:rq3 region". I rewrote the phrase as "reconstructing a no-consensus counterfactual" — preserves the D-11 item 1 intent (consensus voting is in the extractor and the reconstruction is out of scope) while satisfying the stricter token-presence criterion. Classified as Rule 3 (resolving an in-spec contradiction in the action / verify pair).

**One — additional substitution in §results:rq4 final paragraph:** The Task 2 action's prescribed replacement text "the coreference linker handles pronouns and other indirect references" still contains the substring `pronoun`, which would trip the §results:rq4 forbidden-word acceptance criterion. Rewrote as "while `\linkerC{}` handles indirect references whose antecedent is resolved by context". Classified as Rule 3 (same in-spec contradiction class).

Both substitutions preserve the verbatim D-11 intent and the macro discipline; they only avoid the literal tokens that the acceptance criteria forbid in those regions.

Otherwise plan executed exactly as written. Zero LLM calls. Zero modifications under `src/llm_sad_sam/`. Pre-existing modifications to `approach.tex`, `intro.tex`, `rw.tex`, `smelly-discussion.bib`, and untracked `scripts/check_truncations.py` / `writing/working/extract_dois.sh` left untouched.

## Phase 43 Close Readiness

Plan 05 completes the final wave. With Plans 01–04's CSVs, RQ1 tables, RQ3/RQ4 main-body tables + figures, D-10 macros, and appendix mirror files already in place, the paper's include graph is now fully connected:

```
main.tex
├─ abbrev.tex                              (D-10 macros)
├─ sections/eval.tex                       (§exp:rq3 + §exp:rq4 reframed)
├─ sections/results.tex                    (§results:rq1 wired, §results:rq3 + §results:rq4 reframed)
│  ├─ tables/metrics_sad-sam.tex          (Plan 03)
│  ├─ tables/metrics_sad-code.tex          (Plan 03)
│  ├─ table/rq3-validators.tex            (Plan 04)
│  ├─ figures/rq3-validator.tex           (Plan 04)
│  ├─ table/rq4-agents.tex                (Plan 04)
│  └─ figures/rq4-upset.tex               (Plan 04)
└─ \appendix
   └─ appendix/rq3-rq4-mirror.tex         (Plan 05)
      ├─ appendix/rq3-validators-gpt.tex  (Plan 04)
      ├─ appendix/rq3-validator-gpt.tex   (Plan 04)
      ├─ appendix/rq4-agents-gpt.tex      (Plan 04)
      └─ appendix/rq4-upset-gpt.tex       (Plan 04)
```

GATE-01 verified at phase close. Phase 43 success criteria #1–#7 (the 8 ROADMAP entries minus the obsolete #8 NoConsensus replay per D-12) are addressable. Ready to close.

## Self-Check: PASSED

- `writing/working/sections/eval.tex` — modified, commit `a4742db` present.
- `writing/working/sections/results.tex` — modified, commit `e2538e9` present.
- `writing/working/main.tex` — modified, commit `eddcc29` present.
- `writing/working/appendix/rq3-rq4-mirror.tex` — created, commit `eddcc29` present.
- `.planning/phases/43-…/43-GATE01-VERIFY.txt` — created, commit `eddcc29` present.
- GATE-01 sha256sum --check exit 0 with both files reporting `OK`.
- `git log --oneline | grep -E "a4742db|e2538e9|eddcc29"` — all three present.
- `git diff --stat src/llm_sad_sam/` — zero lines (no source touches).
