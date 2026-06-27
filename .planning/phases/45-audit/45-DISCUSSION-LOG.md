# Phase 45: AUDIT - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-07
**Phase:** 45-AUDIT
**Areas discussed:** Verdict rubric, Audit depth, Rewording scope, Cut format / grain

---

## Verdict rubric

### Sub-question 1: how to define `domain-loaded`

| Option | Description | Selected |
|--------|-------------|----------|
| Strict: any SE/SAD jargon | Domain-loaded = uses terms that signal SAD-SAM context. Aggressive — produces a long candidate list for Phase 46. | |
| Pragmatic: only domain-overspecified terms | Domain-loaded = uses domain terms where a universal noun would carry the same meaning to the LLM. Smaller candidate list, lower Phase 46 churn, easier to defend in paper. | ✓ |
| Behavioral: only when removing it would generalize behavior | Domain-loaded = uses surface terms that would change LLM behavior toward a more general task if removed. Hardest to assess from inspection alone. | |

**User's choice:** Pragmatic — only domain-overspecified terms.
**Notes:** Operationalizes PROJECT.md "look general but still SAD/SAM-tuned". Domain terms that are load-bearing stay clean; only over-specified ones are flagged. CONTEXT.md D-01.

### Sub-question 2: how to detect `benchmark-leak`

| Option | Description | Selected |
|--------|-------------|----------|
| Mechanical grep against BENCHMARK_TABOO.md | Automated token lookup; any hit = benchmark-leak. Defensible, reproducible. Risk: false positives on universal-taboo words. | |
| Grep + manual review of universal-taboo hits | Same as above, but universal-taboo hits get a second-pass review against v2.1 cross-dataset isolation methodology. | ✓ |
| Reviewer judgment only, no mechanical pass | Audit purely by reading. Less defensible for the paper. | |

**User's choice:** Grep + manual review of universal-taboo hits.
**Notes:** Per-dataset-section hits auto-classify as leak; Universal-taboo hits trigger second-pass cross-dataset isolation check. CONTEXT.md D-02.

### Sub-question 3: few-shot block example-name classification

| Option | Description | Selected |
|--------|-------------|----------|
| Benchmark-leak per BENCHMARK_TABOO match | Any synthetic example name that overlaps BENCHMARK_TABOO (keywords or aliases) = benchmark-leak, rewording required. | ✓ |
| Benchmark-leak only on direct component-name hits | Only literal benchmark Component-name matches. More permissive. | |
| Whole few-shot block = drop-block candidate regardless | Mark both few-shot blocks as drop-whole-block a priori; skip per-name leak classification. | |

**User's choice:** Benchmark-leak per BENCHMARK_TABOO match (strict).
**Notes:** Per-token leak classification still done so that if a partial replacement wins in Phase 46, the leak verdict carries. Drop-block remains a first-class cut row per REQ-V264-06. CONTEXT.md D-04.

### Sub-question 4: f-string scaffold scope

| Option | Description | Selected |
|--------|-------------|----------|
| Audit all f-string text the same way | Every line inside scaffolds gets the same verdict + cut candidates. | |
| Audit only the prose/instructions, not the schema text | Skip JSON-schema literals (`Return JSON: {…}`, `JSON only:`). Focus only on instruction prose. | ✓ |
| Audit prose + flag schema verbosity separately | Audit prose with 3-verdict rubric; flag schema verbosity in a separate column. | |

**User's choice:** Audit prose only, skip JSON-schema literals.
**Notes:** Schema literals are byte-equality-critical for the parser and not load-bearing for behavior. CONTEXT.md D-03.

---

## Audit depth

### Sub-question 1: harness leverage during audit

| Option | Description | Selected |
|--------|-------------|----------|
| Annotated audit — D-03 cross-reference | Each cut candidate annotated with which test module(s) and which phase tag(s) gate it. No code runs; clean Phase-46 recipe. | ✓ |
| Pure desk audit | Read prompts only; ignore the harness. Phase 46 figures out test mapping itself. | |
| Pilot trial cuts (2–3) | Pick representative candidates and run against the harness as sensitivity check. Bleeds into Phase 46 scope. | |

**User's choice:** Annotated audit with D-03 cross-reference.
**Notes:** `gated_by` column references Phase-44 builder→phase-tag mapping verbatim. Stays read-only. CONTEXT.md D-07.

### Sub-question 2: predictive annotations

| Option | Description | Selected |
|--------|-------------|----------|
| Per-cut risk score: low / med / high | 3-tier prior probability of survival under byte-equality. Cheap, opinionated. | ✓ |
| Per-cut survival hypothesis (yes/no) | Binary call: expect this cut to survive byte-equality on all 97 snapshots? Sharper but binary. | |
| No predictions — stay descriptive | Audit only describes; Phase 46 explores empirically. | |

**User's choice:** Per-cut risk score (low / med / high).
**Notes:** Phase 46 attempts low-risk first, batches. Risk is reviewer judgment, validated empirically by Phase 46. CONTEXT.md D-07.

---

## Rewording scope

### Sub-question 1: scope of rewording proposals

| Option | Description | Selected |
|--------|-------------|----------|
| Benchmark-leak only (strict REQ) | Rewordings only for items classified as benchmark-leak. Domain-loaded items just flagged. | ✓ |
| Also pre-stage rewordings for domain-loaded items | Audit doc proposes rewordings for every domain-loaded item too. Speeds Phase 46 but may waste effort on non-survivors. | |
| Benchmark-leak rewordings + ONE example domain-loaded rewording per builder | Middle ground — strict on leak; one worked example per builder for domain-loaded. | |

**User's choice:** Benchmark-leak only (strict REQ).
**Notes:** Domain-loaded items are listed and characterized; Phase 46 generates their rewordings empirically against the harness. CONTEXT.md D-05.

### Sub-question 2: rewording style

| Option | Description | Selected |
|--------|-------------|----------|
| Synthetic-neutral with universal nouns | Replace benchmark-overlap names with truly neutral synthetic names; same prompt shape, swapped vocab. | (also selected) |
| Concept-only — strip names entirely | Rewrite few-shots to describe the rule abstractly without name examples. | (also selected) |
| Other-domain neutral (e.g., kitchen, robotics) | Use clearly non-SE example domains. | |

**User's choice:** Both (1) and (2) — synthetic-neutral AND concept-only.
**Notes:** User typed "1,2 both try". Each benchmark-leak finding gets rewordings from BOTH families so Phase 46 can test both against the harness. CONTEXT.md D-06.

### Sub-question 3: number of rewording variants per family

| Option | Description | Selected |
|--------|-------------|----------|
| 2–3 of each type | Modest doc bloat, broad coverage. | |
| 1 of each + 'see also' alternates list | Primary + bulleted alternates. | |
| As many as inspection suggests, no fixed count | Audit lists every plausible rewording. Most material for Phase 46. | ✓ |

**User's choice:** As many as inspection suggests, no fixed count.
**Notes:** Phase 46 chooses among them empirically; abundance is cheap here. CONTEXT.md D-06.

---

## Cut format / grain

User chose: "design the best way." Format design was made directly without further questions.

### Design

| Element | Decision | Rationale |
|---|---|---|
| Doc structure | 5 sections by s19 pipeline phase, colocating constants with their builder | Phase 46 reasons about builder-and-imports as a unit; matches mental model |
| Per-item header | Verdict + LOC mini-table | At-a-glance scan |
| Cut-candidate schema | `cut_id`, `file:lines`, `trigger`, `before`, `after`, `risk`, `gated_by` | 1:1 traceability into Phase 46's MINIMIZE-LOG via `cut_id` |
| Cut-id scheme | `CUT-{section-tag}-{NN}` with tags AMB / DKX / DKJ / EXT / VAL / COR | Stable reference across Phases 45→46→47→48 |
| Drop-whole-block | First-class cut row with `after_snippet: ""` | Per REQ-V264-06; few-shot blocks tested as full-removal candidates first |
| Reword variants | Each variant as its own `cut_id` | Phase 46 ships per-variant verdict in MINIMIZE-LOG |
| Long rewordings | Inline detail block under the table when before/after exceed cell width | Readability |

CONTEXT.md D-08.

---

## Claude's Discretion

Captured in CONTEXT.md §"Claude's Discretion":

- Final on-disk location of `s_linker20-PROMPT-AUDIT.md` (milestone tree vs `.planning/phases/45-audit/`).
- Per-cut detail blocks: under each section vs end-of-doc appendix.
- Top-of-doc summary table aggregating per-item verdicts: include or skip.
- Risk-tier justification column: include or leave bare.
- Whether `P1_FOCUS` / `P2_FOCUS` / `COREF_VALIDATION_FOCUS` get standalone audit rows or fold into the validation builder's row (not in REQ-V264-03's enumeration of 9).
- 2-sentence verdict-rubric recap at doc-open or assume reader follows the CONTEXT.md link.

## Deferred Ideas

- **Pilot trial cuts during Phase 45** — Considered, rejected (Phase 46's job). Revisit only if risk tier cannot be decided from inspection alone.
- **Rewordings for domain-loaded items** — Phase 45 catalogs them only; Phase 46 produces rewordings empirically via REQ-V264-07.
- **Standalone rows for `P1_FOCUS` / `P2_FOCUS` / `COREF_VALIDATION_FOCUS`** — Default: fold into validation builder row. Planner discretion to elevate.
- **Claude-backend audit** — Out of scope (v2.6.4 is gpt-5.4 only). v2.6.5 candidate iff v2.6.4 promotes.
- **Per-prompt minimization for `s_linker17e`** — Deferred per `.planning/REQUIREMENTS.md` Future Requirements.

## Reviewed Todos (not folded)

Six low-relevance matches surfaced via keyword scoring (top score 0.6):

- `2026-06-02-improve-prompts-v4-axiom-three-root-cause-fp-fixes.md` — v2.6.1 axiom work; not s_linker19 prompt audit.
- `2026-06-04-replay-s-linker19-checkpoints-for-paper-rq1-rq4-eval.md` — already shipped as Phase 43.
- `2026-06-01-design-better-axioms-section-context-responsibility.md` — axiom design, not prompt audit.
- `260601-flex-tier-integration.md` — cost optimization, future milestone.
- `2026-06-01-implement-refined-v3-axiom-diffs-feasibility-study.md` — feasibility study, not audit.
- `2026-06-02-redesign-voyager-training-gate-and-cross-split-logic.md` — v2.6 training, frozen.

None fit Phase 45's read-only audit scope. They stay in the global backlog.
