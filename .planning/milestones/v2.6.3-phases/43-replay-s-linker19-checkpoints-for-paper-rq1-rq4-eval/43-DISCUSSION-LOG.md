# Phase 43: Replay s_linker19 checkpoints for paper RQ1–RQ4 eval - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-04
**Phase:** 43-replay-s-linker19-checkpoints-for-paper-rq1-rq4-eval
**Areas discussed:** Adapter location & layering, Backend ordering in tables (RQ1 layout), RQ3 NoConsensus replay strategy (became: RQ3 redesign), RQ4 overlap table shape (with single-backend split)

---

## Area 1 — Adapter location & layering

| Option | Description | Selected |
|--------|-------------|----------|
| 2-stage: replay (approach) → format (evaluation) | `approach/scripts/v2.6.3/` emits CSVs; `evaluation/src/lib/metrics_api.py` (unchanged) handles RQ1; new `evaluation/src/paper/{rq3_table,rq4_table}.py` render TeX | ✓ |
| 1-stage end-to-end in approach | Everything in `agent-linker`; `metrics_api.py` imported as a library | |
| Hybrid: RQ1 in evaluation, RQ3/RQ4 in approach | RQ1 stays evaluation-side; RQ3/RQ4 TeX written from approach because they need linker-internal layer dicts | |

**User's choice:** 2-stage.
**Notes:** Clean separation: pickle deserialization stays in `agent-linker` (where the `Link` dataclass lives); TeX formatting stays in `evaluation/` (matching the existing `src/paper/generate_tables.py` pattern). Preserves the stdlib-only constraint on `evaluation/`.

---

## Area 2 — Backend ordering in RQ1 tables

| Option | Description | Selected |
|--------|-------------|----------|
| Wide, Claude-first (5 rows, 2 column groups) | 5 project rows × `Claude \| GPT-5.4` column groups; Claude leads because it is the canonical-default backend with higher s_linker19 numbers | ✓ |
| Wide, GPT-first (5 rows, 2 column groups) | Same shape, GPT-5.4 first | |
| Long, backend-major (10 rows, Claude block then GPT) | Two 5-row blocks stacked | |
| Long, project-major (10 rows, backend pairs per project) | Per-project pairs | |

**User's choice:** Wide, Claude-first.
**Notes:** This sets the paper-wide convention for RQ1 tables. RQ3/RQ4 were later split (Area 4) so the both-backend layout only applies to RQ1.

---

## Area 3 — RQ3 redesign (replaces "NoConsensus replay strategy")

### Initial question (NoConsensus strategy)

A 4-option AskUserQuestion was prepared on how to materialize NoConsensus (accept-by-default lossy AS LOGGED reading, checkpoint replay, hybrid, or skip downstream gates). User asked for clarification first: "where does the diff between union and intersection come from?" and "what's the implemented version?"

Investigation surfaced that:
- The Full pipeline is hard-coded as intersection (`s_linker19.py:637`: `intersected = {k: pass1[k] for k in pass1 if k in pass2}`).
- `pass1` and `pass2` are stashed on `self` only for pickle export — never consumed downstream.
- Therefore `layer3.decisions` and `layer4.coref_decisions` have entries only for intersection members; the symmetric difference `pass1 △ pass2` has no cached validator decisions at all.
- Any NoConsensus reconstruction therefore needs to *invent* a validator decision for the symmetric-difference candidates — which conflicts with the "zero new LLM calls" constraint or produces a methodologically weak counterfactual.

**User's decision:** "remove the nonConsensus validator in that rq, only compare valadators that run LLM calls. lets resign the rq". The original 4-option strategy question was withdrawn.

### Redesigned RQ3 question

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — 3 ablations + Full | RQ3 = {Full, NoEntityValid, NoCitation, NoValidator}. Consensus voting stays inside Full as part of the extractor (not a validator) | ✓ |
| Yes + report consensus voting separately | Same RQ3 ablations, with a side note giving consensus stability metric | |
| Reconsider — keep NoConsensus | Roll back to one of the four reconstruction strategies | |

**User's choice:** 3 ablations + Full.

### Validator labels

| Option | Description | Selected |
|--------|-------------|----------|
| Entity validator + Coref/Citation validator | Matches code: layer3 = entity, layer4 = coref with citation | |
| Entity + Coref (citation as sub-mechanism) | Layer4 framed as coref-with-internal-citation-check | |
| Three named validators (pre-redesign framing) | Consensus + entity + coref | |
| Use TeX macros so labels can be renamed centrally | Define `\entValidator`, `\corefValidator`, variant macros in `abbrev.tex` | ✓ (free text) |

**User's choice:** "use macros so we can easily change that".
**Notes:** Macro convention defined in CONTEXT.md D-10. Sidesteps the labeling question — every label is a single-file edit.

---

## Area 4 — RQ3/RQ4 backend split + RQ4 shape

### Cross-cutting layout decision

User instruction (free text): "for rq3,4, only use result of 1 backend and put the other one at appendix". RQ1 keeps the wide both-backend layout from Area 2; RQ3 and RQ4 main body uses a single backend.

| Option | Description | Selected |
|--------|-------------|----------|
| Claude in main body, GPT-5.4 in appendix | Matches Claude-first convention + default-model policy + stronger s_linker19 numbers | ✓ |
| GPT-5.4 in main body, Claude in appendix | Headline the harder backend (ICSE robustness framing) | |

**User's choice:** Claude main body, GPT-5.4 appendix.

### RQ4 per-linker columns

| Option | Description | Selected |
|--------|-------------|----------|
| TPs caught, Unique TPs, FPs, ΔF1-if-removed | 2-row table (Entity / Coref); footer row carries overlap-TP `|E ∩ C ∩ gold|`. Reuses existing `rq4-agents.tex` structure | ✓ |
| Compact: Unique TPs, Overlap-TP, Recall | Drop FP and ΔF1 columns; pure UpSet decomposition | |
| Per-project rows × overlap columns | 5 project rows × `(only_E, both, only_C, Recall)` | |

**User's choice:** TPs caught + Unique TPs + FPs + ΔF1-if-removed.

### RQ4 column-shape (original question, not asked)

A 4-option AskUserQuestion on RQ4 column structure (compact UpSet / detailed overlap / minimal UpSet / split tables) was prepared but withdrawn when the user redirected to the single-backend split decision above. The 2-row × 4-column shape selected after the redesign covers the equivalent ground.

---

## Claude's Discretion

- Precise filenames inside `approach/scripts/v2.6.3/` (one script per RQ vs combined).
- Whether the appendix backend mirror is a single combined figure or per-RQ duplicates.
- Exact macro names in `writing/working/abbrev.tex` — starter shapes given, planner can align to surrounding conventions.

## Deferred Ideas

- NoConsensus as a future validator-design study (would require new LLM calls).
- Per-project breakdown for RQ3/RQ4 in main body (currently aggregate only; per-project lives in appendix or stays out).
- Consensus-voting stability metric `|pass1 ∩ pass2| / |pass1 ∪ pass2|` as a side diagnostic in §approach or appendix.
- Phase 37 / v2.6 close (GATE-06 'Persistence' taboo fix + v2.6 audit) — already on the deferred list in STATE.md.
