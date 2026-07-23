# Phase 6: EXT-01 — Project-Agnostic Standalone-Mention LLM Primitive — Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in `06-CONTEXT.md` — this log preserves the alternatives considered.

**Date:** 2026-05-30
**Phase:** 06-ext-01-project-agnostic-standalone-mention-llm-primitive
**Areas discussed:** Semantic scope of the LLM primitive, Dotted-path handling in EXT-01 baseline

---

## Gray-Area Selection

| Option | Description | Selected |
|--------|-------------|----------|
| API shape & call topology | Per-(comp,sent) vs per-component batch vs Spike-003 piggyback vs document-level enrichment map. The EXT-01 cost/quality signal driver for Phase 8. | |
| Semantic scope of the LLM primitive | Literal (regex-equivalent) vs semantic (architectural reference) vs hybrid. | ✓ |
| Dotted-path handling in EXT-01 baseline | Regex pre-filter vs LLM prompt semantics vs drop now. | ✓ |
| Cost/quality signal capture for Phase 8 | What metrics get logged & where to feed stack-vs-unify decision. | |

**User's choice:** Discuss only Semantic scope + Dotted-path. API shape and cost-metric capture deferred to Claude's Discretion in CONTEXT.md.

---

## Semantic Scope of the LLM Primitive

### Q1: What should the LLM primitive judge per (comp, sentence)?

| Option | Description | Selected |
|--------|-------------|----------|
| Literal: surface mention only | Mirror regex semantics; LLM decides only surface presence. Safe drop-in. (Recommended) | |
| Semantic: architectural reference | LLM decides 'does this sentence reference the component'. Higher precision, riskier on partial refs. | |
| Hybrid: emit both signals | Single call emits {surface_mention, architectural_ref}. More info, doubles schema. | |
| Other: empirical study first | User declined to pre-lock; run study to choose. | ✓ |

**User's choice:** "do empirical study before to choose"
**Notes:** User prefers data-driven selection over a priori design lock. Drove D-01.

### Q2: What does the empirical study compare?

| Option | Description | Selected |
|--------|-------------|----------|
| Hard-tier probe on all 3 variants | Implement all 3; run TM + BBB only; pick winner; then full 5-project sweep. Matches GATE-05. (Recommended) | |
| Full 5-project sweep on all 3 | Most data, most expensive (3× sweep budget). | |
| Offline anchor-collection diff | Replay anchor collection with each variant; diff which sentences each picks. No full pipeline run. | ✓ |

**User's choice:** Offline anchor-collection diff
**Notes:** Cheapest signal; deliberately no full sweep in the diff stage. Drove D-02 stage 1.

### Q3: What decides the winner?

| Option | Description | Selected |
|--------|-------------|----------|
| Macro F1 — highest wins, ties broken by GATE-06 cleanliness | Standard dual-floor + GATE-06 tie-breaker. (Recommended) | ✓ |
| Macro F1 + cost — top F1 within 1.5× baseline call count | Bakes cost discipline into winner rule. | |
| Hard-tier (TM+BBB) F1 only | Optimizes for the rule-sensitive datasets. Risks overfitting hard tier. | |

**User's choice:** Macro F1 only; GATE-06 tie-breaker
**Notes:** Cost is captured but not part of the winner-decision rule — it lives in the D-06 Phase 8 signal block.

### Q4: How do offline diff and macro F1 combine?

| Option | Description | Selected |
|--------|-------------|----------|
| Diff filters candidates, then sweep finalists | Diff pre-filter → drop catastrophic-diff variants → full sweep on remaining 1-2 → F1 picks winner. (Recommended) | ✓ |
| Diff is the whole study; F1 only on winner | Pick by diff alone; run macro F1 only on the chosen one to confirm dual floor. | |
| Sweep all 3 anyway; diff is diagnostic | Full sweep on all 3; diff explains outcomes in writeup. | |

**User's choice:** Diff filters candidates, then sweep finalists
**Notes:** Resolved tension between Q2 (no sweep) and Q3 (F1 winner). Drove D-02 stage 2 and stitched D-02/D-03 together.

---

## Dotted-Path Handling in EXT-01 Baseline

### Q5: How does EXT-01 keep dotted-path skip behavior?

| Option | Description | Selected |
|--------|-------------|----------|
| Regex pre-filter, LLM only on survivors | Cheap pre-filter; Phase 7 cleanly drops the pre-filter. (Recommended) | |
| Encode in LLM prompt semantics | Prompt teaches "token inside dotted identifier is not standalone". GATE-06 audit harder. | |
| Skip dotted-path handling entirely in Phase 6 | Collapse Phase 6 + Phase 7. Bigger blast radius; conflicts with roadmap gating. | |
| Other: run both 1 and 2 | User declined to pre-lock; run both as competing sub-variants. | ✓ |

**User's choice:** "not sure, run both 1 or 2"
**Notes:** Same empirical-first stance as D-01. Drove D-04 (two sub-variants compete, evaluated as one matrix with the semantic-scope study via the same D-02 protocol).

### Q6: How are the two sub-variants named & promoted?

| Option | Description | Selected |
|--------|-------------|----------|
| Two siblings, winner promoted to s_linker13g | Build as e.g. `s_linker13g_pre.py` + `s_linker13g_sem.py`; winner byte-copied to `s_linker13g.py`. Mirrors v1.0 13f→s_linker13. Loser stays as rejected artifact. (Recommended) | ✓ |
| Two siblings, both registered, no promotion | Ship both as canonical; Phase 8 picks. Doubles ablation rows. | |
| Single variant, A/B switch via flag | One file with constructor flag. Bad for GATE-07 ("standalone file, no flags"). | |

**User's choice:** Two siblings, winner promoted to s_linker13g
**Notes:** Reuses the v1.0 promotion pattern; GATE-07 stays clean. Drove D-05.

---

## Claude's Discretion

Areas user delegated to Claude (researcher + planner):
- API shape & call topology of the new primitive (constrained by D-02 diff-ability and D-06 cost-signal capture)
- Cost/quality metric set beyond the D-06 minimum
- Fallback policy on LLM failure (default: approve-bias per existing pattern)
- Anchor-section vs has_exact_case-flag split (currently unified; may split)
- Prompt-example domains (must come from BENCHMARK_TABOO.md "Safe SE Textbook Examples")

## Deferred Ideas

- EXT-04 (alias-discovery boundary prompting for BBB variance band) — already deferred to v2.1+ per ROADMAP/PROJECT.md, restated here for completeness.

No new scope creep raised during discussion.
