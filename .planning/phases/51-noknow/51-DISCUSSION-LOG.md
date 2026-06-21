# Phase 51: NOKNOW - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-21
**Phase:** 51-noknow
**Areas discussed:** Disable mechanism, Run depth N, Execution & cost gating, GATE-01 evidence + disable scope

> Note: the command was invoked as `/gsd-discuss-phase 11`, but phase 11 is an
> archived, completed phase in milestone v2.1. User confirmed the intent was
> **Phase 51 (NOKNOW)** in the active v2.6.6 milestone; the discussion proceeded
> against phase 51.

---

## Disable mechanism (wiring)

| Option | Description | Selected |
|--------|-------------|----------|
| Env/ctor flag in union | Guarded flag inside `s_linker20_union.py`; default-off path snapshot-identical; minimal diff, no duplication | ✓ |
| Standalone `s_linker20_noknow.py` | Copy the 1086-line variant and null two fields; max standalone-files preference but real drift risk | |
| Thin subclass | `S20NoKnowledge(SLinker20Union)` overriding layer1 builders; small but an inheritance chain (disfavored) | |

**User's choice:** Env/ctor flag in union (Recommended).
**Notes:** GATE-01 constrains full-knowledge *behavior*, not source bytes; NOKNOW-01 literally says "flag/variant".

## Disable mechanism (surfacing to runner/extractor)

| Option | Description | Selected |
|--------|-------------|----------|
| Registered sibling variant | `run_ablation.py` entry `s_linker20_union_noknow` constructing `SLinker20Union(no_knowledge=True)`; distinct name + `_links.csv` prefix; no logic duplicated | ✓ |
| Bare env var | `S20_NO_KNOWLEDGE=1` read in `__init__`; hidden toggle the extractor must be told about out-of-band | |
| You decide | Defer to planning | |

**User's choice:** Registered sibling variant (Recommended).
**Notes:** Clean Full vs No-Knowledge separation for results dirs + the Phase-50 extractor.

---

## Run depth N

| Option | Description | Selected |
|--------|-------------|----------|
| N=1 | 10 live runs; matches REQ floor (N≥1) and milestone scope; cheapest | |
| N=3 (symmetric) | 30 live runs; symmetric with Full; variance bands; ~3× cost | ✓ |
| N=2 (compromise) | 20 live runs; min/max range without full N=3 cost | |

**User's choice:** N=3 (symmetric).
**Notes:** User added: "remember the output folder should annotate as no-knowledge." Pulls the deferred NOKNOW-N variance work into this milestone; supersedes the N≥1 floor.

---

## Execution & cost gating (trigger)

| Option | Description | Selected |
|--------|-------------|----------|
| Phase ships scripts, you trigger | Phase verifies path + ships run scripts; user launches the sweep (Phase-48 gating pattern) | |
| Phase runs the sweep end-to-end | Phase execution kicks off all 30 runs + extract; fully autonomous | ✓ |
| Split: verify in 51, run in follow-up | Disable path + GATE-01 + scripts in 51; sweep in a separate triggered step | |

**User's choice:** Phase runs the sweep end-to-end.
**Notes:** No separate go-ahead checkpoint; build → GATE-01 → 30-run sweep → extract all within the phase.

## Execution & cost gating (cost ceiling)

| Option | Description | Selected |
|--------|-------------|----------|
| Soft cap, log & continue | ~$60 soft budget; log cumulative cost; no hard abort; resumable via `.done` markers | ✓ |
| Hard cap with abort | Dollar ceiling aborts the sweep cleanly if crossed | |
| No cap, just log | No ceiling; just log cost per success criterion | |

**User's choice:** Soft cap, log & continue (Recommended).
**Notes:** Matches "replaceability trumps cost" stance for eval work; No-Knowledge runs are cheaper per run (3 fewer LLM call-sites).

---

## GATE-01 evidence + disable scope (layer1 behavior)

| Option | Description | Selected |
|--------|-------------|----------|
| Skip the layer1 LLM calls | Don't call ambiguity/doc-knowledge prompts; set knowledge empty directly; truly knowledge-free AND cheaper | ✓ |
| Run layer1 calls, then discard | Compute then null; keeps populated layer1 for audit but wastes spend | |
| You decide | Defer to planning after reading call-sites | |

**User's choice:** Skip the layer1 LLM calls (Recommended).
**Notes:** ∅ `ambiguous_names`, `{}` `aliases`; downstream degrades automatically to canonical-name-only.

## GATE-01 evidence + disable scope (proof)

| Option | Description | Selected |
|--------|-------------|----------|
| Snapshot replay vs frozen caches | Zero-LLM: flag-off replay vs 30 frozen phase_caches → byte-identical layer1 + `_links.csv`; plus additive-guard structural check | ✓ |
| Structural guard only | Assert by inspection/AST that change is additive + guarded; no behavioral replay | |
| Per-prompt golden snapshot | v2.6.4-style byte-equality fixtures for the 6 prompt builders; more harness to build | |

**User's choice:** Snapshot replay vs frozen caches (Recommended).
**Notes:** Reuses Phase 50's faithfulness-oracle lineage (set-equality on `(sentence, component_id, source)`).

---

## Claude's Discretion

- Exact No-Knowledge results/extract directory naming (must be unambiguously annotated + machine-distinguishable from Full).
- Whether GATE-01 replay covers all 30 cells or a representative subset (structural guard backstops it).
- Run-script ergonomics — mirror the existing `run_s20union_{gpt,sonnet}_n3.sh` skeleton.

## Deferred Ideas

- 6 pending axiom/prompt-design todos in `.planning/todos/pending/` — v2.6.1/v2.6.2 axiom-era work, out of scope for v2.6.6 eval infra; not folded.
- REQUIREMENTS.md update at phase close: the N≥3 variance work (formerly future `NOKNOW-N`) landed here via D-05 rather than being deferred.
