# Systematic error-mode analysis — replace/union/s21 (before testing more validators)

Run date: 2026-07-04. Method: `pilot/error_mode_analysis.py` reads the predicted-link
CSVs already written by run_ablation and diffs against gold — every FP (with source
stage + sentence text) and every FN (with text, and which variant caught it). NO new
API calls. Purpose: understand WHAT the errors are before deciding whether a stronger
validator can help — instead of bolting on validators blind.

## The ledger (union, one e2e run)

| | teammates | teastore | bbb | total |
|---|---:|---:|---:|---:|
| FP | 4 | 2 | 1 | **7** |
| FN | 8 | 2 | 13 | **23** |

The F1 gap to a win is dominated by **FN (recall), not FP** — 23 vs 7. That already
tells us a stronger (stricter) validator is aimed at the minority side of the ledger.

## FALSE POSITIVE modes (7)

| mode | instances | example | fixable by a stronger validator? |
|---|---|---|---|
| **Code-path / qualified name** | teammates S125, S173, S188 (3) | "Classes in the **storage.entity** package…" → Storage; "**e2e.util** contains helpers" → E2E; "**x.testdriver** contains component test cases" → Test Driver | **YES — but by a structural code-path filter, not an LLM gate.** s21's rules already say to exclude X.Y.Z package paths; the gate just doesn't enforce it structurally. |
| **Wrong referent / anaphora** | teammates S96, bbb S50 (2) | "Although **this component** provides methods…" → Logic; "the **Recording Processor** will take…" → Recording Service | Marginally (a referent check), but only 2 instances |
| **Descriptive / gold-incompleteness** | teastore S35, S36 (2) | "one order-based nearest-neighbor approach is available" → OrderBasedRecommender | No — these arguably ARE correct; a gold issue, not a validator issue |

Only **3 of 7 FP** (the code-path ones) are cleanly targetable, and by a *structural
filter*, not a stronger LLM validator.

## FALSE NEGATIVE modes (23) — the real ceiling

| mode | instances | example | fixable by a stronger validator? |
|---|---|---|---|
| **Generic role-term → specific component** | **bbb ~13** (HTML5 Client / HTML5 Server / WebRTC-SFU) | "the **client** side subscribes to the published collections on the **server** side"; "connecting using **WebRTC**"; "**bbb-html5**" | **NO — a stronger validator REJECTS more, worsening these.** Needs per-sentence client-vs-server disambiguation. |
| **Generic word → component** | teammates ~5 (Logic via "logic", GAE Datastore via "datastore") | "The main **logic** of the application is in POJOs" → Logic; "data is persisted in the **datastore**" → GAE Datastore | No — and see the tension below |
| **Implicit / role reference** | teastore ~2 (Persistence) | "it also acts as a **caching layer**" → Persistence | No — extraction/recall problem |

## Two findings that kill the "stronger validator" premise

1. **The ledger is recall-bound (23 FN vs 7 FP), and the FN modes get WORSE under a
   stricter validator.** bbb (the lowest-F1 dataset) loses 13 links because the doc
   names HTML5 Client/Server/WebRTC by generic role words; a stronger gate rejects
   more of exactly this kind, so it moves F1 the wrong way. This is why the router
   grid showed the router trading 6 TP to kill 3 FP — it is pulling on the wrong end.

2. **The same ambiguous word is an FP in one sentence and an FN in another.** "logic"
   is a FALSE POSITIVE in teammates S79 ("cascade **logic** for create/update/delete"
   → Logic, wrong) and a FALSE NEGATIVE in S7/S8 ("back end **logic**", "main
   **logic**" → Logic, correct, missed). No global validator threshold can be strict
   enough to drop S79 and lenient enough to keep S7/S8 — it is a per-instance semantic
   call. Stronger/weaker validators just slide along this FP↔FN tradeoff curve without
   net F1 gain. This is why s21, replace, and union all cluster at ~93: they hit the
   same semantic ceiling.

## Conclusion — the design space, re-scoped by the errors

- **A stronger validator cannot make replace/union beat s21.** The dominant errors are
  recall-side (generic-term/implicit references to specific components) that a stricter
  gate worsens; the FP side it could help is only ~7, of which the clean 3 are
  code-path cases better handled by a structural filter.
- **The one real, targetable win is a code-path / qualified-name FP filter** (structural,
  GATE-06-safe: match a span occurring only inside an `X.Y[.Z]` identifier path, no
  vocabulary). It removes ~3 teammates FP at ~zero recall cost — worth <+0.5pp on
  teammates, nothing elsewhere. It is NOT a replace-rescue; it is a small precision
  patch that would help s21 too.
- **The F1 ceiling is a disambiguation/recall problem, not a validation problem.** To
  move past ~93 you must recover the generic-role-term references (esp. bbb HTML5
  Client vs Server, WebRTC) — an EXTRACTION + per-sentence disambiguation task — which
  is the opposite of "add a stronger validator."

So: replace/union are parity-not-win because every variant shares one recall ceiling;
no validator fixes it. The productive next direction is client/server-style
disambiguated extraction for role-term references, not a stronger gate.
