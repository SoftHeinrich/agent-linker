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

---

## Iteration 1 — sibling-disambiguation extraction (elegant change, error-driven)

Root cause targeted: bbb's 13 FN were 100% extraction misses (0 reached any gate),
concentrated on the HTML5 Client/Server sibling family referenced by role words.
Change (`proposer._sibling_hint`, opt-in `sibling_disambig`, wired into
`SLinker23._propose`): detect catalog components sharing a distinctive base token
(HTML5 Client/Server, Redis DB/PubSub — purely structural, GATE-06 safe) and instruct
the reader to resolve role/base references to the specific sibling(s). Not a filter/
fallback — a change to how references are resolved.

**Extraction-ceiling test** (`pilot/sibling_extract_probe.py`, the error-mode metric,
no gate): bbb recall 47/62=0.758 → **55/62=0.887** (+13pp); sibling-family gold
20/33 → **29/33** (+9), 0 lost. Recovered exactly the targeted HTML5 Client/Server
misses (S10 client+server dual-ref, S11–13, S19, S76).

**E2E + error RE-ANALYSIS on bbb** (the loop's confirm step):
- s23_verify (router path): the HTML5-family links come through as TP (S9/12/13/19/
  20/21 Server, S76/79 Client recovered). FP stayed the pre-existing modes
  (S50 Recording-Processor referent, S58 FSESL, S79 coref) — **none sibling-caused**.
  → clean win via the augmentation path.
- s23_union (raw-gate path): recall highest but FP blew up to 16, exposing TWO modes
  the router absorbs but the raw gate does not:
  - **sibling over-reach**: S14 names only "HTML5 client" but the "list each" clause
    also emitted HTML5 Server (FP).
  - **diagram-caption dump**: S2/S87 ("the following diagram shows the various
    components…") → the whole catalog linked to one meta-sentence (11 FP on S87).

**Verdict for iteration 1:** the elegant change fixes the target mode at its root and
is safe *through the router* (s23_verify), not through the raw gate (union/replace) —
consistent with the standing finding that union dumps into the gate. Keep
`sibling_disambig` on for the augmentation path.

**New residual modes surfaced for the next iteration (error-driven, not F1-chasing):**
1. **WebRTC-SFU via "WebRTC"** (S65, S73) — a *single* component named by a technology
   word that isn't its catalog name; sibling logic doesn't apply. Needs alias
   discovery ("WebRTC" → WebRTC-SFU).
2. **Diagram-caption / "the various components" dump** (S2, S87) — meta-sentences that
   refer to components collectively; an elegant fix recognizes collective/meta
   reference and declines per-component links, rather than a per-sentence FP filter.

(F1 deltas this run are NOT reported as evidence: the s21 floor sampled low
R=62.9; the reliable signal is the structural recovery of the specific HTML5-family
links, which is noise-robust. F1 effect needs a repeated-run harness.)
