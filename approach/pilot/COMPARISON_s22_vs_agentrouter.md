# Typed router: s22 vs the early agentic router — and a clean generalization

Date: 2026-07-02. Compares the current typed variant (`SLinker22`) against the
earlier type-based router (`SLinker21AgentRouter` + `DocModelAgenticRouter` +
`GroundedTypedProposer`), then motivates the clean reimplementation shipped as
`SLinker23`: an **LLM-decision-driven** augmentation where routing and validation
are chosen by the model from **general guidelines**, with essentially no
structural `if/else` policy in code.

## The two existing designs

Both recover the same *kind* of link s21 misses (component mentions buried in
contrast/negation, plus affirmative mentions Framing-C didn't surface) and both
preserve a gate-floor invariant (an added link is always approved by s21's
two-pass entity gate, so neither can regress below s21). They differ in *how the
route from candidate to validator is decided* and in *how they reuse s21*.

| Axis | `SLinker21AgentRouter` (early) | `SLinker22` (current) |
|---|---|---|
| Candidate source | `GroundedTypedProposer`, per-sentence, own cached LLM client | typed extraction inlined into Phase 2 (reuses s21's `_run_parallel`/`_iter_batches`/`_ask`) |
| Route decision | a **dedicated LLM agent call** emits `VALIDATE / CODE / REJECT` per candidate | the extraction **MODE is the route**, dispatched by hand-written `if/elif` in `_validate_with_evidence` |
| Gate | `StrictGate` — a **reimplementation** of s21's two-pass validator (own prompt `_gate_prompt`, own `_RULES` copy, own client) | the **real** s21 methods (`_run_validation_pass`, `P1_FOCUS`, `P2_FOCUS`) |
| Floor | augments *final* s21 output post-hoc (`super().link()` then a bolted-on second pipeline) | integrates typed extraction into the *live* Phase-2 floor; overrides only two seams |
| LLM subsystems | **three** clients (proposer, agent router, gate) each re-setting env vars | **one** inherited s21 client with phase tags |
| CODE handling | routes CODE candidates to `router_direct` (doc→code) | rejects `CODEPATH` from doc→model; no code escape hatch |
| Measured (macro) | P 0.9592 / R 0.9247 / **F1 0.9402** | P 0.9779 / R 0.9232 / **F1 0.9494** / F2 0.9334 |

Reading: s22 is both **simpler** (no extra agent call, no duplicated gate) and
**stronger** (+0.9pp F1). The agentic router's own docstring frames the LLM
`VALIDATE/CODE/REJECT` step as *replacing* "the hard-coded mode→judge dispatch
table"; s22 quietly shows the dispatch table was the better trade — cheaper
(one fewer LLM round-trip per candidate), and higher precision because it stops
duplicating the gate.

## What is still not clean in s22

s22 wins on behavior but carries two architectural smells:

1. **`_validate_with_evidence` is a 70-line copy-of-s21-plus-branches.** It
   re-derives base case text, re-runs the two-pass loop, and threads a
   mode-branch through the middle. The base path is a near-duplicate of s21's
   method body.
2. **The mode→policy mapping is control flow, not data.** Adding a mode, or
   swapping the AFFIRMATIVE prefilter, means editing branching logic. The three
   "reject" modes, the evidence filter, and the contrast route are all inlined.
3. **Two parallel typed-extraction implementations** now coexist
   (`GroundedTypedProposer` and s22's `_prompt_typed_extraction`), and **two
   notions of "router"** (LLM-action vs mode-dispatch) with no shared vocabulary.

## The design directive that shaped SLinker23

An intermediate `SLinker23` did lift s22's routing into a data-driven
`ModePolicy` registry (`typed_mode_router.py`) that matched s22's decisions
exactly. But that registry still *encoded the policy in code* — regex evidence
filters (`code_like`, `terminal_quote`, `affirmative_evidence_ok`) and a
mode→validator dispatch. The chosen direction is the opposite: **move the decision
into the model, keep only general guidelines in the prompt, and keep almost no
`if/else` policy in Python.** The structural registry was therefore removed.

## The shipped generalization (SLinker23, LLM-driven)

`SLinker23(SLinker21)` runs canonical s21 as the floor, then augments:

1. **Propose (LLM, generic prompt).** `GroundedTypedProposer` reads each sentence
   against a generic English rubric and lists referenced components; the only
   non-LLM step is *grounding* a proposal to a real catalog name (data validation,
   not a decision).
2. **Route (LLM, general guidelines).** `DocModelAgenticRouter` lets the model
   choose ONE action per candidate — `VALIDATE / CODE / REJECT` — from a generic
   rubric. No regex, no mode table, no evidence pre-filter.
3. **Floor (s21's real gate).** A `VALIDATE` candidate is accepted only if s21's
   OWN two-pass entity validator approves it — injected as the router's `gate`, so
   the *real* s21 rubric/passes decide (unlike the earlier agentrouter, which
   reimplemented the gate as `StrictGate`). The agent can only divert or defer;
   it can never add a link the gate rejects.

What this buys, against the two prior designs:

- **vs the structural registry:** no `code_like`/`terminal_quote`/
  `affirmative_evidence_ok` predicates and no mode→validator `if/elif`; the
  keep/route/reject judgement is the model's, guided only by generic text.
- **vs `SLinker21AgentRouter`:** same LLM-decision philosophy, but the gate is
  s21's *actual* validator (zero reimplementation), and it subclasses s21 cleanly
  rather than bolting on a second proposer+router+gate stack.

The bounded-autonomy invariant (accept ⇔ LLM-`VALIDATE` ∧ s21-gate-approve, so no
regression below s21) is proven deterministically in
`pilot/test_s23_gate_floor.py`. Live pilot runs are recorded in `RESULTS.md`
(§ SLinker23).
