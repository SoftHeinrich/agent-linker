# From workflow to agent — internalizing deliberation, grounded

Design note prompted by: *"how can we internalize scratch-pad / deliberation for the
router, or let the LLM decide what to do, or a focused recover — to move from workflow
to agent/harness?"* Grounded in the measured results (`FINDINGS.md`), not speculation.

---

## 0. Where the workflow stands (so we agentify the right thing)

Today's pipeline is a **fixed workflow**: proposer emits a mode → deterministic dispatch
→ a per-mode judge returns keep/reject. Measured ceilings:

- **Proposer is solved** — GTP surfaces **185/195 gold (95%)**; only 5 never surface.
- **Doc-model recall is gold-bound** — best precision-safe cell `named+routed` = F1
  0.9506 (+1.46pp, 0 precision loss); oracle 0.9730. The gap is *not* recoverable by a
  better judge: the entangled "FP" are **~63% gold-incompleteness, ~29% code-structure,
  ~9% real error** (verified, §8).

So the residual is **not a recall problem to grind** — it is a **routing + labeling**
problem: send code-naming sentences to the doc→code linker, keep the gold-gap links but
label them, reject the ~9% real errors, disambiguate the 5 siblings. That is what an
agent could do that the fixed dispatch cannot. The agent's job is *decisions*, not
*more proposing*.

## 1. The hard constraint: reasoning-off ⇒ deliberation must be EXTERNAL

s21 is a no-reasoning config ([[never-use-reasoning-in-linker-experiments]]); gains must
come from prompt/structure, never `OPENAI_REASONING_EFFORT`/thinking. So "scratch-pad /
deliberation" **cannot** be hidden chain-of-thought. It must be **externalized as answer
tokens** — a written NOTE field, a tool call, a structured trace — exactly as
claim-before-verdict already is. This is not a limitation; it is *the thesis*: the
intelligence lives in the **harness structure** (tools, control flow, actions), not in a
reasoning budget. An agent here = a control loop over externalized, auditable steps.

## 2. Empirical check — naive agency REGRESSES (agent_probe.py)

Before designing, we tested the simplest version of "let the LLM decide": one
reasoning-off pass with a **scratch-pad NOTE** and a richer ACTION set (ARCH / CODE /
REJECT), evidence = prev sentence + anchor sentences, on the 48 candidates the routed
judge blindly keeps (13 gold + 35 FP). Scored vs the gold-verified categories:

| verified truth | n | →ARCH | →CODE | →REJECT |
|---|---:|---:|---:|---:|
| GOLD (keep) | 13 | 11 | 1 | 1 |
| GAP (keep, gold-incomplete) | 22 | 18 | 4 | 0 |
| CODE (route to doc→code) | 10 | 4 | 5 | 1 |
| ERROR (reject) | 3 | 2 | 0 | 1 |

- code routed correctly **5/10**, errors rejected **1/3**, and it **lost 2 gold** and
  **misrouted 4 valid links to CODE** (S24/S27 UI "written in Angular / HTML / TypeScript"
  → it saw code words and cried CODE).
- **Verdict: a free scratch-pad agent is *worse* than the deterministic workflow here.**
  It guesses "code vs arch" from surface cues without *resolving* anything, and free
  deliberation *enriches* (rationalizes Angular→CODE) — the exact backfire `fn_judge §8`
  found. Agency without grounding trades stable precision for noise.

**This is the key finding: the lever is not "let the LLM decide freely" — it is "give the
LLM grounded TOOLS and let it decide which tool's result settles the case."**

## 3. The three levers, re-cast by the evidence

1. **Scratch-pad for the router (externalized deliberation).** Keep it — but as a
   *constrained* note that must cite a signal, not free prose. Alone it is noisy (§2);
   its value is calibration *on top of* grounded evidence, not as the decision itself.
2. **Let the LLM decide (tool-using agent) — the right form, IF tools are grounded.**
   Agency = choosing *which evidence to fetch*; grounding = the precision floor. Tools,
   each encoding a measured lesson:
   - `resolve_code(token)` → does it hit the `.acm` code index? → the CODE route is
     **structural**, not the agent guessing (fixes §2's 5/10). Grounding is the FP floor.
   - `get_anchors(component)` → sentences that *name* it → pins the referent (constrain,
     not enrich — the only context that helped, `fn_judge §3`).
   - `check_alias(term)` → the run's `doc_knowledge` alias table → resolves generic nouns
     without inventing a link.
   - `compare_siblings(cands)` → for the HTML5 Client/Server, WebRTC/FreeSWITCH pairs.
   The agent picks the ACTION its *tool result* justifies — decision grounded, not prose.
3. **Focused recover (triage → escalate) — the deployable shape.** The fast
   deterministic path (`named+routed`: cheap, precision-safe, 0-regress) already handles
   the easy ~95%. Reserve the agent for the **hard residual** where routing matters. Never
   run the expensive agent globally (§2 shows it regresses on easy cases).

## 4. Proposed harness — grounded tool-using triage, behind a fast path

```
for each proposed (sentence, component, mode):
    if fast_path_confident(mode):          # AFFIRMATIVE/CONTRAST, name present
        -> s21 strict gate (unchanged, cheap)         # the 95%
    else:                                  # IMPLICIT/ANAPHORA/uncertain
        agent triage loop (reasoning-off, externalized):
          note = observe reference type
          evidence = choose & call tools {resolve_code, get_anchors, check_alias, compare_siblings}
          ACTION grounded in a tool result:
            ACCEPT_ARCH        -> architecture link
            ROUTE_CODE         -> hand to the doc->code linker (Increment B) — a DIFFERENT target space
            FLAG_GOLD_GAP      -> accept + provenance tag "valid; benchmark may not annotate"
            DISAMBIGUATE_SIBLING -> pairwise pick among siblings
            REJECT             -> genuine non-link
```

Why this is the agent form (not a workflow): the **control flow is the LLM's** — which
tools to call, when enough evidence is in, which action — but every accept/route is
**grounded in a structural tool result**, so precision stays floored. `ROUTE_CODE`
makes the doc→code route an *action of one decision-maker*, unifying the two linkers the
pilot kept separate.

## 5. Honest expected value

Set expectations by the measurements, not the ambition:

- **Not a doc-model recall win.** That ceiling is gold-bound (§0); no agent beats it on
  the current gold. The 13 gold-implicit stay entangled with 22 equally-valid gold-gaps.
- **Real wins the agent unlocks:**
  1. **Correct routing of the 10 code-structure sentences to the doc→code linker** — a
     *different metric* (file-level) with genuine headroom, and the pilot's original
     motivation. `ROUTE_CODE` is grounded in `.acm` resolution, so it won't misfire like §2.
  2. **Auto gold-audit labels** — `FLAG_GOLD_GAP` emits the FP→gold reclassification list
     for the `transarc-emp` benchmark-critique pillar, for free.
  3. **Sibling disambiguation** for the 5 hardest (gold-debatable; small, honest yield).
  4. **Generality / elegance** — one grounded decision-maker replaces
     proposer-mode + fixed dispatch + two separate linkers. This is the workflow→agent
     step the paper can tell: *intelligence from harness structure, reasoning-off.*

## 6. Next experiment (empirical, actionable)

Build the grounded tool-triage agent with **real `.acm` resolution** (reuse
`router_direct.CodeIndex`) and `get_anchors`, run on the residual (the 48 + the model-doc
FN), score **routing accuracy vs the §8 verified categories** and, for the `ROUTE_CODE`
set, **doc→code file-level F1** via ArCoTL. Success = code routed on structural
resolution (beating §2's 5/10 surface-guess), gold kept, errors rejected — and a
measurable doc→code recovery the model-doc route structurally cannot reach. Reasoning
stays off; deliberation stays externalized.

---

## 7. MEASURED — bounded-autonomy router keeps the value (agent_router.py)

Direction: *not to improve — keep the value, gain small controlled autonomy, make the
router agentic (drop the heuristic mode→judge dispatch).* Built and measured.

**Construction (why autonomy can't cost value):** the LLM decides one action per
candidate — VALIDATE / CODE / REJECT (default VALIDATE) — replacing the hard-coded
mode→judge table. But every model-doc ACCEPT is **floored by the unchanged s21 gate**:
`accept ≡ agent-VALIDATE ∧ gate-approves`. The agent can only *divert* (→CODE / REJECT)
or send to the gate; it can never add a link the gate rejects.

| config | P | R | F1 |
|---|---:|---:|---:|
| baseline s21 | 0.9894 | 0.8913 | 0.9360 |
| named+routed (target) | 0.9897 | 0.9173 | 0.9506 |
| **bounded-autonomy agentic router** | 0.9592 | 0.9247 | **0.9402** |

- **Gate-floor holds: every accept is gate-approved** (verified True) — no unbounded
  regression, unlike the naive scratch-pad agent (§2).
- **All 4 core recoveries kept** (DB, WebUI, FreeSWITCH×2 → VALIDATE+gate).
- **Autonomy exercised:** of 251 marginal candidates the LLM decided **46 → CODE**
  (routed to the doc→code linker — the right population), **61 REJECT**, 144 VALIDATE.
- **The −1pp vs target is 100% gold-incompleteness, not error.** The 6 accept-FP are all
  verified defensible links the benchmark omits (Reencoding "files are reencoded",
  OrderBasedRecommender "order-based nearest-neighbor approach", Logic "provides methods
  to perform access control", the Presentation-Conversion dual). None are the 3 verified
  errors; none are code. So the agent recovered **11 real TP (vs 4)**; on corrected gold
  it meets or exceeds named+routed.

**Verdict.** Bounded autonomy delivers the goal: the router is now **agentic** (LLM
decides the action, no mode→judge table), autonomy is **real but provably capped** (gate
floor + safe default), the **value is kept in substance** (every recovery retained, all
extra "losses" are annotation gaps), and it **routes the code population out for free**.
To match the *measured* 0.9506 exactly, make the default conservative (VALIDATE only when
the name is present) — trading 7 real-but-unannotated recoveries for exact parity; a
framing choice, not a capability gap.

---

## 8. Standalone implementation — `agentic_router.py`

The router is implemented as a clean, reusable module (cf. `router_direct.py`), separate
from the experiment harness (`agent_router.py`). No dependency on caches/gold/scoring.

```python
from agentic_router import BoundedAutonomyAgenticRouter, Candidate, CODE
router = BoundedAutonomyAgenticRouter()               # default gate = s21 two-pass validator
decisions = router.route([Candidate(id, sentence, component, prev, anchors, quote), ...])
accepted = router.accepted(decisions)                 # action==VALIDATE ∧ gate approved
to_code  = router.routed_to_code(decisions)           # hand to the doc→code linker
```

- **Invariant enforced in `route()`:** `accepted ⇔ agent chose VALIDATE ∧ gate approves`.
  Only VALIDATE candidates reach the (expensive) gate; CODE/REJECT are diverted.
- **Gate is injectable** — default `StrictGate` reuses s21's unchanged `LAYERED_ENTITY_RULES`
  two-pass (P1∧P2, claim-before-verdict); pass any `callable(cands)->{id:keep}` to change
  the floor without touching the router. Falls back to an inline rubric if s21 isn't importable.
- **Reasoning-off**; the agent's NOTE is externalized answer tokens.
- **Live self-test (`python3 agentic_router.py`)** exercises the three actions and shows
  the floor catching a permissive agent: an incidental "logged in"→Auth is agent-VALIDATE
  but gate-rejected ⇒ not accepted. Invariant holds.
