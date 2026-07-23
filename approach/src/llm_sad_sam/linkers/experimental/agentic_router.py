#!/usr/bin/env python3
"""DocModelAgenticRouter — bounded-autonomy agentic router for the DOC->MODEL task.
STANDALONE, reusable module.

NAMING — do not confuse this with ``DocCodeSentenceRouter`` (``router_direct.py``):
that one triages SENTENCES for the DOC->CODE task (ARCH vs CODE, should this sentence
even go through direct code-linking). This one decides per-CANDIDATE for the DOC->MODEL
(sentence->component) task — it operates on candidates a proposer already generated,
and its own CODE action is the escape hatch that hands a candidate to the doc->code
route (`router_direct.DirectCodeLinker`/`DirectLinkJudge`), not a replacement for it.

The router that makes the doc-to-architecture linker *agentic* without losing value:
an LLM decides one ACTION per candidate — VALIDATE / CODE / REJECT — replacing the
hard-coded mode->judge dispatch table (router less heuristic). But autonomy is BOUNDED
by construction:

    accept  ==  agent-chose-VALIDATE  AND  the trusted gate approves

The agent may only DIVERT a candidate (-> CODE-linker / REJECT) or send it to the gate;
it can NEVER add a link the gate rejects. So:
  * downside is provably capped by the (unchanged) gate — cannot regress below it;
  * doing nothing special == the deterministic path;
  * the safe default on any parse failure is VALIDATE (fall back to the trusted gate).

Reasoning-OFF: the agent's deliberation is an externalized NOTE field (answer tokens),
never hidden chain-of-thought — same discipline as s21's claim-before-verdict.

This is the linker component (cf. `router_direct.py`); it has NO dependency on caches,
gold standards, or the scoring harness (that is `agent_router.py`, the experiment).

    router = DocModelAgenticRouter()                 # default gate = s21 two-pass
    decisions = router.route([Candidate(...), ...])
    accepted  = [d.candidate for d in decisions if d.accepted]
    to_code   = [d.candidate for d in decisions if d.action == CODE]

Empirically (see the archived pilot narrative at
.planning/archive/router-pilot-260701/gtp/FINDINGS.md §7): keeps every core recovery,
gate-floor holds, and routes the code population out; its only measured "loss" vs
named+routed is verified gold-incompleteness, not error.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

VALIDATE, CODE, REJECT = "VALIDATE", "CODE", "REJECT"
_ACTIONS = (VALIDATE, CODE, REJECT)


def _make_client(model: str | None = None):
    """Create a client without changing process-wide OpenAI configuration."""
    from llm_sad_sam.llm_client import LLMClient, LLMBackend
    # The ablation runner can explicitly select checkpoint replay.  Preserve the
    # historical standalone OpenAI default when no backend is configured.
    backend = None if os.environ.get("LLM_BACKEND") else LLMBackend.OPENAI
    return LLMClient(backend=backend, model=model, enable_logging=False)


# ── data types ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Candidate:
    """One (sentence, component) trace-link candidate to route."""
    id: str
    sentence: str
    component: str
    prev: str = ""                       # previous sentence — constraining context
    anchors: tuple = ()                  # sentences that NAME the component (pin referent)
    quote: str = ""                      # how the proposer saw it referenced
    is_ambiguous: bool = False           # component name flagged ambiguous (gate context)


@dataclass
class Decision:
    candidate: Candidate
    action: str                          # VALIDATE | CODE | REJECT
    note: str = ""                       # externalized deliberation (answer tokens)
    gate_passed: Optional[bool] = None   # None unless the gate was consulted
    accepted: bool = False               # INVARIANT: action==VALIDATE and gate_passed


# ── the trusted gate (default = s21's unchanged two-pass entity validator) ────

# Fallback rubric if s21 is not importable — keeps the module runnable standalone.
_FALLBACK_RULES = (
    "A component is APPROVED only if the sentence makes a specific architectural claim "
    "that this component is used, provides/consumes a service, is implemented, contains "
    "or is contained, or stores/routes data. Reject negations, contrasts that deny the "
    "component is part of the system, incidental words, and references to a different "
    "entity or a product/brand name.")

try:
    from llm_sad_sam.linkers.experimental.s_linker21 import (
        LAYERED_ENTITY_RULES as _RULES, P1_FOCUS as _P1, P2_FOCUS as _P2)
    _TWO_PASS = True
except Exception:                                    # standalone fallback
    _RULES, _P1, _P2, _TWO_PASS = _FALLBACK_RULES, "", "", False


def _gate_prompt(batch: Sequence[tuple], focus: str) -> str:
    lines = []
    for i, c in batch:
        lines.append(f'Case {i}: "{c.quote or c.component}" -> {c.component}')
        if c.prev:
            lines.append(f'  [prev: "{c.prev}"]')
        lines.append(f'  SENTENCE: "{c.sentence}"')
        if c.is_ambiguous:
            lines.append("  note: this component name is AMBIGUOUS (often an ordinary word)")
        for a in list(c.anchors)[:4]:
            lines.append(f"  anchor: {a}")
    return (f"Validate components in a document. {focus}\n\n{_RULES}\n\n"
            "For each case, first quote the EXACT words stating the architectural claim "
            'about the component (or "none"), then decide approve true/false based only on '
            "that quote.\n\nCASES:\n" + "\n".join(lines) +
            '\n\nReturn JSON: {"validations":[{"case":1,"claim":"<quote or none>",'
            '"approve":true}]}\nJSON only:')


def _parse_validations(txt: str) -> dict:
    a, b = txt.find("{"), txt.rfind("}")
    if a < 0 or b < 0:
        return {}
    try:
        obj = json.loads(txt[a:b + 1])
    except Exception:
        return {}
    out = {}
    for v in obj.get("validations", []):
        try:
            val = v.get("approve", v.get("keep"))
            out[int(v["case"])] = (val is True) or (isinstance(val, str)
                                                    and val.strip().lower() == "true")
        except Exception:
            pass
    return out


class StrictGate:
    """s21's unchanged entity validator (two-pass P1∧P2, claim-before-verdict).

    Callable: gate(candidates) -> {candidate_id: keep_bool}. Parse failure => reject
    (conservative — the router only ever *drops* what the gate cannot confirm)."""

    def __init__(self, client=None, model: str | None = None, batch: int = 8, timeout: int = 180):
        self.client, self.model, self.batch, self.timeout = client, model, batch, timeout

    def _client(self):
        if self.client is None:
            self.client = _make_client(self.model)
        return self.client

    def _pass(self, cands, focus) -> dict:
        out = {}
        for k in range(0, len(cands), self.batch):
            batch = list(enumerate(cands[k:k + self.batch], start=1))
            resp = self._client().query(_gate_prompt(batch, focus), timeout=self.timeout)
            verd = _parse_validations(resp.text if resp.success else "")
            for i, c in batch:
                out[c.id] = bool(verd.get(i, False))
        return out

    def __call__(self, cands: Sequence[Candidate]) -> dict:
        cands = list(cands)
        if not cands:
            return {}
        p1 = self._pass(cands, _P1 or "Judge each candidate.")
        if not _TWO_PASS:
            return p1
        p2 = self._pass(cands, _P2 or "Judge each candidate again, independently.")
        return {c.id: bool(p1.get(c.id) and p2.get(c.id)) for c in cands}


# ── the router ───────────────────────────────────────────────────────────────

_AGENT_PROMPT = (
    "You are the routing step of a documentation-to-architecture trace-link recovery "
    "system. A candidate names a software COMPONENT possibly referenced by a SENTENCE. "
    "For EACH case: (1) write a one-line NOTE — how the component is referenced; use the "
    "anchors ONLY to pin the referent, never to justify. (2) choose ACTION:\n"
    "  VALIDATE - a plausible architecture-level reference; send it to the validator "
    "(the validator makes the final keep/reject — when unsure, prefer VALIDATE).\n"
    "  CODE     - the words name a concrete code element (class/package/file); it belongs "
    "to the code-level linker, not the architecture set.\n"
    "  REJECT   - the component is clearly not the referent (incidental word, other entity).\n"
    "Default to VALIDATE unless CODE or REJECT is clear.\n\nCASES:\n{body}\n\n"
    'Return JSON: {{"decisions":[{{"case":1,"note":"...","action":"VALIDATE|CODE|REJECT"}}]}}'
    "\nJSON only:")


def _agent_case(c: Candidate, i: int) -> str:
    ls = [f'Case {i}: component "{c.component}" referenced by "{c.quote or c.component}"',
          f'  SENTENCE: {c.sentence}']
    if c.prev:
        ls.append(f'  PREV: {c.prev}')
    if c.anchors:
        ls.append("  anchors: " + " | ".join(list(c.anchors)[:3]))
    return "\n".join(ls)


def _parse_actions(txt: str) -> dict:
    a, b = txt.find("{"), txt.rfind("}")
    if a < 0 or b < 0:
        return {}
    try:
        obj = json.loads(txt[a:b + 1])
    except Exception:
        return {}
    out = {}
    for d in obj.get("decisions", []):
        try:
            act = str(d.get("action", VALIDATE)).upper().strip()
            out[int(d["case"])] = (act if act in _ACTIONS else VALIDATE,
                                   str(d.get("note", "")).strip())
        except Exception:
            pass
    return out


class DocModelAgenticRouter:
    """DOC->MODEL agentic router with a gate-floored accept invariant. (Not the
    DOC->CODE sentence triage — see ``router_direct.DocCodeSentenceRouter``.)

    Parameters
    ----------
    client : LLM client for the agent step (reasoning-off). Auto-created if None.
    gate   : callable(candidates)->{id: keep_bool}. Defaults to s21's StrictGate.
             Inject a custom validator (e.g. a contrast gate, or the live s21 pass) to
             change the FLOOR without touching the router.
    """

    def __init__(self, client=None, gate: Optional[Callable] = None,
                 model: str | None = None, batch: int = 8, timeout: int = 180):
        self.client = client
        self.model, self.batch, self.timeout = model, batch, timeout
        self.gate = gate if gate is not None else StrictGate(model=model)

    def _client(self):
        if self.client is None:
            self.client = _make_client(self.model)
        return self.client

    def _run_agent(self, cands: Sequence[Candidate]) -> dict:
        """id -> (action, note). Safe default on any miss = VALIDATE (defer to gate)."""
        out = {}
        for k in range(0, len(cands), self.batch):
            batch = list(enumerate(cands[k:k + self.batch], start=1))
            body = "\n".join(_agent_case(c, i) for i, c in batch)
            resp = self._client().query(_AGENT_PROMPT.format(body=body), timeout=self.timeout)
            parsed = _parse_actions(resp.text if resp.success else "")
            for i, c in batch:
                out[c.id] = parsed.get(i, (VALIDATE, ""))
        return out

    def route(self, candidates: Sequence[Candidate]) -> list:
        """Decide + gate-floor every candidate. Returns a list of Decision.

        INVARIANT enforced here: a Decision is `accepted` iff the agent chose VALIDATE
        AND the trusted gate approved it. CODE/REJECT are diverted (never accepted)."""
        candidates = list(candidates)
        if not candidates:
            return []
        acts = self._run_agent(candidates)

        # only VALIDATE candidates reach the (expensive) gate — the floor
        to_gate = [c for c in candidates if acts[c.id][0] == VALIDATE]
        gate_res = self.gate(to_gate) if to_gate else {}

        decisions = []
        for c in candidates:
            action, note = acts[c.id]
            if action == VALIDATE:
                passed = bool(gate_res.get(c.id, False))
                decisions.append(Decision(c, VALIDATE, note, passed, accepted=passed))
            else:
                decisions.append(Decision(c, action, note, None, accepted=False))
        return decisions

    # ── convenience partitions ───────────────────────────────────────────────
    @staticmethod
    def accepted(decisions):
        return [d.candidate for d in decisions if d.accepted]

    @staticmethod
    def routed_to_code(decisions):
        return [d.candidate for d in decisions if d.action == CODE]

    @staticmethod
    def rejected(decisions):
        return [d.candidate for d in decisions
                if d.action == REJECT or (d.action == VALIDATE and not d.accepted)]


# ── standalone self-test (live) ──────────────────────────────────────────────

def _demo():
    """Three hand-built candidates exercise the three actions + the gate floor."""
    cands = [
        Candidate(id="arch", component="Storage",
                  sentence="The Storage component persists all user records to disk.",
                  quote="Storage"),
        Candidate(id="code", component="Logic",
                  sentence="First, the request is forwarded to the WebApiServlet.",
                  quote="WebApiServlet",
                  anchors=("S7: The Logic component implements the business rules.",)),
        Candidate(id="incidental", component="Auth",
                  sentence="If the user is not logged in, a fallback ranking is used.",
                  quote="logged in"),
    ]
    router = DocModelAgenticRouter()
    print("DocModelAgenticRouter — live self-test (gpt-5.4, reasoning-off)\n")
    decisions = router.route(cands)
    for d in decisions:
        print(f"  {d.candidate.id:<11} action={d.action:<9} "
              f"gate={d.gate_passed}  accepted={d.accepted}")
        print(f"      note: {d.note}")
    invariant_ok = all((d.action == VALIDATE and d.gate_passed) for d in decisions if d.accepted)
    print(f"\n  INVARIANT (accepted => VALIDATE and gate_passed): {invariant_ok}")
    print(f"  accepted={[c.id for c in router.accepted(decisions)]}  "
          f"to_code={[c.id for c in router.routed_to_code(decisions)]}  "
          f"rejected={[c.id for c in router.rejected(decisions)]}")


if __name__ == "__main__":
    _demo()
