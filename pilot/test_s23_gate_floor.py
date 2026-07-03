"""Deterministic bounded-autonomy check for the s23 LLM-driven router.

s23 makes routing/validation an LLM decision, so there is no structural logic to
unit-test. What must hold regardless of what the model decides is the *invariant*
that keeps the augmentation safe: an accepted link is exactly one the LLM sent to
VALIDATE AND s21's own gate approved. The LLM can DIVERT (CODE/REJECT) or defer to
the gate, but it can never add a link the gate rejects — so s23 can never regress
below the s21 floor. This test stubs the agent step and the gate (no network) and
asserts the invariant across every combination of agent action x gate verdict.

Run: `python pilot/test_s23_gate_floor.py` (no API key needed).
"""
from llm_sad_sam.linkers.experimental.agentic_router import (
    DocModelAgenticRouter, Candidate, VALIDATE, CODE, REJECT,
)


def make_case(action, gate_ok, i):
    """One candidate tagged with the agent action it will receive and whether the
    gate would approve it, encoded in the id so the stubs stay pure."""
    return Candidate(id=f"{i}|{action}|{int(gate_ok)}", component=f"C{i}",
                     sentence=f"sentence {i}", quote=f"q{i}")


def main():
    cases = []
    i = 0
    for action in (VALIDATE, CODE, REJECT):
        for gate_ok in (True, False):
            i += 1
            cases.append(make_case(action, gate_ok, i))

    router = DocModelAgenticRouter(
        client=object(),                    # never used: _run_agent is stubbed below
        gate=lambda cands: {c.id: c.id.endswith("|1") for c in cands},
    )
    # stub the LLM agent: action is read straight from the candidate id
    router._run_agent = lambda cands: {c.id: (c.id.split("|")[1], "stub") for c in cands}

    decisions = router.route(cases)
    by_id = {d.candidate.id: d for d in decisions}

    for c in cases:
        _i, action, gate_flag = c.id.split("|")
        d = by_id[c.id]
        gate_ok = gate_flag == "1"
        expect_accept = (action == VALIDATE and gate_ok)
        assert d.accepted == expect_accept, f"{c.id}: accepted={d.accepted} expected {expect_accept}"
        if action != VALIDATE:
            assert not d.accepted, f"{c.id}: diverted action {action} was accepted!"

    accepted = [d.candidate.id for d in decisions if d.accepted]
    assert accepted == ["1|VALIDATE|1"], f"unexpected accepted set: {accepted}"
    # the invariant, stated directly:
    assert all((d.action == VALIDATE and d.gate_passed) for d in decisions if d.accepted)

    print("accepted:", accepted)
    print("PASS: accept <=> (LLM VALIDATE AND s21 gate approved); CODE/REJECT never accepted.")
    print("  => the LLM-driven augmentation is gate-floored and cannot regress below s21.")


if __name__ == "__main__":
    main()
