"""Deterministic end-to-end check of SLinker23's LLM-driven augmentation glue,
with no network. Stubs only the two LLM steps (proposer + router agent) and the
s21 gate; exercises the REAL DocModelAgenticRouter routing/accept invariant and
the real `_augment` wiring (Candidate build, id round-trip, dedup vs the floor,
SadSamLink creation, floor preservation).

Run: `python pilot/test_s23_augment.py` (no API key needed).
"""
from types import SimpleNamespace as NS

from llm_sad_sam.core.data_types_v2 import SadSamLink
import llm_sad_sam.linkers.experimental.s_linker23 as s23mod
from llm_sad_sam.linkers.experimental.s_linker23 import SLinker23
from llm_sad_sam.linkers.experimental.agentic_router import (
    DocModelAgenticRouter, VALIDATE, CODE, REJECT,
)

COMPONENTS = [NS(name="Ayes", id="idA"), NS(name="Bno", id="idB"),
              NS(name="Ccode", id="idC"), NS(name="Dreject", id="idD"),
              NS(name="Edup", id="idE")]

# one proposal per sentence, each exercising a distinct outcome
PROPOSALS = {
    1: [{"component": "Ayes", "mode": "AFFIRMATIVE", "quote": "Ayes"}],   # VALIDATE + gate ok -> ADDED
    2: [{"component": "Bno", "mode": "AFFIRMATIVE", "quote": "Bno"}],     # VALIDATE + gate reject -> dropped
    3: [{"component": "Ccode", "mode": "CODEPATH", "quote": "a.b.Ccode"}],# CODE -> diverted
    4: [{"component": "Dreject", "mode": "IMPLICIT", "quote": "thing"}],  # REJECT -> dropped
    5: [{"component": "Edup", "mode": "AFFIRMATIVE", "quote": "Edup"}],   # VALIDATE + gate ok but already in floor -> deduped
    6: [{"component": "Zzz", "mode": "AFFIRMATIVE", "quote": "Zzz"}],     # ungrounded (not in catalog) -> skipped
}


class StubProposer:
    def __init__(self, *a, **k):
        pass

    def propose_batch(self, sentences, names, roles=None, batch_size=40,
                      strategy="blocks", prev_of=None, key_prefix=""):
        # flatten the per-sentence PROPOSALS into the batched return shape
        out = []
        for s in sentences:
            for r in PROPOSALS.get(s.number, []):
                out.append({"sentence": s.number, "component": r["component"],
                            "quote": r.get("quote", "")})
        return out


def stub_agent(cands):
    """Action per candidate keyed on component name prefix."""
    out = {}
    for c in cands:
        if c.component.startswith(("Ayes", "Bno", "Edup")):
            out[c.id] = (VALIDATE, "stub")
        elif c.component.startswith("Ccode"):
            out[c.id] = (CODE, "stub")
        else:
            out[c.id] = (REJECT, "stub")
    return out


def stub_two_pass(comp_names, cases, focus, phase_tag=None):
    """s21 gate stand-in: approve targets Ayes/Edup, reject Bno."""
    out = {}
    for i, case in enumerate(cases):
        target = case.split("-> ", 1)[1].splitlines()[0].strip() if "-> " in case else ""
        out[i] = target.startswith(("Ayes", "Edup"))
    return out


def main():
    # patch the two LLM seams + loaders; keep the real router + accept invariant
    s23mod.parse_pcm_repository = lambda p: COMPONENTS
    s23mod.load_sentences = lambda p: [NS(number=n, text=f"sentence {n}") for n in range(1, 7)]
    s23mod.build_sent_map = lambda ss: {s.number: s for s in ss}
    s23mod.GroundedTypedProposer = StubProposer
    DocModelAgenticRouter._run_agent = staticmethod(stub_agent)

    obj = object.__new__(SLinker23)
    obj._run_validation_pass = stub_two_pass       # the gate defers to this

    floor = [SadSamLink(5, "idE", "Edup"), SadSamLink(9, "idX", "Other")]
    result = obj._augment(list(floor), "text", "model")

    # floor preserved
    for l in floor:
        assert any(r.sentence_number == l.sentence_number and r.component_id == l.component_id
                   for r in result), f"floor link dropped: {l}"

    added = [(r.sentence_number, r.component_id, r.source) for r in result
             if getattr(r, "source", None) == "llmrouter"]
    assert added == [(1, "idA", "llmrouter")], f"unexpected additions: {added}"

    # CODE candidate diverted, not accepted
    code_routed = [c.component for c in obj.code_routed_candidates]
    assert code_routed == ["Ccode"], f"code_routed: {code_routed}"

    print("floor:", [(l.sentence_number, l.component_id) for l in floor])
    print("additions:", added)
    print("code-routed:", code_routed)
    print("PASS: only VALIDATE+gate-approved, not-already-in-floor candidates added;")
    print("  gate-rejected VALIDATE, CODE, REJECT, and ungrounded proposals all excluded;")
    print("  floor links preserved -> no regression below s21.")


if __name__ == "__main__":
    main()
