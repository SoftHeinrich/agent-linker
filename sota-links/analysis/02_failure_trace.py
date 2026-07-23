#!/usr/bin/env python3
"""02 — For each gold link Artemis recovers but we miss, where in OUR pipeline
does it die?  Traces the phase cache (extraction -> two-pass entity gate ->
coreference) across the 3 gpt runs.

Reads phase-cache-s21/v2.6.6_s21_gpt/run{1,2,3}/.../layer3.pkl (entity gate
decisions).  Needs the agent-linker package importable to unpickle CandidateLink.

Usage:  python3 analysis/02_failure_trace.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

pickle = C.load_phase_cache()
CACHE = os.path.join(C.RL, "phase-cache-s21", "v2.6.6_s21_gpt")

# the 9 Artemis-unique gold links (intersection view); first 5 are never-recovered
CASES = [
    ("mediastore", 33, "FileStorage"), ("teastore", 7, "WebUI"),
    ("teammates", 88, "Logic"), ("bigbluebutton", 59, "FreeSWITCH"),
    ("bigbluebutton", 66, "FreeSWITCH"),
    ("mediastore", 27, "MediaAccess"), ("teammates", 87, "Logic"),
    ("teammates", 101, "Storage"), ("teammates", 186, "E2E"),
]


def trace(proj, sid, tid, run):
    base = os.path.join(CACHE, f"run{run}", "phase_cache", "s_linker21", "openai", proj)
    l3 = pickle.load(open(os.path.join(base, "layer3.pkl"), "rb"))
    cand = [c for c in l3["candidates"] if c.sentence_number == sid and c.component_id == tid]
    dec = l3["decisions"].get((sid, tid))
    eb = l3["evidence_bundles"].get((sid, tid))
    span = eb.get("matched_span") if eb else (cand[0].matched_text if cand else None)
    validated_here = [c.component_name for c in l3["validated"] if c.sentence_number == sid]
    if cand and dec and dec["approved"]:
        status = "APPROVED -> link"
    elif cand and dec:
        status = f"REJECTED by entity gate (p1={dec['p1']}, p2={dec['p2']})"
    elif cand:
        status = "candidate but no decision"
    else:
        status = "NOT EXTRACTED (never a candidate)"
    return status, span, validated_here


def main():
    print("# Failure trace: where each Artemis-unique gold link dies in our pipeline\n")
    for proj, sid, tname in CASES:
        tid = C.name_to_id(proj).get(tname)
        print(f"-- {proj} s{sid} -> {tname} --")
        for run in (1, 2, 3):
            try:
                status, span, val = trace(proj, sid, tid, run)
                print(f"   run{run}: {status}  [span={span!r}; we linked s{sid} to {val}]")
            except Exception as e:  # noqa: BLE001
                print(f"   run{run}: ERROR {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    main()
