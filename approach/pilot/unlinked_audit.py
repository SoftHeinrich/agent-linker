"""`_unlinked` has not been subtracting anything. Deterministic proof, no LLM calls.

    return [c for c in candidates
            if (c.sentence_number, c.component_id) not in linked]

`linked` is the accumulating list of `SadSamLink` that `link()` passes to every linker
(`current`, built by `_union`). The membership test compares a **tuple** against a list
of **dataclass instances**, so it is `False` for every candidate and the comprehension
is the identity. `SadSamLink.__eq__` is the dataclass default; it returns
`NotImplemented` against a tuple, so Python falls back to identity and the comparison
can never succeed.

This file proves it three ways and prices the repair:

    U1  the type test, in isolation: is a (sentence, component) tuple ever `in` a list
        of `SadSamLink`?
    U2  the evidence in the recorded runs: the checkpoint's candidate view is rendered
        *after* the `_unlinked` call, so any already-linked pair appearing in it is a
        pair the subtraction did not remove
    U3  what a working subtraction would remove, per stage and per run, and how much of
        it is gold -- the size of the repair

Why it matters beyond the line itself: this branch's standing explanation for nine
stage-vs-pipeline reversals is that an early stage's admission is "locked into the union
*and stolen from the later, stricter linkers*". The first half is `_union` and is real.
The second half is this predicate, and at the coreference stage it has not been
happening. `pilot/composition_check.py` computes its risk from the same assumption.

    ../.venv/bin/python pilot/unlinked_audit.py
    ../.venv/bin/python pilot/unlinked_audit.py --runs '../results/s6667_e2e_r*_20260817' --arm s_linker65
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import PROJECTS, load_gold, load_project             # noqa: E402
from llm_sad_sam.core.data_types_v2 import SadSamLink                  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker66 import SLinker66      # noqa: E402

#: The candidate view each linker checkpoints, and the stages whose links precede it.
STAGES = [
    ("linker_full_name", "candidates", []),
    ("linker_partial_name", "proposed", ["linker_full_name"]),
    ("linker_coreference", "candidates", ["linker_full_name", "linker_partial_name"]),
]


def u1():
    print("=== U1  the predicate, in isolation ===")
    link = SadSamLink(1, "some-id", "Name")
    print(f"    (1, 'some-id') in [SadSamLink(1, 'some-id', 'Name')]  ->  "
          f"{(1, 'some-id') in [link]}")
    print(f"    SadSamLink(...) == (1, 'some-id')                     ->  "
          f"{link == (1, 'some-id')}")
    kept = SLinker66._unlinked([link], [link])
    print(f"    _unlinked([x], [x]) keeps                             ->  "
          f"{len(kept)} of 1 candidate")
    print("\n    so the comprehension is the identity for every input the linkers "
          "pass it\n")


def keys(view, name_to_id):
    out = set()
    for row in view:
        cid = name_to_id.get(row.get("component"))
        if cid is not None:
            out.add((int(row["sentence"]), cid))
    return out


def u2_u3(runs, arm):
    print("=== U2/U3  the recorded runs, and the size of the repair ===")
    agg = Counter()
    per_stage = Counter()
    for run in runs:
        for name in PROJECTS:
            base = run / "phase_states" / arm / "openai" / name
            if not (base / "linker_coreference.pkl").exists():
                continue
            info = load_project(name)
            gold = set(load_gold(name))
            states = {}
            for phase, _, _ in STAGES:
                with (base / f"{phase}.pkl").open("rb") as handle:
                    states[phase] = pickle.load(handle)
            linked_by_stage = {
                phase: {(l.sentence_number, l.component_id)
                        for l in states[phase]["links"]}
                for phase, _, _ in STAGES
            }
            for phase, view, earlier in STAGES:
                cand = keys(states[phase]["feedback"][view], info["name_to_id"])
                upstream = set().union(*[linked_by_stage[e] for e in earlier]) \
                    if earlier else set()
                stale = cand & upstream
                per_stage[(phase, "candidates")] += len(cand)
                per_stage[(phase, "already linked upstream")] += len(stale)
                per_stage[(phase, "of those, gold")] += len(stale & gold)
                per_stage[(phase, "judged")] += len(
                    states[phase]["feedback"].get("judge_decisions", []))
            agg["units"] += 1
    n = agg["units"] / len(PROJECTS) if agg["units"] else 1
    print(f"\n  per five-project run, over {int(agg['units'] // len(PROJECTS))} runs, "
          f"arm {arm}\n")
    print(f"{'stage (post-_unlinked view)':<28}{'candidates':>12}{'stale':>8}"
          f"{'gold':>7}{'judged':>9}")
    for phase, _, earlier in STAGES:
        print(f"{phase:<28}{per_stage[(phase, 'candidates')] / n:>12.1f}"
              f"{per_stage[(phase, 'already linked upstream')] / n:>8.1f}"
              f"{per_stage[(phase, 'of those, gold')] / n:>7.1f}"
              f"{per_stage[(phase, 'judged')] / n:>9.1f}"
              + ("   (no earlier stage — must be 0)" if not earlier else ""))
    stale = sum(per_stage[(p, "already linked upstream")] for p, _, e in STAGES if e)
    print(f"\n    U2: {stale / n:.1f} stale pairs per run survive a call the code "
          f"makes to remove them.")
    print(f"    U3: repairing the predicate removes exactly those {stale / n:.1f} "
          f"cases per run from the judging batches;")
    print("        it cannot change the final link set (the union already holds "
          "them), only\n        which cases share a batch and how many judge calls "
          "are paid for.\n")
    print("    Note where the stale pairs are NOT: the partial-name stage reads 0.0, "
          "because\n    `skip_when_named` already excludes every sentence that states "
          "a whole name — the\n    two gates overlap, and only one of them is "
          "working.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="../results/s6667_e2e_r*_20260817")
    ap.add_argument("--arm", default="s_linker65")
    args = ap.parse_args()
    runs = sorted(Path().glob(args.runs))
    if not runs:
        raise SystemExit(f"no runs matched {args.runs}")
    print(f"\n_unlinked audit — {args.arm} over {len(runs)} runs\n")
    u1()
    u2_u3(runs, args.arm)


if __name__ == "__main__":
    main()
