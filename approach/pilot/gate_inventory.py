"""What deterministic gates are left, what each one blocks, and which can fold.

No LLM calls. Deterministic pricing of every remaining option in ``SCANS``, plus the
one structural fact that decides whether a gate *can* become a prompt clause.

**The fold law this round established.** A gate folds into a judge's prompt exactly
when that judge is shown the information the gate reads. Three folds and one failure
all obey it:

    skip_qualified   reads the span's shape in the sentence   judge sees the sentence   folded
    skip_stricter    reads the surface against the target     judge sees the target     folded
    unique_owner     reads the whole component catalog        denotation judge sees
                     (name_word row)                          neither target nor
                                                              catalog                   failed
    unique_owner     reads the whole component catalog        full-name judge is given
                     (spelling row)                           `COMPONENTS: ...`         UNTESTED

So the failure of ``foldowner`` was never about the rule being un-statable in English.
It is about *which judge the row reports to*: the partial-name row's gate reads the
catalog and its judge is target-blind by design, while the spelling row's identical
gate reports to a judge that is handed the catalog in its first line. Same predicate,
two rows, two different verdicts predicted by one structural fact.

    G1  the gates that remain, per row, with the judge each row reports to
    G2  what each gate blocks -- pairs and gold -- in the s69 configuration
    G3  the same in the s70 configuration (skip_stricter folded, so the spelling row
        now reaches the ANY_CASE cell and every later gate sees a different mix)
    G4  the information each gate reads against what its judge is shown

    ../.venv/bin/python pilot/gate_inventory.py
"""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))

from design_audit import PROJECTS, load_gold, load_project            # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker69 import (             # noqa: E402
    SCANS, SLinker69, NameForm,
)


class Probe(SLinker69):
    """`SLinker69`'s deterministic layer with no LLM client and an empty alias table."""

    def __init__(self):                                               # noqa: D107
        self.doc_knowledge = type("K", (), {"aliases": {}})()


#: Which judge each row's candidates are handed to, and what that judge's prompt
#: carries. Read off the prompt builders, not asserted: `_prompt_validation` opens
#: with `COMPONENTS: {', '.join(comp_names)}` and names the target in every case
#: line; `_classify_denotations` builds its own prompt with neither.
JUDGE = {
    "stated_name": ("full-name judge (_prompt_validation)", "target + catalog"),
    "spelling": ("full-name judge (_prompt_validation)", "target + catalog"),
    "name_word": ("denotation judge (_classify_denotations)", "neither"),
}

#: What each gate reads, so it can be checked against what its judge is shown.
READS = {
    "unique_owner": "the whole component catalog",
    "skip_when_named": "the target's whole name, in this sentence",
    "skip_stricter": "the surface against the target's name",
    "label_mention": "the surface's case and span shape, in this sentence",
}

GATES = ["unique_owner", "skip_when_named", "skip_stricter"]


def scan_pairs(probe, info, scan):
    return {(c.sentence_number, c.component_id)
            for c in probe._scan(info["sentences"], info["components"], scan)}


def price(probe, table, label):
    """Per gate, per row: the pairs turning it off would add, and how many are gold."""
    freed = Counter()
    gold_freed = Counter()
    base = Counter()
    for name in PROJECTS:
        info = load_project(name)
        info["gold"] = set(load_gold(name))
        for row, scan in table.items():
            on = scan_pairs(probe, info, scan)
            base[row] += len(on)
            for gate in GATES:
                if not getattr(scan, gate):
                    continue
                off = scan_pairs(probe, info, replace(scan, **{gate: False}))
                extra = off - on
                freed[(row, gate)] += len(extra)
                gold_freed[(row, gate)] += len(extra & info["gold"])
    print(f"\n=== {label} ===\n")
    print(f"{'row':<13}{'gate':<18}{'row size':>10}{'freed':>8}{'gold':>7}"
          f"{'gold rate':>11}")
    for row, scan in table.items():
        for gate in GATES:
            if not getattr(scan, gate):
                continue
            f = freed[(row, gate)]
            g = gold_freed[(row, gate)]
            print(f"{row:<13}{gate:<18}{base[row]:>10}{f:>8}{g:>7}"
                  f"{(g / f if f else 0):>11.3f}")
    return freed, gold_freed


def g1():
    print("=== G1  the gates that remain in s69, and the judge each row reports to "
          "===\n")
    print(f"{'row':<13}{'form':<14}{'gates':<40}{'judge is shown':<16}")
    for row, scan in SCANS.items():
        gates = [g for g in GATES + ["label_mention"] if getattr(scan, g)]
        judge, shown = JUDGE[row]
        print(f"{row:<13}{scan.form.value:<14}{', '.join(gates) or '(none)':<40}"
              f"{shown:<16}")
    print("\n    plus, outside the table: `_states_a_name` (used by skip_when_named),")
    print("    `_classify_mention_typed` (the five-value label), and the relation")
    print("    itself at four settings. No predicate here admits a link.\n")


def g4():
    print("\n=== G4  what each gate reads against what its judge is shown ===\n")
    print(f"{'row':<13}{'gate':<18}{'reads':<38}{'judge sees it?':<15}")
    for row, scan in SCANS.items():
        _, shown = JUDGE[row]
        for gate in GATES + ["label_mention"]:
            if not getattr(scan, gate):
                continue
            reads = READS[gate]
            if gate == "unique_owner":
                ok = "catalog" in shown
            elif gate in ("skip_when_named", "skip_stricter"):
                ok = "target" in shown
            else:
                ok = True                      # the sentence is in every prompt
            print(f"{row:<13}{gate:<18}{reads:<38}"
                  f"{'yes -- foldable' if ok else 'NO -- cannot fold':<15}")
    print("\n    `skip_when_named` and `unique_owner` on the partial-name row are")
    print("    blocked by the same single design fact -- the denotation judge is")
    print("    target-blind, and s25's grounded review is what showed why it must")
    print("    stay that way (it traded 5.5 gold for 2.5 spurious). They are not")
    print("    hand-engineering that survived scrutiny; they are the price of a")
    print("    judging design that was measured to be worth its price.\n")


def main():
    probe = Probe()
    g1()
    price(probe, SCANS, "G2  s69 configuration")
    s70 = dict(SCANS)
    s70["spelling"] = replace(SCANS["spelling"], skip_stricter=False)
    price(probe, s70, "G3  s70 configuration (skip_stricter folded into the prompt)")
    g4()


if __name__ == "__main__":
    main()
