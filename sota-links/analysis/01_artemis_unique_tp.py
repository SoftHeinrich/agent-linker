#!/usr/bin/env python3
"""01 — Where does Artemis recover gold links that we (aalinker) do not, and what
are those links made of (lexical name mention vs. semantic match)?

Task: model-doc / SAD-SAM.  "Ours" = aalinker gpt-5.4_s21, union of the 3 runs
(a link counts as ours if recovered in ANY run — the set most favourable to us,
giving the smallest Artemis-unique set).  Run with --intersection for the
adversarial view (a link is ours only if recovered in ALL runs).

Usage:  python3 analysis/01_artemis_unique_tp.py [--variant gpt-5.4_s21] [--intersection]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C


def ours_set(proj, variant, mode):
    base = os.path.join(C.RL, "model-doc", "aalinker", variant)
    runs = [C.links(os.path.join(base, d, f"{proj}.csv"))
            for d in sorted(os.listdir(base)) if d.startswith("run")]
    if not runs:
        return set()
    if mode == "union":
        return set().union(*runs)
    inter = set(runs[0])
    for r in runs[1:]:
        inter &= r
    return inter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="gpt-5.4_s21")
    ap.add_argument("--intersection", action="store_true")
    args = ap.parse_args()
    mode = "intersection" if args.intersection else "union"

    print(f"# Artemis-unique TP study  (ours = aalinker {args.variant}, {mode} of runs)\n")
    print(f"{'project':14}{'gold':>5}{'artTP':>6}{'ourTP':>6}{'artUNIQ':>8}{'ourUNIQ':>8}")
    tot = [0, 0, 0, 0, 0]
    art_unique_rows, our_unique_rows = [], []
    for proj in C.PROJECTS:
        names, sents = C.id_to_name(proj), C.sentences(proj)
        g, art = C.gold(proj), C.artemis(proj)
        ours = ours_set(proj, args.variant, mode)
        art_tp, our_tp = art & g, ours & g
        au, ou = art_tp - our_tp, our_tp - art_tp
        for sid, tid in sorted(au):
            art_unique_rows.append((proj, sid, names.get(tid, "?"), *C.lexical_class(names.get(tid, "?"), sents.get(sid, "")), sents.get(sid, "")))
        for sid, tid in sorted(ou):
            our_unique_rows.append((proj, sid, names.get(tid, "?"), C.lexical_class(names.get(tid, "?"), sents.get(sid, ""))[0]))
        row = [len(g), len(art_tp), len(our_tp), len(au), len(ou)]
        tot = [a + b for a, b in zip(tot, row)]
        print(f"{proj:14}{row[0]:5}{row[1]:6}{row[2]:6}{row[3]:8}{row[4]:8}")
    print(f"{'TOTAL':14}{tot[0]:5}{tot[1]:6}{tot[2]:6}{tot[3]:8}{tot[4]:8}")

    def profile(rows, idx):
        from collections import Counter
        c = Counter(r[idx] for r in rows)
        n = sum(c.values()) or 1
        return "  ".join(f"{k}={c.get(k,0)} ({100*c.get(k,0)//n}%)"
                         for k in ["LEXICAL-full", "LEXICAL-partial", "SEMANTIC-none"])

    print(f"\n## Artemis-unique TP lexical profile (N={len(art_unique_rows)}):  {profile(art_unique_rows, 3)}")
    print(f"## Ours-unique  TP lexical profile (N={len(our_unique_rows)}):  {profile(our_unique_rows, 3)}")

    print("\n## Every Artemis-unique TP:")
    for proj, sid, name, label, ev, sent in art_unique_rows:
        print(f"  {proj:13} s{sid:<4} {name[:16]:17}{label:16}{ev:22} | {sent[:74]}")


if __name__ == "__main__":
    main()
