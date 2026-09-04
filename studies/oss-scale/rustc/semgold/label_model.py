"""Combine the labelling sources into tiered semantic gold and report agreement.

Sources per (sentence, crate):
  A1, A2   ABOUT / REFERS from two grounded annotators of different model families
  SYM      a symbol named in the sentence is defined in the crate (deterministic)
  ANCHOR   the old syntactic gold (hyperlink / verbatim) - now just one vote
  COCHANGE focused commit edited the sentence and the crate
Tiers:
  gold      ABOUT by both annotators
  gold_plus gold  U  (ABOUT by one annotator AND any deterministic vote)
  silver    ABOUT by exactly one annotator, unsupported
  refers    REFERS by either annotator and not ABOUT in gold_plus (excluded from FP in lenient scoring)
"""
from __future__ import annotations

import collections
import csv
import json
import re
import sys
from pathlib import Path

from common import OUT, load_anchor_gold, load_components, load_sentences, write_csv


def load_annotations() -> dict[str, dict]:
    """Primary annotators: one per family, run 1 (no _rN suffix). Repeat runs load separately."""
    out = {}
    for p in sorted(OUT.glob("annotations_*.json")):
        d = json.loads(p.read_text())
        name = p.stem.removeprefix("annotations_")
        if "crateview" in p.name or re.search(r"_r\d+$", name):
            continue
        out[name] = d["labels"]
    return out


def load_repeats(primary: str) -> list[dict]:
    reps = []
    for p in sorted(OUT.glob(f"annotations_{primary}_r*.json")):
        reps.append(json.loads(p.read_text())["labels"])
    return reps


def kappa(a: set, b: set, universe: set) -> float:
    n = len(universe)
    if n == 0:
        return float("nan")
    both = len(a & b)
    only_a = len(a - b)
    only_b = len(b - a)
    neither = n - both - only_a - only_b
    po = (both + neither) / n
    pa = (both + only_a) / n
    pb = (both + only_b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main() -> None:
    rows = load_sentences()
    numbers = [r["number"] for r in rows]
    ann = load_annotations()
    names = list(ann)
    assert len(names) >= 2, f"need two annotators, have {names}"
    # a1 = the linker's own family (openai), a2 = the other family; robustness gold = a2 alone
    a1 = next(n for n in names if n.startswith("openai"))
    a2 = next(n for n in names if n != a1)
    about = {k: {(int(n), c) for n, v in lab.items() for c in v["about"]} for k, lab in ann.items()}
    refers = {k: {(int(n), c) for n, v in lab.items() for c in v["refers"]} for k, lab in ann.items()}
    evidence = {e["number"]: e for e in json.loads((OUT / "evidence.json").read_text())}
    sym = {(n, c) for n, e in evidence.items() for cs in e["symbol_crates"].values() for c in cs}
    anchor = load_anchor_gold()
    cochange = set()
    cc = OUT / "cochange_pairs.csv"
    if cc.exists():
        with open(cc) as h:
            cochange = {(int(r["sentence"]), r["crate"]) for r in csv.DictReader(h)}
    det = sym | anchor | cochange

    # 3-run consistency of the first annotator (Pangakis-style): share of its ABOUT pairs reproduced in every repeat
    reps = load_repeats(a1)
    consistency = {}
    if reps:
        rep_about = [{(int(n), c) for n, v in lab.items() for c in v["about"]} for lab in reps]
        allp = about[a1].union(*rep_about)
        for p in allp:
            consistency[p] = (int(p in about[a1]) + sum(p in r for r in rep_about)) / (1 + len(reps))

    gold = about[a1] & about[a2]
    one = about[a1] ^ about[a2]
    gold_plus = gold | (one & det)
    silver = one - det
    refers_any = (refers[a1] | refers[a2]) - gold_plus

    # is the annotator just re-detecting anchors / code spans? ABOUT rate by sentence surface features
    evid = evidence
    def rate(pred):
        sel = [r for r in rows if pred(r)]
        return {"n": len(sel), a1: round(sum(1 for r in sel if r["number"] in s1_) / max(1, len(sel)), 3),
                a2: round(sum(1 for r in sel if r["number"] in s2_) / max(1, len(sel)), 3),
                "gold_plus": round(sum(1 for r in sel if r["number"] in gp_s) / max(1, len(sel)), 3)}
    s1_ = {n for n, _ in about[a1]}
    s2_ = {n for n, _ in about[a2]}
    gp_s = {n for n, _ in gold_plus}
    surface = {
        "anchored": rate(lambda r: bool(r["link"] or r["verbatim"])),
        "unanchored_with_code_span": rate(lambda r: not (r["link"] or r["verbatim"]) and "`" in r["text"]),
        "unanchored_plain_prose": rate(lambda r: not (r["link"] or r["verbatim"]) and "`" not in r["text"]),
    }

    # crate view (third vote, other direction): agreement with the sentence-view consensus
    crateview = {}
    for p in sorted(OUT.glob("annotations_crateview_*.json")):
        cv = json.loads(p.read_text())["labels"]
        crateview[p.stem.removeprefix("annotations_crateview_")] = {(int(n), c) for n, v in cv.items() for c in v["about"]}
    cv_report = {}
    for name, pairs in crateview.items():
        cv_report[name] = {
            "pairs": len(pairs), "sentences": len({n for n, _ in pairs}),
            "overlap_gold_plus": len(pairs & gold_plus), "gold_plus_recall": round(len(pairs & gold_plus) / max(1, len(gold_plus)), 3),
            "precision_vs_gold_plus": round(len(pairs & gold_plus) / max(1, len(pairs)), 3),
            "precision_vs_any_about": round(len(pairs & (about[a1] | about[a2])) / max(1, len(pairs)), 3),
            "new_pairs_no_sentence_view_vote": len(pairs - about[a1] - about[a2] - refers[a1] - refers[a2]),
            "kappa_vs_gold_plus_on_sentence_x_crate": round(kappa(pairs, gold_plus, {(n, c) for n in numbers for c in load_components()}), 3),
            "gold3_all_three_agree": len(pairs & gold),
        }

    # agreement
    universe = about[a1] | about[a2] | refers[a1] | refers[a2]
    # candidate universe: every crate either annotator mentioned for a sentence, over all sentences
    per_sentence_exact = sum(1 for n in numbers if {c for (m, c) in about[a1] if m == n} == {c for (m, c) in about[a2] if m == n})
    s1 = {n for n, _ in about[a1]}
    s2 = {n for n, _ in about[a2]}
    report = {
        "annotators": names,
        "sentences": len(numbers),
        "about_pairs": {a1: len(about[a1]), a2: len(about[a2])},
        "sentences_with_about": {a1: len(s1), a2: len(s2), "both": len(s1 & s2), "either": len(s1 | s2)},
        "refers_pairs": {a1: len(refers[a1]), a2: len(refers[a2])},
        "pair_agreement": {
            "jaccard_about": len(about[a1] & about[a2]) / max(1, len(about[a1] | about[a2])),
            "kappa_about_on_sentence_x_crate": kappa(about[a1], about[a2], {(n, c) for n in numbers for c in load_components()}),
            "sentence_exact_about_set_agreement": per_sentence_exact / len(numbers),
            "sentence_has_about_agreement": (len(s1 & s2) + len(set(numbers) - s1 - s2)) / len(numbers),
        },
        "tiers": {"gold": len(gold), "gold_plus": len(gold_plus), "silver": len(silver), "refers": len(refers_any)},
        "gold_plus_sentences": len({n for n, _ in gold_plus}),
        "gold_plus_crates": len({c for _, c in gold_plus}),
        "deterministic_support": {"sym_pairs": len(sym), "anchor_pairs": len(anchor), "cochange_pairs": len(cochange),
                                  "one_annotator_supported": len(one & det)},
        "top_gold_plus_crates": collections.Counter(c for _, c in gold_plus).most_common(20),
        "about_rate_by_surface": surface,
        "consistency_runs": 1 + len(reps),
        "consistency_hist_over_a1_union": collections.Counter(round(v, 2) for v in consistency.values()).most_common() if consistency else None,
        "gold_plus_pairs_with_full_consistency": (sum(1 for p in gold_plus if consistency.get(p, 0) == 1.0) if consistency else None),
        "robustness_gold_other_family": {a2: {"pairs": len(about[a2]), "sentences": len(s2)}},
        "crate_view": cv_report,
    }
    (OUT / "label_model_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    tier_rows = []
    for tier, pairs in (("gold", gold), ("gold_plus_only", gold_plus - gold), ("silver", silver), ("refers", refers_any)):
        for n, c in sorted(pairs):
            tier_rows.append({"sentence": n, "crate": c, "tier": tier, "consistency": consistency.get((n, c), ""),
                              "crateview": int(any((n, c) in v for v in crateview.values())),
                              a1: "A" if (n, c) in about[a1] else ("R" if (n, c) in refers[a1] else ""),
                              a2: "A" if (n, c) in about[a2] else ("R" if (n, c) in refers[a2] else ""),
                              "sym": int((n, c) in sym), "anchor": int((n, c) in anchor), "cochange": int((n, c) in cochange)})
    write_csv(OUT / "semantic_labels.csv", tier_rows, ["sentence", "crate", "tier", "consistency", "crateview", a1, a2, "sym", "anchor", "cochange"])
    if crateview:
        cvall = set().union(*crateview.values())
        write_csv(OUT / "gold_semantic_3way.csv", [{"modelElementID": c, "sentence": n} for n, c in sorted(gold & cvall)],
                  ["modelElementID", "sentence"])
    # robustness gold: the family the linker does NOT belong to, alone
    write_csv(OUT / "gold_semantic_a2only.csv", [{"modelElementID": c, "sentence": n} for n, c in sorted(about[a2])],
              ["modelElementID", "sentence"])
    # gold in the runner's format
    write_csv(OUT / "gold_semantic.csv", [{"modelElementID": c, "sentence": n} for n, c in sorted(gold_plus, key=lambda x: (x[0], x[1]))],
              ["modelElementID", "sentence"])
    write_csv(OUT / "gold_semantic_strict.csv", [{"modelElementID": c, "sentence": n} for n, c in sorted(gold)],
              ["modelElementID", "sentence"])


if __name__ == "__main__":
    main()
