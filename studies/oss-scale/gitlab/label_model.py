#!/usr/bin/env python3
"""Label model over the annotation votes for the GitLab document (same tiers as
../rustc/semgold/label_model.py, one fewer source: no symbol index is needed when the
component names are proper nouns).

Votes per (sentence, component):
  terra      ABOUT in >= 2 of the 3 gpt-5.6-terra sentence-view runs (majority)
  claude     ABOUT in the Claude Sonnet sentence-view run
  compview   ABOUT in the terra component-view run (component first, whole document)
  structural the sentence sits in the component's own "Component details" section

Tiers:
  gold            terra AND claude                                   (both families)
  gold_plus_only  exactly one family, supported by compview or structural
  silver          exactly one family, unsupported
  refers          REFERS by either family and not ABOUT by both

Writes out/semantic_labels.csv, out/gold_semantic.csv (= gold ∪ gold_plus_only; runner
format modelElementID,sentence with 1-based sentence numbers), out/gold_semantic_strict.csv
(gold), out/gold_semantic_a2only.csv (Claude family alone), out/gold_semantic_3way.csv
(gold ∩ compview) and out/label_model_report.json.
"""
from __future__ import annotations

import collections
import csv
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
import os as _os
DIR = Path(_os.environ.get("OSS_DIR", HERE)).resolve()  # dataset dir holding data/ and out/
DATA = DIR / "data"
OUT = DIR / "out"


def load_ann(path: Path) -> tuple[set, set]:
    d = json.load(open(path))
    about, refers = set(), set()
    for k, v in d["labels"].items():
        i = int(k)
        about.update((i, c) for c in v.get("about", []))
        refers.update((i, c) for c in v.get("refers", []))
    return about, refers


def kappa(a: set, b: set, universe: int) -> float:
    both = len(a & b)
    only_a = len(a - b)
    only_b = len(b - a)
    neither = universe - both - only_a - only_b
    po = (both + neither) / universe
    pa = (both + only_a) / universe
    pb = (both + only_b) / universe
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def write_gold(path: Path, pairs: set) -> None:
    with open(path, "w") as f:
        f.write("modelElementID,sentence\n")
        for i, c in sorted(pairs, key=lambda p: (p[0], p[1])):
            f.write(f"{c},{i + 1}\n")


def main() -> None:
    sents = DATA.joinpath("sentences.txt").read_text().splitlines()
    comps = [c["id"] for c in json.load(open(DATA / "components.json"))]
    meta = json.load(open(DATA / "sentence_meta.json"))
    structural = {(i, m["own_component"]) for i, m in enumerate(meta) if m["own_component"]}

    terra_runs = []
    for tag in ("gpt-5.6-terra", "gpt-5.6-terra_r2", "gpt-5.6-terra_r3"):
        p = OUT / f"annotations_sentence_{tag}.json"
        if p.exists():
            terra_runs.append(load_ann(p))
    claude_about, claude_refers = load_ann(OUT / "annotations_sentence_sonnet.json")
    cv_path = OUT / "annotations_component_gpt-5.6-terra.json"
    compview = load_ann(cv_path)[0] if cv_path.exists() else set()

    n_runs = len(terra_runs)
    cnt = collections.Counter(itertools.chain.from_iterable(a for a, _ in terra_runs))
    terra_about = {p for p, n in cnt.items() if n * 2 >= n_runs + (n_runs % 2 == 0)}  # majority
    terra_refers = set().union(*(r for _, r in terra_runs)) if terra_runs else set()
    consistency = {p: n / n_runs for p, n in cnt.items()} if n_runs else {}

    universe = len(sents) * len(comps)
    rows = []
    tiers = collections.Counter()
    union = terra_about | claude_about | compview | structural | terra_refers | claude_refers
    for i, c in sorted(union):
        t, cl = (i, c) in terra_about, (i, c) in claude_about
        cv, st = (i, c) in compview, (i, c) in structural
        if t and cl:
            tier = "gold"
        elif (t or cl) and (cv or st):
            tier = "gold_plus_only"
        elif t or cl:
            tier = "silver"
        elif (i, c) in terra_refers or (i, c) in claude_refers:
            tier = "refers"
        else:
            tier = "vote_only"  # compview or structural alone
        tiers[tier] += 1
        rows.append({"sentence": i + 1, "component": c, "tier": tier, "terra": int(t), "claude": int(cl),
                     "compview": int(cv), "structural": int(st),
                     "consistency": f"{consistency.get((i, c), 0):.2f}",
                     "verbatim": int(next(x["name"] for x in json.load(open(DATA / "components.json")) if x["id"] == c).lower() in sents[i].lower()),
                     "text": sents[i]})
    with open(OUT / "semantic_labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    gold = terra_about & claude_about
    gold_plus = gold | {(i, c) for i, c in (terra_about ^ claude_about) if (i, c) in compview or (i, c) in structural}
    write_gold(OUT / "gold_semantic.csv", gold_plus)
    write_gold(OUT / "gold_semantic_strict.csv", gold)
    write_gold(OUT / "gold_semantic_a2only.csv", claude_about)
    write_gold(OUT / "gold_semantic_3way.csv", gold & compview)

    def about_rate(pairs):
        return len({i for i, _ in pairs}) / len(sents)

    report = {
        "sentences": len(sents), "components": len(comps), "terra_runs": n_runs,
        "pairs": {"terra_majority": len(terra_about), "claude": len(claude_about), "compview": len(compview),
                  "structural": len(structural), "gold": len(gold), "gold_plus": len(gold_plus),
                  "3way": len(gold & compview)},
        "tiers": dict(tiers),
        "sentences_with_gold_plus": len({i for i, _ in gold_plus}),
        "share_sentences_with_gold_plus": round(about_rate(gold_plus), 3),
        "kappa_terra_claude": round(kappa(terra_about, claude_about, universe), 3),
        "jaccard_terra_claude": round(len(gold) / len(terra_about | claude_about), 3) if terra_about | claude_about else 0,
        "kappa_compview_vs_gold": round(kappa(compview, gold, universe), 3) if compview else None,
        "terra_full_consistency_share": round(sum(1 for p in gold_plus if consistency.get(p, 0) == 1.0) / len(gold_plus), 3) if gold_plus and n_runs else None,
        "structural_in_gold_plus": round(len(structural & gold_plus) / len(structural), 3) if structural else None,
        "verbatim_share_gold_plus": round(sum(1 for r in rows if r["tier"] in ("gold", "gold_plus_only") and r["verbatim"]) / len(gold_plus), 3) if gold_plus else None,
        "components_with_gold_plus": len({c for _, c in gold_plus}),
    }
    json.dump(report, open(OUT / "label_model_report.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
