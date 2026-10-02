#!/usr/bin/env python3
"""The diff set of the MENTION_COUNTS round: every name-judge verdict that differs
between `control` and `nomention` on the same recorded candidate, with the sentence,
the `naming` row, gold status and both arms' quoted claim.

    python3 approach/pilot/mention_counts_diff.py --stamp 20261002 \\
        > results/mention_counts_round/diff_set_20261002.tsv
"""
import argparse, pickle, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences  # noqa
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository  # noqa
from reading_pilots import BENCH, DATASETS, gold_pairs  # noqa

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stamp", required=True)
    ap.add_argument("--arms", nargs=2, default=["control", "nomention"])
    ap.add_argument("--cand-stamp", default=None, help="stamp of the second arm, if it ran in another batch")
    a = ap.parse_args()
    base, cand = a.arms
    print("\t".join(["knowledge", "model", "run", "project", "sentence", "component",
                     "naming", "gold", "flip", f"{base}_claim", f"{cand}_claim", "text"]))
    for kn in ("", "_noknow"):
        for m in ("terra", "luna"):
            for i in (1, 2, 3):
                for p, (text, repo, gold) in DATASETS.items():
                    g = gold_pairs(BENCH / gold)
                    names = {c.id: c.name for c in parse_pcm_repository(str(BENCH / repo))}
                    sm = build_sent_map(load_sentences(str(BENCH / text)))
                    d = {}
                    for arm in (base, cand):
                        s = pickle.load(open(ROOT.parent / "results" /
                            f"mcreplay_{arm}{kn}_e2e_{m}_r{i}_{a.cand_stamp if arm == cand and a.cand_stamp else a.stamp}" / "phase_states" /
                            "s_linker126/openai" / p / "linker_name.pkl", "rb"))
                        d[arm] = {(x["sentence"], x["component_id"]): x
                                  for x in s["feedback"]["judge_decisions"]}
                    for k, x in sorted(d[base].items()):
                        y = d[cand][k]
                        if x["approved"] == y["approved"]:
                            continue
                        print("\t".join(map(str, [kn.strip("_") or "full", m, i, p, k[0],
                            names[k[1]], x["naming"], int(k in g),
                            "lost" if x["approved"] else "gained",
                            x["claim"], y["claim"], sm[k[0]].text])))

if __name__ == "__main__":
    main()
