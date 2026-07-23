#!/usr/bin/env python3
"""Score the FN-recovery experiment: judge approval by label + recall/precision tradeoff.

Reads cases.json + verdicts.json (produced by run_judges.py). Reports, per judge structure:
  - reject-pool recovery:  approval on R-TP (gold links the s21 validator dropped) vs
                           leakage on R-TN (non-gold links it correctly dropped) -> precision cost
  - ceiling:               approval on NP-FN (gold links never proposed) vs
                           over-link on NP-CTRL (sibling distractors)          -> precision cost
  - remaining-FN slice:    approval restricted to the actual s21 FN (consistent = missed in all 3
                           runs; per-run = mean over runs), the number the user cares about.
"""
import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

ARD = Path("/mnt/hostshare/ardoco-home")
BENCH = Path(os.environ.get(
    "TRANSARC_BENCHMARK",
    ARD / "ardoco/core/tests-base/src/main/resources/benchmark"))
EXTRACTS = ARD / "agent-linker/results/v2.6.6_extracts_s21/gpt"
PROJECTS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]
RUNS = ["run1", "run2", "run3"]
GS_SAD_SAM = {
    "mediastore":    "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv",
    "teastore":      "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv",
    "teammates":     "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "bigbluebutton": "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv",
    "jabref":        "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv",
}
HERE = Path(__file__).resolve().parent
JUDGES = ["J0_s21", "J0_amb", "J1_soft", "J2_recover", "J3_vote"]


def load_gold(p):
    g = set()
    with (BENCH / GS_SAD_SAM[p]).open() as f:
        for r in csv.DictReader(f):
            g.add((int(r["sentence"]), r["modelElementID"]))
    return g


def fn_status():
    """returns consistent_fn set{(proj,s,cid)}, perrun_fn_count per project, total gold."""
    consistent = set()
    perrun_total = [0, 0, 0]
    total_gold = 0
    for p in PROJECTS:
        gold = load_gold(p)
        total_gold += len(gold)
        finals = []
        for run in RUNS:
            d = json.load(open(EXTRACTS / run / f"{p}.json"))
            finals.append({(l["s"], l["c"]) for l in d["final"]["links"]})
        for i in range(3):
            perrun_total[i] += len(gold - finals[i])
        cons = gold - (finals[0] | finals[1] | finals[2])
        for (s, cid) in cons:
            consistent.add((p, s, cid))
    return consistent, perrun_total, total_gold


def approve(verdicts, judge, cid, mode="majority"):
    v = verdicts.get(f"{judge}|{cid}")
    if isinstance(v, dict):        # J3_vote
        return bool(v.get(mode, False))
    return bool(v)


def main():
    cases = json.loads((HERE / "cases.json").read_text())
    verdicts = json.loads((HERE / "verdicts.json").read_text())
    consistent, perrun_total, total_gold = fn_status()
    by_label = defaultdict(list)
    for c in cases:
        by_label[c["label"]].append(c)
    ncons = len(consistent)

    def rate(judge, subset, mode="majority"):
        if not subset:
            return 0, 0
        k = sum(1 for c in subset if approve(verdicts, judge, c["id"], mode))
        return k, len(subset)

    print("=" * 90)
    print("FN-RECOVERY EXPERIMENT — judge approval by label (gpt-5.4 s21 slot)")
    print("=" * 90)
    print(f"gold(model-doc)={total_gold}  per-run FN={perrun_total} (mean {sum(perrun_total)/3:.1f})"
          f"  consistent FN (missed all 3 runs)={ncons}")
    print("\nDistinct judge cases by label:")
    for lab in ("R-TP", "R-TN", "NP-FN", "NP-CTRL"):
        print(f"  {lab:<8} {len(by_label[lab])}")

    print("\n" + "-" * 90)
    print("APPROVAL RATE per judge structure  (want HIGH on R-TP/NP-FN [recall],")
    print("                                    LOW on R-TN/NP-CTRL [precision cost])")
    print("-" * 90)
    hdr = f"{'judge':<22}{'R-TP(recall)':>16}{'R-TN(leak)':>14}{'NP-FN(ceil)':>14}{'NP-CTRL(over)':>16}"
    print(hdr)
    for j in JUDGES:
        modes = ["majority"] if j != "J3_vote" else ["any", "majority"]
        for m in modes:
            tag = j if j != "J3_vote" else f"{j}:{m}"
            rtp = rate(j, by_label["R-TP"], m); rtn = rate(j, by_label["R-TN"], m)
            npf = rate(j, by_label["NP-FN"], m); npc = rate(j, by_label["NP-CTRL"], m)
            print(f"{tag:<22}"
                  f"{rtp[0]:>6}/{rtp[1]:<3}{100*rtp[0]/rtp[1] if rtp[1] else 0:>5.0f}%"
                  f"{rtn[0]:>5}/{rtn[1]:<3}{100*rtn[0]/rtn[1] if rtn[1] else 0:>4.0f}%"
                  f"{npf[0]:>5}/{npf[1]:<3}{100*npf[0]/npf[1] if npf[1] else 0:>4.0f}%"
                  f"{npc[0]:>6}/{npc[1]:<3}{100*npc[0]/npc[1] if npc[1] else 0:>4.0f}%")

    # remaining-FN slice: the actual consistent FN (22). Which judge approves which?
    print("\n" + "-" * 90)
    print(f"REMAINING-FN SLICE — approval on the {ncons} CONSISTENT FN (missed in all 3 runs)")
    print("  these are the real recall gap; R-TP/NP-FN cases whose (proj,s,cid) is a consistent FN")
    print("-" * 90)
    cons_cases = [c for c in cases
                  if (c["project"], c["sentence_num"], c["component_id"]) in consistent]
    # dedup (one case per consistent FN)
    seen = set(); cons_cases2 = []
    for c in cons_cases:
        k = (c["project"], c["sentence_num"], c["component_id"])
        if k not in seen:
            seen.add(k); cons_cases2.append(c)
    print(f"  (matched {len(cons_cases2)}/{ncons} consistent FN as judge cases)")
    hdr2 = f"  {'judge':<22}{'approved / '+str(len(cons_cases2)):>18}{'rate':>8}"
    print(hdr2)
    for j in JUDGES:
        modes = ["majority"] if j != "J3_vote" else ["any", "majority"]
        for m in modes:
            tag = j if j != "J3_vote" else f"{j}:{m}"
            k, n = rate(j, cons_cases2, m)
            print(f"  {tag:<22}{k:>10}{'':>8}{100*k/n if n else 0:>6.0f}%")

    # per-FN detail
    print("\n  per-consistent-FN verdicts (E=entity-rej C=coref-rej .=never-proposed):")
    order = {"R-TP": 0, "NP-FN": 1}
    def catcode(c):
        if c["label"] == "NP-FN":
            return "."
        return "C" if (c.get("coref") and not c.get("matched_text")) else "E"
    hdr3 = "    " + f"{'proj s#':<18}{'component':<16}{'m':<2}" + "".join(f"{j.split('_')[0]:>6}" for j in JUDGES)
    print(hdr3)
    for c in sorted(cons_cases2, key=lambda c: (c["project"], order.get(c["label"], 2), c["sentence_num"])):
        row = f"    {c['project'][:8]+' s'+str(c['sentence_num']):<18}{str(c['component'])[:15]:<16}{catcode(c):<2}"
        for j in JUDGES:
            m = "majority"
            row += f"{'Y' if approve(verdicts,j,c['id'],m) else '·':>6}"
        print(row)

    # estimated recall impact (per-run mean): recovered FN net of leakage
    print("\n" + "-" * 90)
    print("ESTIMATED RECALL IMPACT if the judge structure re-scores s21's REJECT pool")
    print("  (reject-pool only; NP-FN needs a proposer and is a separate ceiling)")
    print("-" * 90)
    for j in JUDGES:
        m = "majority"
        rtp_k, rtp_n = rate(j, by_label["R-TP"], m)
        rtn_k, rtn_n = rate(j, by_label["R-TN"], m)
        print(f"  {j:<12} re-approves {rtp_k}/{rtp_n} rejected-TP (recall+)  "
              f"and leaks {rtn_k}/{rtn_n} rejected-TN (precision-)")


if __name__ == "__main__":
    main()
