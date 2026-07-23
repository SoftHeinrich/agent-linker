#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from studies.common import (
    ARDOCO_HOME,
    RUNS,
    SLOT,
    load_metrics,
    load_router_direct,
    load_sentences,
    prf,
)


def main() -> None:
    M = load_metrics()
    rd = load_router_direct()

    bench = M.BENCHMARK
    projects = M.PROJECTS
    acm = {p: bench / M.ACM_FILES[p] for p in projects}
    rec = ARDOCO_HOME / "sota" / "recovered-links"

    print("DIRECT MATCHER STUDY")
    print("=" * 80)

    header = (
        "project | direct-only SC | candidate TP | candidate FP | candidate precision | "
        "bridge-gap SC | direct-addressable SC | implicit SC | no-component files"
    )
    print(header)
    print("-" * len(header))

    overall = defaultdict(int)

    for proj in projects:
        sents = load_sentences(bench, proj)
        idx = rd.CodeIndex(rd.load_code_units(acm[proj]))
        dl = rd.DirectCodeLinker(idx, include_test=True, max_files_per_package=None)
        gold = M.enroll(M.load_gs_sad_code_raw(proj), M.load_code_model_files(proj))
        code_files = M.load_code_model_files(proj)
        f2c = M.load_file_to_comps(proj, code_files)

        # direct-only opportunity: gold sentence ids not in SAM gold
        sc_raw = M.load_gs_sad_code_raw(proj)
        ss = M.load_gs_sad_sam(proj)
        sc_sents = {s for s, _ in sc_raw}
        ss_sents = {s for _, s in ss}
        direct_sents = sc_sents - ss_sents

        # candidate quality
        emit = tp = fp = 0
        for sid, text in sents.items():
            for ident, kind, paths in dl.candidates(text):
                for code_path in paths:
                    emit += 1
                    if (sid, code_path) in gold:
                        tp += 1
                    else:
                        fp += 1

        cand_prec = tp / emit if emit else 0.0

        # residual gaps after best config, categorized at sentence/component level
        trans = set()
        for run in RUNS:
            trans |= M.load_result(rec / "doc-code/aalinker-composed" / SLOT / run / f"{proj}.csv", "sad-code")

        direct = set()
        has_codeid = {}
        for sid, text in sents.items():
            cand = dl.candidates(text)
            has_codeid[sid] = bool(cand)
            for ident, kind, paths in cand:
                for code_path in paths:
                    direct.add((sid, code_path))
        recovered = trans | direct

        md_union = set()
        for run in RUNS:
            with open(ARDOCO_HOME / "sota" / "recovered-links" / "model-doc" / "aalinker" / SLOT / run / f"{proj}.csv") as handle:
                for row in csv.DictReader(handle):
                    md_union.add((row["target_id"], row["sentence_id"]))

        # project-level miss counts
        gold_sc = defaultdict(set)
        no_component = 0
        for s, code_path in gold:
            comps = f2c.get(code_path, ())
            if not comps:
                if (s, code_path) not in recovered:
                    no_component += 1
                continue
            for comp in comps:
                gold_sc[(s, comp)].add(code_path)

        bridge_gap = direct_addr = implicit = 0
        for (s, comp), gfiles in gold_sc.items():
            if any((s, code_path) in recovered for code_path in gfiles):
                continue
            if (comp, s) in md_union:
                bridge_gap += 1
            elif has_codeid.get(s):
                direct_addr += 1
            else:
                implicit += 1

        overall["direct_sents"] += len(direct_sents)
        overall["sc_links"] += len(sc_raw)
        overall["emit"] += emit
        overall["tp"] += tp
        overall["cand_fp"] += fp
        overall["bridge_gap"] += bridge_gap
        overall["direct_addr"] += direct_addr
        overall["implicit"] += implicit
        overall["no_component"] += no_component

        print(
            f"{proj:<11} | {len(direct_sents):>13} | {tp:>11} | {fp:>11} | {cand_prec:>19.3f} | "
            f"{bridge_gap:>13} | {direct_addr:>21} | {implicit:>11} | {no_component:>18}"
        )

    print("-" * 80)
    print(
        f"TOTAL      | {overall['direct_sents']:>13} | {overall['tp']:>11} | {overall['cand_fp']:>11} | "
        f"{overall['tp']/overall['emit'] if overall['emit'] else 0.0:>19.3f} | "
        f"{overall['bridge_gap']:>13} | {overall['direct_addr']:>21} | {overall['implicit']:>11} | {overall['no_component']:>18}"
    )

    print("\nInterpretation:")
    print("- direct-only opportunity is concentrated in teammates")
    print("- bigbluebutton has a smaller residual direct-addressable tail")
    print("- mediastore, teastore, jabref are mostly implicit/model-doc misses")


if __name__ == "__main__":
    main()
