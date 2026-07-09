#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from studies.common import (
    ARDOCO_HOME,
    RUNS,
    SLOT,
    load_json,
    load_metrics,
    load_router_direct,
    load_sentences,
    mean,
    prf,
)


def main() -> None:
    M = load_metrics()
    rd = load_router_direct()

    bench = M.BENCHMARK
    projects = M.PROJECTS
    acm = {p: bench / M.ACM_FILES[p] for p in projects}
    rec = ARDOCO_HOME / "sota" / "recovered-links"
    full_router = load_json(
        Path("/mnt/hostshare/ardoco-home/agent-linker/.planning/archive/router-pilot-260701/cache/router_cache_full.json")
    )

    print("ROUTER DECOUPLING STUDY")
    print("=" * 80)

    header = "project | rule CODE sentences | llm CODE sentences | rule deltaF1 | llm deltaF1 | note"
    print(header)
    print("-" * len(header))

    for proj in projects:
        sents = load_sentences(bench, proj)
        idx = rd.CodeIndex(rd.load_code_units(acm[proj]))
        dl = rd.DirectCodeLinker(idx, include_test=True, max_files_per_package=None)
        gold = M.enroll(M.load_gs_sad_code_raw(proj), M.load_code_model_files(proj))

        rule = {sid: rd.rule_route(text, dl) for sid, text in sents.items()}
        llm = {sid: ("CODE" if full_router.get(f"{proj}:{sid}", {}).get("route") == "CODE" else "ARCH") for sid in sents}

        base_scores = []
        rule_scores = []
        llm_scores = []
        base = set()
        for run in RUNS:
            base_run = M.load_result(rec / "doc-code/aalinker-composed" / SLOT / run / f"{proj}.csv", "sad-code")
            base_scores.append(prf(gold, base_run)[2])
            rule_scores.append(prf(gold, rd.augment_doc_code(base_run, sents, dl, rule))[2])
            llm_scores.append(prf(gold, rd.augment_doc_code(base_run, sents, dl, llm))[2])

        rule_code = sum(1 for v in rule.values() if v == "CODE")
        llm_code = sum(1 for v in llm.values() if v == "CODE")
        note = "teammates carries the gain" if proj == "teammates" else "little recall headroom"
        print(
            f"{proj:<11} | {rule_code:>18} | {llm_code:>17} | "
            f"{mean(rule_scores) - mean(base_scores):>11.4f} | {mean(llm_scores) - mean(base_scores):>10.4f} | {note}"
        )

    print("\nInterpretation:")
    print("- rule and LLM routers are alternative policies over the same matcher")
    print("- the router mainly changes exposure, not recall, outside teammates")
    print("- if you want generalization, improve the matcher and candidate extraction")


if __name__ == "__main__":
    main()
