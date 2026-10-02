#!/usr/bin/env python3
"""Stage replay: the s126 name judge with and without `MENTION_COUNTS`.

`MENTION_COUNTS` is the rule sentence "A mention that says nothing further about the
component still counts as a valid link." This pilot measures what deleting that one
sentence (and nothing else) from `TRACE_LINK_RULE` does to the paper's RQ1-RQ4
numbers, without a full end-to-end re-run.

Why a single-stage replay is exact here: in `s_linker126` no linker receives the links
an earlier one produced (`SLinker126._run_linker`), and the final set is the by-pair
merge of the name linker's links and the coreference linker's links (`link`). The name
judge's inputs are the document plus the recorded alias table (`knowledge.pkl`), and
the candidate scan over them is deterministic. So for every recorded run this pilot

  1. loads that run's `knowledge.pkl` and rebuilds the name candidates (checked
     against the run's recorded `linker_name.pkl` candidates),
  2. re-asks ONLY the name union judge, once per arm, in this one invocation:
       control    -- `TRACE_LINK_RULE` byte-identical to the head
       nomention  -- the same rule with `MENTION_COUNTS` removed
  3. merges the new name links with the run's recorded coreference links exactly as
     `link` does, and writes a run directory per (arm, recorded run) in the
     `run_ablation.py` layout (link CSVs, `phase_states`, `ablation_*.json`,
     `llm_logs`), so `build_alinker_extracts.py`, `rq12.py` and `rq34.py` score it
     unchanged.

The arm's delta is read against `control` from the same invocation, not against the
recorded run: the recorded name judge ran on 2026-09-16 and API drift would confound it.

    OPENAI_API_KEY=... OPENAI_SERVICE_TIER=default \\
      .venv/bin/python approach/pilot/mention_counts_replay.py --stamp 20261002
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences  # noqa: E402
from llm_sad_sam.linkers.experimental import s_linker126 as s126  # noqa: E402
from llm_sad_sam.linkers.experimental.linker_infra import linker_feedback  # noqa: E402
from llm_sad_sam.llm_client import LLMBackend  # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository  # noqa: E402

from reading_pilots import BENCH, DATASETS, gold_pairs  # noqa: E402

RESULTS = ROOT.parent / "results"
PROJECTS = ("mediastore", "teammates", "teastore", "bigbluebutton", "jabref")
MODELS = ("terra", "luna")

assert s126.TRACE_LINK_RULE.count(s126.MENTION_COUNTS) == 1
RULES = {
    "control": s126.TRACE_LINK_RULE,
    "nomention": s126.TRACE_LINK_RULE.replace(s126.MENTION_COUNTS, "", 1),
}


class ReplayLinker(s126.SLinker126):
    """s126 with the union prompt's rule swapped for one arm's rule, nothing else."""

    def __init__(self, rule: str, **kwargs):
        super().__init__(**kwargs)
        self.rule = rule

    def _prompt_union(self, comp_names, sentence_table, cases) -> str:
        prompt = super()._prompt_union(comp_names, sentence_table, cases)
        assert prompt.count(s126.TRACE_LINK_RULE) == 1
        return prompt.replace(s126.TRACE_LINK_RULE, self.rule, 1)


def recorded_dir(model: str, i: int, noknow: bool) -> Path:
    tag = "greedymerge_noknow_e2e" if noknow else "greedymerge_e2e"
    return RESULTS / f"{tag}_{model}_r{i}_20260916v2"


def out_dir(arm: str, model: str, i: int, noknow: bool, stamp: str) -> Path:
    tag = f"mcreplay_{arm}_noknow_e2e" if noknow else f"mcreplay_{arm}_e2e"
    return RESULTS / f"{tag}_{model}_r{i}_{stamp}"


def load_state(run: Path, project: str, name: str):
    path = run / "phase_states" / "s_linker126" / "openai" / project / f"{name}.pkl"
    with open(path, "rb") as handle:
        return pickle.load(handle)


def replay(arm: str, model: str, i: int, noknow: bool, project: str, stamp: str) -> dict:
    src = recorded_dir(model, i, noknow)
    text, repo, gold_path = DATASETS[project]
    components = parse_pcm_repository(str(BENCH / repo))
    sentences = load_sentences(str(BENCH / text))
    sent_map = build_sent_map(sentences)
    name_to_id = {c.name: c.id for c in components}

    linker = ReplayLinker(RULES[arm], backend=LLMBackend.OPENAI,
                          model=f"gpt-5.6-{model}", no_knowledge=noknow)
    linker.doc_knowledge = load_state(src, project, "knowledge")["doc_knowledge"]
    recorded_name = load_state(src, project, "linker_name")
    coref = load_state(src, project, "linker_coreference")

    started = time.time()
    links, feedback = linker._run_name_linker(
        sentences, components, name_to_id, sent_map)
    # The candidate scan is deterministic over (document, alias table): if it does not
    # reproduce the recorded candidates, the replay is not a single-stage replay.
    view = lambda rows: sorted((r["sentence"], r["component"], r["source"]) for r in rows)
    if view(feedback["candidates"]) != view(recorded_name["feedback"]["candidates"]):
        raise SystemExit(f"{src.name}/{project}: candidate set differs from recorded")

    final, seen = [], set()
    for link in list(links) + list(coref["links"]):
        key = (link.sentence_number, link.component_id)
        if key not in seen:
            final.append(link)
            seen.add(key)
    history = [{"linker": "name", "feedback": linker_feedback(feedback)},
               *coref["workflow"][1:]]

    dst = out_dir(arm, model, i, noknow, stamp)
    pdir = dst / "phase_states" / "s_linker126" / "openai" / project
    pdir.mkdir(parents=True, exist_ok=True)
    states = {
        "knowledge": load_state(src, project, "knowledge"),
        "linker_name": {"links": links, "feedback": feedback, "workflow": history[:1]},
        "linker_coreference": coref,
        "final": {"final": final, "workflow": history,
                  "elapsed_s": round(time.time() - started, 2)},
    }
    for name, state in states.items():
        with open(pdir / f"{name}.pkl", "wb") as handle:
            pickle.dump(state, handle)

    variant = "s_linker126_noknow" if noknow else "s_linker126"
    with open(dst / f"{variant}_{project}_links.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sentence", "component_id", "component_name", "confidence",
                         "source"])
        for link in final:
            writer.writerow([link.sentence_number, link.component_id,
                             link.component_name, f"{link.confidence:.2f}", link.source])
    (dst / "llm_logs").mkdir(exist_ok=True)
    with open(dst / "llm_logs" / f"{variant}_{project}_name_judge_calls.json", "w") as h:
        json.dump(linker._llm_calls, h, indent=1, default=str)

    gold = gold_pairs(BENCH / gold_path)
    tp = len(seen & gold)
    recorded_keys = {(l.sentence_number, l.component_id) for l in recorded_name["links"]}
    new_keys = {(l.sentence_number, l.component_id) for l in links}
    return {
        "arm": arm, "model": model, "run": i, "noknow": noknow, "project": project,
        "variant": variant, "dst": str(dst),
        "tp": tp, "fp": len(seen) - tp, "fn": len(gold - seen), "n_links": len(seen),
        "name_kept": len(new_keys), "name_gold": len(new_keys & gold),
        "name_candidates": len(feedback["candidates"]),
        "vs_recorded_added": len(new_keys - recorded_keys),
        "vs_recorded_dropped": len(recorded_keys - new_keys),
        "llm_calls": len(linker._llm_calls),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--projects", nargs="+", default=list(PROJECTS))
    ap.add_argument("--arms", nargs="+", default=list(RULES))
    ap.add_argument("--knowledge", nargs="+", default=["full", "noknow"],
                    choices=["full", "noknow"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    print("rule delta (nomention removes):", repr(s126.MENTION_COUNTS))
    tasks = [(arm, model, i, kn == "noknow", project, args.stamp)
             for kn in args.knowledge for model in args.models for i in args.runs
             for project in args.projects for arm in args.arms]
    rows = []
    with cf.ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(replay, *task): task for task in tasks}
        for future in cf.as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{row['arm']:9s} {row['model']} r{row['run']} "
                  f"{'noknow' if row['noknow'] else 'full  '} {row['project']:13s} "
                  f"tp={row['tp']:3d} fp={row['fp']:3d} fn={row['fn']:3d} "
                  f"name_kept={row['name_kept']:3d} calls={row['llm_calls']}",
                  file=sys.stderr, flush=True)

    # One ablation JSON per output run directory: the tp/fp/fn oracle rq34 validates.
    by_dir: dict[str, dict] = {}
    for row in rows:
        by_dir.setdefault(row["dst"], {}).setdefault(row["project"], {})[row["variant"]] = {
            k: row[k] for k in ("variant", "tp", "fp", "fn", "n_links", "llm_calls")}
    for dst, data in by_dir.items():
        with open(Path(dst) / f"ablation_{args.stamp}_replay.json", "w") as handle:
            json.dump(data, handle, indent=1)

    summary = RESULTS / "mention_counts_round" / f"mcreplay_summary_{args.stamp}.csv"
    summary.parent.mkdir(exist_ok=True)
    rows.sort(key=lambda r: (r["noknow"], r["model"], r["run"], r["project"], r["arm"]))
    with open(summary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[k for k in rows[0] if k != "dst"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if k != "dst"})
    print("summary:", summary, file=sys.stderr)


if __name__ == "__main__":
    main()
