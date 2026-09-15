"""Print the actual antecedent sentences behind every surviving coreference link.

The `written` audit says `whole name (qualified)` and off-list antecedents hold 17
false positives and 0 gold. A cross-tab is not an inspection, and a refusal that
closes a case has to be read case by case before it is written -- `s_linker109`'s
first version consulted the alias table and cost 3 gold links in one run, which only
a per-pair reading caught.

So this prints, for every surviving coreference link: the target sentence, the
referring expression the resolver quoted, the antecedent sentence it cited, what the
antecedent sentence writes of the component, and whether the link is gold.

    python3 pilot/coref_antecedent_dump.py ../results [--only qualified|offlist|gold]
"""
from __future__ import annotations

import csv
import glob
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from llm_sad_sam.core.document_loader_v2 import load_sentences  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker123 import SLinker123  # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository  # noqa: E402

from reading_pilots import BENCH, DATASETS, gold_pairs  # noqa: E402

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT.parent / "results")
ONLY = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--only" else None
ARMS = ("s_linker123", "s_linker124")


def phase(run_dir, variant, project, name):
    path = run_dir / "phase_states" / variant / "openai" / project / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def resolutions(run_dir, variant, project):
    out = []
    for path in sorted(glob.glob(str(
            run_dir / "llm_logs" / f"{variant}_openai_{project}_*_calls.json"))):
        for call in json.load(open(path)):
            if call.get("phase") != "phase_25_coreference":
                continue
            try:
                data = json.loads(call.get("response_text") or "{}")
            except json.JSONDecodeError:
                continue
            for r in data.get("resolutions", []):
                if r.get("component") and isinstance(r.get("sentence"), int) \
                        and isinstance(r.get("antecedent_sentence"), int):
                    out.append(r)
    return out


def survivors(run_dir, variant, project):
    path = run_dir / f"{variant}_{project}_links.csv"
    if not path.exists():
        return None
    with path.open() as fh:
        return {(int(r["sentence"]), r["component_name"])
                for r in csv.DictReader(fh) if r.get("source") == "coreference"}


def main():
    seen = set()
    shown = 0
    for model in ("terra", "luna"):
        for run in (1, 2, 3):
            run_dir = RESULTS / f"shortlistmark_e2e_{model}_r{run}_20260914"
            if not run_dir.is_dir():
                continue
            for project in sorted(DATASETS):
                text, repo, gold_path = DATASETS[project]
                sent_text = {s.number: s.text
                             for s in load_sentences(str(BENCH / text))}
                components = parse_pcm_repository(str(BENCH / repo))
                gold = gold_pairs(BENCH / gold_path)
                by_name = {c.name: c.id for c in components}

                for variant in ARMS:
                    know = phase(run_dir, variant, project, "knowledge")
                    surv = survivors(run_dir, variant, project)
                    if know is None or surv is None:
                        continue
                    probe = SLinker123.__new__(SLinker123)
                    probe.doc_knowledge = know["doc_knowledge"]

                    for r in resolutions(run_dir, variant, project):
                        sent, comp = r["sentence"], r["component"]
                        ant = r["antecedent_sentence"]
                        if (sent, comp) not in surv or comp not in by_name:
                            continue
                        body = sent_text.get(ant)
                        if body is None:
                            continue
                        value = probe._written_as(body, comp)
                        listed = ant < sent and probe._states_a_name(body, comp)
                        hit = (sent, by_name[comp]) in gold
                        bucket = ("gold" if hit else
                                  "offlist" if not listed else
                                  "qualified" if value == "whole name (qualified)"
                                  else "other-fp")
                        if ONLY and bucket != ONLY:
                            continue
                        # one line per distinct case, not per run
                        key = (project, sent, comp, ant)
                        if key in seen:
                            continue
                        seen.add(key)
                        shown += 1
                        print(f"\n[{project} S{sent} -> {comp}]  "
                              f"{'GOLD' if hit else 'FALSE POSITIVE'}  "
                              f"({bucket}, written={value}, on-list={listed})")
                        print(f"  reference  : {r.get('reference')!r}")
                        print(f"  target  S{sent}: {sent_text.get(sent, '')[:180]}")
                        print(f"  antecedent S{ant}: {body[:180]}")
                        print(f"  quoted as  : {r.get('antecedent_text')!r}")
    print(f"\n{shown} distinct cases"
          + (f" in bucket {ONLY}" if ONLY else ""))


main()
