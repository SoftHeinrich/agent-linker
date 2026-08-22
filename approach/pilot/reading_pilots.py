"""Level-2 stage pilot for the reading round: does ONE prompt propose what TWO do?

Level 1 (``pilot/test_s91_reading.py``) pins what is structurally unchanged and
prices the routing shift at 2.1 pairs a project-run.  It cannot answer the only
question that matters, because that question is about LLM behaviour: asked both
reference forms in a single call, does the model still report what two dedicated
calls report?

This replays both arms on the same fixed inputs -- the same sentences, the same
recorded alias table -- and scores their PROPOSALS against gold.  No judging, no
composition: a stage arm screens candidates, it does not decide them.

    control   the head's own two calls: _prompt_extraction + _prompt_coref
    merged    s94's single call:        _prompt_reading
    ordered   s95's single call, the named section committed before the refer-backs
    narrow    s93: both calls kept, the resolver asked only about nameless sentences
    grain     s96: the merged call, asked at the resolution question's batch size

Every arm uses its variant's own prompt builders, so the byte-identity the
measurement policy requires holds by construction rather than by assertion.
**Put every arm a claim rests on in the same invocation** -- absolute levels drift
between invocation sets, so ``--arms control merged ordered narrow`` in one run is
the comparison, not four runs compared afterwards.

Usage (per the branch's credential convention -- never write OAI_KEY anywhere):

    OPENAI_API_KEY="$OAI_KEY" LLM_BACKEND=openai \\
    OPENAI_MODEL_NAME=gpt-5.6-terra OPENAI_REASONING_EFFORT=none \\
      ../.venv/bin/python pilot/reading_pilots.py --samples 3 \\
        --arms control merged ordered narrow \\
        --datasets mediastore teastore teammates bigbluebutton jabref

Run it on BOTH models before drawing a conclusion: the typed and compaction
rounds each refused arms on the second model that the first accepted.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_sad_sam.core.document_loader_v2 import build_sent_map, load_sentences  # noqa: E402
from llm_sad_sam.linkers.experimental.helper_v3 import get_comp_names, parse_snum  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker90 import SLinker90  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker94 import SLinker94  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker95 import SLinker95  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker96 import SLinker96  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker97 import SLinker97  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker98 import SLinker98  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker99 import SLinker99  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker100 import SLinker100  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker102 import SLinker102  # noqa: E402
from llm_sad_sam.linkers.experimental.s_linker93 import SLinker93  # noqa: E402
from llm_sad_sam.llm_client import LLMBackend  # noqa: E402
from llm_sad_sam.pcm_parser_v2 import parse_pcm_repository  # noqa: E402

BENCH = Path(os.environ.get("ALINKER_BENCHMARK", ROOT.parent / "benchmark"))
DATASETS = {
    "mediastore": ("mediastore/text_2016/mediastore.txt",
                   "mediastore/model_2016/pcm/ms.repository",
                   "mediastore/goldstandards/goldstandard_sad_2016-sam_2016.csv"),
    "teastore": ("teastore/text_2020/teastore.txt",
                 "teastore/model_2020/pcm/teastore.repository",
                 "teastore/goldstandards/goldstandard_sad_2020-sam_2020.csv"),
    "teammates": ("teammates/text_2021/teammates.txt",
                  "teammates/model_2021/pcm/teammates.repository",
                  "teammates/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "bigbluebutton": ("bigbluebutton/text_2021/bigbluebutton.txt",
                      "bigbluebutton/model_2021/pcm/bbb.repository",
                      "bigbluebutton/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
    "jabref": ("jabref/text_2021/jabref.txt",
               "jabref/model_2021/pcm/jabref.repository",
               "jabref/goldstandards/goldstandard_sad_2021-sam_2021.csv"),
}
#: Where to read a recorded alias table from, so both arms see the same one and
#: the alias module's own run-to-run swing (2.8 terms) is held out of the delta.
DEFAULT_RUN = ROOT.parent / "results/anchors_e2e_terra_r1_20260821"


def recorded_aliases(run: Path, project: str) -> dict[str, str]:
    """The alias table a recorded run's judge approved: term -> component."""
    table: dict[str, str] = {}
    for path in glob.glob(str(run / "llm_logs" / f"*_{project}_*_calls.json")):
        for call in json.load(open(path)):
            if not call.get("phase", "").endswith("doc_judge"):
                continue
            try:
                data = json.loads(call["response_text"])
            except Exception:
                continue
            for m in data.get("approved", []) or []:
                if m.get("term") and m.get("component"):
                    table[m["term"]] = m["component"]
    return table


def gold_pairs(gold_path: Path) -> set[tuple[int, str]]:
    return {(int(r["sentence"]), r["modelElementID"].strip())
            for r in csv.DictReader(open(gold_path))
            if r.get("sentence") and r.get("modelElementID")}


def _mk(cls, backend, model):
    linker = cls(backend=backend, model=model)
    return linker


def control_proposals(linker, sentences, components, name_to_id, sent_map, mappings):
    """The head's two calls, using its own prompt builders."""
    comp_names = get_comp_names(components)
    out: set[tuple[int, str]] = set()

    linker.llm.set_phase("pilot_control_extract")
    for _, batch in linker._iter_batches(sentences, linker.EXTRACTION_BATCH):
        data = linker._ask(linker._prompt_extraction(comp_names, mappings, batch),
                           timeout=240, label="extract", require="references")
        for ref in data.get("references", []) or []:
            snum, cname = parse_snum(ref.get("sentence")), ref.get("component")
            if snum in sent_map and cname in name_to_id:
                out.add((snum, name_to_id[cname]))

    linker.llm.set_phase("pilot_control_coref")
    for _, batch in linker._iter_batches(sentences, linker.COREFERENCE_BATCH):
        targets, window_ids = [], set()
        for i, sent in enumerate(batch, 1):
            window = [w.number for w in linker._window(sent.number, sentences)]
            window_ids.update(window)
            targets.append({"case": i, "target": sent.number,
                            "text": sent.text, "context": window})
        table = [{"sentence": n, "text": sent_map[n].text}
                 for n in sorted(window_ids) if n in sent_map]
        data = linker._ask(linker._prompt_coref(comp_names, table, targets),
                           timeout=600, label="coref", require_present="resolutions")
        for res in data.get("resolutions", []) or []:
            snum, cname = parse_snum(res.get("sentence")), res.get("component")
            ant = parse_snum(res.get("antecedent_sentence"))
            if snum in sent_map and cname in name_to_id and ant in sent_map:
                out.add((snum, name_to_id[cname]))
    return out


def narrow_proposals(linker, sentences, components, name_to_id, sent_map, mappings):
    """Arm D: the head's extraction, plus the resolver asked only about the
    sentences that write no name. Uses s93's own narrowed loop."""
    comp_names = get_comp_names(components)
    out: set[tuple[int, str]] = set()

    linker.llm.set_phase("pilot_narrow_extract")
    for _, batch in linker._iter_batches(sentences, linker.EXTRACTION_BATCH):
        data = linker._ask(linker._prompt_extraction(comp_names, mappings, batch),
                           timeout=240, label="extract", require="references")
        for ref in data.get("references", []) or []:
            snum, cname = parse_snum(ref.get("sentence")), ref.get("component")
            if snum in sent_map and cname in name_to_id:
                out.add((snum, name_to_id[cname]))

    resolved, _meta = linker._resolve_references(
        sentences, components, name_to_id, sent_map)
    for link in resolved:
        out.add((link.sentence_number, link.component_id))
    return out


def glean_proposals(linker, sentences, components, name_to_id, sent_map, mappings):
    """Rung I: its own two-pass read, scored the same way as every other arm."""
    linker.doc_knowledge = type("K", (), {"aliases": dict(
        m.split("=", 1) for m in mappings if "=" in m)})()
    linker._reading = None
    r = linker._read_document(sentences, components, name_to_id, sent_map)
    return ({(k[0], k[1]) for k in r["named"]}
            | {(l.sentence_number, l.component_id) for l in r["coref"]})


def merged_proposals(linker, sentences, components, name_to_id, sent_map, mappings):
    """s94's single call, using its own prompt builder."""
    comp_names = get_comp_names(components)
    out: set[tuple[int, str]] = set()
    established: dict[str, int] = {}
    linker.batch_loads = []
    linker.llm.set_phase("pilot_merged_reading")
    for _, batch in linker._iter_batches(sentences, linker.EXTRACTION_BATCH):
        data = linker._ask(
            linker._prompt_reading(comp_names, mappings, batch, established),
            timeout=600, label="reading", require_present="references")
        for ref in data.get("references", []) or []:
            snum, cname = parse_snum(ref.get("sentence")), ref.get("component")
            if snum not in sent_map or cname not in name_to_id:
                continue
            out.add((snum, name_to_id[cname]))
            if linker._states_a_name(sent_map[snum].text, cname):
                established[cname] = max(established.get(cname, 0), snum)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--run", default=str(DEFAULT_RUN),
                    help="recorded run supplying the fixed alias table")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL_NAME"))
    ap.add_argument("--dump", default="", help="write every arm's proposals here")
    ap.add_argument("--arms", nargs="+", default=["control", "merged"],
                    choices=["control", "merged", "ordered", "narrow", "grain", "cases", "window", "multi", "glean", "member"],
                    help="every arm a claim rests on goes in the SAME invocation")
    args = ap.parse_args()
    backend = (LLMBackend.OPENAI
               if os.environ.get("LLM_BACKEND", "openai").lower() == "openai"
               else LLMBackend.CLAUDE)

    DUMP: list[dict] = []
    totals = collections.defaultdict(collections.Counter)
    for project in args.datasets:
        text, model_path, gold_path = DATASETS[project]
        sentences = load_sentences(str(BENCH / text))
        sent_map = build_sent_map(sentences)
        components = parse_pcm_repository(str(BENCH / model_path))
        name_to_id = {c.name: c.id for c in components}
        gold = gold_pairs(BENCH / gold_path)
        table = recorded_aliases(Path(args.run), project)
        mappings = [f"{t}={c}" for t, c in sorted(table.items())]
        print(f"\n=== {project}: {len(sentences)} sentences, {len(components)} components, "
              f"{len(gold)} gold, {len(mappings)} aliases held fixed ===")

        ARMS = {
            "control": (SLinker90, control_proposals),
            "merged": (SLinker94, merged_proposals),
            "ordered": (SLinker95, merged_proposals),
            "narrow": (SLinker93, narrow_proposals),
            "grain": (SLinker96, merged_proposals),
            "cases": (SLinker97, merged_proposals),
            "window": (SLinker98, merged_proposals),
            "multi": (SLinker99, merged_proposals),
            "glean": (SLinker100, glean_proposals),
            "member": (SLinker102, merged_proposals),
        }
        for arm in args.arms:
            cls, fn = ARMS[arm]
            for s in range(args.samples):
                linker = _mk(cls, backend, args.model)
                proposals = fn(linker, sentences, components,
                               name_to_id, sent_map, mappings)
                tp, fp = len(proposals & gold), len(proposals - gold)
                if args.dump:
                    DUMP.append({"project": project, "arm": arm, "sample": s,
                                 "pairs": sorted(proposals), "gold": sorted(gold),
                                 "loads": getattr(linker, "batch_loads", [])})
                totals[arm]["pairs"] += len(proposals)
                totals[arm]["gold"] += tp
                totals[arm]["spurious"] += fp
                print(f"  {arm:<8} sample {s + 1}: {len(proposals):>4} proposals, "
                      f"{tp:>3} gold, {fp:>3} spurious")

    n = args.samples * len(args.datasets)
    print(f"\n{'arm':<10}{'proposals':>11}{'gold':>8}{'spurious':>10}{'precision':>11}")
    for arm in args.arms:
        t = totals[arm]
        prec = t["gold"] / t["pairs"] if t["pairs"] else 0.0
        print(f"{arm:<10}{t['pairs'] / n:>11.1f}{t['gold'] / n:>8.1f}"
              f"{t['spurious'] / n:>10.1f}{prec:>11.3f}")
    base = totals["control"]
    for arm in args.arms:
        if arm == "control":
            continue
        d_gold = (totals[arm]["gold"] - base["gold"]) / n
        d_sp = (totals[arm]["spurious"] - base["spurious"]) / n
        print(f"\n{arm} - control: gold {d_gold:+.1f}, "
              f"spurious {d_sp:+.1f} per project-run")
    if args.dump:
        json.dump(DUMP, open(args.dump, "w"))
        print(f"\nproposals dumped to {args.dump}")
    print("Read this against the recorded null floor (FP 10.7, TP 4.8), and run the")
    print("same pilot on the second model before concluding anything.")


if __name__ == "__main__":
    main()
