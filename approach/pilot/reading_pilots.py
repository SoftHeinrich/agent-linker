"""Shared benchmark loading helpers for the pilot suite.

This module originally also drove a level-2 stage pilot comparing several
retired reading-stage variants' PROPOSALS against gold. That comparison, its
argparse CLI and the retired-variant imports it needed were archived along
with those variants; what remains is the benchmark/gold-loading surface that
``pilot/coref_exact_pilots.py`` (and, through it, ``pilot/test_s126.py``)
still imports: ``BENCH``, ``DATASETS``, ``gold_pairs``, and the recorded-alias
helpers used to hold a fixed alias table across a stage pilot.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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
