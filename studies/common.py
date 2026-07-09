from __future__ import annotations

import csv
import glob
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
ARDOCO_HOME = Path("/mnt/hostshare/ardoco-home")
MINI = ARDOCO_HOME / "transarc-emp" / "mini-src" / "metrics.py"
ROUTER_DIRECT = ROOT / "src" / "llm_sad_sam" / "linkers" / "experimental" / "router_direct.py"
SLOT = "gpt-5.4_s21"
RUNS = ["run1", "run2", "run3"]


def load_metrics():
    spec = importlib.util.spec_from_file_location("metrics", MINI)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["metrics"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_router_direct():
    spec = importlib.util.spec_from_file_location("router_direct", ROUTER_DIRECT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["router_direct"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_sentences(bench, project: str) -> dict[str, str]:
    hits = glob.glob(str(bench / project / "text_*" / f"{project}.txt"))
    out: dict[str, str] = {}
    if not hits:
        return out
    with open(hits[0], errors="replace") as handle:
        for i, line in enumerate(handle, 1):
            text = line.strip()
            if text:
                out[str(i)] = text
    return out


def prf(gold: set[tuple[str, str]], res: set[tuple[str, str]]) -> tuple[float, float, float]:
    if not res:
        return 0.0, 0.0, 0.0
    tp = len(gold & res)
    p = tp / len(res)
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))
