"""Shared filesystem roots for the tail-metric side studies.

These scripts live in ``studies/`` rather than ``evaluation/`` — they probe
candidate metrics and none of them writes a paper table — but they still score
with ``evaluation/mini-src/metrics.py`` so the numbers stay comparable with the
reported panels. Resolving the three roots here keeps ten scripts from drifting
apart, and makes the link dump overridable without editing any of them.

    SOTA_LINKS=/path/to/dump python3 explore.py
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]      # studies/explore-tail -> <repo>
MINI_SRC = REPO / "evaluation" / "mini-src"     # metrics.py: loaders + the RQ panels
MINI_RQ34 = REPO / "evaluation" / "mini-rq34"   # rq34.py: phase-state reader
SOTA = Path(os.environ.get("SOTA_LINKS", REPO / "sota-links"))
