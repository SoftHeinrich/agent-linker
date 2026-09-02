#!/usr/bin/env python3
"""Thin wrapper: guard the paper's gold_concentration.{tex,csv} against drift.

The full table guard now lives in ``evaluation/mini-src/sync_paper.py`` (it covers every paper-bound
table, deriving the set from ``csv_to_tex.SPECS``). This entry point is kept for back-compat
and delegates the gold_concentration (OUT-02) check to it via ``--only gold``. Same exit
codes: 0 in sync, 1 drift (prints a unified diff), 2 paper dir not found.

    python3 check_paper_table.py
    PAPER_TABLE_DIR=/path/to/alinker-paper/table python3 check_paper_table.py
    python3 check_paper_table.py /path/to/alinker-paper/table

For the complete guard (RQ tables + gold): ``python3 evaluation/mini-src/sync_paper.py --check``.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "evaluation" / "mini-src"))
import sync_paper  # noqa: E402


def _paper_root():
    """Map the legacy 'table dir' arg / PAPER_TABLE_DIR onto the paper root sync_paper wants."""
    raw = (sys.argv[1] if len(sys.argv) > 1 else None) or os.environ.get("PAPER_TABLE_DIR")
    if not raw:
        return None
    p = Path(raw)
    return str(p.parent if p.name == "table" else p)


if __name__ == "__main__":
    argv = ["--check", "--only", "gold"]
    root = _paper_root()
    if root:
        argv.append(root)
    raise SystemExit(sync_paper.main(argv))
