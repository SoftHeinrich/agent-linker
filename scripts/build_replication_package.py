#!/usr/bin/env python3
"""Rebuild the s126 replication package from the active runtime and a recorded sweep.

    python3 scripts/build_replication_package.py --stamp 20261002 [--check-only]

What it does, in `replication/agentlinker-s126/`:

  src/       every runtime module re-copied from `approach/src/` with its Python
             comments blanked (each COMMENT token replaced by nothing, the line then
             right-stripped, so line numbers are kept). Each copy's syntax tree is
             asserted equal to its source's.
  recorded/  the 12 run directories of the sweep (`greedymerge{,_noknow}_e2e_
             {terra,luna}_r{1,2,3}_<stamp>`), restricted to s126 output: the link CSVs,
             the `s_linker126*` LLM logs, and `phase_states/s_linker126/`.
  SHA256SUMS every packaged file except itself.

`run.py`, `pyproject.toml`, `data/`, `nltk_data/`, `BENCHMARK-LICENSE` and the empty
`src/llm_sad_sam/core/__init__.py` are the package's own static parts (the active
`core/__init__.py` re-exports modules the package does not ship) and are left as they are. `README.md` and
`VERIFICATION.txt` are written by hand, because they describe checks that are run
against the built package.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import shutil
import sys
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "replication" / "agentlinker-s126"
SOURCE = REPO / "approach" / "src"
RUNTIME = [
    "llm_sad_sam/__init__.py",
    "llm_sad_sam/core/data_types_v2.py",
    "llm_sad_sam/core/document_loader_v2.py",
    "llm_sad_sam/linkers/__init__.py",
    "llm_sad_sam/linkers/experimental/__init__.py",
    "llm_sad_sam/linkers/experimental/helper_v3.py",
    "llm_sad_sam/linkers/experimental/linker_infra.py",
    "llm_sad_sam/linkers/experimental/s_linker126.py",
    "llm_sad_sam/llm_client.py",
    "llm_sad_sam/pcm_parser_v2.py",
]


def strip_comments(text: str) -> str:
    lines = text.splitlines(keepends=True)
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            row, col = token.start
            line = lines[row - 1]
            lines[row - 1] = line[:col] + line[token.end[1]:]
    out = "".join(line.rstrip() + "\n" if line.endswith("\n") else line.rstrip()
                  for line in lines)
    assert ast.dump(ast.parse(out)) == ast.dump(ast.parse(text))
    assert not any(t.type == tokenize.COMMENT
                   for t in tokenize.generate_tokens(io.StringIO(out).readline))
    return out


def runs(stamp: str):
    for tag in ("greedymerge_e2e", "greedymerge_noknow_e2e"):
        for model in ("terra", "luna"):
            for i in (1, 2, 3):
                yield f"{tag}_{model}_r{i}_{stamp}"


def copy_run(name: str) -> int:
    src, dst = REPO / "results" / name, PACKAGE / "recorded" / name
    files = sorted(src.glob("s_linker126*_links.csv"))
    files += sorted((src / "llm_logs").glob("s_linker126*"))
    files += sorted(p for p in (src / "phase_states" / "s_linker126").rglob("*") if p.is_file())
    assert len(list(src.glob("s_linker126*_links.csv"))) == 5, f"{name}: incomplete run"
    for path in files:
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return len(files)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--check-only", action="store_true",
                    help="compare stripped sources with the package's src/; write nothing")
    args = ap.parse_args(argv)

    differ = []
    for rel in RUNTIME:
        stripped = strip_comments((SOURCE / rel).read_text())
        current = PACKAGE / "src" / rel
        if not current.exists() or current.read_text() != stripped:
            differ.append(rel)
        if not args.check_only:
            current.parent.mkdir(parents=True, exist_ok=True)
            current.write_text(stripped)
    print("runtime modules that differ from the package before this build:", differ or "none")
    if args.check_only:
        return

    shutil.rmtree(PACKAGE / "recorded", ignore_errors=True)
    for name in runs(args.stamp):
        print(f"recorded/{name}: {copy_run(name)} files")

    files = sorted(p for p in PACKAGE.rglob("*")
                   if p.is_file() and p.name != "SHA256SUMS" and ".venv" not in p.parts
                   and "__pycache__" not in p.parts)
    with open(PACKAGE / "SHA256SUMS", "w") as handle:
        for path in files:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            handle.write(f"{digest}  {path.relative_to(PACKAGE)}\n")
    print(f"SHA256SUMS: {len(files)} files")


if __name__ == "__main__":
    sys.exit(main())
