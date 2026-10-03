#!/usr/bin/env python3
"""Rebuild the generated parts of the replication package in `replication/`.

    python3 scripts/build_replication_package.py --stamp 20261002

The package code (`agentlinker/agentlinker/`, `agentlinker/run.py`, `artemis/run.sh`),
the static inputs (`agentlinker/data/`, `agentlinker/nltk_data/`) and the hand-written
README.md are maintained in place. This script writes:

  agentlinker/recorded/{terra,luna}/{full,no-aliases}/run{1,2,3}/
      the s126 E2E sweep `results/greedymerge{,_noknow}_e2e_<model>_r<i>_<stamp>`,
      converted to the package's format: `<dataset>_links.csv` without the constant
      confidence column, and `<dataset>_calls.json` holding each call's prompt and
      response byte-for-byte, with phase labels renamed and per-call bookkeeping
      (timestamps, timeout, retry bound, success flag) dropped.
  artemis/recorded/{terra,luna}/run{1,2,3}/{doc-model,doc-code}/
      the ArTEMiS re-run dumps from `sota-links/{model-doc,doc-code}/artemis/<model>_5.6/`,
      and `artemis/logs/` with the run and build logs from `sota-links/_build-logs/`,
      local absolute paths replaced by <taas25>, <ner>, <workspace> and <home>.
  artemis/taas25/, artemis/ner/
      the ArTEMiS source (upstream TAAS25 replication package at TAAS_COMMIT) and its
      unpublished NER dependency (upstream at NER_COMMIT), each overlaid with the local
      changes the re-run used, without git files. Comments those changes added are removed; upstream
      code, including its licence headers, is left as published.
  SHA256SUMS
      every packaged file except itself.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import io
import json
import shutil
import subprocess
import sys
import re
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "replication"
HOME = REPO.parent
TAAS = HOME / "sota/Replication-Package-TAAS25_LLM-assisted-Software-Traceability-with-Architecture-Entity-Recognition"
TAAS_COMMIT = "ed07208"
NER = HOME / "sota/ner-1.0.0-SNAPSHOT"
NER_COMMIT = "74ccb33"
DATASETS = ("mediastore", "teammates", "teastore", "bigbluebutton", "jabref")
PHASES = {
    "phase_25_doc_extract": "alias_extract",
    "phase_25_doc_judge": "alias_judge",
    "phase_25_name_union_judge": "name_judge",
    "phase_25_coreference": "coreference",
    "phase_25_coreference_judge": "coreference_judge",
}


def convert_agentlinker_run(source: Path, target: Path, variant: str) -> None:
    target.mkdir(parents=True)
    for dataset in DATASETS:
        with open(source / f"{variant}_{dataset}_links.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        with open(target / f"{dataset}_links.csv", "w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["sentence", "component_id", "component_name", "source"])
            for row in rows:
                writer.writerow([row["sentence"], row["component_id"],
                                 row["component_name"], row["source"]])
        [log] = (source / "llm_logs").glob(f"s_linker126_openai_{dataset}_*_calls.json")
        calls = []
        for call in json.loads(log.read_text()):
            assert call["success"] and call["error"] is None, log
            calls.append({
                "phase": PHASES[call["phase"]],
                "prompt": call["prompt"],
                "response": call["response_text"],
                "model": call["model"],
                "usage": call["token_usage"],
                "latency_ms": call["latency_ms"],
            })
        with open(target / f"{dataset}_calls.json", "w", encoding="utf-8") as handle:
            json.dump(calls, handle, indent=1, ensure_ascii=False)


def build_agentlinker(stamp: str) -> None:
    recorded = PACKAGE / "agentlinker" / "recorded"
    shutil.rmtree(recorded, ignore_errors=True)
    for model in ("terra", "luna"):
        for arm, tag, variant in (("full", "greedymerge_e2e", "s_linker126"),
                                  ("no-aliases", "greedymerge_noknow_e2e", "s_linker126_noknow")):
            for run in (1, 2, 3):
                source = REPO / "results" / f"{tag}_{model}_r{run}_{stamp}"
                convert_agentlinker_run(source, recorded / model / arm / f"run{run}", variant)
                print(f"agentlinker/recorded/{model}/{arm}/run{run} <- {source.name}")


def strip_added_comments(upstream: str, local: str, suffix: str) -> str:
    """Drop the comments that `local` adds to `upstream`; keep licence headers."""
    upstream_lines, local_lines = upstream.splitlines(), local.splitlines()
    added = set()
    matcher = difflib.SequenceMatcher(None, upstream_lines, local_lines, autojunk=False)
    for tag, _, _, start, end in matcher.get_opcodes():
        if tag in ("insert", "replace"):
            added.update(range(start, end))
    out, in_xml_comment = [], False
    for index, line in enumerate(local_lines):
        stripped = line.strip()
        if index not in added:
            out.append(line)
        elif suffix == ".xml":
            if in_xml_comment or stripped.startswith("<!--"):
                in_xml_comment = "-->" not in stripped
            else:
                out.append(line)
        elif stripped.startswith("/* Licensed under"):
            out.append(line)
        elif not stripped.startswith(("/**", "*", "//")):
            code, mark, _ = line.partition(" //")
            if mark and code.count('"') % 2 == 0:
                line = code.rstrip() + (" //" if code.rstrip().endswith(",") else "")
            out.append(line)
    return "\n".join(out) + "\n"


def export_tree(repo: Path, commit: str, subdir: str, target: Path, changed: list[str]) -> None:
    archive = subprocess.run(["git", "-C", str(repo), "archive", commit, subdir or "."],
                             check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if not member.isfile() or any(part.startswith(".git") for part in Path(member.name).parts):
                continue
            relative = Path(member.name).relative_to(subdir) if subdir else Path(member.name)
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tar.extractfile(member).read())
    root = repo / subdir if subdir else repo
    for name in changed:
        path = target / name
        upstream = path.read_text() if path.exists() else ""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(strip_added_comments(upstream, (root / name).read_text(), path.suffix))


def build_artemis() -> None:
    artemis = PACKAGE / "artemis"
    for name in ("taas25", "ner", "recorded", "logs"):
        shutil.rmtree(artemis / name, ignore_errors=True)
    export_tree(TAAS, TAAS_COMMIT, "Replication-Package-TAAS25", artemis / "taas25", [
        "aggregator-pom.xml",
        "pom.xml",
        "tlr/stages-tlr/model-provider/src/main/java/edu/kit/kastel/mcse/ardoco/tlr/models/informants/LargeLanguageModel.java",
        "tlr/tests-tlr/src/test/java/edu/kit/kastel/mcse/ardoco/tlr/tests/integration/RawTraceLinksIT.java",
    ])
    shutil.copyfile(TAAS / "LICENSE.md", artemis / "taas25" / "LICENSE.md")
    export_tree(NER, NER_COMMIT, "", artemis / "ner", [
        "pom.xml",
        "src/main/java/edu/kit/kastel/mcse/ardoco/naer/recognizer/NamedEntityRecognizer.java",
        "src/main/java/edu/kit/kastel/mcse/ardoco/naer/recognizer/TwoPartPrompt.java",
    ])
    for model in ("terra", "luna"):
        for run in (1, 2, 3):
            for task in ("model-doc", "doc-code"):
                source = REPO / "sota-links" / task / "artemis" / f"{model}_5.6" / f"run{run}"
                target = artemis / "recorded" / model / f"run{run}" / task.replace("model-doc", "doc-model")
                shutil.copytree(source, target)
    logs = REPO / "sota-links" / "_build-logs"
    (artemis / "logs").mkdir()
    for model in ("terra", "luna"):
        for path in sorted(logs.glob(f"*artemis*gpt-5.6-{model}*.log")):
            name = path.name.replace(f"gpt-5.6-{model}.log", f"gpt-5.6-{model}-run1.log") \
                if path.name.startswith("run-") else path.name
            text = path.read_text(errors="replace")
            for local, placeholder in ((str(TAAS / "Replication-Package-TAAS25"), "<taas25>"),
                                       (str(NER), "<ner>"), (str(HOME), "<workspace>"),
                                       (str(Path.home()), "<home>")):
                text = text.replace(local, placeholder)
            (artemis / "logs" / name).write_text(text)
    print("artemis/: taas25, ner, recorded, logs")


def write_checksums() -> None:
    files = sorted(p for p in PACKAGE.rglob("*")
                   if p.is_file() and p.name != "SHA256SUMS"
                   and not {".venv", "__pycache__", "run-output"} & set(p.parts))
    with open(PACKAGE / "SHA256SUMS", "w") as handle:
        for path in files:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            handle.write(f"{digest}  {path.relative_to(PACKAGE)}\n")
    print(f"SHA256SUMS: {len(files)} files")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stamp", required=True)
    args = parser.parse_args(argv)
    build_agentlinker(args.stamp)
    build_artemis()
    write_checksums()


if __name__ == "__main__":
    sys.exit(main())
