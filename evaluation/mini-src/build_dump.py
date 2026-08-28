#!/usr/bin/env python3
"""Build one aalinker config slot (doc-model + composed doc-code) in the sota dump.

Self-contained, stdlib only. Reads one arm's neutral extracts — one JSON per
(backend, run, project) cell, of which only ``final.links`` is used — and writes a
config slot under the sota recovered-links tree:

    <sota>/model-doc/aalinker/<config>/<run>/<proj>.csv
    <sota>/doc-code/aalinker-composed/<config>/<run>/<proj>.csv

composing model-doc -> code via the prebuilt ArCoTL bridge
(model-code/arcotl/<proj>.csv). Writing is additive: each run creates or refreshes
exactly the one slot its env names, so arms sit side by side and rq12.py can score
whichever the roster lists.

Defaults build the CANONICAL arm's body backend: s_linker92a on GPT-5.6-terra
(``terra_s92a``), whose extracts come from ``build_alinker_extracts.py``. Every
knob is env-overridable, so the SAME builder serves every other cell — the mirror
backend, the no-knowledge sweep, and the retired s21 lineage:

    # mirror backend (luna)
    EXTRACTS_DIR=$PWD/results/s92a_extracts DUMP_BE_DIR=luna \
      DUMP_BE_TAG=gpt-5.6-luna DUMP_CONFIG=luna_s92a DUMP_MANIFEST_TAG=s92a_luna \
      python3 mini-src/build_dump.py

    # a no-knowledge sweep: tags the manifest rows knowledge=noknow
    EXTRACTS_DIR=<...noknow extracts> DUMP_CONFIG=terra_s92a_noknow \
      DUMP_MANIFEST_TAG=s92a_noknow DUMP_KNOW=noknow python3 mini-src/build_dump.py

This is the version-controlled companion to sota/recovered-links/build_unified.py
(that tree is not a git repo). The three writer helpers below (sha256/write_norm/
write_raw) are copied verbatim from build_unified.py so the dump is byte-identical;
the manifest's P/R/F1 integrity figure comes from ``metrics.prf``, the tree's one
F-measure. Gold and bridge are read from the already-built sota dump rather than
rebuilt from raw sources.

    python3 mini-src/build_dump.py
"""
import csv, glob, hashlib, json, os
from pathlib import Path

import metrics as m   # same directory: the tree's one P/R/F1 (see metrics.prf)

_HERE = Path(__file__).resolve().parent              # .../evaluation/mini-src
_ARDOCO_HOME = _HERE.parents[1]                       # .../ardoco-home
ROOT = os.environ.get("SOTA_LINKS", str(_ARDOCO_HOME / "sota/recovered-links"))
EXTRACTS = os.environ.get(
    "EXTRACTS_DIR", str(_ARDOCO_HOME / "agent-linker/results/s92a_extracts"))

PROJECTS = m.PROJECTS
RUNS = ["run1", "run2", "run3"]
# Which cell of the extracts tree to read, and what to call the slot it writes.
# Defaults = the canonical arm's body backend (s_linker92a / GPT-5.6-terra); see the
# module docstring for the mirror-backend and no-knowledge invocations.
BE_DIR = os.environ.get("DUMP_BE_DIR", "terra")          # <extracts>/<BE_DIR>/<run>/<proj>.json
BE_TAG = os.environ.get("DUMP_BE_TAG", "gpt-5.6-terra")  # backend column in the manifest
CONFIG = os.environ.get("DUMP_CONFIG", "terra_s92a")     # the sota config slot to write
MANIFEST_TAG = os.environ.get("DUMP_MANIFEST_TAG", "s92a_terra")   # _manifest_<tag>.csv
# Knowledge tier recorded in the manifest; set DUMP_KNOW=noknow for a no-knowledge sweep.
KNOW = os.environ.get("DUMP_KNOW", "full")


# ---- writers (verbatim from sota/recovered-links/build_unified.py) ----------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_norm(path, rows, header=("sentence_id", "target_id")):
    """Write deduped, sorted normalized links."""
    uniq = sorted(set(rows), key=lambda t: (str(t[0]), str(t[1])))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(uniq)
    return len(uniq)


def write_raw(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def write_manifest(path, rows):
    cols = ["task", "system", "config", "backend", "knowledge", "run", "project",
            "n_links", "P", "R", "F1", "src", "sha"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---- inputs read from the already-built sota dump ---------------------------
def load_gold():
    """{project: set((int sentence, component_id))} from model-doc/gold."""
    gold = {}
    for proj in PROJECTS:
        rows = set()
        with open(f"{ROOT}/model-doc/gold/{proj}.csv") as f:
            for r in csv.DictReader(f):
                rows.add((int(r["sentence_id"]), r["target_id"]))
        gold[proj] = rows
    return gold


def load_bridge():
    """{project: {component_id: [code_paths]}} from model-code/arcotl."""
    bridge = {}
    for proj in PROJECTS:
        b = {}
        with open(f"{ROOT}/model-code/arcotl/{proj}.csv") as f:
            for r in csv.DictReader(f):
                b.setdefault(r["source_id"], []).append(r["target_id"])
        bridge[proj] = b
    return bridge


def build_slot(md_gold, arcotl_bridge):
    md_man, dc_man = [], []
    for run in RUNS:
        for proj in PROJECTS:
            jpath = os.path.join(EXTRACTS, BE_DIR, run, f"{proj}.json")
            if not os.path.exists(jpath):
                print(f"  MISSING {run}/{proj}: {jpath}")
                continue
            d = json.load(open(jpath))
            links = d["final"]["links"]

            # --- model-doc (native sentence -> component) ---
            md_pairs = [(int(l["s"]), l["c"]) for l in links]
            base = f"{ROOT}/model-doc/aalinker/{CONFIG}/{run}/{proj}"
            n_md = write_norm(f"{base}.csv", md_pairs)
            write_raw(f"{base}.raw.csv",
                      ["sentence", "component_id", "component_name", "confidence", "source"],
                      [[l["s"], l["c"], l.get("component_name", ""),
                        l.get("confidence", ""), l.get("source", "")] for l in links])
            P, R, F = m.prf(md_gold[proj], set(md_pairs))
            md_man.append(dict(task="model-doc", system="aalinker", config=CONFIG,
                               backend=BE_TAG, knowledge=KNOW, run=run, project=proj,
                               n_links=n_md, P=f"{P:.4f}", R=f"{R:.4f}", F1=f"{F:.4f}",
                               src=os.path.relpath(jpath, _ARDOCO_HOME), sha=sha256(jpath)))

            # --- doc-code (composed: ours model-doc o ArCoTL model-code) ---
            bridge = arcotl_bridge[proj]
            dc_pairs, raw_rows = [], []
            for s, cid in md_pairs:
                for code in bridge.get(cid, []):
                    dc_pairs.append((s, code))
                    raw_rows.append([s, cid, code])
            cbase = f"{ROOT}/doc-code/aalinker-composed/{CONFIG}/{run}/{proj}"
            n_dc = write_norm(f"{cbase}.csv", dc_pairs)
            write_raw(f"{cbase}.raw.csv", ["sentence_id", "via_component", "target_id"], raw_rows)
            dc_man.append(dict(task="doc-code", system="aalinker-composed", config=CONFIG,
                               backend=BE_TAG, knowledge=KNOW, run=run, project=proj,
                               n_links=n_dc, P="", R="", F1="",
                               src=f"model-doc/aalinker/{CONFIG}/{run}/{proj}.csv o model-code/arcotl/{proj}.csv",
                               sha=""))
    return md_man, dc_man


def rebuild_unified(root):
    """Aggregate every per-task manifest into UNIFIED_MANIFEST.csv.

    Globs `_manifest.csv` (the arcotl + baseline base, written by build_unified.py)
    plus all `_manifest_*.csv` add-ons (one per arm/backend slot, written here)
    under each task dir, in canonical task order
    (model-doc -> doc-code -> model-code). Decoupled from which builder produced
    each manifest, so the unified file is complete regardless of run order and a
    fresh `build_dump.py` run alone refreshes it from the persisted dump.
    Idempotent; dedupes on (task, config, run, project)."""
    task_dirs = [
        ("model-doc",  f"{root}/model-doc/aalinker"),
        ("doc-code",   f"{root}/doc-code/aalinker-composed"),
        ("model-code", f"{root}/model-code/arcotl"),
    ]
    seen, rows = set(), []
    for _task, d in task_dirs:
        manifests = sorted(glob.glob(f"{d}/_manifest.csv")) + sorted(glob.glob(f"{d}/_manifest_*.csv"))
        for mf in manifests:
            with open(mf) as f:
                for r in csv.DictReader(f):
                    key = (r["task"], r["config"], r["run"], r["project"])
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(r)
    write_manifest(f"{root}/UNIFIED_MANIFEST.csv", rows)
    return len(rows)


def main():
    md_gold = load_gold()
    arcotl_bridge = load_bridge()
    md_man, dc_man = build_slot(md_gold, arcotl_bridge)
    if not md_man:
        # Nothing was read: almost always a wrong EXTRACTS_DIR/DUMP_BE_DIR. Bail out
        # BEFORE writing, or the empty result overwrites a good manifest with a bare
        # header and then rebuilds UNIFIED_MANIFEST.csv without the slot's rows.
        raise SystemExit(
            f"no cells found under {EXTRACTS}/{BE_DIR}/<run>/<project>.json "
            f"— nothing written. Check EXTRACTS_DIR / DUMP_BE_DIR.")

    write_manifest(f"{ROOT}/model-doc/aalinker/_manifest_{MANIFEST_TAG}.csv", md_man)
    write_manifest(f"{ROOT}/doc-code/aalinker-composed/_manifest_{MANIFEST_TAG}.csv", dc_man)

    n_unified = rebuild_unified(ROOT)

    fs = [float(r["F1"]) for r in md_man]
    print(f"\n== {CONFIG} model-doc F1 vs gold (integrity) ==")
    print(f"  {CONFIG:14s} macro-F1 = {sum(fs)/len(fs):.4f}  ({len(fs)} cells)")
    print(f"wrote {len(md_man)} model-doc + {len(dc_man)} doc-code(composed) entries into {ROOT}.")
    print(f"rebuilt UNIFIED_MANIFEST.csv: {n_unified} rows (all per-task manifests aggregated).")


if __name__ == "__main__":
    main()
