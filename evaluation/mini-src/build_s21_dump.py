#!/usr/bin/env python3
"""Build the S21 (canonical Full = s_linker21) aalinker dump for RQ1/RQ2.

Self-contained, stdlib only. Reads the S21 gpt-5.4 neutral extracts
(agent-linker/results/v2.6.6_extracts_s21) and writes a NEW `gpt-5.4_s21` config
slot under the sota recovered-links tree:

    <ardoco-home>/sota/recovered-links/model-doc/aalinker/gpt-5.4_s21/<run>/<proj>.csv
    <ardoco-home>/sota/recovered-links/doc-code/aalinker-composed/gpt-5.4_s21/<run>/<proj>.csv

composing model-doc -> code via the prebuilt ArCoTL bridge
(model-code/arcotl/<proj>.csv). The existing `gpt-5.4_full` (s20_union) slot is left
untouched, so rq12.py can score S21 and s20_union side by side.

This is the version-controlled companion to sota/recovered-links/build_unified.py
(that tree is not a git repo). The four helpers below (sha256/write_norm/write_raw/f1)
are copied verbatim from build_unified.py so the dump is byte-identical; gold and bridge
are read from the already-built sota dump rather than rebuilt from raw sources.

    python3 mini-src/build_s21_dump.py
"""
import csv, glob, hashlib, json, os
from pathlib import Path

_HERE = Path(__file__).resolve().parent              # .../transarc-emp/mini-src
_ARDOCO_HOME = _HERE.parents[1]                       # .../ardoco-home
ROOT = os.environ.get("SOTA_LINKS", str(_ARDOCO_HOME / "sota/recovered-links"))
EXTRACTS_S21 = os.environ.get(
    "EXTRACTS_S21", str(_ARDOCO_HOME / "agent-linker/results/v2.6.6_extracts_s21"))

PROJECTS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]
RUNS = ["run1", "run2", "run3"]
# Backend knobs are env-overridable so the SAME builder serves both S21 backends
# (D-04 REVISED: gpt-5.4 = body, Claude/Sonnet = appendix mirror). Defaults keep the
# original gpt-5.4 behaviour byte-identical. For the Sonnet appendix dump, run with
#   EXTRACTS_S21=<…/v2.6.6_extracts_s21_sonnet> S21_BE_DIR=sonnet S21_BE_TAG=claude
#   S21_CONFIG=sonnet_s21 S21_MANIFEST_TAG=s21_sonnet
# so it writes a NEW sonnet_s21 config slot and a distinct manifest (no gpt clobber).
BE_DIR = os.environ.get("S21_BE_DIR", "gpt")
BE_TAG = os.environ.get("S21_BE_TAG", "gpt-5.4")
CONFIG = os.environ.get("S21_CONFIG", "gpt-5.4_s21")
MANIFEST_TAG = os.environ.get("S21_MANIFEST_TAG", "s21")
# Knowledge tier recorded in the manifest. Default "full" keeps the canonical S21
# build byte-identical. For the no-knowledge sweep, run with S21_KNOW=noknow plus
# the noknow EXTRACTS_S21/S21_CONFIG/S21_MANIFEST_TAG overrides so it writes a
# distinct *_s21_noknow config slot and tags the rows knowledge=noknow.
KNOW = os.environ.get("S21_KNOW", "full")


# ---- helpers (verbatim from sota/recovered-links/build_unified.py) ----------
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


def f1(pred, gold):
    pred, gold = set(pred), set(gold)
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    rec = tp / len(gold) if gold else 0.0
    fm = 2 * p * rec / (p + rec) if (p + rec) else 0.0
    return p, rec, fm, tp, len(pred), len(gold)


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


def build_s21(md_gold, arcotl_bridge):
    md_man, dc_man = [], []
    for run in RUNS:
        for proj in PROJECTS:
            jpath = os.path.join(EXTRACTS_S21, BE_DIR, run, f"{proj}.json")
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
            P, R, F, tp, npred, ngold = f1(md_pairs, md_gold[proj])
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

    Globs `_manifest.csv` (the s20_union/full + arcotl base, written by
    build_unified.py) plus all `_manifest_*.csv` add-ons (S21 backends, written
    here) under each task dir, in canonical task order
    (model-doc -> doc-code -> model-code). Decoupled from which builder produced
    each manifest, so the unified file is complete regardless of run order and a
    fresh `build_s21_dump.py` run alone refreshes it from the persisted dump.
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
    md_man, dc_man = build_s21(md_gold, arcotl_bridge)

    write_manifest(f"{ROOT}/model-doc/aalinker/_manifest_{MANIFEST_TAG}.csv", md_man)
    write_manifest(f"{ROOT}/doc-code/aalinker-composed/_manifest_{MANIFEST_TAG}.csv", dc_man)

    n_unified = rebuild_unified(ROOT)

    fs = [float(r["F1"]) for r in md_man]
    print(f"\n== S21 model-doc F1 vs gold (integrity) ==")
    print(f"  {CONFIG:14s} macro-F1 = {sum(fs)/len(fs):.4f}  ({len(fs)} cells)")
    print(f"wrote {len(md_man)} model-doc + {len(dc_man)} doc-code(composed) S21 entries into {ROOT}.")
    print(f"rebuilt UNIFIED_MANIFEST.csv: {n_unified} rows (all per-task manifests aggregated).")


if __name__ == "__main__":
    main()
