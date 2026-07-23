#!/usr/bin/env python3
"""Build the unified recovered-links archive.

Generalizes the SOTA-baseline recovered-links folder into one place holding all
three traceability tasks, well separated, with our approach's runs organized by
running config. Additive: it never touches the existing flat baseline CSVs.

Layout produced (alongside the existing baselines):

  model-doc/   SAD -> architecture-model element        (doc->model)
    aalinker/<backend>_<know>/run<N>/<project>.csv       normalized: sentence_id,target_id
                                       <project>.raw.csv  verbatim extract links (5 cols)
    gold/<project>.csv  + .raw.csv
  doc-code/    SAD -> source-code file                   (doc->code)
    aalinker-composed/<backend>_<know>/run<N>/<project>.csv      normalized
                                          <project>.raw.csv      bridge: sentence_id,via_component,target_id
    gold/<project>.csv  + .raw.csv
  model-code/  architecture-model element -> code (ArCoTL)
    arcotl/<project>.csv + .raw.csv      deterministic; shared across our runs
    gold/<project>.csv   + .raw.csv

Two copies coexist per file: `.csv` (normalized, deduped+sorted, Implementation/
prefix stripped to match the existing harness) and `.raw.csv` (verbatim source,
for integrity checks). A `_manifest.csv` per task records config + provenance +
row counts + sha256, and (where exact id-matching applies) P/R/F1 vs gold.
Idempotent.
"""
import csv, glob, hashlib, json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.abspath(os.path.join(ROOT, "..", ".."))            # /mnt/hostshare/ardoco-home
EXTRACTS = os.environ.get(
    "EXTRACTS", os.path.join(HOME, "agent-linker", "results", "v2.6.6_extracts"))
# ArCoTL model-code bridge (samCodeTlr_*.csv). The canonical pruned copy lives in
# transarc-emp/mini-data (the 15 TransArc result CSVs the studies use); fall back to
# the legacy results/ tree if present. Env-overridable.
TRANSARC = os.environ.get("TRANSARC", next(
    (os.path.join(HOME, "transarc-emp", d)
     for d in ("mini-data", "results")
     if os.path.isdir(os.path.join(HOME, "transarc-emp", d))),
    os.path.join(HOME, "transarc-emp", "mini-data")))
BENCH = os.path.join(HOME, "ardoco", "core", "tests-base",
                     "src", "main", "resources", "benchmark")
PROJECTS = ["mediastore", "teastore", "teammates", "bigbluebutton", "jabref"]
BACKENDS = {"gpt": "gpt-5.4", "sonnet": "sonnet"}                 # extract subdir -> model tag
RUNS = ["run1", "run2", "run3"]
KNOW = "full"                                                     # full-knowledge only (per decision)


# ----- helpers ---------------------------------------------------------------
def strip_impl(p):
    return p[len("Implementation/"):] if p.startswith("Implementation/") else p


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


def gold_file(project, kind):
    """kind in {sad-sam, sad-code, sam-code} -> benchmark gold path (year-agnostic)."""
    pat = {"sad-sam": "goldstandard_sad_*-sam_*.csv",
           "sad-code": "goldstandard_sad_*-code_*.csv",
           "sam-code": "goldstandard_sam_*-code_*.csv"}[kind]
    hits = [p for p in glob.glob(os.path.join(BENCH, project, "goldstandards", pat))
            if "_UME" not in p]
    return hits[0] if hits else None


def read_csv_rows(path):
    with open(path, newline="") as f:
        r = list(csv.reader(f))
    return r[0], r[1:]


def f1(pred, gold):
    pred, gold = set(pred), set(gold)
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    rec = tp / len(gold) if gold else 0.0
    fm = 2 * p * rec / (p + rec) if (p + rec) else 0.0
    return p, rec, fm, tp, len(pred), len(gold)


# ----- gold (all three tasks) ------------------------------------------------
def build_gold():
    md_gold, mc_gold = {}, {}          # cache for F1 + composition checks
    for proj in PROJECTS:
        # model-doc gold: modelElementID,sentence  ->  sentence_id,target_id
        gp = gold_file(proj, "sad-sam")
        _, rows = read_csv_rows(gp)
        pairs = [(int(s), mid) for mid, s in ((r[0], r[1]) for r in rows if len(r) >= 2)]
        write_norm(f"{ROOT}/model-doc/gold/{proj}.csv", pairs)
        write_raw(f"{ROOT}/model-doc/gold/{proj}.raw.csv",
                  ["modelElementID", "sentence"], rows)
        md_gold[proj] = set(pairs)

        # doc-code gold: sentenceID,codeID  ->  sentence_id,target_id (strip Impl/)
        gp = gold_file(proj, "sad-code")
        _, rows = read_csv_rows(gp)
        dc = [(int(r[0]), strip_impl(r[1])) for r in rows if len(r) >= 2 and r[0].strip().isdigit()]
        write_norm(f"{ROOT}/doc-code/gold/{proj}.csv", dc)
        write_raw(f"{ROOT}/doc-code/gold/{proj}.raw.csv", ["sentenceID", "codeID"], rows)

        # model-code gold: ae_id,ae_name,ce_ids  ->  source_id,target_id (strip Impl/)
        gp = gold_file(proj, "sam-code")
        _, rows = read_csv_rows(gp)
        mc = [(r[0], strip_impl(r[-1])) for r in rows if len(r) >= 2]
        write_norm(f"{ROOT}/model-code/gold/{proj}.csv", mc, ("source_id", "target_id"))
        write_raw(f"{ROOT}/model-code/gold/{proj}.raw.csv",
                  ["ae_id", "ae_name", "ce_ids"], rows)
        mc_gold[proj] = mc
    return md_gold, mc_gold


# ----- model-code: ArCoTL recovered (deterministic, shared) ------------------
def build_model_code():
    """Returns {project: {component_id: [code_paths]}} from recovered ArCoTL."""
    bridge = {}
    man = []
    for proj in PROJECTS:
        src = glob.glob(os.path.join(TRANSARC, proj, "sam-code", "*.csv"))[0]
        _, rows = read_csv_rows(src)
        pairs = [(r[0], strip_impl(r[1])) for r in rows if len(r) >= 2]
        n = write_norm(f"{ROOT}/model-code/arcotl/{proj}.csv", pairs, ("source_id", "target_id"))
        write_raw(f"{ROOT}/model-code/arcotl/{proj}.raw.csv", ["sentenceID(=ae_id)", "codeID"], rows)
        b = {}
        for cid, code in pairs:
            b.setdefault(cid, []).append(code)
        bridge[proj] = b
        man.append(dict(task="model-code", system="arcotl", config="deterministic",
                        backend="-", knowledge="-", run="-", project=proj,
                        n_links=n, P="", R="", F1="",
                        src=os.path.relpath(src, HOME), sha=sha256(src)))
    return bridge, man


# ----- model-doc + composed doc-code: our approach ---------------------------
def build_aalinker(md_gold, arcotl_bridge):
    md_man, dc_man = [], []
    for be_dir, be_tag in BACKENDS.items():
        config = f"{be_tag}_{KNOW}"
        for run in RUNS:
            for proj in PROJECTS:
                jpath = os.path.join(EXTRACTS, be_dir, run, f"{proj}.json")
                if not os.path.exists(jpath):
                    continue
                d = json.load(open(jpath))
                links = d["final"]["links"]
                meta = d.get("meta", {})
                variant = meta.get("variant", "s_linker20_union")

                # --- model-doc (native) ---
                md_pairs = [(int(l["s"]), l["c"]) for l in links]
                base = f"{ROOT}/model-doc/aalinker/{config}/{run}/{proj}"
                n_md = write_norm(f"{base}.csv", md_pairs)
                write_raw(f"{base}.raw.csv",
                          ["sentence", "component_id", "component_name", "confidence", "source"],
                          [[l["s"], l["c"], l.get("component_name", ""),
                            l.get("confidence", ""), l.get("source", "")] for l in links])
                P, R, F, tp, npred, ngold = f1(md_pairs, md_gold[proj])
                md_man.append(dict(task="model-doc", system="aalinker", config=config,
                                   backend=be_tag, knowledge=KNOW, run=run, project=proj,
                                   n_links=n_md, P=f"{P:.4f}", R=f"{R:.4f}", F1=f"{F:.4f}",
                                   src=os.path.relpath(jpath, HOME), sha=sha256(jpath)))

                # --- doc-code (composed: ours model-doc o ArCoTL model-code) ---
                bridge = arcotl_bridge[proj]
                dc_pairs, raw_rows = [], []
                for s, cid in md_pairs:
                    for code in bridge.get(cid, []):
                        dc_pairs.append((s, code))
                        raw_rows.append([s, cid, code])
                cbase = f"{ROOT}/doc-code/aalinker-composed/{config}/{run}/{proj}"
                n_dc = write_norm(f"{cbase}.csv", dc_pairs)
                write_raw(f"{cbase}.raw.csv", ["sentence_id", "via_component", "target_id"], raw_rows)
                dc_man.append(dict(task="doc-code", system="aalinker-composed", config=config,
                                   backend=be_tag, knowledge=KNOW, run=run, project=proj,
                                   n_links=n_dc, P="", R="", F1="",
                                   src=f"model-doc/aalinker/{config}/{run}/{proj}.csv o model-code/arcotl/{proj}.csv",
                                   sha=""))
    return md_man, dc_man


def write_manifest(path, rows):
    cols = ["task", "system", "config", "backend", "knowledge", "run", "project",
            "n_links", "P", "R", "F1", "src", "sha"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def rebuild_unified():
    """Aggregate every per-task manifest into UNIFIED_MANIFEST.csv.

    Globs each task dir's `_manifest.csv` (this builder's s20_union/full + arcotl
    base) plus all `_manifest_*.csv` add-ons (the S21 backends, written by the
    git-tracked transarc-emp/mini-src/build_s21_dump.py companion), in canonical
    task order. Decoupled from which builder produced each manifest, so the unified
    file is complete regardless of run order. Idempotent; dedupes on
    (task, config, run, project). Kept byte-identical to build_s21_dump.rebuild_unified."""
    task_dirs = [
        ("model-doc",  f"{ROOT}/model-doc/aalinker"),
        ("doc-code",   f"{ROOT}/doc-code/aalinker-composed"),
        ("model-code", f"{ROOT}/model-code/arcotl"),
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
    write_manifest(f"{ROOT}/UNIFIED_MANIFEST.csv", rows)
    return len(rows)


def main():
    md_gold, _ = build_gold()
    arcotl_bridge, mc_man = build_model_code()
    md_man, dc_man = build_aalinker(md_gold, arcotl_bridge)

    write_manifest(f"{ROOT}/model-doc/aalinker/_manifest.csv", md_man)
    write_manifest(f"{ROOT}/doc-code/aalinker-composed/_manifest.csv", dc_man)
    write_manifest(f"{ROOT}/model-code/arcotl/_manifest.csv", mc_man)
    # Aggregate base + S21 add-on manifests into the unified file (complete regardless
    # of run order); see rebuild_unified. With no S21 manifests present this reproduces
    # exactly md_man + dc_man + mc_man.
    n_unified = rebuild_unified()

    # integrity print: our model-doc F1 vs gold, macro per config
    print("== our approach model-doc F1 vs gold (integrity check) ==")
    by_cfg = {}
    for r in md_man:
        by_cfg.setdefault(r["config"], []).append(float(r["F1"]))
    for cfg, fs in sorted(by_cfg.items()):
        print(f"  {cfg:18s} macro-F1 = {sum(fs)/len(fs):.4f}  ({len(fs)} cells)")
    print(f"\nwrote {len(md_man)} model-doc, {len(dc_man)} doc-code(composed), "
          f"{len(mc_man)} model-code(arcotl) entries + gold for 3 tasks.")
    print(f"UNIFIED_MANIFEST.csv: {n_unified} rows (base + any S21 add-on manifests).")


if __name__ == "__main__":
    main()
