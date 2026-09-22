#!/usr/bin/env python3
"""Why the teammates doc-model result does not transfer to doc-code (s126, terra).

Reads the committed ``sota-links`` dump and the benchmark gold standards and
re-partitions what ``rq12.py`` already scores; it computes no new metric.

    python3 studies/teammates-doc-code-transfer/analyze.py

The partition: composing a doc-model link set with the model-code map yields
file links, so the doc-code gold splits into the pairs a doc-model system can
reach that way -- the closure of the doc-model GOLD -- and the pairs beyond it,
which are gold at the file grain with no gold (sentence, component) link behind
them. Every doc-code recall difference lands in one half or the other.
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation" / "mini-src"))
import metrics as m                                                    # noqa: E402

SL = m.REPO / "sota-links"
ARM, BACKEND, RUNS = "s126", "terra", ("run1", "run2", "run3")
LOAD = lambda p, t: m.load_result(SL / p, t)                           # noqa: E731


def view(project):
    """Every link set this study needs for one project."""
    code_files = m.load_code_model_files(project)
    names, sam_gold = m.load_sam_code(project, code_files)
    arcotl = defaultdict(set)
    with open(SL / "model-code" / "arcotl" / f"{project}.csv") as f:
        for r in csv.DictReader(f):
            arcotl[r["source_id"]].add(m.normalize_path(r["target_id"]))
    v = dict(
        project=project, names=names, arcotl=arcotl,
        files_per_comp={ae: {fp for a, fp in sam_gold if a == ae} for ae, _ in sam_gold},
        file_to_comps=m.load_file_to_comps(project, code_files),       # interfaces dropped
        gold_dm=m.load_gs_sad_sam(project),                            # (component, sentence)
        gold_dc=m.enroll(m.load_gs_sad_code_raw(project), code_files),  # (sentence, file)
        app_dm=[LOAD(f"model-doc/aalinker/{BACKEND}_{ARM}/{r}/{project}.csv", "sad-sam")
                for r in RUNS],
        app_dc=[LOAD(f"doc-code/aalinker-composed/{BACKEND}_{ARM}/{r}/{project}.csv", "sad-code")
                for r in RUNS],
        sw_dm=LOAD(f"model-doc/swattr-{project}.csv", "sad-sam"),
        ta_dc=LOAD(f"doc-code/transarc-{project}.csv", "sad-code"))
    v["closure"] = {(s, f) for (c, s) in v["gold_dm"] for f in arcotl.get(c, ())}
    return v


def scored_sets(v):
    print(f"== {v['project']}, {ARM} {BACKEND}: the link sets the paper scores ==")
    pairs = [(f"approach {r}", dm, dc) for r, dm, dc in zip(RUNS, v["app_dm"], v["app_dc"])]
    pairs.append(("SWATTR/TransArC", v["sw_dm"], v["ta_dc"]))
    for tag, dm, dc in pairs:
        p1, r1, f1 = m.prf(v["gold_dm"], dm)
        p2, r2, f2 = m.prf(v["gold_dc"], dc)
        print(f"  {tag:16s} doc-model P={p1:.3f} R={r1:.3f} F1={f1:.3f} |pred|={len(dm):3d}"
              f"  |  doc-code P={p2:.3f} R={r2:.3f} F1={f2:.3f} |pred|={len(dc)}")
    print(f"  gold: {len(v['gold_dm'])} doc-model links, "
          f"{len(v['gold_dc'])} doc-code (sentence,file) pairs")


def closure_table(views):
    print("\n== is the doc-code gold the closure of the doc-model gold? ==")
    print(f"{'project':14s}{'|gold DM|':>10s}{'|gold DC|':>10s}{'|closure|':>10s}"
          f"{'closure in gold':>17s}{'gold DC beyond closure':>24s}")
    for v in views:
        cl, gdc = v["closure"], v["gold_dc"]
        beyond = gdc - cl
        print(f"{v['project']:14s}{len(v['gold_dm']):10d}{len(gdc):10d}{len(cl):10d}"
              f"{len(cl & gdc)/max(len(cl), 1):16.1%}"
              f"{len(beyond)/max(len(gdc), 1):18.1%} ({len(beyond)} pairs)")


def per_component(v):
    print("\n== per component: size, gold links at each grain, and who recovers them ==")
    print(f"{'component':22s}{'files':>6s}{'gDM':>5s}{'gDC':>7s}{'app DM':>8s}{'sw DM':>7s}"
          f"{'app DC':>8s}{'ta DC':>7s}")
    for ae in sorted(v["files_per_comp"], key=lambda a: -len(v["files_per_comp"][a])):
        nm = v["names"].get(ae, ae)
        if nm.startswith("Interface:"):                                # D-12, as in metrics.py
            continue
        g_dm = {(a, s) for (a, s) in v["gold_dm"] if a == ae}
        g_dc = {(s, f) for (s, f) in v["gold_dc"] if ae in v["file_to_comps"].get(f, ())}
        print(f"{nm[:22]:22s}{len(v['files_per_comp'][ae]):6d}{len(g_dm):5d}{len(g_dc):7d}"
              f"{sum(len(g_dm & d) for d in v['app_dm'])/3:8.1f}{len(g_dm & v['sw_dm']):7d}"
              f"{sum(len(g_dc & d) for d in v['app_dc'])/3:8.0f}{len(g_dc & v['ta_dc']):7d}")


def recall_split(v):
    inside, beyond = v["gold_dc"] & v["closure"], v["gold_dc"] - v["closure"]
    print("\n== doc-code recall, split by the two halves of the gold ==")
    for tag, dc in [(f"approach {r}", d) for r, d in zip(RUNS, v["app_dc"])] + \
                   [("TransArC", v["ta_dc"])]:
        print(f"  {tag:12s} inside closure {len(dc & inside)/len(inside):6.1%}"
              f"   beyond closure {len(dc & beyond)/len(beyond):6.1%}"
              f"   ({len(dc & beyond)}/{len(beyond)} pairs)")
    app, ta = v["app_dc"][0], v["ta_dc"]
    n = len(v["gold_dc"])
    print(f"  gap (run1 vs TransArC): {(len(ta & inside)-len(app & inside))/n*100:+.1f}pp "
          f"inside, {(len(ta & beyond)-len(app & beyond))/n*100:+.1f}pp beyond")


def false_positives_at_file_grain(v):
    print("\n== are the doc-model false positives wrong at the file grain too? ==")
    for tag, dm in [("approach run1", v["app_dm"][0]), ("SWATTR", v["sw_dm"])]:
        fps = dm - v["gold_dm"]
        rows = [(v["names"].get(c, c), s,
                 len({(s, f) for f in v["arcotl"].get(c, ())} & v["gold_dc"]),
                 len(v["arcotl"].get(c, ())))
                for (c, s) in sorted(fps)]
        good = sum(hit for _, _, hit, _ in rows)
        print(f"  {tag}: {len(fps)} doc-model FPs -> {good} gold doc-code pairs recovered,"
              f" {sum(tot for *_, tot in rows) - good} spurious file links")
        for nm, s, hit, tot in sorted(rows, key=lambda r: -r[2])[:6]:
            print(f"      FP sentence {s:>4s} -> {nm[:22]:22s} {hit:5d}/{tot:4d}"
                  f" of its files are gold doc-code links")


def orphan_sentences(v):
    s_dm = {s for (_, s) in v["gold_dm"]}
    s_dc = {s for (s, _) in v["gold_dc"]}
    orphans = sorted(s_dc - s_dm, key=int)
    text = (m.BENCHMARK / v["project"] / "text_2021" / f"{v['project']}.txt").read_text()
    sent = {str(i + 1): t.strip() for i, t in enumerate(text.splitlines())}
    n_pairs = sum(1 for (s, _) in v["gold_dc"] if s in set(orphans))
    print(f"\n== {len(orphans)} sentences carry gold doc-code links but no gold doc-model link"
          f" ({n_pairs} pairs) ==")
    print(f"  gold doc-model covers {len(s_dm)} sentences, gold doc-code {len(s_dc)}")
    for s in orphans[:12]:
        comps = sorted({v["names"].get(c, c).replace("Component: ", "")
                        for (x, f) in v["gold_dc"] if x == s
                        for c in v["file_to_comps"].get(f, ())})
        n = sum(1 for (x, _) in v["gold_dc"] if x == s)
        print(f"  s{s:>4s} ({n:4d} file links; {', '.join(comps) or 'no component'}): "
              f"{sent.get(s, '')[:96]}")
    print(f"  ... {max(len(orphans) - 12, 0)} more")
    cc = Counter(v["names"].get(c, c)
                 for (s, f) in (v["gold_dc"] - v["closure"])
                 for c in v["file_to_comps"].get(f, ()))
    print("  beyond-closure pairs by component: "
          + ", ".join(f"{nm.replace('Component: ', '')} {n}" for nm, n in cc.most_common()))


def main():
    views = [view(p) for p in m.PROJECTS]
    tm = next(v for v in views if v["project"] == "teammates")
    scored_sets(tm)
    closure_table(views)
    per_component(tm)
    recall_split(tm)
    false_positives_at_file_grain(tm)
    orphan_sentences(tm)


if __name__ == "__main__":
    main()
