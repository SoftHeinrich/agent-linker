#!/usr/bin/env python3
"""Is the teammates doc-code loss a component-size effect, or a coverage effect?

Companion to ``analyze.py``, which located the loss in the half of the doc-code
gold that no doc-model link can reach. This one asks the other question: the
file grain weights each (sentence, component) decision by how many code files
the component owns, so a system can lose at the file grain while covering the
same components. It re-partitions the same committed link sets by component
size; it computes no new metric.

    python3 studies/teammates-doc-code-transfer/analyze_size.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation" / "mini-src"))
import metrics as m                                                    # noqa: E402
from analyze import RUNS, view                                         # noqa: E402


def comp_name(v, ae):
    return v["names"].get(ae, ae).replace("Component: ", "")


def gold_components(v):
    """Gold components of the doc-code task, largest file extent first."""
    comps = {c for (_, f) in v["gold_dc"] for c in v["file_to_comps"].get(f, ())}
    return sorted(comps, key=lambda c: -len(v["files_per_comp"].get(c, ())))


def slice_by_comp(v, pairs, ae):
    return {(s, f) for (s, f) in pairs if ae in v["file_to_comps"].get(f, ())}


def size_table(v):
    """Per-component doc-code recall and F1, ordered by how many files it owns."""
    print(f"\n== {v['project']}: doc-code per component, largest first ==")
    print(f"{'component':16s}{'files':>6s}{'gold DC':>9s}{'app R':>8s}{'ta R':>8s}"
          f"{'dR':>8s}{'app F1':>8s}{'ta F1':>8s}{'pairs lost':>12s}")
    app, ta = v["app_dc"][0], v["ta_dc"]
    for ae in gold_components(v):
        g = slice_by_comp(v, v["gold_dc"], ae)
        a, t = slice_by_comp(v, app, ae), slice_by_comp(v, ta, ae)
        ra, rt = len(g & a) / len(g), len(g & t) / len(g)
        _, _, fa = m.prf(g, a)
        _, _, ft = m.prf(g, t)
        print(f"{comp_name(v, ae)[:16]:16s}{len(v['files_per_comp'].get(ae, ())):6d}"
              f"{len(g):9d}{ra:8.1%}{rt:8.1%}{(ra-rt)*100:+8.1f}{fa:8.3f}{ft:8.3f}"
              f"{len(g & t) - len(g & a):+12d}")


def grain_ladder(v):
    """The same two link sets scored at four grains, coarse to size-blind."""
    print(f"\n== {v['project']}: the same link sets, scored at four grains ==")
    print(f"{'system':16s}{'file R':>9s}{'file F1':>9s}{'(sent,comp) F1':>16s}"
          f"{'macro comp R':>14s}{'worst C F1':>12s}{'harm C F1':>11s}")
    rows = [(f"approach {r}", d) for r, d in zip(RUNS, v["app_dc"])] + \
           [("TransArC", v["ta_dc"])]
    for tag, dc in rows:
        s = m.compute_sad_code(v["project"], dc)
        recalls = []
        for ae in gold_components(v):
            g = slice_by_comp(v, v["gold_dc"], ae)
            recalls.append(len(g & slice_by_comp(v, dc, ae)) / len(g))
        print(f"{tag:16s}{s['file_r']:9.3f}{s['file_f1']:9.3f}{s['component_f1']:16.3f}"
              f"{sum(recalls)/len(recalls):14.3f}{s['worst_component_f1']:12.3f}"
              f"{s['harmonic_component_f1']:11.3f}")


def link_mass(v):
    """What one doc-model decision is worth at the file grain, per component."""
    print(f"\n== {v['project']}: file mass of one doc-model link ==")
    app, sw = v["app_dm"][0], v["sw_dm"]
    print(f"{'component':16s}{'files/link':>11s}{'app DM':>8s}{'sw DM':>7s}"
          f"{'app mass':>10s}{'sw mass':>9s}")
    for ae in gold_components(v):
        n = len(v["arcotl"].get(ae, ()))
        a = len([1 for (c, _) in app if c == ae])
        s = len([1 for (c, _) in sw if c == ae])
        print(f"{comp_name(v, ae)[:16]:16s}{n:11d}{a:8d}{s:7d}{a*n:10d}{s*n:9d}")
    for tag, dm in [("approach run1", app), ("SWATTR", sw)]:
        mass = sum(len(v["arcotl"].get(c, ())) for (c, _) in dm)
        print(f"  {tag:14s} {len(dm):3d} doc-model links -> {mass:5d} composed file links"
              f"  ({mass/len(dm):.0f} files per link on average)")


def size_advantage(views):
    """Across projects: does the per-component doc-code gap follow component size?"""
    print("\n== across projects: per-component recall gap (approach run1 - TransArC) "
          "vs component size ==")
    print(f"{'project':14s}{'comps':>6s}{'app>ta':>8s}{'app<ta':>8s}"
          f"{'median size won':>17s}{'median size lost':>18s}{'spearman(size,dR)':>19s}")
    all_sizes, all_gaps = [], []
    for v in views:
        sizes, gaps = [], []
        for ae in gold_components(v):
            g = slice_by_comp(v, v["gold_dc"], ae)
            if not g:
                continue
            ra = len(g & slice_by_comp(v, v["app_dc"][0], ae)) / len(g)
            rt = len(g & slice_by_comp(v, v["ta_dc"], ae)) / len(g)
            sizes.append(len(v["files_per_comp"].get(ae, ())))
            gaps.append(ra - rt)
        won = sorted(s for s, d in zip(sizes, gaps) if d > 1e-9)
        lost = sorted(s for s, d in zip(sizes, gaps) if d < -1e-9)
        med = lambda xs: f"{xs[len(xs)//2]:d}" if xs else "--"          # noqa: E731
        rho = m.spearman(sizes, gaps) if len(sizes) > 2 else float("nan")
        print(f"{v['project']:14s}{len(sizes):6d}{len(won):8d}{len(lost):8d}"
              f"{med(won):>17s}{med(lost):>18s}{rho:19.2f}")
        all_sizes += sizes
        all_gaps += gaps
    print(f"  pooled over {len(all_sizes)} gold components: "
          f"spearman(size, recall gap) = {m.spearman(all_sizes, all_gaps):.2f}")


def main():
    views = [view(p) for p in m.PROJECTS]
    tm = next(v for v in views if v["project"] == "teammates")
    size_table(tm)
    grain_ladder(tm)
    link_mass(tm)
    size_advantage(views)


if __name__ == "__main__":
    main()
