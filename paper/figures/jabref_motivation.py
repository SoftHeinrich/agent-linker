#!/usr/bin/env python3
"""Reproducible generator for the JabRef motivation figure (\\autoref{fig:motivation}).

Two stacked panels over JabRef's gold components, ordered by link-pair share:

  TOP    The doc-link distribution that file-level \\fone implicitly weights ---
         each component's share of all gold sentence--file link pairs (grey bars).
         Overlaid are two size-independent importance axes: each component's share
         of the *documented sentences* (orange markers) and its share of the code's
         *cross-component dependencies* (purple markers; from jabref_depshare.csv).
         The gap is the point: the small components (preferences/cli/globals) own
         almost none of the link pairs, yet the architecture document describes each
         of them with a sizeable fraction of its sentences AND much of the code
         depends on them. The link-level metric under-weights them by orders of
         magnitude relative to both importance axes.

  BOTTOM Per-component F1 for two real, strong recovery tools (TransArc, Artemis;
         Artemis = mean of the three GPT-5.6-terra runs). Artemis scores lowest on the
         small `preferences` component -- 0.53 against TransArc's 0.80, and 0 outright
         in one run of three -- while the link-level average barely moves, because
         preferences owns only 0.44% of the
         link pairs the link-level metric counts.

Every number is COMPUTED from the ARDoCo benchmark gold standard and the two
systems' recovered sad-code links -- nothing is hardcoded. The per-component F1
matches the component suite's scoring (set-overlap F1 over the mapped-only
sentence universe). Paths derive from the repo layout (siblings under
ardoco-home) and may be overridden:

    TRANSARC_BENCHMARK   benchmark/ root (.../tests-base/src/main/resources/benchmark)
    RECOVERED_LINKS      doc-code recovered-links store (default sota/recovered-links/doc-code;
                         transarc-jabref.csv + artemis-jabref-gpt-5.4.csv, schema sentence_id,target_id)

    python3 figures/jabref_motivation.py        # writes jabref_motivation{,_linear,_log}.{pdf,png}

Requires matplotlib (figure tooling only; not a dependency of the paper build).
"""
import csv
import sys
import os
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir() + "/mpl-jabref-motiv")


def _plt():
    """Import matplotlib lazily. compute() and the CSV dump are stdlib-only, so
    `--data-only` regenerates the provenance data on a machine without it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

# ── Repo layout ───────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent                 # …/paper/figures
ARDOCO_HOME = HERE.parents[1]                           # …/agent-linker
# The evaluation tree's shared core supplies the benchmark root, the gold path
# maps, the gold loaders, the enrolment rule and the F-measure, so this figure
# describes the same gold the reported metrics score.
sys.path.insert(0, str(ARDOCO_HOME / "evaluation" / "mini-src"))
import metrics as m  # noqa: E402

BENCHMARK = m.BENCHMARK                                 # $TRANSARC_BENCHMARK overrides
# Recovered doc-code links for the two reference systems live in the unified sota store
# (sota/recovered-links/doc-code, schema: sentence_id,target_id). Earlier runs read them from
# transarc-emp/{results,results_artemis_gpt54}/...; that layout is gone, so default to the store.
SOTA_LINKS = Path(os.environ.get("SOTA_LINKS", ARDOCO_HOME / "sota-links"))
# The old sota/recovered-links/doc-code layout is gone; the normalized dump is the
# sota-links store this repo ships (see HOWTO-REGENERATE-RQ.md).
RECOVERED = Path(os.environ.get("RECOVERED_LINKS", SOTA_LINKS / "doc-code"))
# Per-component code-dependency share (share of all cross-component afferent coupling).
# Provenanced static-analysis result — see jabref_depshare.csv header and the replication
# package transarc-emp/mini-depimport. Cannot be recomputed from the benchmark alone.
DEPSHARE_CSV = HERE / "jabref_depshare.csv"

PROJECT = "jabref"

# (label, [recovered sad-code links csv, ...]) — order = plotting order in the legend.
# A label with several paths is scored per run and AVERAGED, which is how sec:results
# reports it. \Artemis{} moved from the released single gpt-5.4 run to the GPT-5.6-terra
# re-run (mean of 3) on 2026-08-27, so that this figure and tab:rq2 describe one system.
SYSTEMS = [
    ("TransArc", [RECOVERED / "transarc-jabref.csv"]),
    ("Artemis", [SOTA_LINKS / f"doc-code/artemis/terra_5.6/run{i}/jabref.csv"
                 for i in (1, 2, 3)]),
]
SYS_STYLE = {
    "TransArc": dict(color="#159e8c", marker="o"),
    "Artemis":  dict(color="#9b2226", marker="s"),
}
ARTEMIS_LABEL = "Artemis"
DROPPED = "preferences"                                  # the component Artemis misses

# ── Loaders: the shared core's, except where this figure needs another grain ──
normalize_path = m.normalize_path
enroll = m.enroll


def load_code_model_files():
    return m.load_code_model_files(PROJECT)


def _short(c):
    return c.replace("Component: ", "").replace("Interface: ", "")


def load_file_to_comps(code_files):
    """SAM-CODE gold -> {file: {component}} (enrolled, mapped-only).

    NOT ``metrics.load_file_to_comps``, deliberately: this figure keys components
    by their short DISPLAY name (it labels an axis with them) and keeps
    interfaces, where the suite keys by ``ae_id`` and applies the D-12 interface
    drop. The read itself is shared -- ``metrics.load_sam_code`` -- so only the
    keying differs.
    """
    names, sam_enrolled = m.load_sam_code(PROJECT, code_files)
    file_to_comps = defaultdict(set)
    for ae, fp in sam_enrolled:
        file_to_comps[fp].add(_short(names[ae]))
    return file_to_comps


def load_gold_sad_code(code_files):
    return enroll(m.load_gs_sad_code_raw(PROJECT), code_files)   # set[(sentence, file)]


def load_links(path, code_files):
    raw = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            s = r.get("sentenceID") or r.get("modelElementID") or r.get("sentence_id")
            cid = r.get("codeID") or r.get("codeId") or r.get("target_id")
            if s and cid:
                raw.add((s, normalize_path(cid)))
    return enroll(raw, code_files)


def load_dep_share():
    """component -> share (%) of the code's cross-component dependencies (optional series)."""
    out = {}
    if not DEPSHARE_CSV.exists():
        return out
    with open(DEPSHARE_CSV) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if parts[0] == "component":
                continue
            out[parts[0]] = float(parts[2])
    return out


# ── Metric: per-component F1 (set-overlap over the component's sentence set) ───
def f1(gold_links, pred_links):
    """``metrics.prf``'s F1 at the grain ``metric.tex`` eq:worst defines.

    A component owns a set of code files; its F1 is computed over the LINKS whose
    target file belongs to it -- the same grain, and the same shared ``prf``, as
    the worst-/harmonic-component metrics in tab:rq2. (Until 2026-08-29 this
    figure scored a component over its SENTENCE set instead; on JabRef the two
    agree to four decimals, but the definitions had drifted apart.) Only gold
    components are scored, so the empty-gold case never arises.
    """
    return m.prf(gold_links, pred_links)[2]


def collapse(pairs, file_to_comps):
    """(sentence, file) -> {(sentence, component)} (mapped-only).

    Used for the documentation-footprint columns (sent_n / sent_pct), which ask
    which SENTENCES mention a component and are independent of the F1 grain.
    """
    out = set()
    for s, fp in pairs:
        for c in file_to_comps.get(fp, ()):
            out.add((s, c))
    return out


def links_by_comp(pairs, file_to_comps):
    """(sentence, file) -> {component: {(sentence, file)}} -- the eq:worst slice."""
    out = defaultdict(set)
    for s, fp in pairs:
        for c in file_to_comps.get(fp, ()):
            out[c].add((s, fp))
    return out


def compute():
    code = load_code_model_files()
    file_to_comps = load_file_to_comps(code)
    gold_pairs = load_gold_sad_code(code)                # (sentence, file)
    gold_sc = collapse(gold_pairs, file_to_comps)        # (sentence, component)

    # link-pair share (the file-level metric's implicit weight)
    linkpairs = defaultdict(int)
    for s, fp in gold_pairs:
        for c in file_to_comps.get(fp, ()):
            linkpairs[c] += 1
    tot_lp = sum(linkpairs.values()) or 1

    # documented-sentence share (a component's footprint in the architecture doc)
    gold_by_c = defaultdict(set)
    for s, c in gold_sc:
        gold_by_c[c].add(s)
    tot_sent = len({s for s, _ in gold_sc}) or 1

    # per-system per-component F1
    gold_links_by_c = links_by_comp(gold_pairs, file_to_comps)      # eq:worst grain
    sys_f1 = {}
    for label, paths in SYSTEMS:
        per_run = []
        for path in paths:
            pred_links_by_c = links_by_comp(load_links(path, code), file_to_comps)
            per_run.append({c: f1(gold_links_by_c.get(c, set()),
                                  pred_links_by_c.get(c, set()))
                            for c in gold_by_c})
        sys_f1[label] = {c: sum(r[c] for r in per_run) / len(per_run)
                         for c in gold_by_c}

    dep = load_dep_share()                                # component -> code-dependency share %
    comps = sorted(gold_by_c, key=lambda c: -linkpairs[c])  # by link-pair share desc
    rows = []
    for c in comps:
        rows.append({
            "component": c,
            "linkpair_pct": 100 * linkpairs[c] / tot_lp,
            "sent_n": len(gold_by_c[c]),
            "sent_pct": 100 * len(gold_by_c[c]) / tot_sent,
            "dep_share": dep.get(c),
            **{label: sys_f1[label][c] for label, _ in SYSTEMS},
        })
    return rows


# ── Figure ────────────────────────────────────────────────────────────────────
def draw(rows, logscale):
    comps = [r["component"] for r in rows]
    x = list(range(len(comps)))
    lp = [r["linkpair_pct"] for r in rows]
    sp = [r["sent_pct"] for r in rows]
    ds = [r.get("dep_share") for r in rows]
    have_ds = all(v is not None for v in ds)
    drop_i = comps.index(DROPPED) if DROPPED in comps else None

    plt = _plt()
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(6.6, 5.0), sharex=True,
        gridspec_kw=dict(height_ratios=[1.05, 1.0], hspace=0.12))

    # ── top: link-pair share (bars) + documented-sentence share (markers) ──
    bar_bottom = 0.03 if logscale else 0
    bars = ax0.bar(x, lp, width=0.6, bottom=bar_bottom, color="#bdbdbd",
                   edgecolor="#7d7d7d", linewidth=0.7, zorder=2)
    line, = ax0.plot(x, sp, linestyle="--", linewidth=1.3, color="#d9822b",
                     marker="D", markersize=6, markeredgecolor="white",
                     markeredgewidth=0.6, zorder=4)
    depline = None
    if have_ds:
        depline, = ax0.plot(x, ds, linestyle=":", linewidth=1.3, color="#6A4C93",
                            marker="^", markersize=7, markeredgecolor="white",
                            markeredgewidth=0.6, zorder=4)

    def _fmt_lp(v):
        # sub-1% shares get 2 decimals so preferences reads 0.44 (matches the
        # motivation prose: 20 / 0.4352 = 46x; 0.4 would invite the wrong 20/0.4=50).
        return f"{v:.2f}" if v < 1 else f"{v:.1f}"
    # The bar labels sit under two series lines; a tight white box keeps them legible
    # where a line crosses (gui's 34.2 sat directly under the 40% sentence-share line).
    for xi, v in zip(x, lp):
        ax0.annotate(_fmt_lp(v), (xi, v + bar_bottom), textcoords="offset points",
                     xytext=(0, 2), ha="center", va="bottom", fontsize=8,
                     color="#555555", zorder=6,
                     bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none",
                               alpha=0.85))
    for xi, v in zip(x, sp):
        ax0.annotate(f"{v:.0f}", (xi, v), textcoords="offset points",
                     xytext=(7, 4), ha="left", va="bottom", fontsize=8,
                     color="#b5651d", fontweight="bold")
    if have_ds:
        for xi, v in zip(x, ds):
            ax0.annotate(f"{v:.0f}", (xi, v), textcoords="offset points",
                         xytext=(-7, 4), ha="right", va="bottom", fontsize=8,
                         color="#6A4C93", fontweight="bold")

    if logscale:
        ax0.set_yscale("log")
        ax0.set_ylim(0.03, 400)
        ax0.set_ylabel("share (%, log)")
    else:
        ax0.set_ylim(0, max(max(lp), max(sp), max(ds) if have_ds else 0) * 1.55)
        ax0.set_ylabel("share (%)")
    handles = [bars, line] + ([depline] if have_ds else [])
    labels = (["share of gold link pairs", "share of documented sentences"]
              + (["share of code dependencies"] if have_ds else []))
    ax0.legend(handles, labels,
               loc="upper center" if logscale else "upper right",
               fontsize=8, framealpha=0.92, handlelength=1.6)

    # under-weighting callout on the dropped component
    if drop_i is not None:
        r = rows[drop_i]
        if have_ds:
            txt = (f"{_fmt_lp(r['linkpair_pct'])}% of links, but\n{r['sent_pct']:.0f}% of doc"
                   f" sentences and\n{r['dep_share']:.0f}% of code deps\n→ under-weighted")
        else:
            txt = (f"{_fmt_lp(r['linkpair_pct'])}% of links\nbut {r['sent_pct']:.0f}% of"
                   " doc sentences\n→ under-weighted")
        if logscale:
            # text in the empty mid-band; arrow up to the sentence marker
            ax0.annotate(txt, xy=(drop_i, r["sent_pct"]), xycoords="data",
                         xytext=(drop_i - 0.45, 1.1), textcoords="data",
                         fontsize=8.0, color="#9b2226", ha="left", va="center",
                         arrowprops=dict(arrowstyle="->", color="#9b2226", lw=1.0))
        else:
            ax0.annotate(txt, xy=(drop_i, r["sent_pct"]), xycoords="data",
                         xytext=(drop_i + 0.30, max(max(lp), max(sp)) * 0.78),
                         textcoords="data", fontsize=8.0, color="#9b2226",
                         ha="left", va="center",
                         arrowprops=dict(arrowstyle="->", color="#9b2226", lw=1.0))

    # ── bottom: per-component F1 for the two tools ──
    for label, _ in SYSTEMS:
        st = SYS_STYLE[label]
        ax1.plot(x, [rw[label] for rw in rows], linewidth=1.8, markersize=7,
                 marker=st["marker"], color=st["color"], label=label, zorder=3)
    ax1.set_ylim(-0.06, 1.10)
    ax1.set_yticks([0, 0.5, 1.0])
    ax1.set_ylabel("per-component F1")
    ax1.axhline(0, color="#cccccc", linewidth=0.8, zorder=1)
    ax1.legend(loc="lower left", fontsize=9, framealpha=0.9)
    if drop_i is not None:
        # Anchor on the value actually plotted. The released gpt-5.4 arm scored a flat
        # 0 here, so this used to read "entire component missed" and point at the zero
        # line; the mean of three GPT-5.6-terra runs scores 0.53 (0 in one run of three),
        # so an arrow to 0.0 would point at empty axis.
        drop_y = rows[drop_i][ARTEMIS_LABEL]
        ax1.annotate(
            f"weakest component\n({_fmt_lp(rows[drop_i]['linkpair_pct'])}% of links,"
            f" {rows[drop_i]['sent_pct']:.0f}% of doc sentences,"
            f" {rows[drop_i]['dep_share']:.0f}% of code deps)",
            xy=(drop_i, drop_y), xycoords="data", xytext=(drop_i - 2.1, 0.22),
            textcoords="data", fontsize=8.0, color="#9b2226", ha="left",
            arrowprops=dict(arrowstyle="->", color="#9b2226", lw=1.0))

    # shared highlight band on the dropped component
    if drop_i is not None:
        for ax in (ax0, ax1):
            ax.axvspan(drop_i - 0.5, drop_i + 0.5, color="#f6d3d1", alpha=0.45,
                       zorder=0)

    ax1.set_xticks(x)
    ax1.set_xticklabels(comps, rotation=20, ha="right")
    for ax in (ax0, ax1):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.985, bottom=0.12, hspace=0.12)
    return fig


def main():
    data_only = "--data-only" in sys.argv
    rows = compute()
    print(f"{'component':12}{'link%':>8}{'sent_n':>8}{'sent%':>8}"
          + "".join(f"{lbl:>10}" for lbl, _ in SYSTEMS))
    for r in rows:
        print(f"{r['component']:12}{r['linkpair_pct']:>8.2f}{r['sent_n']:>8}"
              f"{r['sent_pct']:>8.1f}"
              + "".join(f"{r[lbl]:>10.3f}" for lbl, _ in SYSTEMS))

    # data dump for provenance
    with open(HERE / "jabref_motivation_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["component", "linkpair_pct", "sent_n", "sent_pct", "dep_share"] + [l for l, _ in SYSTEMS]
        w.writerow(cols)
        for r in rows:
            ds = "" if r.get("dep_share") is None else f"{r['dep_share']:.4f}"
            w.writerow([r["component"], f"{r['linkpair_pct']:.4f}", r["sent_n"],
                        f"{r['sent_pct']:.4f}", ds] + [f"{r[l]:.4f}" for l, _ in SYSTEMS])

    if data_only:
        print("\n[jabref_motivation] --data-only: wrote jabref_motivation_data.csv; "
              "figures NOT rebuilt (needs matplotlib).")
        return
    plt = _plt()
    for stem, logscale in [("jabref_motivation_linear", False),
                           ("jabref_motivation_log", True)]:
        fig = draw(rows, logscale)
        for ext in ("pdf", "png"):
            fig.savefig(HERE / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)
    # generic name = the linear variant the paper \includegraphics
    fig = draw(rows, logscale=False)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"jabref_motivation.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[jabref_motivation] wrote figures + jabref_motivation_data.csv to {HERE}")


if __name__ == "__main__":
    main()
