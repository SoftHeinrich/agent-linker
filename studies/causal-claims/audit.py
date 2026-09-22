#!/usr/bin/env python3
"""Error analysis for the CAUSAL claims in paper/sections/results.tex.

The provenance check (verification/2026-09-17-s126-paper-results-provenance.md)
established that every NUMBER in results.tex regenerates from the committed s126
CSVs. This module checks the other half: the "This is because ..." clauses, i.e.
the MECHANISM each number is attributed to.

Method
------
Every test re-derives its evidence from the recorded s126 terra sweep and scores
it through ``evaluation/mini-src/metrics.py`` -- the tree's sole metric
implementation -- so no definition is re-implemented here. Section 0 prints a
reproduction gate against the committed tables; if that gate passes, every
count below is on the same basis as the paper's own floats.

The unit of analysis is the individual (sentence, component) link. Doc-model
gold is 195 links over five projects, so every finding below is enumerable by
hand from ``items.csv``, which this script also writes.

    python3 studies/causal-claims/audit.py            # full report
    python3 studies/causal-claims/audit.py --csv-only # just refresh items.csv

Stdlib only. Reads, never writes, the run directories.
"""
import argparse
import csv
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "evaluation" / "mini-src"))
import metrics as m          # noqa: E402
import rq34                  # noqa: E402
import rq34_rq2 as R2        # noqa: E402

rq34.install_unpickler()

PROJECTS = m.PROJECTS
RUNS = ["r1", "r2", "r3"]
#: the arm the paper's body reports (evaluation/HOWTO-REGENERATE-RQ.md)
RUNDIR = {r: REPO / f"results/greedymerge_e2e_terra_{r}_20260916v2" for r in RUNS}
#: the knowledge ablation, same variant and model with the alias table off
NOKNOW = {r: REPO / f"results/greedymerge_noknow_e2e_terra_{r}_20260916v2" for r in RUNS}
SOTA = REPO / "sota-links"
VARIANT = "s_linker126"

#: the SAD, one sentence per line, 1-based -- the same numbering the gold uses
TEXT = {
    "mediastore":    "mediastore/text_2016/mediastore.txt",
    "teastore":      "teastore/text_2020/teastore.txt",
    "teammates":     "teammates/text_2021/teammates.txt",
    "bigbluebutton": "bigbluebutton/text_2021/bigbluebutton.txt",
    "jabref":        "jabref/text_2021/jabref.txt",
}
#: the doc-model component catalog, so a component no system ever links is still named
UML = {
    "mediastore":    "mediastore/model_2016/uml/ms.uml",
    "teastore":      "teastore/model_2020/uml/teastore.uml",
    "teammates":     "teammates/model_2021/uml/teammates.uml",
    "bigbluebutton": "bigbluebutton/model_2021/uml/bbb.uml",
    "jabref":        "jabref/model_2021/uml/jabref.uml",
}
_COMP_RE = re.compile(r'xmi:type="uml:Component"\s+xmi:id="([^"]+)"\s+name="([^"]+)"')


# ── loaders ──────────────────────────────────────────────────────────────────
def sentences(project):
    lines = (m.BENCHMARK / TEXT[project]).read_text(encoding="utf-8").splitlines()
    return {i + 1: t for i, t in enumerate(lines)}


def catalog(project):
    return dict(_COMP_RE.findall((m.BENCHMARK / UML[project]).read_text(encoding="utf-8")))


def gold(project):
    return rq34.load_gold(project)


def state(rundir, project, phase):
    p = rundir / "phase_states" / VARIANT / "openai" / project / f"{phase}.pkl"
    with p.open("rb") as f:
        return pickle.load(f)


def final_links(rundir, project):
    """{(sentence, component_id): source-form} -- what the pipeline emitted."""
    return {(int(x.sentence_number), str(x.component_id)):
            (getattr(x, "source", "") or "") for x in state(rundir, project, "final")["final"]}


def judged(rundir, project, phase):
    """(kept, rejected) for one judge, via rq34's own reader."""
    return rq34._judged_sets(state(rundir, project, phase))


def kept_by_form(rundir, project):
    """The three PROPOSAL forms, split off the stage label each kept link carries."""
    st = state(rundir, project, "linker_name")
    by_src = rq34._kept_by_source(st)
    kn, _ = rq34._judged_sets(st)
    kc, _ = judged(rundir, project, "linker_coreference")
    return {"FullName": by_src.get("full_name", set()) & kn,
            "PartialName": by_src.get("partial_name", set()) & kn,
            "Coref": kc}


def aliases(rundir, project):
    """The run's own document-alias table: surface string -> component name."""
    dk = state(rundir, project, "knowledge")["doc_knowledge"]
    out = {}
    for field in ("aliases", "abbreviations", "synonyms", "partial_references"):
        for k, v in (getattr(dk, field, None) or {}).items():
            out[str(k)] = str(v)
    return out


def dump(rel):
    return m.load_result(SOTA / rel, "sad-sam" if rel.startswith("model-doc") else "sad-code")


def artemis_dm(project, i):
    return {(int(s), c) for c, s in dump(f"model-doc/artemis/terra_5.6/run{i}/{project}.csv")}


def swattr_dm(project):
    return {(int(s), c) for c, s in dump(f"model-doc/swattr-{project}.csv")}


# ── surface-form classification ──────────────────────────────────────────────
# How the SENTENCE refers to the COMPONENT. A property of the document, not of
# the pipeline -- except that `alias` consults the run's own knowledge table,
# which is flagged wherever the category is used.
TOK = re.compile(r"[A-Za-z0-9]+")
#: architectural head nouns that name no component on their own
GENERIC = {"component", "components", "the", "a", "an", "of", "and", "system",
           "module", "layer", "service", "manager", "handler", "server", "client"}


def toks(s):
    return [t.lower() for t in TOK.findall(s)]


def contains_seq(hay, needle):
    return bool(needle) and any(hay[i:i + len(needle)] == needle
                                for i in range(len(hay) - len(needle) + 1))


def classify(sent_text, comp_name, alias_map):
    """(category, matched surface). First rule that fires wins.

      canonical  the canonical name appears as a contiguous token run
      alias      a document alias bound to this component appears likewise
      partial_d  a DISTINCTIVE token of the canonical name appears
      partial_g  only a GENERIC head noun of the name appears ("client", "server")
      none       no token of the component name appears in the sentence
    """
    h, n = toks(sent_text), toks(comp_name)
    if contains_seq(h, n):
        return "canonical", comp_name
    hits = sorted((a for a, c in alias_map.items()
                   if c.lower() == comp_name.lower() and contains_seq(h, toks(a))),
                  key=lambda s: -len(s))
    if hits:
        return "alias", hits[0]
    present = [t for t in n if len(t) >= 2 and t in h]
    distinct = [t for t in present if t not in GENERIC]
    if distinct:
        return "partial_d", " ".join(distinct)
    if present:
        return "partial_g", " ".join(present)
    return "none", ""


# ── scoring helpers ──────────────────────────────────────────────────────────
def prf(pred, g):
    tp, fp, fn = len(pred & g), len(pred - g), len(g - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return tp, fp, fn, p, r, f1, m.fbeta(p, r)


def per_component_dc(project, sad_sam):
    """{component_id: (F1, gold-link count)} on doc-code, at metric.tex's eq:worst grain."""
    res = R2.compose_doc_code(project, sad_sam)
    code_files = m.load_code_model_files(project)
    g = m.enroll(m.load_gs_sad_code_raw(project), code_files)
    f2c = m.load_file_to_comps(project, code_files)
    gb, rb = defaultdict(set), defaultdict(set)
    for s, f in g:
        for c in f2c.get(f, ()):
            gb[c].add((s, f))
    for s, f in res:
        for c in f2c.get(f, ()):
            rb[c].add((s, f))
    return {c: (m.prf(gb[c], rb.get(c, set()))[2], len(gb[c])) for c in gb}


def sec(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


# ── the item table ───────────────────────────────────────────────────────────
def build_items():
    rows = []
    for p in PROJECTS:
        g, cat, sents = gold(p), catalog(p), sentences(p)
        ap = {r: final_links(RUNDIR[r], p) for r in RUNS}
        nk = {r: final_links(NOKNOW[r], p) for r in RUNS}
        ar = {r: artemis_dm(p, r[1]) for r in RUNS}
        sw = swattr_dm(p)
        al = aliases(RUNDIR["r1"], p)
        keys = set(g) | sw
        for r in RUNS:
            keys |= set(ap[r]) | ar[r]
        for (s, cid) in sorted(keys):
            name = cat.get(cid, f"<unknown:{cid}>")
            text = sents.get(s, f"<no sentence {s}>")
            klass, surf = classify(text, name, al)
            rows.append(dict(
                project=p, sentence=s, component=name, component_id=cid,
                gold=int((s, cid) in g),
                approach_runs=sum((s, cid) in ap[r] for r in RUNS),
                approach_forms=",".join(sorted({ap[r][(s, cid)] for r in RUNS
                                                if (s, cid) in ap[r]})),
                noknow_runs=sum((s, cid) in nk[r] for r in RUNS),
                artemis_runs=sum((s, cid) in ar[r] for r in RUNS),
                swattr=int((s, cid) in sw),
                surface=klass, matched=surf, text=text))
    return rows


def write_items(rows):
    out = HERE / "items.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out


# ── per-item sheets for the link-by-link analyses ────────────────────────────
def write_sheets(rows, out: Path):
    """Two build products, both derived -- never hand-authored, never edited.

    ch1_gold_sheet.csv  every GOLD doc-model link (195), with how each system fared
                        and which proposal form reached it -> the CH1 analysis
    ch2_fp_sheet.csv    every FALSE POSITIVE any system produced, with each judge's
                        decision on it -> the CH2 analysis
    """
    out.mkdir(parents=True, exist_ok=True)

    ch1 = [dict(project=r["project"], sentence=r["sentence"], component=r["component"],
                auto_surface=r["surface"], auto_matched=r["matched"],
                swattr=r["swattr"], artemis_runs=r["artemis_runs"],
                approach_runs=r["approach_runs"], approach_forms=r["approach_forms"],
                noknow_runs=r["noknow_runs"], sentence_text=r["text"])
            for r in rows if r["gold"]]

    fp = defaultdict(lambda: dict(name_judged=0, name_kept=0, coref_judged=0,
                                  coref_kept=0, approach_final=0, artemis=0, swattr=0))
    for r in RUNS:
        for p in PROJECTS:
            g, cat = gold(p), catalog(p)
            kn, rn = judged(RUNDIR[r], p, "linker_name")
            kc, rc = judged(RUNDIR[r], p, "linker_coreference")
            fin = set(final_links(RUNDIR[r], p))
            art = artemis_dm(p, r[1])
            sw = swattr_dm(p) if r == "r1" else set()   # single-shot: count once
            for (s, cid) in (kn | rn | kc | rc | fin | art | sw) - g:
                d = fp[(p, s, cat.get(cid, cid))]
                d["name_judged"] += (s, cid) in (kn | rn)
                d["name_kept"] += (s, cid) in kn
                d["coref_judged"] += (s, cid) in (kc | rc)
                d["coref_kept"] += (s, cid) in kc
                d["approach_final"] += (s, cid) in fin
                d["artemis"] += (s, cid) in art
                d["swattr"] += (s, cid) in sw
    ch2 = []
    for (p, s, comp), d in sorted(fp.items()):
        sents, al, cat = sentences(p), aliases(RUNDIR["r1"], p), catalog(p)
        k, matched = classify(sents.get(s, ""), comp, al)
        here = sorted(cat.get(c, c) for (x, c) in gold(p) if x == s)
        ch2.append(dict(project=p, sentence=s, wrong_component=comp,
                        auto_surface=k, auto_matched=matched,
                        gold_components_for_this_sentence=";".join(here) or "(none)",
                        **d, sentence_text=sents.get(s, "")))

    for name, table in (("ch1_gold_sheet.csv", ch1), ("ch2_fp_sheet.csv", ch2)):
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
        print(f"wrote {out / name} ({len(table)} rows)")


def write_mode2_sheet(out: Path):
    """The CH2 'type 2' population: false positives on sentences carrying NO surface
    form of the wrongly linked component -- the inference mode the paper does not name.

    Carries the diagnostics needed to separate topical inference from neighbour bleed:
    the neighbouring sentences, the gold components of this sentence, and the distance
    to the nearest sentence that IS a gold link for the wrongly linked component.
    """
    out.mkdir(parents=True, exist_ok=True)
    acc = defaultdict(lambda: dict(name_judged=0, name_kept=0, coref_judged=0,
                                   coref_kept=0, approach_final=0, artemis=0, swattr=0))
    for r in RUNS:
        for p in PROJECTS:
            g, cat, sents = gold(p), catalog(p), sentences(p)
            al = aliases(RUNDIR[r], p)
            kn, rn = judged(RUNDIR[r], p, "linker_name")
            kc, rc = judged(RUNDIR[r], p, "linker_coreference")
            fin = set(final_links(RUNDIR[r], p))
            art = artemis_dm(p, r[1])
            sw = swattr_dm(p) if r == "r1" else set()
            for (s, cid) in (kn | rn | kc | rc | fin | art | sw) - g:
                name = cat.get(cid, cid)
                if classify(sents.get(s, ""), name, al)[0] != "none":
                    continue          # a surface form IS present -> mode 1, not mode 2
                d = acc[(p, s, name, cid)]
                d["name_judged"] += (s, cid) in (kn | rn)
                d["name_kept"] += (s, cid) in kn
                d["coref_judged"] += (s, cid) in (kc | rc)
                d["coref_kept"] += (s, cid) in kc
                d["approach_final"] += (s, cid) in fin
                d["artemis"] += (s, cid) in art
                d["swattr"] += (s, cid) in sw
    rows = []
    for (p, s, name, cid), d in sorted(acc.items()):
        sents, cat, g = sentences(p), catalog(p), gold(p)
        gold_sents = sorted(x for (x, c) in g if c == cid)
        near = min((abs(x - s) for x in gold_sents), default=None)
        rows.append(dict(
            project=p, sentence=s, wrong_component=name,
            gold_here=";".join(sorted(cat.get(c, c) for (x, c) in g if x == s)) or "(none)",
            gold_sentences_of_this_component=";".join(str(x) for x in gold_sents) or "(none)",
            distance_to_nearest_gold_sentence=near if near is not None else "",
            **d,
            prev_sentence=sents.get(s - 1, ""), sentence_text=sents.get(s, ""),
            next_sentence=sents.get(s + 1, "")))
    with (out / "ch2_mode2_sheet.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / 'ch2_mode2_sheet.csv'} ({len(rows)} rows)")
    return rows


# ── 0. reproduction gate ─────────────────────────────────────────────────────
def gate():
    sec("0. REPRODUCTION GATE -- this module's own numbers against the committed tables")
    ok = True

    def check(label, got, want, tol=0.05):
        nonlocal ok
        hit = abs(got - want) <= tol
        ok &= hit
        print(f"  {'OK  ' if hit else 'FAIL'}  {label:52s} got {got:8.3f}  committed {want:8.3f}")

    agg = defaultdict(lambda: defaultdict(list))
    for r in RUNS:
        for p in PROJECTS:
            g = gold(p)
            f = set(final_links(RUNDIR[r], p))
            _, rn = judged(RUNDIR[r], p, "linker_name")
            _, rc = judged(RUNDIR[r], p, "linker_coreference")
            for name, s in (("Full", f), ("NoNameValid", f | rn),
                            ("NoCitation", f | rc), ("NoValidator", f | rn | rc)):
                agg[name][p].append(prf(s, g))

    def macro(name, idx):
        return sum(sum(x[idx] for x in v) / len(v) for v in agg[name].values()) / len(agg[name])

    check("macro F1, Full (rq3_variants.csv)", macro("Full", 5), 0.934737, 0.002)
    check("macro F2, Full", macro("Full", 6), 0.949061, 0.002)
    check("macro F1, NoValidator", macro("NoValidator", 5), 0.784290, 0.002)
    check("macro F1, NoNameValid", macro("NoNameValid", 5), 0.835413, 0.002)
    check("macro F1, NoCitation", macro("NoCitation", 5), 0.853710, 0.002)

    rej_fp = rej_tp = lost = 0.0
    comb_fp = comb_tp = 0.0
    for r in RUNS:
        for p in PROJECTS:
            g = gold(p)
            kn, rn = judged(RUNDIR[r], p, "linker_name")
            kc, rc = judged(RUNDIR[r], p, "linker_coreference")
            rej_fp += len(rc - g) / 3
            rej_tp += len(rc & g) / 3
            lost += len((rc & g) - kn) / 3
            comb_fp += len((rn | rc) - g) / 3
            comb_tp += len(((rn | rc) & g) - (set(final_links(RUNDIR[r], p)))) / 3
    check("coref rejected-FP  (rq3_validators.csv)", rej_fp, 53.33, 0.2)
    check("coref rejected-TP", rej_tp, 84.33, 0.2)
    check("coref unique rejected-TP", lost, 2.33, 0.2)
    check("both judges, distinct FP rejected", comb_fp, 145.67, 0.2)
    check("both judges, TP lost outright", comb_tp, 10.67, 0.2)

    u = Counter()
    for r in RUNS:
        for p in PROJECTS:
            g = gold(p)
            f = kept_by_form(RUNDIR[r], p)
            for k in f:
                others = set().union(*(f[o] for o in f if o != k))
                u[k] += len((f[k] - others) & g) / 3
    for k, want in (("FullName", 136.67), ("PartialName", 13.0), ("Coref", 15.0)):
        check(f"unique TPs, {k} (rq4_linkers.csv)", u[k], want, 0.2)

    dc_full = dc_nk = w_full = w_nk = 0.0
    for p in PROJECTS:
        for r in RUNS:
            a = m.compute_sad_code(p, R2.compose_doc_code(p, set(final_links(RUNDIR[r], p))))
            b = m.compute_sad_code(p, R2.compose_doc_code(p, set(final_links(NOKNOW[r], p))))
            dc_full += a["file_f1"] / 15
            dc_nk += b["file_f1"] / 15
            w_full += a["worst_component_f1"] / 15
            w_nk += b["worst_component_f1"] / 15
    check("doc-code worst-component F1, Full (tab:rq4)", w_full, 0.77, 0.01)
    check("doc-code worst-component F1, no-knowledge", w_nk, 0.44, 0.02)
    check("doc-code file F1 delta, no-knowledge (pp)", 100 * (dc_nk - dc_full), -10.9, 0.3)
    print(f"\n  GATE {'PASSED' if ok else 'FAILED'} -- the analysis below is on the paper's own basis."
          if ok else "\n  GATE FAILED -- do not read the analysis below as comparable.")
    return ok


# ── 1. RQ1 mechanism ─────────────────────────────────────────────────────────
def test_rq1(rows):
    sec("1. RQ1 (results.tex L38-39) -- is the gain over Artemis carried by ALIAS mentions?")
    g = [r for r in rows if r["gold"]]
    print(f"A. the 195 gold doc-model links, by how the sentence refers to the component")
    c = Counter(r["surface"] for r in g)
    for k in ("canonical", "alias", "partial_d", "partial_g", "none"):
        print(f"     {k:10s} {c[k]:4d}  ({100 * c[k] / len(g):4.1f}%)")

    print("\nB. recall by surface class, majority of 3 runs")
    print(f"     {'class':10s} {'gold':>5s} {'approach':>9s} {'Artemis':>8s} {'SWATTR':>7s} {'gain':>6s}")
    t = defaultdict(lambda: [0, 0, 0, 0])
    for r in g:
        k = r["surface"]
        t[k][0] += 1
        t[k][1] += r["approach_runs"] >= 2
        t[k][2] += r["artemis_runs"] >= 2
        t[k][3] += r["swattr"]
    for k in ("canonical", "alias", "partial_d", "partial_g", "none"):
        n, a, b, s = t[k]
        if n:
            print(f"     {k:10s} {n:5d} {a:9d} {b:8d} {s:7d} {a - b:+6d}")
    tot = [sum(t[k][i] for k in t) for i in range(4)]
    print(f"     {'TOTAL':10s} {tot[0]:5d} {tot[1]:9d} {tot[2]:8d} {tot[3]:7d} {tot[1] - tot[2]:+6d}")

    gain = [r for r in g if r["approach_runs"] >= 2 and r["artemis_runs"] < 2]
    print(f"\nC. THE GAIN SET -- {len(gain)} gold links \\approach recovers and Artemis does not")
    cg = Counter(r["surface"] for r in gain)
    for k, n in cg.most_common():
        print(f"     {k:10s} {n:3d}  ({100 * n / len(gain):4.1f}% of the gain)")
    print(f"     proposer form: {dict(Counter(r['approach_forms'] for r in gain))}")
    print("\n     every item (audit by hand against items.csv):")
    for i, r in enumerate(sorted(gain, key=lambda x: (x["project"], x["sentence"])), 1):
        print(f"     {i:>3} {r['project']:14s} s{r['sentence']:<4d} {r['component'][:20]:20s} "
              f"{r['surface']:9s} {r['approach_forms'][:13]:13s} | {r['text'][:74]}")

    loss = [r for r in g if r["artemis_runs"] >= 2 and r["approach_runs"] < 2]
    print(f"\nD. THE LOSS SET -- {len(loss)} gold links Artemis recovers and \\approach does not")
    for r in loss:
        print(f"       {r['project']:14s} s{r['sentence']:<4d} {r['component'][:20]:20s} "
              f"{r['surface']:9s} | {r['text'][:74]}")

    fa = [r for r in rows if not r["gold"] and r["approach_runs"] >= 2]
    fb = [r for r in rows if not r["gold"] and r["artemis_runs"] >= 2]
    print(f"\nE. false positives, majority of 3 runs -- 'the plausible-but-wrong links an LLM produces'")
    print(f"     \\approach {len(fa):3d}  by surface {dict(Counter(r['surface'] for r in fa))}")
    print(f"     Artemis   {len(fb):3d}  by surface {dict(Counter(r['surface'] for r in fb))}")


# ── 2. RQ1/RQ3 -- the scan-plus-judge split ──────────────────────────────────
def test_judges():
    sec("2. RQ1 L43 / RQ3 L128-134 -- does the scan-plus-judge split buy BOTH "
        "precision and recall?")
    agg = defaultdict(lambda: defaultdict(list))
    for r in RUNS:
        for p in PROJECTS:
            g = gold(p)
            f = set(final_links(RUNDIR[r], p))
            _, rn = judged(RUNDIR[r], p, "linker_name")
            _, rc = judged(RUNDIR[r], p, "linker_coreference")
            for name, s in (("Full (scan+judges)", f), ("scan only (no judge)", f | rn | rc)):
                agg[name][p].append(prf(s, g))
        for p in PROJECTS:
            agg["Artemis (same backend)"][p].append(prf(artemis_dm(p, r[1]), gold(p)))
    for p in PROJECTS:
        agg["SWATTR (lexical)"][p].append(prf(swattr_dm(p), gold(p)))
    print(f"  {'system':24s} {'P':>7s} {'R':>7s} {'F1':>7s} {'F2':>7s}")
    for name in ("Full (scan+judges)", "scan only (no judge)",
                 "Artemis (same backend)", "SWATTR (lexical)"):
        v = [sum(sum(x[i] for x in runs) / len(runs) for runs in agg[name].values()) / 5
             for i in (3, 4, 5, 6)]
        print(f"  {name:24s} {v[0]:7.3f} {v[1]:7.3f} {v[2]:7.3f} {v[3]:7.3f}")

    print("\n  L131 'a judge ... can never add recall' -- per (run, project) cell,")
    print("  is the judges-on TP set a SUBSET of the judges-off TP set?")
    bad = 0
    for r in RUNS:
        for p in PROJECTS:
            g = gold(p)
            f = set(final_links(RUNDIR[r], p))
            _, rn = judged(RUNDIR[r], p, "linker_name")
            _, rc = judged(RUNDIR[r], p, "linker_coreference")
            bad += not ((f & g) <= ((f | rn | rc) & g))
    print(f"     {15 - bad}/15 cells hold. Violations: {bad}")

    sec("3. RQ3 L126/L132 -- do the two judges read a different KIND OF CASE, and is the "
        "coref judge's input 'the least precise'?")
    n = c = both = agree = disagree = 0.0
    stats = {}
    for lbl, ph, other in (("name", "linker_name", "linker_coreference"),
                           ("coref", "linker_coreference", "linker_name")):
        cand = kept = cand_g = excl = excl_g = 0.0
        for r in RUNS:
            for p in PROJECTS:
                g = gold(p)
                k, rj = judged(RUNDIR[r], p, ph)
                ko, ro = judged(RUNDIR[r], p, other)
                cs, co = k | rj, ko | ro
                cand += len(cs) / 3
                kept += len(k) / 3
                cand_g += len(cs & g) / 3
                excl += len(cs - co) / 3
                excl_g += len((cs - co) & g) / 3
        stats[lbl] = (cand, kept, cand_g, excl, excl_g)
    for r in RUNS:
        for p in PROJECTS:
            kn, rn = judged(RUNDIR[r], p, "linker_name")
            kc, rc = judged(RUNDIR[r], p, "linker_coreference")
            cn, cc = kn | rn, kc | rc
            n += len(cn) / 3
            c += len(cc) / 3
            both += len(cn & cc) / 3
            for k in cn & cc:
                if (k in kn) == (k in kc):
                    agree += 1 / 3
                else:
                    disagree += 1 / 3
    for lbl in ("name", "coref"):
        cand, kept, cand_g, excl, excl_g = stats[lbl]
        print(f"  {lbl + 'Validator':16s} candidates {cand:6.1f}  survive {kept:6.1f} "
              f"({100 * kept / cand:4.1f}%)   INPUT precision vs gold {100 * cand_g / cand:4.1f}%"
              f"   (on its EXCLUSIVE {excl:5.1f} candidates: {100 * excl_g / excl:4.1f}%)")
    print(f"\n  same (sentence, component) pair seen by BOTH judges: {both:.1f}"
          f"  = {100 * both / c:.1f}% of the coref judge's input, {100 * both / n:.1f}% of the name judge's")
    print(f"  on those shared cases the two judges agree {agree:.1f} / disagree {disagree:.1f}"
          f"  ({100 * disagree / both:.1f}% disagreement)")
    print(f"\n  per-case weight: name {9.93 / 300.7:.4f} pp F1 per candidate, "
          f"coref {8.10 / 174.0:.4f} pp F1 per candidate")


# ── 4. RQ4 -- the proposal forms ─────────────────────────────────────────────
def test_forms():
    sec("4. RQ4 L166 -- do the two smaller forms reach mentions the FullName scan "
        "'cannot quote in full'?")
    uniq = defaultdict(list)
    for r in RUNS:
        for p in PROJECTS:
            g, cat, sents = gold(p), catalog(p), sentences(p)
            al = aliases(RUNDIR[r], p)
            f = kept_by_form(RUNDIR[r], p)
            for name in f:
                others = set().union(*(f[o] for o in f if o != name))
                for (s, cid) in (f[name] - others) & g:
                    k, surf = classify(sents.get(s, ""), cat.get(cid, "?"), al)
                    uniq[name].append((p, s, cat.get(cid, "?"), k, sents.get(s, "")))
    for name in ("FullName", "PartialName", "Coref"):
        print(f"  {name:12s} unique TPs/run {len(uniq[name]) / 3:6.1f}   "
              f"surface classes (3 runs pooled): {dict(Counter(x[3] for x in uniq[name]))}")
    print("\n  the two smaller forms, deduplicated over runs -- does the sentence write the whole name?")
    for name in ("PartialName", "Coref"):
        for k in sorted({(x[0], x[1], x[2], x[3]) for x in uniq[name]}, key=lambda y: (y[0], y[1])):
            txt = next(x[4] for x in uniq[name] if (x[0], x[1], x[2], x[3]) == k)
            print(f"    {name:11s} {k[0]:14s} s{k[1]:<4d} {k[2][:20]:20s} {k[3]:9s} | {txt[:72]}")


# ── 5. RQ4 -- the knowledge module ───────────────────────────────────────────
def test_knowledge(rows):
    sec("5. RQ4 L178 -- 'an alias is the only route to a component the document never "
        "names canonically, and those components are small'")
    print("A. gold doc-model links the FULL arm has and the NO-KNOWLEDGE arm loses")
    lost = defaultdict(list)
    for r in RUNS:
        for p in PROJECTS:
            g, cat, sents = gold(p), catalog(p), sentences(p)
            al = aliases(RUNDIR[r], p)
            full = set(final_links(RUNDIR[r], p))
            nk = set(final_links(NOKNOW[r], p))
            for (s, cid) in (full & g) - nk:
                k, surf = classify(sents.get(s, ""), cat.get(cid, "?"), al)
                lost[(p, s, cat.get(cid, "?"), k, surf)].append(r)
    n = sum(len(v) for v in lost.values())
    print(f"   {n} losses over 3 runs, by surface class: "
          f"{dict(Counter(k[3] for k, v in lost.items() for _ in v))}")
    for k in sorted(lost, key=lambda x: (x[0], x[1])):
        print(f"     [{len(lost[k])}/3] {k[0]:14s} s{k[1]:<4d} {k[2][:20]:20s} "
              f"{k[3]:9s} {k[4][:18]:18s}")

    print("\nB. does the SAD ever write the canonical name of the components that collapse?")
    for p in ("mediastore", "teastore"):
        cat = catalog(p)
        sents = sentences(p)
        for cid, name in sorted(cat.items(), key=lambda x: x[1]):
            hits = [s for s, t in sents.items() if contains_seq(toks(t), toks(name))]
            print(f"     {p:12s} {name[:26]:26s} canonical name in the SAD? "
                  f"{'YES ' + str(hits[:5]) if hits else 'NO'}")

    print("\nC. per-component doc-code F1, Full vs no-knowledge, WITH component size")
    print(f"   {'project':14s} {'component':26s} {'gold links':>10s} {'share':>6s} "
          f"{'Full':>6s} {'noknow':>7s} {'delta':>7s}")
    hurt, fine = [], []
    for p in PROJECTS:
        code_files = m.load_code_model_files(p)
        names, _ = m.load_sam_code(p, code_files)
        acc = defaultdict(lambda: [0.0, 0.0, 0])
        for r in RUNS:
            a = per_component_dc(p, set(final_links(RUNDIR[r], p)))
            b = per_component_dc(p, set(final_links(NOKNOW[r], p)))
            for c, (f1, sz) in a.items():
                acc[c][0] += f1 / 3
                acc[c][1] += b.get(c, (0.0, sz))[0] / 3
                acc[c][2] = sz
        tot = sum(v[2] for v in acc.values())
        for c in sorted(acc, key=lambda x: acc[x][1] - acc[x][0]):
            f, nkf, sz = acc[c]
            share = 100 * sz / tot
            nm = names.get(c, c).replace("Component: ", "")
            (hurt if nkf - f < -0.05 else fine).append((p, nm, sz, share, f, nkf))
            if nkf - f < -0.05 or nkf - f > 0.05:
                print(f"   {p:14s} {nm[:26]:26s} {sz:10d} {share:5.1f}% "
                      f"{f:6.3f} {nkf:7.3f} {nkf - f:+7.3f}")
    print(f"\n   components LOSING >5pp without the alias table: n={len(hurt)}, "
          f"gold-link share min {min(x[3] for x in hurt):.1f}% "
          f"median {sorted(x[3] for x in hurt)[len(hurt) // 2]:.1f}% "
          f"max {max(x[3] for x in hurt):.1f}%")
    print(f"   components not hurt:                            n={len(fine)}, "
          f"median share {sorted(x[3] for x in fine)[len(fine) // 2]:.1f}%")

    print("\nD. per-project effect of removing the alias table (mean of 3 runs)")
    print(f"   {'project':14s} {'DM F1':>7s} {'d':>7s} {'DM F2':>7s} {'d':>7s} | "
          f"{'DC F1':>7s} {'d':>7s} {'worst':>7s} {'d':>7s}")
    for p in PROJECTS:
        g = gold(p)
        a = [0.0] * 4
        b = [0.0] * 4
        for r in RUNS:
            x = prf(set(final_links(RUNDIR[r], p)), g)
            y = prf(set(final_links(NOKNOW[r], p)), g)
            ca = m.compute_sad_code(p, R2.compose_doc_code(p, set(final_links(RUNDIR[r], p))))
            cb = m.compute_sad_code(p, R2.compose_doc_code(p, set(final_links(NOKNOW[r], p))))
            a = [a[0] + x[5] / 3, a[1] + x[6] / 3, a[2] + ca["file_f1"] / 3,
                 a[3] + ca["worst_component_f1"] / 3]
            b = [b[0] + y[5] / 3, b[1] + y[6] / 3, b[2] + cb["file_f1"] / 3,
                 b[3] + cb["worst_component_f1"] / 3]
        print(f"   {p:14s} {a[0]:7.3f} {b[0] - a[0]:+7.3f} {a[1]:7.3f} {b[1] - a[1]:+7.3f} | "
              f"{a[2]:7.3f} {b[2] - a[2]:+7.3f} {a[3]:7.3f} {b[3] - a[3]:+7.3f}")


# ── 6. RQ1 L62 / RQ2 L92 ─────────────────────────────────────────────────────
def test_inherit():
    sec("6. RQ1 L62 -- does the doc-code advantage 'inherit' the doc-model one? "
        "and RQ2 L92 -- is the link-level gap driven by the large components?")
    print(f"  {'project':14s} {'DM appr':>8s} {'DM Art':>7s} {'DM gain':>8s} | "
          f"{'DC appr':>8s} {'DC Art':>7s} {'DC gain':>8s}")
    pts = []
    for p in PROJECTS:
        def mean(tmpl, task, fn):
            return sum(fn(p, m.load_result(SOTA / tmpl.format(r=i, p=p), task))
                       for i in (1, 2, 3)) / 3
        dma = mean("model-doc/aalinker/terra_s126/run{r}/{p}.csv", "sad-sam",
                   lambda x, y: m.compute_sad_sam(x, y)["link_f1"])
        dmb = mean("model-doc/artemis/terra_5.6/run{r}/{p}.csv", "sad-sam",
                   lambda x, y: m.compute_sad_sam(x, y)["link_f1"])
        dca = mean("doc-code/aalinker-composed/terra_s126/run{r}/{p}.csv", "sad-code",
                   lambda x, y: m.compute_sad_code(x, y)["file_f1"])
        dcb = mean("doc-code/artemis/terra_5.6/run{r}/{p}.csv", "sad-code",
                   lambda x, y: m.compute_sad_code(x, y)["file_f1"])
        pts.append((dma - dmb, dca - dcb))
        print(f"  {p:14s} {dma:8.3f} {dmb:7.3f} {dma - dmb:+8.3f} | "
              f"{dca:8.3f} {dcb:7.3f} {dca - dcb:+8.3f}")
    mx = sum(a for a, _ in pts) / 5
    my = sum(b for _, b in pts) / 5
    num = sum((a - mx) * (b - my) for a, b in pts)
    den = (sum((a - mx) ** 2 for a, _ in pts) * sum((b - my) ** 2 for _, b in pts)) ** .5
    print(f"  Pearson r(doc-model gain, doc-code gain) = {num / den:+.3f}  (n=5, no significance)")
    print(f"  doc-model gain positive on {sum(a > 0 for a, _ in pts)}/5, "
          f"doc-code gain positive on {sum(b > 0 for _, b in pts)}/5")

    print("\n  gold doc-code link mass by component size, and the per-component gap vs Artemis")
    print("  (BOTH systems composed through the SAME sam-code map here, so this isolates the\n   doc-model difference; Artemis's own released doc-code dump is what the table above uses)")
    for p in PROJECTS:
        code_files = m.load_code_model_files(p)
        names, _ = m.load_sam_code(p, code_files)
        sizes = {c: n for c, (_f, n) in per_component_dc(
            p, set(final_links(RUNDIR["r1"], p))).items()}
        tot = sum(sizes.values())
        A = defaultdict(float)
        B = defaultdict(float)
        for r in RUNS:
            for c, (f1, _n) in per_component_dc(p, set(final_links(RUNDIR[r], p))).items():
                A[c] += f1 / 3
            for c, (f1, _n) in per_component_dc(p, artemis_dm(p, r[1])).items():
                B[c] += f1 / 3
        order = sorted(sizes, key=lambda c: -sizes[c])
        top = order[:2]
        rest = order[2:]
        dtop = sum(A[c] - B[c] for c in top) / len(top)
        drest = sum(A[c] - B[c] for c in rest) / len(rest) if rest else 0.0
        print(f"    {p:14s} top-2 carry {100 * sum(sizes[c] for c in top) / tot:5.1f}% of link mass, "
              f"mean per-component gap there {dtop:+.3f}; on the other "
              f"{len(rest)} components {drest:+.3f}")



# ── 7. the three challenges of approach.tex L9-27 ────────────────────────────
def test_challenges(rows):
    """Evidence indexed by CHALLENGE rather than by RQ.

    approach.tex L9-27 states three: (CH1) links are carried by various reference
    forms; (CH2) a recovered link can read as plausible with no support in the
    sentence; (CH3) the project vocabulary coins aliases as the document goes.
    """
    g = [r for r in rows if r["gold"]]
    CL = ("canonical", "alias", "partial_d", "partial_g", "none")

    sec("7. CH1 -- do the gold links really span the reference forms, and does each "
        "module reach the form it was designed for?")
    print(f"  {'the sentence writes':22s} {'gold':>5s} | {'SWATTR':>9s} {'Artemis':>9s} "
          f"{'approach':>9s} | proposed by")
    label = {"canonical": "the whole name", "alias": "a document alias",
             "partial_d": "a distinctive word", "partial_g": "a generic word",
             "none": "no name at all"}
    prop = defaultdict(Counter)
    for r in g:
        if r["approach_runs"] >= 2:
            prop[r["surface"]][r["approach_forms"]] += 1
    for k in CL:
        sub = [r for r in g if r["surface"] == k]
        if not sub:
            continue
        n = len(sub)
        print(f"  {label[k]:22s} {n:5d} | {sum(r['swattr'] for r in sub):4d}/{n:<4d} "
              f"{sum(r['artemis_runs'] >= 2 for r in sub):4d}/{n:<4d} "
              f"{sum(r['approach_runs'] >= 2 for r in sub):4d}/{n:<4d} | {dict(prop[k])}")
    nn = [r for r in g if r["surface"] != "canonical"]
    print(f"\n  gold links that do NOT write the canonical name: {len(nn)}/{len(g)} "
          f"= {100 * len(nn) / len(g):.1f}%")
    for lbl, f in (("SWATTR", lambda r: r["swattr"]),
                   ("Artemis", lambda r: r["artemis_runs"] >= 2),
                   ("approach", lambda r: r["approach_runs"] >= 2)):
        hit = sum(1 for r in nn if f(r))
        print(f"    recall there: {lbl:9s} {hit:3d}/{len(nn)} = {100 * hit / len(nn):5.1f}%")
    print("\n  NOTE the `alias` row is classified with run 1's knowledge table, which is"
          "\n  rediscovered per run; four mediastore links read `none` here and `alias` under"
          "\n  their own run's table (AudioAccess, DataStorage).")

    sec("8. CH2 -- what shape are the links the judges remove?")
    rej = Counter()
    per = defaultdict(Counter)
    for r in RUNS:
        for p in PROJECTS:
            gl, cat, sents = gold(p), catalog(p), sentences(p)
            al = aliases(RUNDIR[r], p)
            for ph, lbl in (("linker_name", "nameValidator"),
                            ("linker_coreference", "corefValidator")):
                _k, rj = judged(RUNDIR[r], p, ph)
                for (s, cid) in rj - gl:
                    k, _ = classify(sents.get(s, ""), cat.get(cid, "?"), al)
                    rej[k] += 1 / 3
                    per[lbl][k] += 1 / 3
    tot = sum(rej.values())
    print(f"  rejected false positives per run: {tot:.1f}, by what the sentence contains")
    for k in CL:
        if rej[k]:
            print(f"    {label[k]:22s} {rej[k]:6.1f}  ({100 * rej[k] / tot:4.1f}%)")
    present = tot - rej["none"]
    print(f"  a surface form of the component IS present for {100 * present / tot:.1f}% of them"
          f" -- the 'plausible' shape CH2 names")
    for lbl in per:
        print(f"    {lbl:15s} {dict((k, round(v, 1)) for k, v in per[lbl].items())}")

    print("\n  the running example of motivation.tex (fig:example), mediastore:")
    cat = catalog("mediastore")
    print(f"    S27 {sentences('mediastore')[27]}")
    print(f"        gold: {sorted(cat.get(c, c) for (s, c) in gold('mediastore') if s == 27)}"
          f"  -- the DB link is the paper's false-positive example")
    for r in RUNS:
        _kn, rn = judged(RUNDIR[r], "mediastore", "linker_name")
        _kc, rc = judged(RUNDIR[r], "mediastore", "linker_coreference")
        fin = set(final_links(RUNDIR[r], "mediastore"))
        print(f"        {r}: rejected {sorted(cat.get(c, c) for (s, c) in (rn | rc) if s == 27)}"
              f" | emitted {sorted(cat.get(c, c) for (s, c) in fin if s == 27)}")
    print(f"    S24 {sentences('mediastore')[24]}")
    print(f"        gold: {sorted(cat.get(c, c) for (s, c) in gold('mediastore') if s == 24)}"
          f"  -- the pronoun example")
    for r in RUNS:
        f = kept_by_form(RUNDIR[r], "mediastore")
        who = sorted(k for k, v in f.items() if any(s == 24 for (s, _c) in v))
        print(f"        {r}: emitted by {who}")

    sec("9. CH1 honest datum -- the weakest cell: a distinctive word of a multi-word name")
    for r in rows:
        if r["gold"] and r["surface"] == "partial_d" and r["approach_runs"] < 2:
            txt = r["text"]
            whole = [c for c in set(catalog(r["project"]).values())
                     if contains_seq(toks(txt), toks(c))]
            print(f"  {r['project']:14s} s{r['sentence']:<4d} {r['component'][:18]:18s} "
                  f"writes '{r['matched']}'; SWATTR gets it: {bool(r['swattr'])}; "
                  f"whole names also in the sentence: {whole}")
    print("  The word scan skips any sentence that already writes a whole name"
          " (fig:approach-overview),\n  which is exactly what costs bigbluebutton s6.")




# ── 10. mechanism attribution ────────────────────────────────────────────────
def test_attribution(rows):
    """A 2x2 over {alias table on/off} x {judges on/off}.

    Both arms recorded per-phase state, so all four cells are measurable; only two
    of them appear in any committed table. This is what lets success be traced to a
    mechanism rather than to the arm as a whole.
    """
    sec("10. MECHANISM ATTRIBUTION -- doc-model, macro over 5 projects, mean of 3 runs")

    def conf(run_map, judges):
        out = [0.0] * 4
        for p in PROJECTS:
            g = gold(p)
            acc = [0.0] * 4
            for r in RUNS:
                rd = run_map[r]
                s = set(final_links(rd, p))
                if not judges:
                    _k1, r1 = judged(rd, p, "linker_name")
                    _k2, r2 = judged(rd, p, "linker_coreference")
                    s |= r1 | r2
                v = prf(s, g)
                acc = [acc[i] + v[j] / 3 for i, j in enumerate((3, 4, 5, 6))]
            out = [out[i] + acc[i] / 5 for i in range(4)]
        return out

    print(f"  {'configuration':34s} {'P':>7s} {'R':>7s} {'F1':>7s} {'F2':>7s}")
    grid = {}
    for know, rm in (("alias table ON", RUNDIR), ("alias table OFF", NOKNOW)):
        for jd, lbl in ((True, "judges ON"), (False, "judges OFF")):
            v = conf(rm, jd)
            grid[(know, lbl)] = v
            print(f"  {know + ', ' + lbl:34s} {v[0]:7.3f} {v[1]:7.3f} {v[2]:7.3f} {v[3]:7.3f}")
    for name, sets in (("Artemis (same backend)", lambda p: [artemis_dm(p, r[1]) for r in RUNS]),
                       ("SWATTR (lexical)", lambda p: [swattr_dm(p)])):
        out = [0.0] * 4
        for p in PROJECTS:
            g = gold(p)
            ss = sets(p)
            for s in ss:
                v = prf(s, g)
                out = [out[i] + v[j] / len(ss) / 5 for i, j in enumerate((3, 4, 5, 6))]
        print(f"  {name:34s} {out[0]:7.3f} {out[1]:7.3f} {out[2]:7.3f} {out[3]:7.3f}")

    a = grid[("alias table ON", "judges ON")][2]
    b = grid[("alias table ON", "judges OFF")][2]
    c = grid[("alias table OFF", "judges ON")][2]
    d = grid[("alias table OFF", "judges OFF")][2]
    print("\n  F1 main effects and interaction")
    print(f"    judges, WITH the alias table : {100 * (a - b):+6.1f}pp")
    print(f"    judges, without it           : {100 * (c - d):+6.1f}pp")
    print(f"    alias table, WITH judges     : {100 * (a - c):+6.1f}pp")
    print(f"    alias table, without judges  : {100 * (b - d):+6.1f}pp   <-- the whole point")
    print(f"    interaction (non-additivity) : {100 * ((a - c) - (b - d)):+6.1f}pp")
    print("    i.e. the alias table's ENTIRE F1 contribution is interaction with the judges:")
    print("    on its own it trades recall for precision at par.")

    print("\n  where each mechanism's recall contribution lands, by reference form")
    g = [r for r in rows if r["gold"]]
    label = {"canonical": "the whole name", "alias": "a document alias",
             "partial_d": "a distinctive word", "partial_g": "a generic word",
             "none": "no name at all"}
    print(f"  {'the sentence writes':22s} {'gold':>5s} {'alias ON':>9s} {'alias OFF':>10s} "
          f"{'Artemis':>8s} {'SWATTR':>7s}")
    for k in ("canonical", "alias", "partial_d", "partial_g", "none"):
        sub = [r for r in g if r["surface"] == k]
        if not sub:
            continue
        print(f"  {label[k]:22s} {len(sub):5d} {sum(r['approach_runs'] >= 2 for r in sub):9d} "
              f"{sum(int(r['noknow_runs']) >= 2 for r in sub):10d} "
              f"{sum(r['artemis_runs'] >= 2 for r in sub):8d} {sum(r['swattr'] for r in sub):7d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-only", action="store_true")
    ap.add_argument("--sheets", metavar="DIR",
                    help="also emit the CH1/CH2 per-item sheets into DIR")
    a = ap.parse_args()
    rows = build_items()
    out = write_items(rows)
    print(f"wrote {out} ({len(rows)} rows, {sum(r['gold'] for r in rows)} of them gold)")
    if a.sheets:
        write_sheets(rows, Path(a.sheets))
        write_mode2_sheet(Path(a.sheets))
    if a.csv_only:
        return 0
    gate()
    test_rq1(rows)
    test_judges()
    test_forms()
    test_knowledge(rows)
    test_inherit()
    test_challenges(rows)
    test_attribution(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
