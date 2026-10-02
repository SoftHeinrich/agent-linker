#!/usr/bin/env python3
"""Recompute every run-derived number the paper's prose quotes and assert it appears.

    python3 verification/paper_number_audit.py [--paper paper]

Each check recomputes a quantity from the committed table CSVs (the files
`sync_paper.py` copies into the paper) or from `evaluation/reports/`, formats it the
way the prose writes it, and asserts that exact string occurs in the named file.
Percentage-point gaps are differences of the unrounded scores, rounded afterwards.
Exit 1 if any check fails. Baseline-only, gold-only and literature figures are listed
at the end as out of scope; this script does not check them.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def rows(path):
    return list(csv.DictReader(open(path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=str(REPO / "paper"))
    paper = Path(ap.parse_args().paper)
    text = {}

    def src(name):
        if name not in text:
            text[name] = (paper / name).read_text()
        return text[name]

    fails = []

    def check(name, snippet, why):
        body = src(name)
        # LaTeX line breaks inside a sentence are irrelevant to the claim.
        if snippet not in body and re.sub(r"\s+", " ", snippet) not in re.sub(r"\s+", " ", body):
            fails.append(f"{name}: missing {snippet!r}  ({why})")
        else:
            print(f"ok   {name:26s} {snippet[:70]!r}")

    f2 = lambda x: f"{x:.2f}"
    f3 = lambda x: f"{x:.3f}"
    pp = lambda a, b: f"{round(100 * (a - b))}"

    rq1 = {(r["project"], r["task"]): r for r in rows(paper / "table/rq1-results.csv")}
    g = lambda p, t, k: float(rq1[(p, t)][k])
    A, B = "Average", "approach"
    dm = lambda p, s, m: g(p, "DM", f"{s}_{m}")
    dc = lambda p, s, m: g(p, "DC", f"{s}_{m}")
    rq2 = {}
    for r in rows(paper / "table/rq2-results.csv"):
        rq2[(r["left_project"], r["system"])] = {k[5:]: r[k] for k in r if k.startswith("left_")}
        rq2[(r["right_project"], r["system"])] = {k[6:]: r[k] for k in r if k.startswith("right_")}
    q2 = lambda p, s, k: float(rq2[(p, s)][k])
    rq3 = {r["judge"]: r for r in rows(paper / "table/rq3-confusion.csv")}
    rq4 = {r["variant"]: r for r in rows(paper / "table/rq4-results.csv")}
    cost = {r["system"]: r for r in rows(paper / "table/inference-cost.csv")}
    big = {(r["system"], r["run"]): r for r in rows(REPO / "evaluation/reports/RQ12_BIGTABLE.csv")}
    per = {(r["system"], r["project"]): r for r in rows(REPO / "evaluation/reports/RQ12_PERPROJECT.csv")}

    R = "sections/results.tex"
    # --- RQ1, terra -----------------------------------------------------------
    leads = {p: dm(p, B, "f1") - dm(p, "Artemis", "f1")
             for p in ("mediastore", "teastore", "teammates", "bigbluebutton", "jabref")}
    small = min(leads, key=leads.get)
    check(R, f"The gain is smallest on JabRef, where the doc-model \\fone{{}} differs by $+{pp(dm('jabref',B,'f1'),dm('jabref','Artemis','f1'))}$pp and \\ftwo{{}} by $+{pp(dm('jabref',B,'f2'),dm('jabref','Artemis','f2'))}$pp.", f"smallest lead is {small}")
    check(R, f"ranges from $+{pp(dm('mediastore',B,'f1'),dm('mediastore','Artemis','f1'))}$pp (MediaStore) to $+{pp(dm('teastore',B,'f1'),dm('teastore','Artemis','f1'))}$pp (TeaStore)", "per-project lead range")
    check(R, f"from \\Artemis{{}}'s ${f2(dm(A,'Artemis','f1'))}$ to ${f2(dm(A,B,'f1'))}$, a gain of $+{pp(dm(A,B,'f1'),dm(A,'Artemis','f1'))}$pp", "avg DM F1")
    check(R, f"the lexical SWATTR pipeline reaches ${f2(dm(A,'pipeline','f1'))}$", "SWATTR DM F1")
    check(R, f"${f2(dm(A,'Artemis','f2'))} \\rightarrow {f2(dm(A,B,'f2'))}$ ($+{pp(dm(A,B,'f2'),dm(A,'Artemis','f2'))}$pp)", "avg DM F2")
    check(R, f"precision ${f2(dm(A,B,'p'))}$ against ${f2(dm(A,'Artemis','p'))}$ ($+{pp(dm(A,B,'p'),dm(A,'Artemis','p'))}$pp) and recall ${f2(dm(A,B,'r'))}$ against ${f2(dm(A,'Artemis','r'))}$ ($+{pp(dm(A,B,'r'),dm(A,'Artemis','r'))}$pp)", "DM P/R")
    check(R, f"leads by $+{pp(dc(A,B,'p'),dc(A,'Artemis','p'))}$pp precision and $+{pp(dc(A,B,'r'),dc(A,'Artemis','r'))}$pp recall", "DC P/R gaps")
    check(R, f"from \\Artemis{{}}'s ${f2(dc(A,'Artemis','f1'))}$ to ${f2(dc(A,B,'f1'))}$ ($+{pp(dc(A,B,'f1'),dc(A,'Artemis','f1'))}$pp) and the \\avgftwo\\ from ${f2(dc(A,'Artemis','f2'))}$ to ${f2(dc(A,B,'f2'))}$ ($+{pp(dc(A,B,'f2'),dc(A,'Artemis','f2'))}$pp)", "DC F1/F2")
    check(R, f"pipeline reaches ${f2(dc(A,'pipeline','f1'))}$ and ${f2(dc(A,'pipeline','f2'))}$", "TransArC DC")
    projects = ("mediastore", "teastore", "teammates", "bigbluebutton", "jabref")
    over_art = sum(dc(p, B, "f1") > dc(p, "Artemis", "f1") for p in projects)
    over_ta = sum(dc(p, B, "f1") > dc(p, "pipeline", "f1") for p in projects)
    words = {5: "all five", 4: "four of the five", 3: "three"}
    check(R, f"The direction holds on {words[over_art]} projects over \\Artemis and {words[over_ta]} over \\TransArc.", f"DC direction {over_art}/{over_ta}")
    check(R, f"a $+{pp(dm('bigbluebutton',B,'f1'),dm('bigbluebutton','pipeline','f1'))}$pp doc-model \\fone gain becomes a ${pp(dc('bigbluebutton',B,'f1'),dc('bigbluebutton','pipeline','f1'))}$pp doc-code \\fone decrease, with doc-code precision ${f2(dc('bigbluebutton',B,'p'))}$ against ${f2(dc('bigbluebutton','pipeline','p'))}$", "BBB vs TransArC")
    check(R, f"\\approach{{}} leads \\Artemis{{}} by ${pp(dm(A,B,'f1'),dm(A,'Artemis','f1'))}$pp doc-model and ${pp(dc(A,B,'f1'),dc(A,'Artemis','f1'))}$pp doc-code", "RQ1 box")
    check(R, f"Corresponding gaps to SWATTR and \\TransArc{{}} are ${pp(dm(A,B,'f1'),dm(A,'pipeline','f1'))}$pp and ${pp(dc(A,B,'f1'),dc(A,'pipeline','f1'))}$pp", "RQ1 box baselines")

    # --- luna (RQ12_BIGTABLE / PERPROJECT) --------------------------------------
    al, rl = big[("approach (GPT-5.6-luna)", "average")], big[("Artemis (GPT-5.6-luna)", "average")]
    at, rt = big[("approach (GPT-5.6-terra)", "average")], big[("Artemis (GPT-5.6-terra)", "average")]
    v = lambda r, k: float(r[k])
    check(R, f"doc-model \\avgfone{{}} of ${f2(v(al,'doc_to_model_link_f1'))}$ against \\Artemis{{}}'s ${f2(v(rl,'doc_to_model_link_f1'))}$ ($+{pp(v(al,'doc_to_model_link_f1'),v(rl,'doc_to_model_link_f1'))}$pp), and a doc-code \\avgfone{{}} of ${f2(v(al,'doc_to_code_file_f1'))}$ against ${f2(v(rl,'doc_to_code_file_f1'))}$ ($+{pp(v(al,'doc_to_code_file_f1'),v(rl,'doc_to_code_file_f1'))}$pp)", "luna same-backend")
    lo = lambda s, k: min(float(big[(s, f"run{i}")][k]) for i in (1, 2, 3))
    hi = lambda s, k: max(float(big[(s, f"run{i}")][k]) for i in (1, 2, 3))
    sep = all(lo("approach (GPT-5.6-luna)", k) > hi("Artemis (GPT-5.6-luna)", k)
              for k in ("doc_to_model_link_f1", "doc_to_code_file_f1"))
    if sep:
        check(R, "the lowest of \\approach{}'s three luna runs exceeds the highest of \\Artemis{}'s", "run separation")
    else:
        fails.append("run-separation claim no longer holds")
    check(R, f"\\approach{{}} by ${pp(v(at,'doc_to_model_link_f1'),v(al,'doc_to_model_link_f1'))}$pp doc-model \\avgfone{{}} (${f2(v(at,'doc_to_model_link_f1'))} \\rightarrow {f2(v(al,'doc_to_model_link_f1'))}$), mostly in precision (${f2(v(at,'doc_to_model_link_precision'))} \\rightarrow {f2(v(al,'doc_to_model_link_precision'))}$), and \\Artemis{{}} by ${pp(v(rt,'doc_to_model_link_f1'),v(rl,'doc_to_model_link_f1'))}$pp (${f2(v(rt,'doc_to_model_link_f1'))} \\rightarrow {f2(v(rl,'doc_to_model_link_f1'))}$)", "terra vs luna")
    check(R, f"from $+{pp(v(al,'doc_to_code_file_f1'),v(rl,'doc_to_code_file_f1'))}$pp link-level to $+{pp(v(al,'doc_to_code_worst_component_f1'),v(rl,'doc_to_code_worst_component_f1'))}$pp worst-component and $+{pp(v(al,'doc_to_code_harmonic_component_f1'),v(rl,'doc_to_code_harmonic_component_f1'))}$pp harmonic", "luna RQ2")
    check(R, f"\\cmrname{{}} is ${v(al,'doc_to_model_component_miss_rate'):.1f}\\%$ against \\Artemis{{}}'s ${v(rl,'doc_to_model_component_miss_rate'):.1f}\\%$", "luna CMR")

    # --- inference cost -------------------------------------------------------------
    ai, ao = float(cost[B]["Total_input_k"]), float(cost[B]["Total_output_k"])
    ri, ro = float(cost["Artemis"]["Total_input_k"]), float(cost["Artemis"]["Total_output_k"])
    check(R, f"used {ai:.1f} thousand input tokens", "approach input")
    check(R, f"Its output total was {ao:.1f} thousand tokens", "approach output")
    check(R, f"\\Artemis{{}} used {ri:.1f} thousand input tokens and {ro:.1f} thousand output tokens", "Artemis tokens")
    check(R, f"consumes {ai/ri:.1f}$\\times$ more input tokens", "input ratio")
    usd = lambda i, o: (i * 2 + o * 12) / 1000
    check(R, f"approximately US\\${usd(ai,ao):.2f} for \\approach{{}} and US\\${usd(ri,ro):.2f} for \\Artemis{{}}", "cost")
    check("sections/discussion.tex", f"approximately US\\$${usd(ai,ao):.2f}$", "discussion cost")
    check("sections/discussion.tex", f"US\\$${usd(ri,ro):.2f}$", "discussion Artemis cost")

    # --- RQ2 --------------------------------------------------------------------------
    cmr = lambda s: q2(A, s, "dm_cmr")
    check(R, f"${cmr(B):.1f}\\%$ for \\approach{{}} against ${cmr('Artemis'):.1f}\\%$ and ${cmr('TransArC'):.1f}\\%$", "CMR")
    check(R, f"\\approach{{}} leads \\Artemis{{}} by $+{pp(q2(A,B,'dc_file_f1'),q2(A,'Artemis','dc_file_f1'))}$pp after rounding", "RQ2 link gap")
    check(R, f"worst-component \\fone\\ of ${f2(q2(A,B,'dc_worst_f1'))}$ against \\Artemis{{}}'s ${f2(q2(A,'Artemis','dc_worst_f1'))}$", "worst")
    check(R, f"($+{pp(q2(A,B,'dc_worst_f1'),q2(A,'Artemis','dc_worst_f1'))}$pp), and a harmonic per-component \\fone\\ of ${f2(q2(A,B,'dc_harm_f1'))}$ against ${f2(q2(A,'Artemis','dc_harm_f1'))}$ ($+{pp(q2(A,B,'dc_harm_f1'),q2(A,'Artemis','dc_harm_f1'))}$pp)", "harmonic")
    check(R, f"$+{pp(q2(A,B,'dc_file_f1'),q2(A,'TransArC','dc_file_f1'))}$pp link-level, $+{pp(q2(A,B,'dc_worst_f1'),q2(A,'TransArC','dc_worst_f1'))}$pp worst-component, $+{pp(q2(A,B,'dc_harm_f1'),q2(A,'TransArC','dc_harm_f1'))}$pp harmonic", "vs TransArC")
    check(R, f"despite a doc-code \\avgfone{{}} of ${f2(q2(A,B,'dc_file_f1'))}$, reaches a worst-component \\fone{{}} of only ${f2(q2('bigbluebutton',B,'dc_worst_f1'))}$ on BigBlueButton and ${f2(q2('teammates',B,'dc_worst_f1'))}$ on Teammates", "headroom")
    check(R, f"on JabRef, \\TransArc{{}} exceeds \\approach{{}} on file \\fone{{}} (${f2(q2('jabref','TransArC','dc_file_f1'))}$ versus ${f2(q2('jabref',B,'dc_file_f1'))}$), but trails on worst-component \\fone{{}} (${f2(q2('jabref','TransArC','dc_worst_f1'))}$ versus ${f2(q2('jabref',B,'dc_worst_f1'))}$)", "JabRef reversal")
    check(R, f"\\Artemis{{}} exceeds \\TransArc{{}} on file \\fone{{}} (${f2(q2(A,'Artemis','dc_file_f1'))}$ versus ${f2(q2(A,'TransArC','dc_file_f1'))}$), but trails on worst-component \\fone{{}} (${f2(q2(A,'Artemis','dc_worst_f1'))}$ versus ${f2(q2(A,'TransArC','dc_worst_f1'))}$)", "macro reversal")
    check(R, f"the standard \\fone\\ gap is $+{pp(q2(A,B,'dc_file_f1'),q2(A,'Artemis','dc_file_f1'))}$pp; the size-aware suite opens it to $+{pp(q2(A,B,'dc_worst_f1'),q2(A,'Artemis','dc_worst_f1'))}$pp worst-component and $+{pp(q2(A,B,'dc_harm_f1'),q2(A,'Artemis','dc_harm_f1'))}$pp harmonic", "RQ2 box")

    # --- RQ3 --------------------------------------------------------------------------
    n, c, o, z = rq3["name"], rq3["coref"], rq3["full_on"], rq3["no_judge"]
    x = lambda r, k: float(r[k])
    check(R, f"rejects ${x(n,'rej_fp'):.1f}$ false positives while costing ${x(n,'rej_tp'):.1f}$ true trace links", "name judge")
    check(R, f"rejects ${x(c,'rej_fp'):.1f}$ against ${x(c,'rej_tp'):.1f}$", "coref judge")
    check(R, f"reject ${x(o,'rej_fp'):.1f}$ distinct false positives and lose ${x(o,'rej_tp'):.1f}$ true links", "both judges")
    check(R, f"from ${f2(x(o,'dm_f1'))}$ to ${f2(x(z,'dm_f1'))}$, a loss of ${pp(x(o,'dm_f1'),x(z,'dm_f1'))}$pp, but link-level \\ftwo\\ only from ${f2(x(o,'dm_f2'))}$ to ${f2(x(z,'dm_f2'))}$, a loss of ${pp(x(o,'dm_f2'),x(z,'dm_f2'))}$pp", "judges off")
    check(R, f"costs ${pp(x(o,'dm_f1'),x(n,'dm_f1'))}$ and ${pp(x(o,'dm_f1'),x(c,'dm_f1'))}$pp of \\fone\\ (\\judgeOne{{}}, \\judgeTwo{{}}) against ${pp(x(o,'dm_f2'),x(n,'dm_f2'))}$ and ${pp(x(o,'dm_f2'),x(c,'dm_f2'))}$pp", "one at a time")
    check(R, f"precision falls by ${pp(x(o,'dm_p'),x(z,'dm_p'))}$pp (${f2(x(o,'dm_p'))} \\rightarrow {f2(x(z,'dm_p'))}$) while recall rises by only ${pp(x(z,'dm_r'),x(o,'dm_r'))}$pp (${f2(x(o,'dm_r'))} \\rightarrow {f2(x(z,'dm_r'))}$)", "P/R judges off")
    check("sections/discussion.tex", f"raises recall by ${pp(x(z,'dm_r'),x(o,'dm_r'))}$\\,pp but lowers precision by ${pp(x(o,'dm_p'),x(z,'dm_p'))}$\\,pp", "discussion judges")
    check(R, f"reject ${x(o,'rej_fp'):.1f}$ distinct false positives while losing ${x(o,'rej_tp'):.1f}$ true links; removing both costs ${pp(x(o,'dm_f1'),x(z,'dm_f1'))}$pp link-level \\fone\\ but only ${pp(x(o,'dm_f2'),x(z,'dm_f2'))}$pp", "RQ3 box")

    # --- RQ4 --------------------------------------------------------------------------
    F, N, C, K = rq4["Full"], rq4["Name"], rq4["Coref"], rq4["No knowledge"]
    y = lambda r, k: float(r[k])
    MF1, MF2, MP, MR = ("doc_to_model_macro_f1", "doc_to_model_macro_f2",
                        "doc_to_model_macro_precision", "doc_to_model_macro_recall")
    check(R, f"\\routeOne{{}} alone reaches a doc-model \\avgfone\\ of ${f2(y(N,MF1))}$ and \\avgftwo\\ of ${f2(y(N,MF2))}$", "name alone")
    check(R, f"raises them to ${f2(y(F,MF1))}$ and ${f2(y(F,MF2))}$", "full")
    check(R, f"recall rises from ${f2(y(N,MR))}$ to ${f2(y(F,MR))}$ while precision is ${f2(y(N,MP))}$ with \\routeOne{{}} alone and ${f2(y(F,MP))}$ with both", "route recall")
    g2, g1 = y(F, MF2) - y(N, MF2), y(F, MF1) - y(N, MF1)
    check(R, f"Its ${round(100*g2)}$pp \\ftwo{{}} gain is about ${g2/g1:.1f}$ times its ${round(100*g1)}$pp \\fone{{}} gain", "ratio")
    check(R, f"contributes ${int(float(N['unique_tps']))}$ true positives no other route reaches and \\routeTwo{{}} another ${int(float(C['unique_tps']))}$", "unique TPs")
    W, H, DF = "dc_worst_component_f1", "dc_harmonic_component_f1", "dc_file_f1"
    check(R, f"worst-component \\fone\\ falls from ${f2(y(F,W))}$ to ${f2(y(N,W))}$ and the harmonic per-component \\fone\\ from ${f2(y(F,H))}$ to ${f2(y(N,H))}$, against ${pp(y(F,DF),y(N,DF))}$pp", "name-only components")
    check(R, f"${f2(y(C,MF1))}$ \\avgfone\\ at a ${y(C,'doc_to_model_component_miss_rate'):.1f}\\%$", "coref alone")
    check(R, f"is ${pp(y(F,MF1),y(K,MF1))}$\\,pp lower in \\avgfone\\ (${f2(y(F,MF1))} \\rightarrow {f2(y(K,MF1))}$) and ${pp(y(F,MF2),y(K,MF2))}$\\,pp \\avgftwo\\ (${f2(y(F,MF2))} \\rightarrow {f2(y(K,MF2))}$)", "no knowledge")
    check(R, f"recall falls from ${f2(y(F,MR))}$ to ${f2(y(K,MR))}$", "no-knowledge recall")
    check(R, f"worst-component \\fone\\ falls from ${f2(y(F,W))}$ to ${f2(y(K,W))}$ (${pp(y(K,W),y(F,W))}$pp) and the harmonic per-component \\fone\\ from ${f2(y(F,H))}$ to ${f2(y(K,H))}$ (${pp(y(K,H),y(F,H))}$pp), against ${pp(y(K,DF),y(F,DF))}$pp of doc-code file \\fone, and the \\cmrname{{}} rises from ${y(F,'doc_to_model_component_miss_rate'):.1f}\\%$ to ${y(K,'doc_to_model_component_miss_rate'):.1f}\\%$", "no-knowledge components")
    check(R, f"\\avgfone\\ from ${f2(y(N,MF1))}$ to ${f2(y(F,MF1))}$ and \\avgftwo\\ from ${f2(y(N,MF2))}$ to ${f2(y(F,MF2))}$", "RQ4 box routes")
    check(R, f"differs by ${pp(y(F,MF1),y(K,MF1))}$pp \\avgfone\\ and ${pp(y(F,MF2),y(K,MF2))}$pp \\avgftwo{{}}", "RQ4 box knowledge")
    check(R, f"by ${pp(y(F,W),y(K,W))}$pp of worst-component \\fone", "RQ4 box worst")
    check("sections/conclusion.tex", f"worst-component \\fone{{}} falls by ${pp(y(F,W),y(K,W))}$\\,pp", "conclusion knowledge")

    # --- abstract / intro / conclusion / discussion headline ----------------------
    for name in ("sections/intro.tex", "sections/conclusion.tex"):
        check(name, f"doc-model \\avgfone{'' if name.endswith('intro.tex') else '{}'} of ${f3(dm(A,B,'f1'))}$, ${pp(dm(A,B,'f1'),dm(A,'Artemis','f1'))}$\\,pp above", "headline DM")
    check("sections/intro.tex", f"${f3(dm(A,B,'f2'))}$ \\avgftwo, ${pp(dm(A,B,'f2'),dm(A,'Artemis','f2'))}$\\,pp above", "headline F2")
    check("sections/intro.tex", f"\\avgfone of ${f3(dc(A,B,'f1'))}$ ($+{pp(dc(A,B,'f1'),dc(A,'Artemis','f1'))}$\\,pp) and an \\avgftwo of ${f3(dc(A,B,'f2'))}$ ($+{pp(dc(A,B,'f2'),dc(A,'Artemis','f2'))}$\\,pp)", "headline DC")
    check("sections/intro.tex", f"leads \\TransArc{{}} by ${pp(q2(A,B,'dc_harm_f1'),q2(A,'TransArC','dc_harm_f1'))}$\\,pp on the per-component harmonic mean, against ${pp(q2(A,B,'dc_file_f1'),q2(A,'TransArC','dc_file_f1'))}$\\,pp at the link level", "intro vs TransArC")
    check("sections/intro.tex", f"by ${pp(dm(A,B,'f1'),dm(A,'Artemis','f1'))}$\\,pp \\fone\\ (${pp(dm(A,B,'f2'),dm(A,'Artemis','f2'))}$\\,pp \\ftwo) on doc-model and ${pp(dc(A,B,'f1'),dc(A,'Artemis','f1'))}$\\,pp \\fone\\ (${pp(dc(A,B,'f2'),dc(A,'Artemis','f2'))}$\\,pp \\ftwo) on doc-code", "contributions")
    check("sections/conclusion.tex", f"doc-code \\avgfone{{}} of ${f3(dc(A,B,'f1'))}$ ($+{pp(dc(A,B,'f1'),dc(A,'Artemis','f1'))}$\\,pp)", "conclusion DC")
    check("sections/conclusion.tex", f"widens to $+{pp(q2(A,B,'dc_worst_f1'),q2(A,'Artemis','dc_worst_f1'))}$\\,pp worst-component \\fone{{}} and $+{pp(q2(A,B,'dc_harm_f1'),q2(A,'Artemis','dc_harm_f1'))}$\\,pp harmonic", "conclusion size-aware")
    check("sections/conclusion.tex", f"worst-component \\fone{{}} is ${f2(q2(A,B,'dc_worst_f1'))}$", "conclusion worst")
    check("main.tex", f"by {pp(dm(A,B,'f1'),dm(A,'Artemis','f1'))} percentage points (pp) in architecture doc-model \\avgfone, and by {pp(dm(A,B,'f2'),dm(A,'Artemis','f2'))}\\,pp", "abstract DM")
    check("main.tex", f"carries through to {pp(dc(A,B,'f1'),dc(A,'Artemis','f1'))}\\,pp of \\avgfone", "abstract DC")
    check("main.tex", f"widens to {pp(q2(A,B,'dc_worst_f1'),q2(A,'Artemis','dc_worst_f1'))} and {pp(q2(A,B,'dc_harm_f1'),q2(A,'Artemis','dc_harm_f1'))}\\,pp", "abstract size-aware")
    D = "sections/discussion.tex"
    check(D, f"worst-component \\fone{{}} on doc-code is ${f2(q2(A,B,'dc_worst_f1'))}$, and on BigBlueButton it falls to ${f2(q2('bigbluebutton',B,'dc_worst_f1'))}$", "discussion worst")
    check(D, f"the worst-component \\fone{{}} is ${f2(q2('teammates',B,'dc_worst_f1'))}$, well below the link-level \\fone{{}} of ${f2(q2('teammates',B,'dc_file_f1'))}$", "discussion Teammates")
    check(D, f"gains ${pp(dm('bigbluebutton',B,'f1'),dm('bigbluebutton','pipeline','f1'))}$\\,pp of doc-model \\fone{{}} over \\TransArc{{}} but loses ${pp(dc('bigbluebutton',B,'f1'),dc('bigbluebutton','pipeline','f1'))[1:]}$\\,pp on doc-code \\fone{{}}, where its doc-code precision is ${f2(dc('bigbluebutton',B,'p'))}$ against ${f2(dc('bigbluebutton','pipeline','p'))}$", "discussion BBB")
    # Motivation: the Artemis-vs-TransArC reversal is baseline-only but recomputable.
    check("sections/motivation.tex", f"\\Artemis{{}} leads \\TransArc{{}} by ${100*(q2(A,'Artemis','dc_file_f1')-q2(A,'TransArC','dc_file_f1')):.1f}$pp of link-level \\fone\\ while behind it by ${100*(q2(A,'TransArC','dc_worst_f1')-q2(A,'Artemis','dc_worst_f1')):.1f}$pp on the worst component and ${100*(q2(A,'TransArC','dc_harm_f1')-q2(A,'Artemis','dc_harm_f1')):.1f}$pp", "motivation reversal")

    print("\nout of scope (not run-derived by s126): motivation/metric gold-concentration and "
          "Gini figures, the JabRef `preferences` example, intro's cited 84% F1, "
          "discussion's BigBlueButton component/sentence counts, metric's worked example.")
    if fails:
        print(f"\n{len(fails)} FAIL(S):")
        for f in fails:
            print("  " + f)
        return 1
    print("\nPASS: every checked number matches the recomputed value")
    return 0


if __name__ == "__main__":
    sys.exit(main())
