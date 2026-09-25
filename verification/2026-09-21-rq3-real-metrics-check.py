"""Independent re-derivation of the rendered RQ3 table (body + appendix).

Reads the ENGINE csvs (rq3_validators.csv = per-judge audit incl. the all_combined and
no-judge rows, rq3_variants.csv = doc-model, rq34_rq2_variants.csv = doc-code incl. the
size-aware pair) directly -- not rq_tables.py -- and asserts every printed cell of
reports/tex/rq3-confusion.tex and rq3-runs.tex, plus the LaTeX structure
(band spans == column count == cells per row).
"""
import csv, re, sys
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1] / "evaluation"
RQ34 = EVAL / "reports/rq34/s126"
TEX = EVAL / "reports/tex"

def rows(p):
    return list(csv.DictReader(p.open(encoding="utf-8")))

val = {(r["backend"], r["run"], r["validator"]): r for r in rows(RQ34 / "rq3_validators.csv")}
dm = {(r["backend"], r["run"], r["variant"]): r for r in rows(RQ34 / "rq3_variants.csv")}
dc = {(r["backend"], r["run"], r["variant"]): r for r in rows(RQ34 / "rq34_rq2_variants.csv")}
body_csv = rows(EVAL / "reports/tex_src/rq3.csv")
runs_csv = rows(EVAL / "reports/tex_src/rq3_runs.csv")

# (printed label, rq3_validators row the counts come from, rq3 variant the metrics come from)
ROWS = [("Full", "all_combined", "Full"),
        ("\\nameValidator{}", "name", "NoNameValid"),
        ("\\corefValidator{}", "coref", "NoCitation"),
        ("No judge", "none", "NoValidator")]

def f3(x):
    """Score cells: csv_to_tex.SCORE_DP decimals (2), leading zero dropped."""
    s = f"{float(x):.2f}"
    return s[1:] if s.startswith("0.") else s

def f1(x):
    return f"{float(x):.1f}"

def f2(x):
    s = f"{float(x):.2f}"
    return s[1:] if s.startswith("0.") else s

def num(x):
    f = float(x)
    return str(round(f)) if abs(f - round(f)) < 1e-9 else f"{f:.1f}"

def strip_bold(cell):
    """Unwrap every \\textbf{...}, allowing one level of nested braces (\\macro{})."""
    pat = re.compile(r"\\textbf\{((?:[^{}]|\{[^{}]*\})*)\}")
    prev = None
    cell = cell.strip()
    while prev != cell:
        prev, cell = cell, pat.sub(r"\1", cell)
    return cell.strip()


def body_rows(path):
    """Data rows only: everything between the first \\midrule and \\bottomrule."""
    out, started = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("\\midrule"):
            started = True
            continue
        if line.startswith("\\bottomrule"):
            break
        if started and line.endswith("\\\\") and "&" in line:
            out.append([strip_bold(c) for c in line[:-2].split("&")])
    return out

def expect(backend, run, count_fmt):
    exp = []
    for label, judge, variant in ROWS:
        a = val[(backend, run, judge)]
        m, c = dm[(backend, run, variant)], dc[(backend, run, variant)]
        counts = [count_fmt(a["rejected_fp"]), count_fmt(a["unique_rejected_tp"]),
                  count_fmt(a["kept_tp"]), count_fmt(a["kept_fp"])]
        exp.append((label, counts + [
            f"{f3(m['macro_precision'])}/{f3(m['macro_recall'])}\\,;\\,{f3(m['macro_f1'])}/{f3(m['macro_f2'])}",
            f1(m["component_miss_rate"]),
            f"{f3(c['doc_to_code_file_precision'])}/{f3(c['doc_to_code_file_recall'])}\\,;\\,"
            f"{f3(c['doc_to_code_file_f1'])}/{f3(c['doc_to_code_file_f2'])}",
            f"{f2(c['doc_to_code_worst_component_f1'])}/{f2(c['doc_to_code_worst_component_f2'])}",
            f"{f2(c['doc_to_code_harmonic_component_f1'])}/{f2(c['doc_to_code_harmonic_component_f2'])}"]))
    return exp


def pool_checks(backend, run):
    """The no-judge row is the candidate pool: nothing rejected, everything kept.

    Its true positives must equal what the full layer keeps plus the true links the
    layer costs outright (``all_combined`` kept_tp + unique_rejected_tp) -- exact per
    run, +-0.01 on the ``average`` row, where the engine writes each count rounded to
    two decimals. The false positives do NOT add up that way and are not asserted: a
    link one judge rejects can be kept by the other, so it is inside both
    ``rejected_fp`` and ``kept_fp``.
    """
    none, comb = val[(backend, run, "none")], val[(backend, run, "all_combined")]
    zeros = all(float(none[k]) == 0
                for k in ("rejected_tp", "unique_rejected_tp", "rejected_fp"))
    pool_tp = float(comb["kept_tp"]) + float(comb["unique_rejected_tp"])
    return zeros, abs(float(none["kept_tp"]) - pool_tp) <= 0.02


def markdown_table():
    """The body table as markdown, re-derived from the engine CSVs (for the note).

    So the write-up quotes generated numbers too: no cell in this repo's prose is typed
    by hand. Run `python3 <this file> --md`.
    """
    head = ("| Judges | rej FP | rej TP | keep TP | keep FP | doc-model P/R; F1/F2 | CMR% "
            "| doc-code P/R; F1/F2 | worst F1/F2 | harm. F1/F2 |")
    out = [head, "|" + "---|" * 10]
    plain = {"\\nameValidator{}": "name", "\\corefValidator{}": "coref"}
    for label, cells in expect("terra", "average", f1):
        cells = [c.replace("\\,;\\,", "; ") for c in cells]
        out.append("| " + " | ".join([plain.get(label, label)] + cells) + " |")
    return "\n".join(out)


if "--md" in sys.argv:
    print(markdown_table())
    sys.exit(0)

fails = 0

# ---- table CSV keys agree with the display configuration ----
for name, table, block_size in (("rq3.csv", body_csv, 4),
                                ("rq3_runs.csv", runs_csv, 4)):
    labels = [r["judge"] for r in table]
    ok = len(labels) % block_size == 0 and all(
        labels[i:i + block_size] == ["full_on", "name", "coref", "no_judge"]
        for i in range(0, len(labels), block_size))
    print(f"[{'OK' if ok else 'FAIL'}] {name}: No judge is the last row of each block")
    fails += not ok

# ---- structure ----
for name, nlab in (("rq3-confusion.tex", 1), ("rq3-runs.tex", 3)):
    text = (TEX / name).read_text(encoding="utf-8")
    spans = [int(n) for n in re.findall(r"\\multicolumn\{(\d+)\}\{c\}", text.splitlines()[9 if nlab == 1 else 8])]
    band = next(l for l in text.splitlines() if "multicolumn" in l)
    spans = [int(n) for n in re.findall(r"\\multicolumn\{(\d+)\}", band)]
    ncol = sum(spans)
    header = next(l for l in text.splitlines() if l.startswith(("Judges &", "Backend &")))
    ncells = len(header.split("&"))
    ok = ncol + nlab == ncells
    print(f"[{'OK' if ok else 'FAIL'}] {name}: bands {spans} = {ncol} cols + {nlab} label(s) == {ncells} header cells")
    fails += not ok
    for r in body_rows(TEX / name):
        if len(r) != ncells:
            print(f"[FAIL] {name}: row has {len(r)} cells, header has {ncells}: {r}")
            fails += 1

# ---- body table cells ----
got = body_rows(TEX / "rq3-confusion.tex")
exp = expect("terra", "average", f1)
for (label, cells), g in zip(exp, got):
    ok = g[0] == label and g[1:] == cells
    print(f"[{'OK' if ok else 'FAIL'}] body {label:24s} {' '.join(g[1:])}")
    if not ok:
        print(f"        expected {[label] + cells}")
    fails += not ok

# ---- appendix cells (both backends, 3 runs + average) ----
got = body_rows(TEX / "rq3-runs.tex")
i = 0
for backend in ("terra", "luna"):
    for run in ("run1", "run2", "run3", "average"):
        for label, cells in expect(backend, run, num):
            g = got[i]; i += 1
            gl, gc = g[2], g[3:]
            ok = gl == label and gc == cells
            if not ok:
                print(f"[FAIL] appendix {backend}/{run} {label}: {gc} != {cells}")
            fails += not ok
print(f"[{'OK' if not fails else 'FAIL'}] appendix: {i} rows checked against the engine CSVs")

# ---- the No judge row rejects nothing and keeps the whole candidate pool ----
for backend in ("terra", "luna"):
    for run in ("run1", "run2", "run3", "average"):
        zeros, pool_ok = pool_checks(backend, run)
        if not zeros:
            print(f"[FAIL] {backend}/{run}: the no-judge row rejects something")
        if not pool_ok:
            print(f"[FAIL] {backend}/{run}: no-judge kept_tp != full-layer kept_tp + "
                  "unique_rejected_tp")
        fails += (not zeros) + (not pool_ok)
print(f"[{'OK' if not fails else 'FAIL'}] no-judge row: 0 rejects and kept_tp == the "
      "candidate pool, in every backend/run block")

print("\nRESULT:", "PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
