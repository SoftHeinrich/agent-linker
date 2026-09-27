"""Audit the generated RQ1 task panels and gold-table Gini display.

Run from the repository root: python3 verification/verify_rq1_side_by_side.py
"""

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "evaluation"
PAPER = ROOT / "paper/table"
PROJECTS = ("mediastore", "teastore", "teammates", "bigbluebutton", "jabref", "Average")
ABBR = ("MS", "TS", "TM", "BBB", "JR", "Avg")
SYSTEMS = ("approach", "Artemis", "pipeline")
METRICS = ("p", "r", "f1", "f2")


def rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def displayed(value):
    text = f"{float(value):.2f}"
    return text.removeprefix("0")


source = {(row["project"], row["task"]): row
          for row in rows(EVAL / "reports/tex_src/rq1_transposed.csv")}
panel_csv = EVAL / "reports/tex_src/rq1_side_by_side.csv"
panel = rows(panel_csv)
assert [row["project"] for row in panel] == list(PROJECTS)
for row in panel:
    for task, source_task in (("dm", "DM"), ("dc", "DC")):
        original = source[(row["project"], source_task)]
        for system in SYSTEMS:
            for metric in METRICS:
                assert row[f"{task}_{system}_{metric}"] == original[f"{system}_{metric}"]

panel_tex = EVAL / "reports/tex/rq1-results.tex"
assert panel_csv.read_bytes() == (PAPER / "rq1-results.csv").read_bytes()
assert panel_tex.read_bytes() == (PAPER / "rq1-results.tex").read_bytes()
tex = panel_tex.read_text()
assert r"\centering\footnotesize" in tex
assert r"\setlength{\tabcolsep}{2pt}" in tex
assert r"\multicolumn{3}{c}{Doc-model}" in tex
assert r"\multicolumn{3}{c}{Doc-code}" in tex
assert " & ".join([""] + [r"P/R; \fone/\ftwo"] * 6) + r" \\" in tex.splitlines()
tex_rows = [line for line in tex.splitlines()
            if line.startswith(("MS &", "TS &", "TM &", "BBB &", "JR &", r"\textbf{Avg} &"))]
assert len(tex_rows) == 6
checked = 0
for row, abbr, line in zip(panel, ABBR, tex_rows):
    cells = [cell.strip() for cell in line.removesuffix(r"\\").split(" & ")]
    assert len(cells) == 7 and cells[0] == (r"\textbf{Avg}" if abbr == "Avg" else abbr)
    for task_index, task in enumerate(("dm", "dc")):
        for system_index, system in enumerate(SYSTEMS):
            cell = cells[1 + task_index * 3 + system_index]
            assert cell.count(";") == 1 and r"\makecell" not in cell
            numbers = re.findall(r"(\\textbf\{)?(1\.00|\.[0-9]{2})", cell)
            assert len(numbers) == 4, (row["project"], task, system, cell)
            for metric, (mark, shown) in zip(METRICS, numbers):
                raw = row[f"{task}_{system}_{metric}"]
                peers = [row[f"{task}_{other}_{metric}"] for other in SYSTEMS]
                assert shown == displayed(raw)
                assert bool(mark) == (float(displayed(raw)) ==
                                      max(float(displayed(peer)) for peer in peers))
                checked += 1
assert checked == 144
print("PASS RQ1: 6 project rows, 144 values copied from evaluation CSV; TeX panels, P/R; F1/F2 subheaders, and bold marks match")

gold_csv = EVAL / "mini-inequality/reports/out02_concentration.csv"
gold_tex = EVAL / "mini-inequality/reports/out02_concentration.tex"
assert gold_csv.read_bytes() == (PAPER / "gold_concentration.csv").read_bytes()
assert gold_tex.read_bytes() == (PAPER / "gold_concentration.tex").read_bytes()
gold_lines = gold_tex.read_text().splitlines()
gold = rows(gold_csv)
assert len(gold) == 5
for row in gold:
    matches = [line for line in gold_lines if line.startswith(row["project"] + " (")]
    assert len(matches) == 1
    values = [cell.strip().replace("{,}", "")
              for cell in matches[0].removesuffix(r"\\").split(" & ")]
    assert len(values) == 14
    for index, key in ((7, "doc_model_gini"), (12, "doc_code_gini")):
        assert values[index] == displayed(row[key])
    for index, key in enumerate(row):
        if index not in (0, 7, 12):
            assert float(values[index]) == float(row[key])
print("PASS gold: 10 Gini cells use .xx; other 55 numeric cells match parsed CSV")
