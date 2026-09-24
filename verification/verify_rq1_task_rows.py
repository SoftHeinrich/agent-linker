"""Check that the generated RQ1 table has one printed row per task.

Run from the repository root: python3 verification/verify_rq1_task_rows.py
"""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "evaluation/reports/tex_src/rq1_transposed.csv"
GENERATED = ROOT / "evaluation/reports/tex/rq1-results.tex"
PAPER = ROOT / "paper/table/rq1-results.tex"
PROJECTS = {"mediastore": "MS", "teastore": "TS", "teammates": "TM",
            "bigbluebutton": "BBB", "jabref": "JR", "Average": "Avg"}
TASKS = {"DM": "doc-model", "DC": "doc-code"}
SYSTEMS = ("approach", "Artemis", "pipeline")


def score(value):
    return f"{float(value):.2f}".removeprefix("0")


def shown(cell):
    return (cell.replace(r"\textbf{", "").replace("}", "")
            .replace(r"$\pm$", "±").replace(r"\,", ""))


def expected(row, system):
    values = []
    for metric in ("p", "r", "f1", "f2"):
        value = score(row[f"{system}_{metric}"])
        if metric in ("p", "r") and system != "pipeline":
            value += "±" + score(row[f"{system}_{metric}_sd"])
        values.append(value)
    return "/".join(values[:2]) + ";" + "/".join(values[2:])


with SOURCE.open(newline="") as source:
    rows = list(csv.DictReader(source))
tex = GENERATED.read_text()
assert GENERATED.read_bytes() == PAPER.read_bytes()
assert "Project & Task &" in tex
assert r"\makecell" not in tex
body = [line for line in tex.splitlines()
        if line.startswith(("MS &", "TS &", "TM &", "BBB &", "JR &",
                            " & doc-code", r"\textbf{Avg} &"))]
assert len(rows) == len(body) == 12
checked = 0
for index, (row, line) in enumerate(zip(rows, body)):
    cells = [cell.strip() for cell in line.removesuffix(r"\\").split(" & ")]
    assert len(cells) == 5, (index, cells)
    project = PROJECTS[row["project"]]
    assert shown(cells[0]) == ("" if row["task"] == "DC" and project != "Avg"
                               else project)
    assert shown(cells[1]) == TASKS[row["task"]]
    for system, cell in zip(SYSTEMS, cells[2:]):
        assert shown(cell) == expected(row, system), (index, system, shown(cell))
        checked += 1
assert "rounded to two decimals" in tex
print(f"PASS RQ1: {len(body)} single-line task rows; {checked} system cells match source scores and two-decimal SD")
