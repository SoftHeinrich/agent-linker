"""Check token means, estimated costs, and the generated paper table.

Run from the repository root: python3 verification/verify_inference_cost_compact.py
"""

import csv
from decimal import Decimal
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation/mini-src"))
from inference_cost import PRICES, TABLE_ROWS

USAGE = ROOT / "evaluation/reports/INFERENCE_COST_PERRUN.csv"
SOURCE = ROOT / "evaluation/reports/tex_src/inference_cost.csv"
COMPACT = ROOT / "evaluation/reports/tex_src/inference_cost_by_system.csv"
GENERATED = ROOT / "evaluation/reports/tex/inference-cost.tex"
PAPER = ROOT / "paper/table"
PROJECTS = (("mediastore", "MS"), ("teastore", "TS"), ("teammates", "TM"),
            ("bigbluebutton", "BBB"), ("jabref", "JR"), ("Total", "Total"))
SYSTEMS = {"approach": r"\approach{}", "Artemis": r"\Artemis{}"}


def read(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


source = {row["project"]: row for row in read(SOURCE)}
usage = read(USAGE)
compact = read(COMPACT)
assert [(row["system"], row["backend"]) for row in compact] == [
    (shown, model.removeprefix("gpt-5.6-")) for _, shown, model in TABLE_ROWS]
assert GENERATED.read_bytes() == (PAPER / "inference-cost.tex").read_bytes()
assert COMPACT.read_bytes() == (PAPER / "inference-cost.csv").read_bytes()
tex = GENERATED.read_text()
assert "System & Backend & " + " & ".join(abbr for _, abbr in PROJECTS) + r" & (US\$) \\" in tex
assert r"\multicolumn{6}{c}{Input/Output (k tokens)}" in tex

matched = 0
for row, (system, shown, model) in zip(compact, TABLE_ROWS):
    abbr = SYSTEMS[shown]
    prefix = abbr + " & " + row["backend"] + " & "
    line = next(line for line in tex.splitlines() if line.startswith(prefix))
    cells = [cell.strip() for cell in line.removesuffix(r"\\").split(" & ")]
    assert len(cells) == 9 and cells[:2] == [abbr, row["backend"]]
    totals = []
    for (project, _), cell in zip(PROJECTS, cells[2:-1]):
        selected = [r for r in usage if r["system"] == system
                    and (project == "Total" or r["project"] == project)]
        assert len(selected) == (15 if project == "Total" else 3)
        assert {r["run"] for r in selected} == {"1", "2", "3"}
        assert all(r["model"] == model for r in selected)
        values = []
        for metric in ("input_k", "output_k"):
            raw = source[project][f"{system}_{metric}"]
            mean = sum(Decimal(r[metric]) for r in selected) / 3
            assert abs(mean - Decimal(raw)) <= Decimal("0.0000005")
            if project == "Total":
                totals.append(mean)
            assert row[f"{project}_{metric}"] == raw
            values.append(f"{float(raw):.1f}")
            matched += 1
        assert cell == "/".join(values), (system, project, cell)
    price = PRICES[model]
    cost = sum(tokens * Decimal(str(rate))
               for tokens, rate in zip(totals, price)) / 1000
    assert abs(cost - Decimal(row["cost_usd"])) < Decimal("0.000001")
    assert cells[-1] == f"{cost:.2f}", (system, model, cells[-1], cost)
    print(f"PASS cost: {shown}/{row['backend']} = US${cost:.6f} -> US${cost:.2f}")

assert sum(line.startswith((r"\approach{} &", r"\Artemis{} &"))
           for line in tex.splitlines()) == len(TABLE_ROWS)
print(f"PASS cost: {len(TABLE_ROWS)} system/backend rows, {matched} token values, all costs, and paper copies match")
