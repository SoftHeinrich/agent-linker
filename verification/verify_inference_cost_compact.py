"""Check the two-row inference-cost table against its project-oriented source.

Run from the repository root: python3 verification/verify_inference_cost_compact.py
"""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "evaluation/reports/tex_src/inference_cost.csv"
COMPACT = ROOT / "evaluation/reports/tex_src/inference_cost_by_system.csv"
GENERATED = ROOT / "evaluation/reports/tex/inference-cost.tex"
PAPER = ROOT / "paper/table"
PROJECTS = (("mediastore", "MS"), ("teastore", "TS"), ("teammates", "TM"),
            ("bigbluebutton", "BBB"), ("jabref", "JR"), ("Total", "Total"))
SYSTEMS = (("approach", r"\approach{}"), ("Artemis", r"\Artemis{}"))


def read(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


source = {row["project"]: row for row in read(SOURCE)}
compact = read(COMPACT)
assert [row["system"] for row in compact] == [system for system, _ in SYSTEMS]
assert GENERATED.read_bytes() == (PAPER / "inference-cost.tex").read_bytes()
assert COMPACT.read_bytes() == (PAPER / "inference-cost.csv").read_bytes()
tex = GENERATED.read_text()
assert "System & " + " & ".join(abbr for _, abbr in PROJECTS) + r" \\" in tex
assert r"\multicolumn{6}{c}{Input/Output (k tokens)}" in tex

matched = 0
for row, (system, abbr) in zip(compact, SYSTEMS):
    line = next(line for line in tex.splitlines() if line.startswith(abbr + " & "))
    cells = [cell.strip() for cell in line.removesuffix(r"\\").split(" & ")]
    assert len(cells) == 7 and cells[0] == abbr
    for (project, _), cell in zip(PROJECTS, cells[1:]):
        values = []
        for metric in ("input_k", "output_k"):
            raw = source[project][f"{system}_{metric}"]
            assert row[f"{project}_{metric}"] == raw
            values.append(f"{float(raw):.1f}")
            matched += 1
        assert cell == "/".join(values), (system, project, cell)

assert sum(line.startswith((r"\approach{} &", r"\Artemis{} &"))
           for line in tex.splitlines()) == 2
assert r"\approach{} uses GPT-5.6-terra; \Artemis{} uses GPT-5.6-luna" in tex
print(f"PASS cost: 2 system rows, 12 input/output tuples, {matched} source values match")
