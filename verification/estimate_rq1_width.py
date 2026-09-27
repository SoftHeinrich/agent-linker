"""Estimate RQ1 table width from Libertine font metrics (not a TeX compile).

Run from repo root after downloading the regular and bold Libertine OTF files:
    python3 verification/estimate_rq1_width.py REGULAR.otf BOLD.otf
"""

import re
import sys
from pathlib import Path

from PIL import ImageFont


if len(sys.argv) != 3:
    raise SystemExit("usage: estimate_rq1_width.py REGULAR.otf BOLD.otf")
regular = ImageFont.truetype(sys.argv[1], 11)  # 8.25 pt at 96 DPI
bold = ImageFont.truetype(sys.argv[2], 11)
tex = (Path(__file__).resolve().parent.parent /
       "evaluation/reports/tex/rq1-results.tex").read_text()
assert r"\centering\footnotesize" in tex
colsep = float(re.search(r"\\setlength\{\\tabcolsep\}\{([0-9.]+)pt\}", tex).group(1))
gap = float(re.search(r"@\{\\hspace\{([0-9.]+)pt\}\}", tex).group(1))
assert r"\makecell" not in tex


def points(cell):
    total = 0.0
    for part in re.split(r"(\\textbf\{[^{}]*\})", cell):
        glyphs = (bold.getlength(part[8:-1]) if part.startswith(r"\textbf{")
                  else regular.getlength(part))
        total += glyphs * 72 / 96
    return total


headers = ("Proj.", "ArchLinker", "Artemis", "SWATTR",
           "ArchLinker", "Artemis", "TransArc")
widths = [points(header) for header in headers]
subheaders = [line for line in tex.splitlines()
              if line.startswith(r" & P/R; \fone/\ftwo")]
assert len(subheaders) == 1
subheader_cells = [cell.strip() for cell in subheaders[0].removesuffix(r"\\").split(" & ")[1:]]
assert subheader_cells == [r"P/R; \fone/\ftwo"] * 6
for index, cell in enumerate(subheader_cells, start=1):
    widths[index] = max(widths[index], points(cell.replace(r"\fone", "F1").replace(r"\ftwo", "F2")))
data = [line for line in tex.splitlines()
        if line.startswith(("MS &", "TS &", "TM &", "BBB &", "JR &", r"\textbf{Avg} &"))]
assert len(data) == 6
for line in data:
    cells = [part.strip() for part in line.removesuffix(r"\\").split(" & ")]
    assert len(cells) == 7
    for index, cell in enumerate(cells):
        widths[index] = max(widths[index], points(cell))

# acmart acmsmall: 6.75 in paper, 46 pt inner and outer margins.
available = 6.75 * 72.27 - 2 * 46
needed = sum(widths) + len(widths) * 2 * colsep + gap
print(f"ESTIMATE RQ1 width {needed:.1f} pt / {available:.1f} pt available "
      f"({available - needed:.1f} pt spare); font and scores stay at paper size")
assert needed < available
