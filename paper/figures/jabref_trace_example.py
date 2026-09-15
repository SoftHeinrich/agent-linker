#!/usr/bin/env python3
"""Render the MediaStore documentation->model->code ground-truth example.

Mirrors figures/drawio/jabref_trace_example.drawio. Outputs jabref_trace_example.{pdf,png}.
Kept deliberately compact (wide) so it costs little vertical space at column width.

The sentence text and links come from the benchmark. S23 links "Database" to DB;
S24 refers back to DB; S27 links MediaAccess but not DB; and S28 refers back to
MediaAccess. The dashed S27 -> DB edge is the candidate rejected by the reported
judge. The S24--S27 gap is explicit, and code-file sets are summarized by counts.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["mathtext.fontset"] = "dejavusans"   # inline $\mathbf{...}$ -> bold sans, matches body

GREY_F, GREY_S, GREY_T = "#EDEDED", "#8C8C8C", "#333333"
BLUE_F, BLUE_S, BLUE_T = "#DCE3F2", "#4664AA", "#1F2D52"
GREEN_F, GREEN_S, GREEN_T = "#E7F2E2", "#5C8A3A", "#33491F"
RED, HDR, PURPLE = "#C0392B", "#666666", "#6A4C93"

W, H = 575, 236
fig, ax = plt.subplots(figsize=(7.0, 2.85))
ax.set_xlim(0, W)
ax.set_ylim(H, 0)          # invert y so (0,0) is top-left, as in draw.io
ax.axis("off")


def box(x, y, w, h, text, fc, ec, tc, fs=8, mono=False, bold=False, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0,rounding_size=6",
                 linewidth=1.0, facecolor=fc, edgecolor=ec, mutation_aspect=1.0))
    fam = "monospace" if mono else "sans-serif"
    fw = "bold" if bold else "normal"
    if align == "left":
        ax.text(x + 7, y + h / 2, text, ha="left", va="center",
                fontsize=fs, color=tc, family=fam, fontweight=fw)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, family=fam, fontweight=fw)


def arrow(x1, y1, x2, y2, color="#333333", dashed=False, lw=1.3):
    ls = (0, (5, 3)) if dashed else "-"
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=10, color=color, lw=lw, linestyle=ls,
                 shrinkA=0, shrinkB=0))


def curved_arrow(x1, y1, x2, y2, color="#333333", rad=0.16, lw=1.4, dashed=False):
    ls = (0, (5, 3)) if dashed else "-"
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=11, color=color, lw=lw, linestyle=ls,
                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


# column headers
ax.text(105, 7, "Documentation", ha="center", va="center", fontsize=8.5, fontweight="bold", color="black")
ax.text(310, 7, "Architecture model", ha="center", va="center", fontsize=8.5, fontweight="bold", color="black")
ax.text(495, 7, "Code", ha="center", va="center", fontsize=8.5, fontweight="bold", color="black")

# Verbatim MediaStore benchmark sentences; line breaks are typographic only.
box(0, 20, 225, 32, 'S23  "The $\\mathbf{Database}$ component represents\nan actual database (e.g., MySQL)."',
    GREY_F, GREY_S, GREY_T, fs=7.1, align="left")
box(0, 58, 225, 36, 'S24  "$\\mathbf{It}$ stores user information and meta-data\nof audio files such as the name and the genre."',
    GREY_F, GREY_S, GREY_T, fs=6.9, align="left")
ax.text(112, 101, "$\\vdots$", ha="center", va="center", fontsize=10, color=HDR)
box(0, 108, 225, 34, 'S27  "The $\\mathbf{MediaAccess}$ component encapsulates\n$\\mathbf{database\\ access}$ for meta-data of audio files."',
    GREY_F, GREY_S, GREY_T, fs=6.8, align="left")
box(0, 148, 225, 32, 'S28  "Furthermore, $\\mathbf{it}$ fetches a list of all\navailable audio files."',
    GREY_F, GREY_S, GREY_T, fs=7.1, align="left")

# architecture-model components
box(270, 48, 110, 30, "DB", BLUE_F, BLUE_S, BLUE_T, fs=9, mono=True, bold=True)
box(270, 126, 110, 30, "MediaAccess", BLUE_F, BLUE_S, BLUE_T, fs=8.4, mono=True, bold=True)

# benchmark code-file sets, summarized by count
box(425, 48, 145, 30, "4 code files", GREEN_F, GREEN_S, GREEN_T, fs=8, mono=True)
box(425, 126, 145, 30, "2 code files", GREEN_F, GREEN_S, GREEN_T, fs=8, mono=True)

# doc -> model links (sentence -> component)
arrow(225, 36, 269, 60)        # S23 -> DB (gold, Database alias)
arrow(225, 76, 269, 68)        # S24 -> DB through "It" (gold)
arrow(225, 125, 269, 140)      # S27 -> MediaAccess (gold)
arrow(225, 164, 269, 148)      # S28 -> MediaAccess through "it" (gold)
# model -> code links (component -> package)
arrow(380, 63, 424, 63)        # DB -> four gold code files
arrow(380, 141, 424, 141)      # MediaAccess -> two gold code files
# false positive
arrow(225, 128, 271, 76, color=RED, dashed=True)   # rejected S27 -> DB candidate
ax.text(247, 104, "rejected", ha="center", va="center", fontsize=6.5, style="italic", color=RED)
# direct doc -> code links (the doc-code link = composition of doc-model + model-code), one per chain
curved_arrow(180, 20, 455, 50, color=PURPLE, rad=-0.20)    # S23 -> DB code
curved_arrow(180, 180, 475, 156, color=PURPLE, rad=0.18)   # S28 -> MediaAccess code
ax.text(310, 194, "alias  ·  coreference  ·  evidence check", ha="center", va="center",
        fontsize=6.5, style="italic", color=HDR)
ax.text(310, 220, "doc-code links (direct)", ha="center", va="center",
        fontsize=7, style="italic", color=PURPLE)

for ext in ("pdf", "png"):
    fig.savefig(f"jabref_trace_example.{ext}", bbox_inches="tight", pad_inches=0.02,
                dpi=200 if ext == "png" else None)
print("wrote jabref_trace_example.pdf / .png")
