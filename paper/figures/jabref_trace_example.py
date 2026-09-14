#!/usr/bin/env python3
"""Render the documentation->model->code trace example for the Motivation section.

Mirrors figures/drawio/jabref_trace_example.drawio. Outputs jabref_trace_example.{pdf,png}.
Kept deliberately compact (wide) so it costs little vertical space at column width.

Layout: gui <- S6 (the alias "UI") and S7 ("It" refers to S6); the Data Model
<- S8 (the partial name "model"); preferences <- S11 (the exact name).
Each component maps to a code package (model-code). Dashed red S7 -> preferences
is the false positive a judge rejects.
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

# documentation sentences (S6 reaches gui via alias, S7 reaches gui by coreference,
# S8 reaches the multi-word Data Model by a partial name, and S11 is exact).
box(0, 20, 210, 28, 'S6  "The $\\mathbf{UI}$ renders the\nmain application window."',
    GREY_F, GREY_S, GREY_T, fs=7.5, align="left")
box(0, 54, 210, 34, 'S7  "$\\mathbf{It}$ knows the user and\ntheir $\\mathbf{preferences}$."',
    GREY_F, GREY_S, GREY_T, fs=7.5, align="left")
box(0, 94, 210, 30, 'S8  "The $\\mathbf{model}$ stores entries\nand metadata."',
    GREY_F, GREY_S, GREY_T, fs=7.5, align="left")
box(0, 130, 210, 34, 'S11  "The $\\mathbf{preferences}$ component\nstores user settings."',
    GREY_F, GREY_S, GREY_T, fs=7.5, align="left")

# architecture-model components
box(255, 43, 110, 30, "gui", BLUE_F, BLUE_S, BLUE_T, fs=9, mono=True, bold=True)
box(255, 94, 110, 30, "Data Model", BLUE_F, BLUE_S, BLUE_T, fs=8.5, mono=True, bold=True)
box(255, 134, 110, 30, "preferences", BLUE_F, BLUE_S, BLUE_T, fs=9, mono=True, bold=True)

# code packages (one per component)
box(420, 43, 150, 30, "org/jabref/gui/", GREEN_F, GREEN_S, GREEN_T, fs=8, mono=True)
box(420, 94, 150, 30, "org/jabref/model/", GREEN_F, GREEN_S, GREEN_T, fs=8, mono=True)
box(420, 134, 150, 30, "org/jabref/\npreferences/", GREEN_F, GREEN_S, GREEN_T, fs=8, mono=True)

# doc -> model links (sentence -> component)
arrow(210, 34, 254, 58)        # S6 -> gui (via alias UI)
arrow(210, 71, 254, 64)        # S7 -> gui (resolved via It and S6)
arrow(210, 109, 254, 109)      # S8 -> Data Model (partial model)
arrow(210, 147, 254, 149)      # S11 -> preferences (exact)
# model -> code links (component -> package)
arrow(365, 58, 419, 58)        # gui -> org/jabref/gui/
arrow(365, 109, 419, 109)      # Data Model -> org/jabref/model/
arrow(365, 149, 419, 149)      # preferences -> org/jabref/preferences/
# false positive
arrow(210, 80, 256, 140, color=RED, dashed=True)   # S7 -> preferences
ax.text(238, 112, "false positive", ha="center", va="center", fontsize=6.5, style="italic", color=RED)
# direct doc -> code links (the doc-code link = composition of doc-model + model-code), one per chain
curved_arrow(210, 30, 440, 46, color=PURPLE, rad=-0.22)    # S6 -> org/jabref/gui/ (bows up)
curved_arrow(160, 164, 470, 164, color=PURPLE, rad=0.16)   # S11 -> org/jabref/preferences/ (bows down)
curved_arrow(160, 124, 470, 124, color=PURPLE, rad=0.08)   # S8 -> org/jabref/model/
ax.text(310, 184, "alias  ·  coref  ·  partial  ·  exact", ha="center", va="center",
        fontsize=6.5, style="italic", color=HDR)
ax.text(310, 220, "doc-code links (direct)", ha="center", va="center",
        fontsize=7, style="italic", color=PURPLE)

for ext in ("pdf", "png"):
    fig.savefig(f"jabref_trace_example.{ext}", bbox_inches="tight", pad_inches=0.02,
                dpi=200 if ext == "png" else None)
print("wrote jabref_trace_example.pdf / .png")
