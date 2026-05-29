"""render_ablation.py — single source-of-truth render for the Phase 5 ablation table.

Reads canonical JSONs under results/ablation_results/; emits ABLATION-TABLE.md and
ABLATION-TABLE.tex via tabulate. Re-runnable: produces deterministic byte-identical
output given the same JSONs.

Per .planning/phases/05-promote-and-ablation-artifact/05-CONTEXT.md D-47 / D-49 / D-52.

JSON schema (verified Plan 05-02 Task 1a, observed across all 7 canonical JSONs):
    {<dataset>: {<variant>: {"variant": str, "P": float, "R": float, "F1": float,
                              "tp": int, "fp": int, "fn": int, "n_links": int}}}
    Datasets: mediastore, teastore, teammates, bigbluebutton, jabref.
    Hard-tier-only JSONs (13d) have only "teammates" and "bigbluebutton" top-level keys.

JSON-vs-D-47 sanity check (Plan 05-02 Task 1c):
    All rows match D-47 macro within +/- 0.0005 (sub-rounding noise from computing
    macro on native JSON floats vs D-47's table of 3-decimal-rounded F1 cells).
    Per D-47a, the JSON wins. Computed macros:
        12c: 0.9404 (D-47: 0.9405; native float macro = 0.9404305 from native F1s)
        13a: 0.9363 (D-47: 0.9364)
        13b: 0.9519 (D-47: 0.9519)
        13c: 0.9314 (D-47: 0.9314)
        13e: 0.9379 (D-47: 0.9380)
        13f: 0.9506 (D-47: 0.9509; native float macro = 0.9505847)
        s_linker13: 0.9506 (= 13f by D-44a)
    The D-47-stated rounded macros are quoted in the markdown's provenance footer
    so the canonical paper numbers (0.9405 / 0.9509) remain visible alongside the
    JSON-derived render values.
    The 12c BBB cell is explicitly sourced from BBB re-run ablation_20260514_185017.json
    (BBB F1 = 0.844) per D-47 row 1; the full-sweep JSON ablation_20260513_192513.json
    native BBB = 0.818 is NOT used.
"""
from __future__ import annotations

import json
from pathlib import Path

from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[3]  # repo root
JSON_DIR = ROOT / "results" / "ablation_results"
OUT_DIR = ROOT / ".planning" / "phases" / "05-promote-and-ablation-artifact"

DATASETS = ("mediastore", "teastore", "teammates", "bigbluebutton", "jabref")
DATASET_SHORT = {
    "mediastore": "MS",
    "teastore": "TS",
    "teammates": "TM",
    "bigbluebutton": "BBB",
    "jabref": "JAB",
}

# Per-row data dict (D-47 canonical).
# (variant, parent, rule_short, json_name, hard_tier_only, status, note)
# Per D-44a: s_linker13 sources its numbers from ablation_20260529_215932.json
# (same JSON as 13f); no separate sweep was run for the canonical promotion.
ROWS = [
    ("s_linker12c", None, "— (baseline)",
     "ablation_20260513_192513.json", False, "BASELINE", None),
    ("s_linker13a", "s_linker12c",
     "_split_component_name (Spike 001 LLM trailing-word, partial)",
     "ablation_20260528_173020.json", False, "PASS",
     "loosened BBB 4pp tolerance"),
    ("s_linker13b", "s_linker13a",
     "_is_structurally_unambiguous (post-filter)",
     "ablation_20260528_190916.json", False, "PASS", None),
    ("s_linker13c", "s_linker13b",
     "_is_ambiguous_name_component (inline-remove)",
     "ablation_20260528_201851.json", False, "PASS",
     "loosened BBB 6pp tolerance"),
    ("s_linker13d", "s_linker13c",
     "_classify_mention (4-regex -> LLM enum, Spike 003)",
     "ablation_20260529_110532.json", True, "RETIRED",
     "TM -0.188 vs 12c, dotted-path classification failure"),
    ("s_linker13e", "s_linker13c",
     "_is_strong_alias + _get_strong_alias_mappings (LLM scope field)",
     "ablation_20260529_201324.json", False, "PASS", None),
    ("s_linker13f", "s_linker13e",
     "_has_strong_alias_mention (coref antecedent_via_alias fold)",
     "ablation_20260529_215932.json", False, "PASS", None),
    # D-44a: s_linker13 row is byte-equivalent to 13f's numbers; sources 13f JSON.
    ("s_linker13",  "s_linker13f",
     "— (promotion: cumulative chain, 6 rules removed)",
     "ablation_20260529_215932.json", False, "PROMOTED",
     "defined as 13f per D-44a"),
]


def _read_variant_f1(json_path: Path, variant: str, dataset: str) -> float:
    """Return per-dataset F1 for one variant; NaN if missing."""
    data = json.loads(json_path.read_text())
    if dataset not in data:
        return float("nan")
    ds = data[dataset]
    if variant not in ds:
        return float("nan")
    return float(ds[variant].get("F1", float("nan")))


def load_per_dataset_f1(json_path: Path, variant: str) -> dict[str, float]:
    """Read per-dataset F1 from an ablation JSON. Returns {dataset: f1}."""
    return {ds: _read_variant_f1(json_path, variant, ds) for ds in DATASETS}


def macro(per_ds: dict[str, float]) -> float:
    vals = [v for v in per_ds.values() if v == v]  # filter NaN
    return sum(vals) / len(vals) if vals else float("nan")


def build_rows() -> list[dict]:
    rows = []
    for variant, parent, rule_short, json_name, hard_tier, status, note in ROWS:
        if variant == "s_linker12c":
            # 12c uses full-sweep JSON ablation_20260513_192513.json, BUT the
            # canonical BBB cell comes from the BBB re-run ablation_20260514_185017.json
            # per CONTEXT.md D-47 row 1.
            per_ds = load_per_dataset_f1(JSON_DIR / json_name, variant)
            bbb_rerun = load_per_dataset_f1(
                JSON_DIR / "ablation_20260514_185017.json", variant
            )
            per_ds["bigbluebutton"] = bbb_rerun["bigbluebutton"]
        elif variant == "s_linker13":
            # 13 is defined as 13f per D-44a; reuse 13f's JSON with 13f variant key.
            per_ds = load_per_dataset_f1(JSON_DIR / json_name, "s_linker13f")
        elif hard_tier:
            # 13d: hard-tier only — TM only; MS/TS/BBB/JAB are blank (NaN) per D-46c.
            tm = _read_variant_f1(JSON_DIR / json_name, variant, "teammates")
            per_ds = {
                "mediastore": float("nan"),
                "teastore": float("nan"),
                "teammates": tm,
                "bigbluebutton": float("nan"),
                "jabref": float("nan"),
            }
        else:
            per_ds = load_per_dataset_f1(JSON_DIR / json_name, variant)
        # For hard-tier-only rows (e.g., 13d) the macro is undefined per D-46c
        # (no full-sweep was run); leave it NaN so it renders as a dash.
        row_macro = float("nan") if hard_tier else macro(per_ds)
        rows.append({
            "variant": variant,
            "parent": parent,
            "rule": rule_short,
            "per_ds": per_ds,
            "macro": row_macro,
            "status": status,
            "note": note,
            "json": json_name,
        })
    return rows


def deltas(rows: list[dict]) -> list[dict]:
    by_variant = {r["variant"]: r for r in rows}
    baseline = by_variant["s_linker12c"]["macro"]
    for r in rows:
        r["delta_12c"] = (r["macro"] - baseline) if r["macro"] == r["macro"] else float("nan")
        if r["parent"] is None:
            r["delta_parent"] = float("nan")
        elif r["variant"] == "s_linker13":
            r["delta_parent"] = 0.0  # promotion: 13 == 13f by D-44a
        else:
            p_macro = by_variant[r["parent"]]["macro"]
            r["delta_parent"] = (r["macro"] - p_macro) if (
                r["macro"] == r["macro"] and p_macro == p_macro
            ) else float("nan")
    return rows


def fmt(v: float, digits: int = 4) -> str:
    if v != v:  # NaN
        return "—"
    return f"{v:.{digits}f}"


def fmt_delta(v: float) -> str:
    if v != v:
        return "—"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.4f}"


def latex_escape(s: str) -> str:
    """No-op transform — tabulate's `latex_booktabs` formatter already escapes
    `_`, `&`, `%`, `#`, `$` automatically. Kept as an extension point.
    """
    return s


def render_table(rows: list[dict], tablefmt: str, escape: bool = False) -> str:
    headers = [
        "variant", "parent", "rule removed",
        "MS", "TS", "TM", "BBB", "JAB",
        "macro", "Δ vs 12c", "Δ vs parent", "status",
    ]
    if escape:
        headers = [latex_escape(h) if h != "Δ vs 12c" and h != "Δ vs parent" else h
                   for h in headers]
        # delta headers contain Δ which is unicode-safe in modern LaTeX; keep raw
    body = []
    for r in rows:
        cells = [
            r["variant"], r["parent"] or "—", r["rule"],
            fmt(r["per_ds"]["mediastore"], 3),
            fmt(r["per_ds"]["teastore"], 3),
            fmt(r["per_ds"]["teammates"], 3),
            fmt(r["per_ds"]["bigbluebutton"], 3),
            fmt(r["per_ds"]["jabref"], 3),
            fmt(r["macro"], 4),
            fmt_delta(r["delta_12c"]),
            fmt_delta(r["delta_parent"]),
            r["status"],
        ]
        if escape:
            cells = [latex_escape(c) if isinstance(c, str) else c for c in cells]
        body.append(cells)
    return tabulate(body, headers=headers, tablefmt=tablefmt)


def main() -> None:
    rows = deltas(build_rows())
    md_table = render_table(rows, "github", escape=False)
    tex_table = render_table(rows, "latex_booktabs", escape=True)

    md_path = OUT_DIR / "ABLATION-TABLE.md"
    tex_path = OUT_DIR / "ABLATION-TABLE.tex"

    md_path.write_text(
        "# Ablation Table — 13-Series Chain\n"
        "\n"
        "_Generated by `render_ablation.py` (Phase 5, PROMO-03). Source: "
        "`results/ablation_results/` JSONs cited per row._\n"
        "\n"
        f"{md_table}\n"
        "\n"
        "**13d footnote:** RETIRED 2026-05-29 — TM F1 = 0.750 (Δ −0.188 vs 12c) on "
        "dotted-path regression. LLM enum classifier cannot reproduce the project-specific "
        "Java-package convention (`ui.website`, `logic.api`, `storage.entity`) encoded in "
        "12c's regex `_classify_mention`. Milestone-level finding: classification of "
        "language-construct references is regex territory; the no-hand-crafted-rules thesis "
        "holds with this caveat. See "
        "`.planning/phases/03-mention-classifier-migration/03-01-SUMMARY.md` "
        "§\"Failure-Mode Analysis\".\n"
        "\n"
        "**s_linker13 row:** numbers are 13f's (per D-44a — `s_linker13.py` is byte-equivalent "
        "to `s_linker13f.py` modulo `_VARIANT_NAME`, class name, docstring, banner; no separate "
        "sweep was run). Source JSON: `ablation_20260529_215932.json`.\n"
        "\n"
        "**12c BBB cell provenance:** the full-sweep JSON `ablation_20260513_192513.json` "
        "reports BBB F1 = 0.818; the canonical 12c BBB baseline cell is sourced from the BBB "
        "re-run `ablation_20260514_185017.json` (BBB F1 = 0.844) per D-47 row 1.\n"
        "\n"
        "**D-47 paper macros (rounded reference, JSON-derived macros differ by sub-0.0005):** "
        "12c = 0.9405, 13a = 0.9364, 13b = 0.9519, 13c = 0.9314, 13e = 0.9380, "
        "13f = 0.9509, s_linker13 = 0.9509 (per D-44a). The render script computes macro "
        "from native JSON floats and prints to 4 decimals; per D-47a the JSON wins on any "
        "discrepancy.\n"
        "\n"
        "**Source JSONs (per row):**\n"
        "\n"
        + "".join(
            f"- `{r['variant']}`: `results/ablation_results/{r['json']}`"
            + (f" (BBB cell from `ablation_20260514_185017.json`)"
               if r["variant"] == "s_linker12c" else "")
            + "\n"
            for r in rows
        )
    )
    # Prepend a LaTeX comment block that names every variant by its raw identifier.
    # (Underscore inside a LaTeX tabular cell must be escaped as `\_` and tabulate
    # already does that, which means a literal `grep 's_linker13'` does not match
    # cells of the rendered table. The comment block makes the raw identifier
    # visible for greps and reviewers without breaking LaTeX compilation — `%`
    # starts a comment line so `_` is harmless there.)
    tex_header = (
        "% Phase 5 ablation table — generated by render_ablation.py (PROMO-03).\n"
        "% Variants covered (raw identifiers, for greps / reviewers):\n"
        "%   s_linker12c (BASELINE)\n"
        "%   s_linker13a (PASS)\n"
        "%   s_linker13b (PASS)\n"
        "%   s_linker13c (PASS)\n"
        "%   s_linker13d (RETIRED — TM dotted-path regression)\n"
        "%   s_linker13e (PASS)\n"
        "%   s_linker13f (PASS)\n"
        "%   s_linker13  (PROMOTED — canonical promotion of s_linker13f per D-44a)\n"
        "% Canonical artifact-of-record: s_linker13 (Phase 5 chain-end).\n"
        "% Source JSONs: see results/ablation_results/ablation_20260529_215932.json\n"
        "%   (and the per-row JSONs listed in ABLATION-TABLE.md).\n"
        "% Within the tabular below, underscores are LaTeX-escaped as `\\_` per\n"
        "% tabulate's latex_booktabs formatter; the raw identifiers above are the\n"
        "% canonical form referenced by METHODOLOGY.md and PROJECT.md.\n"
    )
    tex_path.write_text(tex_header + tex_table + "\n")

    print("ABLATION TABLES RENDERED")
    for r in rows:
        print(f"  {r['variant']}: macro={fmt(r['macro'])}, "
              f"Δ12c={fmt_delta(r['delta_12c'])}, "
              f"Δparent={fmt_delta(r['delta_parent'])}, "
              f"status={r['status']}")


if __name__ == "__main__":
    main()
