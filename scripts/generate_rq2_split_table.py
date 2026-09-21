#!/usr/bin/env python3
"""Generate the paired-panel RQ2 CSV and LaTeX table.

The source report contains more systems and metrics than the paper table. This
script selects the GPT-5.6-terra rows used in RQ2, aligns three projects on
each side, writes the displayed values to a compact CSV, and renders that CSV
as LaTeX. Running the script twice must produce byte-identical outputs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "evaluation/reports/tex_src/bigtable_rq12_perproject.csv"
DEFAULT_CSV = ROOT / "paper/table/rq2-wide-comparison.csv"
DEFAULT_TEX = ROOT / "paper/table/rq2-wide-comparison.tex"

LEFT_PROJECTS = ("mediastore", "teastore", "teammates")
RIGHT_PROJECTS = ("bigbluebutton", "jabref", "Average")
PROJECT_LABELS = {
    "mediastore": "MediaStore",
    "teastore": "TeaStore",
    "teammates": "Teammates",
    "bigbluebutton": "BigBlueButton",
    "jabref": "JabRef",
    "Average": "Average",
}
SYSTEMS = (
    ("approach (GPT-5.6-terra)", "ArchLinker"),
    ("Artemis (GPT-5.6-terra)", "Artemis"),
    ("TransArC", "Pipeline"),
)
METRICS = (
    "doc_to_model_link_f1",
    "doc_to_model_link_f2",
    "doc_to_model_component_miss_rate",
    "doc_to_code_file_f1",
    "doc_to_code_file_f2",
    "doc_to_code_worst_component_f1",
    "doc_to_code_worst_component_f2",
    "doc_to_code_harmonic_component_f1",
    "doc_to_code_harmonic_component_f2",
)
SHORT_METRICS = (
    "dm_link_f1",
    "dm_link_f2",
    "dm_cmr_pct",
    "dc_link_f1",
    "dc_link_f2",
    "dc_worst_f1",
    "dc_worst_f2",
    "dc_harmonic_f1",
    "dc_harmonic_f2",
)


@dataclass(frozen=True)
class DisplayRow:
    left_project: str
    approach: str
    left_values: tuple[str, ...]
    right_project: str
    right_values: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--tex-output", type=Path, default=DEFAULT_TEX)
    return parser.parse_args()


def format_score(raw: str) -> str:
    value = float(raw)
    rendered = f"{value:.2f}"
    return rendered[1:] if 0 <= value < 1 else rendered


def format_cmr(raw: str) -> str:
    return f"{float(raw):.1f}"


def load_source(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(row["system"], row["project"]): row for row in rows}

    expected = {
        (system, project)
        for system, _ in SYSTEMS
        for project in LEFT_PROJECTS + RIGHT_PROJECTS
    }
    missing = sorted(expected - indexed.keys())
    if missing:
        raise ValueError(f"source is missing required rows: {missing}")
    return indexed


def values_for(row: dict[str, str]) -> tuple[str, ...]:
    values = []
    for metric in METRICS:
        raw = row[metric]
        if not raw:
            raise ValueError(f"empty required metric {metric!r} in {row}")
        values.append(format_cmr(raw) if metric.endswith("miss_rate") else format_score(raw))
    return tuple(values)


def build_rows(source: dict[tuple[str, str], dict[str, str]]) -> list[DisplayRow]:
    output = []
    for left_project, right_project in zip(LEFT_PROJECTS, RIGHT_PROJECTS, strict=True):
        for system, approach in SYSTEMS:
            output.append(
                DisplayRow(
                    left_project=PROJECT_LABELS[left_project],
                    approach=approach,
                    left_values=values_for(source[(system, left_project)]),
                    right_project=PROJECT_LABELS[right_project],
                    right_values=values_for(source[(system, right_project)]),
                )
            )
    return output


def write_csv(path: Path, rows: list[DisplayRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        ["left_project", "approach"]
        + [f"left_{metric}" for metric in SHORT_METRICS]
        + ["right_project"]
        + [f"right_{metric}" for metric in SHORT_METRICS]
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fields)
        for row in rows:
            writer.writerow(
                [row.left_project, row.approach, *row.left_values, row.right_project, *row.right_values]
            )


def latex_approach(name: str) -> str:
    return {
        "ArchLinker": r"\approach{}",
        "Artemis": r"\Artemis{}",
        "Pipeline": r"Pipeline$^{\dagger}$",
    }[name]


def latex_row(row: DisplayRow) -> str:
    cells = [*row.left_values, latex_approach(row.approach), *row.right_values]
    return " & ".join(cells) + r" \\"


def render_tex(source: Path, csv_output: Path, rows: list[DisplayRow]) -> str:
    relative_source = source.resolve().relative_to(ROOT)
    relative_csv = csv_output.resolve().relative_to(ROOT)
    body = []
    previous_left = None
    for row in rows:
        if row.left_project != previous_left:
            if previous_left is not None:
                body.append(r"\addlinespace[1.5pt]")
            body.append(
                rf"\multicolumn{{9}}{{@{{}}l}}{{\textit{{{row.left_project}}}}}"
                rf" & & \multicolumn{{9}}{{l@{{}}}}{{\textit{{{row.right_project}}}}} \\[-1pt]"
            )
        body.append(latex_row(row))
        previous_left = row.left_project

    return rf"""% GENERATED by scripts/generate_rq2_split_table.py.
% Source: {relative_source}
% Intermediate data: {relative_csv}
% Do not edit by hand; rerun the generator.
\begin{{table}}[t]
\caption{{Alternative RQ2 per-project layout on GPT-5.6-terra. Each half reports doc-model link \fone/\ftwo and component miss rate (CMR), followed by doc-code link, worst-component, and harmonic-component \fone/\ftwo. \approach{{}} and \Artemis{{}} are means of three runs. Pipeline denotes SWATTR for doc-model and \TransArc{{}} for doc-code.}}
\label{{tab:rq2-wide-comparison}}
\centering\small
\setlength{{\tabcolsep}}{{1pt}}
\renewcommand{{\arraystretch}}{{0.96}}
\begin{{tabular*}}{{\linewidth}}{{@{{}}r@{{\extracolsep{{\fill}}}}*{{8}}{{r}}c*{{9}}{{r}}@{{}}}}
\toprule
\multicolumn{{3}}{{c}}{{doc-model}} & \multicolumn{{6}}{{c}}{{doc-code}}
& & \multicolumn{{3}}{{c}}{{doc-model}} & \multicolumn{{6}}{{c}}{{doc-code}} \\
\cmidrule(lr){{1-3}}\cmidrule(lr){{4-9}}\cmidrule(lr){{11-13}}\cmidrule(l){{14-19}}
\multicolumn{{2}}{{c}}{{Link}} & CMR & \multicolumn{{2}}{{c}}{{Link}} & \multicolumn{{2}}{{c}}{{Worst}} & \multicolumn{{2}}{{c}}{{Harm.}}
& Approach & \multicolumn{{2}}{{c}}{{Link}} & CMR & \multicolumn{{2}}{{c}}{{Link}} & \multicolumn{{2}}{{c}}{{Worst}} & \multicolumn{{2}}{{c}}{{Harm.}} \\
\fone & \ftwo & \% & \fone & \ftwo & \fone & \ftwo & \fone & \ftwo
& & \fone & \ftwo & \% & \fone & \ftwo & \fone & \ftwo & \fone & \ftwo \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular*}}
\par\smallskip\footnotesize $^{{\dagger}}$The pipeline row uses SWATTR for doc-model and \TransArc{{}} for doc-code.
\end{{table}}
"""


def main() -> None:
    args = parse_args()
    source = load_source(args.source)
    rows = build_rows(source)
    write_csv(args.csv_output, rows)
    args.tex_output.parent.mkdir(parents=True, exist_ok=True)
    args.tex_output.write_text(render_tex(args.source, args.csv_output, rows), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.csv_output}")
    print(f"wrote LaTeX table to {args.tex_output}")


if __name__ == "__main__":
    main()
