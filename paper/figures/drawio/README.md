# figures/drawio

Editable **draw.io / diagrams.net** sources for the paper's figures. Keep the
`.drawio` here as the source of truth; export a vector copy into `../` for
`\includegraphics`.

> **Format:** plain, uncompressed mxGraph XML (human-readable / git-diffable).
>
> **Gotcha — cell `id`s must not collide with JS prototype names.** draw.io keys
> its internal cell map with a plain `{}`, so a cell `id` like `at`, `map`,
> `filter`, `constructor`, or `hasOwnProperty` resolves to an inherited
> `Object`/`Array`/`String` method instead of a cell. Symptom: the file refuses
> to open with **`d.setId is not a function`**, or CLI export dies with
> *"Export failed"*. Use descriptive ids (here `atbl` for the Alias Table, not
> `at`). Verified via the draw.io CLI (`drawio -x -f pdf …`).

| Source | Figure | Notes |
|--------|--------|-------|
| `approach-overview.drawio` | §\ref{sec:approach} pipeline | Wang-style "in a nutshell" overview: three numbered stages (knowledge layer → proposal by reference form → judging & consolidation), matching the reported `s_linker120` arm — one knowledge table (alias table), **three proposers** (whole name and one word by scanning, coreference by an LLM resolution) and **two** single-pass judges, because the two written forms are judged together under one rule. Redrawn 2026-09-01 for `s_linker110`, rewired 2026-09-11 for the union arm; the revision before that drew the retired `s_linker21` design (two linkers, two validators, a Model-Understanding/Ambiguity-Map second table, and a `p₁ ∧ p₂` validation conjunction), none of which any reported arm runs. |

> **Arm: `s_linker120`.** The figure was drawn against `s_linker92a` and re-labelled for
> `s_linker110`, whose two changes (the sibling-name refusal in the word scan, the
> resolver's code-computed antecedent shortlist) sit below the granularity this overview
> draws. `s_linker120`, the reported arm since 2026-09-11, is the first change that does
> not: it merges the full-name and partial-name **judges** into one, so the third column
> has two boxes rather than three while the second still has three. What did NOT change:
> the alias table, the three reference forms, and the one-pass character of every judge.
> Written up in `../../sections/approach.tex` (§\ref{sec:name-linker},
> §\ref{sec:coref-linker}).

> **Re-export needed (still).** `../approach-overview.pdf` predates BOTH redraws: it is
> the retired `s_linker21` render, and the paper has been shipping it. This source has
> moved ahead of it twice. Regenerate with
> `drawio -x -f pdf --crop -o ../approach-overview.pdf approach-overview.drawio`
> (no draw.io CLI was available on the machine that made the edit).

## Editing

- Web: open at <https://app.diagrams.net> → *Open Existing Diagram*.
- VS Code: the *Draw.io Integration* extension (`hediet.vscode-drawio`) edits
  `.drawio` files inline.

## Exporting for LaTeX (no CLI available in this repo)

Export to PDF (preferred for vector text) or SVG and drop it next to the other
figures, e.g. `figures/approach-overview.pdf`, then:

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/approach-overview.pdf}
  \caption{Overview of \approach{}. \ldots}
  \label{fig:approach-overview}
\end{figure*}
```

If a `drawio` CLI is later installed, regenerate with:

```bash
drawio -x -f pdf --crop -o figures/approach-overview.pdf \
  figures/drawio/approach-overview.drawio
```

## Palette

KIT brand colours — `kit-green #009682` (knowledge artifacts / output),
`kit-blue #4664AA` (LLM analyses / linkers), purple `#A3107C` (validators),
amber `#D08B16` (consolidation).
