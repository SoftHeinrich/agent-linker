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
| `approach-overview.drawio` | §\ref{sec:approach} pipeline | Wang-style "in a nutshell" overview: three numbered stages (knowledge layer → reference-form-specialised linkers → judging & consolidation), matching the reported `s_linker110` arm — one knowledge table (alias table), **three** linkers (full-name and partial-name propose by scanning, coreference by an LLM resolution), and **three** single-pass judges. Redrawn 2026-09-01; the previous revision drew the retired `s_linker21` design (two linkers, two validators, a Model-Understanding/Ambiguity-Map second table, and a `p₁ ∧ p₂` validation conjunction), none of which the reported arm runs. |

> **Arm: `s_linker110`.** The figure was drawn against `s_linker92a`; s110 became the
> reported arm on 2026-09-02 and is `s92a` plus two changes, neither of which is at the
> granularity this overview draws: `s109` has the partial-name scan refuse a word written
> only inside another component's whole name, and `s110` hands the coreference resolver a
> per-case, code-computed shortlist of the components the sentences above it name. Three
> linkers, three judges, one alias table — unchanged. Both are written up in
> `../../sections/approach.tex` (§\ref{sec:partial-linker}, §\ref{sec:coref-linker}).

> **Re-export needed.** `../approach-overview.pdf` is still the old two-linker
> render — this source has moved ahead of it. Regenerate with
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
