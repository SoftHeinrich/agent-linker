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
| `approach-overview.drawio` | §\ref{sec:approach} pipeline | Wang-style "in a nutshell" overview: three numbered stages (knowledge layer → reference-form-specialised linkers → validation & consolidation), matching the canonical `s_linker21` design (a thin subclass of `s_linker20_union` that swaps only the validation-gate prompt; 2-pass alias-aware extraction **unioned** for recall; `p₁ ∧ p₂` conjunction at validation for precision). |

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
