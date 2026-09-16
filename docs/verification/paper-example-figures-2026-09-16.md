# Paper example-figure integration verification

Date: 2026-09-16

## Scope

The paper now includes the named-link example in the name-linker subsection and
the coreference example in the coreference-linker subsection. Both figures are
cropped to their drawn content before inclusion.

## Commands and results

### Figure preparation

Command:

```text
pdfcrop --margins '6 6 6 6' paper/figures/named_link_approach.pdf <tmp>/named_link_approach.pdf
pdfcrop --margins '6 6 6 6' paper/figures/coref-link-approach.pdf <tmp>/coref-link-approach.pdf
```

Result:

```text
named_link_approach.pdf: 1 page, 365 x 122 pt
coref-link-approach.pdf: 1 page, 365 x 129 pt
```

### Manuscript build

Command:

```text
./scripts/build-paper.sh
```

The first build completed with exit code `0` and wrote `paper/main.pdf` with 17
pages. A repeat invocation encountered an empty generated `paper/main.aux` and
failed with exit code `12` at `! Text line contains an invalid character.` The
forced rebuild below completed from the same source tree:

```text
(cd paper && latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex)
```

Result:

```text
Output written on main.pdf (17 pages, 820082 bytes).
FORCED_BUILD_EXIT=0
```

The build retains existing diagnostics: three unresolved `fig:example`
references from the earlier figure without a caption, 14 BibTeX metadata
warnings, and PDF-version warnings for the existing overview figure and the two
new assets. No diagnostics identify either new figure label or caption as
unresolved.

### Output checks

```text
pdftotext -layout paper/main.pdf - | rg -n -C 3 'Example decisions for (named|coreference) links'
```

Result: the output contains `Fig. 3. Example decisions for named links` and
`Fig. 4. Example decisions for coreference links`; both labels are recorded in
`paper/main.aux` on page 7.

```text
git -C paper diff --check -- sections/approach.tex
```

Result: exit code `0`, with no whitespace errors.

## Follow-up: updated figure export

The paper submodule later received updated artwork in commit `e7a4abd`. The
parent initially still pointed to `04a5bfa`, so that child update was not
included when the package was checked out. The updated PDFs also retained their
full-page export margins. At full width, the named figure made its float 30.77
pt too tall; LaTeX deferred both figures to pages 18--19.

The updated artwork was cropped without changing its drawn content:

```text
pdfcrop --margins '6 6 6 6' paper/figures/named_link_approach.pdf <tmp>/named_link_approach.pdf
pdfcrop --margins '6 6 6 6' paper/figures/coref-link-approach.pdf <tmp>/coref-link-approach.pdf
```

The resulting assets are one page each, sized `354 x 118.08 pt` and
`364.08 x 121.92 pt`, respectively. They are committed in paper commit
`934ead3`.

Command:

```text
(cd paper && latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex)
```

Result:

```text
Output written on main.pdf (17 pages, 817179 bytes).
FORCED_BUILD_EXIT=0
```

The rebuilt output contains both figure captions and records
`fig:approach-named` and `fig:approach-coref` on page 7. The final log contains
no `Float too large` diagnostic. Existing `fig:example` and bibliography
warnings remain unchanged.
