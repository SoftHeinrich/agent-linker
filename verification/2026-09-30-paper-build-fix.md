# Paper build fix (2026-09-30)

## Failure

`paper/sections/intro.tex:53` cites `clelandhuang2012handbook`, but the
uncommitted edit to `paper/agent-linker.bib` (made alongside the
`sections/rw.tex` rewrite) deleted that entry. BibTeX reported
`I didn't find a database entry for "clelandhuang2012handbook"` and
LaTeX left the citation undefined. Overleaf reports this as an error.

## Fix (paper/agent-linker.bib, working tree)

- Restored the `clelandhuang2012handbook` entry, identical to the one that was
  deleted.
- `lago2009scoped`: changed `doi` from `https://doi.org/10.1016/...` to the
  bare DOI, because ACM-Reference-Format adds the resolver prefix itself.
- `spanoudakis2005software` stays removed; nothing cites it.

## Verification

```text
$ cd paper && latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
exit 0
Output written on main.pdf (19 pages, 827338 bytes).

$ grep -nE "^!|Citation .* undefined|Reference .* undefined|didn't find" main.log main.blg
(no matches)
```

One earlier run exited 12 with "LaTeX didn't generate the expected log file".
`main.synctex(busy)` was present at the time, which points to another build
running in the same directory at once. Rerunning after it finished gave the
clean result above.

The paper changes are still uncommitted in the submodule working tree, together
with the in-progress `sections/rw.tex` edits.
