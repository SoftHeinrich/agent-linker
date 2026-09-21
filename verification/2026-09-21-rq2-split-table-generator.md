# RQ2 paired-panel table generator verification

Date: 2026-09-21

## Scope

The generator selects the GPT-5.6-terra ArchLinker and Artemis rows and the
deterministic pipeline rows from
`evaluation/reports/tex_src/bigtable_rq12_perproject.csv`. It writes a
nine-row display CSV and renders the alternative RQ2 LaTeX table. The left
project block contains MediaStore, TeaStore, and Teammates; the aligned right
block contains BigBlueButton, JabRef, and the reported macro average.

Each data row contains 18 separate numeric columns: doc-model link F1/F2 and
CMR for both aligned projects, plus doc-code link, worst-component, and
harmonic-component F1/F2 for both projects. The shared center approach column
avoids repeating the system label.

## Generation and deterministic replay

Command, from the repository root:

```bash
python scripts/generate_rq2_split_table.py
```

Result:

```text
wrote 9 rows to /mnt/hostshare/ardoco-home/agent-linker/paper/table/rq2-wide-comparison.csv
wrote LaTeX table to /mnt/hostshare/ardoco-home/agent-linker/paper/table/rq2-wide-comparison.tex
```

The generated files were copied to `/tmp`, the command was rerun, and both
were compared byte-for-byte:

```bash
cmp /tmp/rq2-split.csv paper/table/rq2-wide-comparison.csv
cmp /tmp/rq2-split.tex paper/table/rq2-wide-comparison.tex
```

Both comparisons exited 0 with no output. Final SHA-256 digests:

```text
e3f55e890a3003219de5156044efde1dde0a1b9321eb26998879f54c64597d5f  paper/table/rq2-wide-comparison.csv
34d68a23270657f394f77067c9366dca2265fa92883496086d029dead5e897ec  paper/table/rq2-wide-comparison.tex
```

The generator also passed a syntax check:

```bash
PYTHONPYCACHEPREFIX=/tmp/rq2-pycache python -m py_compile scripts/generate_rq2_split_table.py
```

Result: exit 0, no output.

## ACM-template compilation

Command, from `verification/`:

```bash
/tmp/rq2-tectonic.grZsvV/tectonic rq2-split-table-test.tex
```

Result: exit 0; Tectonic wrote `rq2-split-table-test.pdf` (45.26 KiB). The
log contained no overfull or underfull box warning. The rendered table page is
preserved as `verification/rq2-split-table-test-page2.png`.

The test uses the paper's `acmsmall` class, top-matter configuration,
abbreviations, `\small` table font, and the generated table file directly.

## Full-paper check

Command, from `paper/`:

```bash
/tmp/rq2-tectonic.grZsvV/tectonic -Z shell-escape-cwd=. main.tex
```

The build parsed the generated RQ2 table without a table-specific layout
warning, then stopped at a later, unrelated input:

```text
error: sections/results:171: ! LaTeX Error: File `table/rq5-knowledge-judges.tex' not found.
```

At verification time that RQ5 file was already staged for deletion in the
paper worktree. This change does not restore or otherwise modify it.
