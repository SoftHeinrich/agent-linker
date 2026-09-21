# Standalone related-work draft: API metadata and verification

Date: 2026-09-21 UTC. Scope: root `draft.tex` and its new `draft-rw-extra.bib`; no paper-submodule source was changed for this task.

## Configuration and source rule

The draft contains 32 bibliography items: the original 18 plus 14 distinct items selected by evidence ID from `reviewer_paper_evidence.csv` (E007, E009, E010, E022, E030, E031, E039, E040, E041, E059, E060, E062, E063, E065). The sheet supplied IDs and DOI/arXiv locators only. For each new item, the command fetched its own Crossref JSON and BibTeX transform or arXiv Atom record. The snapshot in `research/fse-2027-reviewer-audit/rw_draft_extra_api_records.json` includes the source URL, returned metadata, and bibliography entry for every item. The original 18 were fetched independently into `citation_metadata_check.json`.

Crossref's transform sometimes uses the online year while its JSON record supplies a later print or published year. The draft uses the JSON year for E007 (2010), E041 (2013), E059 (2024), and E063 (2024); both API values remain in the snapshot. Local BibTeX keys were changed where the transform key collided with an existing key or carried the earlier year. Other bibliography fields come from the API response.

## Commands and text results

```text
$ python3 research/fse-2027-reviewer-audit/fact_check_metadata.py
Result: 18 records fetched; 8 inventory rows with title, author, or year differences.
CIT-02: authors
CIT-04: authors
CIT-09: authors
CIT-10: authors
CIT-15: authors
CIT-16: authors
CIT-17: authors
CIT-18: authors

$ python3 research/fse-2027-reviewer-audit/expand_rw_draft.py
Result: 14 API records fetched; 10 Crossref and 4 arXiv; all BibTeX keys unique.

$ python3 research/fse-2027-reviewer-audit/sync_rw_draft.py
PASS: draft.tex cites 32 unique API-sourced entries; 14 added rows

$ python3 research/fse-2027-reviewer-audit/verify_rw_draft.py
PASS: 18 original + 14 new API entries; 32 unique keys and citations
PASS: 18 original entries match individual API title, ordered authors, year, DOI/arXiv ID, and pages where supplied
PASS: 14 extra BibTeX entries byte-match the API snapshot; titles and IDs have matching draft rows
PASS: new titles, ordered authors, identifiers, and publication years match their individual API records

$ python3 research/fse-2027-reviewer-audit/verify.py
PASS offline: 461 roster rows; 461 unique official profiles
PASS offline: role counts Chair=2, Area Chair=19, Member=440
PASS offline: 33 shortlisted reviewers across C1-C5
PASS offline: 67 paper-evidence rows; counts and score aggregation checked
PASS offline: 18 citation candidates; PC-author and HTTPS checks passed
PASS offline: local BibTeX PRESENT/ABSENT statuses and conflict flag checked

$ git -C paper diff --exit-code -- agent-linker.bib sections/rw.tex
PASS: zero diff
```

The eight differences concern the pre-existing search inventory, not the API-derived bibliography entries; the inventory was left unchanged. Crossref abbreviates the authors of CIT-18, so that full-name difference cannot be resolved from its record alone. The full per-item comparison is in `citation_metadata_check.md`.

## Build limitation

Command/configuration: `latexmk -pdf -interaction=nonstopmode -halt-on-error draft.tex` from the repository root. Result: exit 127, `BLOCKED: latexmk is not installed; draft.tex could not be compiled.` The text output is preserved in `verification/2026-09-21-rw-draft-build.log`. Static citation and metadata verification passed, but rendered layout remains unchecked.
