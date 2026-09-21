#!/usr/bin/env python3
"""Check the standalone draft against its captured API bibliography records."""

import json
import re
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent


def keys(text):
    return re.findall(r"@\w+\{([^,]+),", text)


def normalize(value):
    value = unicodedata.normalize("NFKD", value.casefold())
    return re.sub(r"[^a-z0-9]+", "", "".join(c for c in value if not unicodedata.combining(c)))


def bib_field(entry, name):
    match = re.search(r"\b" + name + r"\s*=\s*\{([^}]*)\}", entry, re.I)
    return match.group(1) if match else ""


def main():
    draft = (REPO / "draft.tex").read_text()
    extra = (REPO / "draft-rw-extra.bib").read_text()
    first = re.search(r"\\begin\{filecontents\*\}\{draft-api.bib\}(.*?)\\end\{filecontents\*\}", draft, re.S)
    assert first, "original embedded API bibliography missing"
    original_keys = keys(first.group(1))
    extra_keys = keys(extra)
    assert len(original_keys) == 18, f"original entry count: {len(original_keys)}"
    assert len(extra_keys) == 14, f"extra entry count: {len(extra_keys)}"
    assert len(set(original_keys + extra_keys)) == 32, "duplicate BibTeX keys"
    original_snapshot = json.loads((ROOT / "citation_metadata_check.json").read_text())
    assert len(original_snapshot["records"]) == 18
    original_records = {r["id"]: r["source_record"] for r in original_snapshot["records"]}
    original_chunks = re.split(r"(?=% citation_id: CIT-\d+)", first.group(1))
    checked_original = 0
    for chunk in original_chunks:
        match = re.search(r"% citation_id: (CIT-\d+)", chunk)
        if not match:
            continue
        citation_id = match.group(1)
        source = original_records[citation_id]
        assert len(keys(chunk)) == 1, f"entry missing for {citation_id}"
        assert normalize(bib_field(chunk, "title")) == normalize(source["title"]), f"title mismatch: {citation_id}"
        assert bib_field(chunk, "year") == str(source["year"]), f"year mismatch: {citation_id}"
        authors = []
        for author in bib_field(chunk, "author").split(" and "):
            if "," in author:
                family, given = author.split(",", 1)
                authors.append(given.strip() + " " + family.strip())
            else:
                authors.append(author.strip())
        assert [normalize(a) for a in authors] == [normalize(a) for a in source["authors"]], f"authors mismatch: {citation_id}"
        if "doi" in source:
            assert normalize(bib_field(chunk, "doi")) == normalize(source["doi"]), f"DOI mismatch: {citation_id}"
            assert bib_field(chunk, "journal") or bib_field(chunk, "booktitle"), f"venue missing: {citation_id}"
            if source["pages"]:
                assert normalize(bib_field(chunk, "pages")) == normalize(source["pages"]), f"pages mismatch: {citation_id}"
        else:
            assert bib_field(chunk, "eprint") == source["arxiv_id"], f"arXiv ID mismatch: {citation_id}"
        checked_original += 1
    assert checked_original == 18
    nocite = re.search(r"\\nocite\{([^}]*)\}", draft, re.S)
    assert nocite, "nocite list missing"
    cited = [k.strip() for k in nocite.group(1).split(",") if k.strip()]
    assert len(cited) == 32 and set(cited) == set(original_keys + extra_keys), "citation list does not match entries"
    assert r"\bibliography{draft-api,draft-rw-extra}" in draft
    snapshot = json.loads((ROOT / "rw_draft_extra_api_records.json").read_text())
    records = snapshot["records"]
    assert len(records) == 14
    expected_extra = "\n\n".join(f"% {r['id']} | API: {r.get('bibtex_api', r['metadata_api'])}\n{r['bibtex']}" for r in records) + "\n"
    assert extra == expected_extra, "extra BibTeX does not match API snapshot"
    generated_rows = re.search(r"% BEGIN API RW EXTRA ROWS(.*?)% END API RW EXTRA ROWS", draft, re.S)
    assert generated_rows, "generated citation-map rows missing"
    row_ids = re.findall(r"\\texttt\{(E\d+)\}", generated_rows.group(1))
    assert row_ids == [r["id"] for r in records], "citation-map rows do not match API record order"
    for r in records:
        assert r["metadata_api"].startswith(("https://api.crossref.org/works/", "https://export.arxiv.org/api/query?"))
        assert normalize(bib_field(r["bibtex"], "title")) == normalize(r["title"]), f"title differs from API: {r['id']}"
        authors = []
        for author in bib_field(r["bibtex"], "author").split(" and "):
            if "," in author:
                family, given = author.split(",", 1)
                authors.append(given.strip() + " " + family.strip())
            else:
                authors.append(author.strip())
        assert [normalize(a) for a in authors] == [normalize(a) for a in r["authors"]], f"authors differ from API: {r['id']}"
        if r["kind"] == "Crossref":
            assert re.search(r"\byear=\{" + str(r["year"]) + r"\}", r["bibtex"]), f"year differs from API: {r['id']}"
            assert r["doi"].casefold() in r["bibtex"].casefold(), f"DOI missing: {r['id']}"
        else:
            assert f"eprint = {{{r['arxiv_id']}}}" in r["bibtex"], f"arXiv ID missing: {r['id']}"
    print("PASS: 18 original + 14 new API entries; 32 unique keys and citations")
    print("PASS: 18 original entries match individual API title, ordered authors, year, DOI/arXiv ID, and pages where supplied")
    print("PASS: 14 extra BibTeX entries byte-match the API snapshot; titles and IDs have matching draft rows")
    print("PASS: new titles, ordered authors, identifiers, and publication years match their individual API records")


if __name__ == "__main__":
    main()
